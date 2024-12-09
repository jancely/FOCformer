import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from math import log2
from utils.masking import TriangularCausalMask, ProbMask, LogSparseMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import copy
import math

def softmax(x):
    """
    softmax function
    """
    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出
        x -= tmp.reshape((x.shape[0], 1))  # 利用性质缩放元素
        x = np.exp(x)  # 计算所有值的指数
        tmp = np.sum(x, axis=1)  # 每行求和
        x /= tmp.reshape((x.shape[0], 1))  # 求softmax
    else:
        # 向量
        tmp = np.max(x)  # 得到最大值
        x -= tmp  # 利用最大值缩放数据
        x = np.exp(x)  # 对所有元素求指数
        tmp = np.sum(x)  # 求元素和
        x /= tmp  # 求somftmax
    return x


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class MultiScaleAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(MultiScaleAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.attention_name = type(attention).__name__
        self.factor = attention.factor
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
            queries: (B, L_q, d_model)
            keys: (B, L_k, d_model)
            values: (B, L_v=L_k, d_model)
            attn_mask: (B, 1, L, L)
            return: (B, L_q, d_model)
        """
        B, L_q, _ = queries.shape
        _, L_k, _ = keys.shape
        H = self.n_heads
        L_min = min(L_q, L_k)
        scale_num = math.floor(log2(L_min // self.factor)) + 1
        attn_list = clones(self.inner_attention, scale_num-1)
        scale_weight = np.zeros(scale_num)
        for i in range(scale_num):
            scale_weight[i] = 1 / (2 ** i)
        scale_weight = scale_weight / sum(scale_weight)
        # scale_weight = softmax(scale_weight)
        # scale_weight[:] = 1 / scale_num

        queries = self.query_projection(queries).view(B, L_q, H, -1)  # (B, L_q, H, D_q)
        keys = self.key_projection(keys).view(B, L_k, H, -1)  # (B, L_k, H, D_k=D_q)
        values = self.value_projection(values).view(B, L_k, H, -1)  # (B, L_v=L_k, H, D_v)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out * scale_weight[0]

        head_flag = self.inner_attention.head_flag
        for i in range(1, scale_num):
            head_flag = not head_flag
            attn_list[i-1].factor = self.factor * (2 ** i)
            attn_list[i-1].head_flag = head_flag
            out1, _ = attn_list[i-1](
                queries,
                keys,
                values,
                attn_mask
            )
            out = out + out1 * scale_weight[i]
        out = out.view(B, L_q, -1)  # (B, L_q, n_heads * D_v)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out

class LogSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(LogSparseAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # attn_mask: the positions of elements which do not participate in calculation are True
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        assert L_k == L_v
        assert D_q == D_k
        scale = self.scale or 1. / sqrt(D_q)

        scores = torch.einsum("blhd,bshd->bhls", queries, keys)  # (B, H, L_q, L_k), softmax on the last dim

        if L_q == L_k:  # 说明是自注意
            logmask = LogSparseMask(B, H, L_q, L_k, device=queries.device)
            scores.masked_fill_(logmask.mask, -np.inf)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L_q, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)  # Fill elements of the tensor with -inf where mask is True

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)  # (B, L_q, H, D_v)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class SegmentCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        # factor is segment length
        super(SegmentCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag  # if True drop first else drop last

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)

        correlation_scores = torch.einsum("bmlhd,bnlhd->bhmnd", seg_queries, seg_keys)  # (b, h, 5, 3, d_q)
        A = torch.softmax(scale * correlation_scores, dim=-2)  # (b, h, 5, 3, d_q)
        V = torch.einsum("bhmnd,bnlhd->bmlhd", A, seg_values)  # (b, 5, l_s, h, d_v)

        V = V.reshape(B, -1, H, D_v)  # (b, l_q, h, d_v)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SegmentCorrelation2(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        # factor is segment length
        super(SegmentCorrelation2, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag  # if True drop first else drop last

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s * D_q)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)

        correlation_scores = torch.einsum("bmlhd,bnlhd->bhmn", seg_queries, seg_keys)  # (b, h, 5, 3)
        A = torch.softmax(scale * correlation_scores, dim=-1)  # (b, h, 5, 3)
        V = torch.einsum("bhmn,bnlhd->bmlhd", A, seg_values)  # (b, 5, l_s, h, d_v)

        V = V.reshape(B, -1, H, D_v)  # (b, l_q, h, d_v)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SegmentCorrelation3(nn.Module):
    # shift 1 segment
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        # factor is segment length
        super(SegmentCorrelation3, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag  # if True drop first else drop last

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s * D_q)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_keys_pre = seg_keys[:, :-1, ...]  # (b, 2, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)
        seg_values_aft = seg_values[:, 1:, ...]  # (b, 2, l_s, h, d_v)

        correlation_scores = torch.einsum("bmlhd,bnlhd->bhmn", seg_queries, seg_keys_pre)  # (b, h, 5, 2)
        A = torch.softmax(scale * correlation_scores, dim=-1)  # (b, h, 5, 2)
        tmp_V = torch.einsum("bhmn,bnlhd->bmlhd", A, seg_values_aft)  # (b, 5, l_s, h, d_v)
        V = torch.roll(tmp_V, shifts=1, dims=1)

        V = V.reshape(B, -1, H, D_v)  # (b, l_q, h, d_v)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SegmentCorrelation4(nn.Module):
    # shift 1 segment
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, flag=True):
        # factor is segment length
        super(SegmentCorrelation4, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.head_flag = flag  # if True drop first else drop last

    def forward(self, queries, keys, values, attn_mask):
        B, L_q, H, D_q = queries.shape
        _, L_k, _, D_k = keys.shape
        _, L_v, _, D_v = values.shape
        L_s = self.factor  # segment length
        scale = self.scale or 1. / sqrt(L_s * D_q)
        assert L_k == L_v
        assert D_q == D_k == D_v
        assert L_s <= L_q
        assert L_s <= L_v
        addition_len_q = L_q % L_s
        addition_len_v = L_v % L_s

        if self.head_flag:  # drop first
            queries = queries[:, addition_len_q:, ...]
            keys = keys[:, addition_len_v:, ...]
            values = values[:, addition_len_v:, ...]
            addition_Q = queries[:, :addition_len_q, ...] if addition_len_q != 0 else None
        else:  # drop last
            queries = queries[:, :-addition_len_q, ...] if addition_len_q != 0 else queries
            keys = keys[:, :-addition_len_v, ...] if addition_len_v != 0 else keys
            values = values[:, :-addition_len_v, ...] if addition_len_v != 0 else values
            addition_Q = queries[:, -addition_len_q:, ...] if addition_len_q != 0 else None

        seg_queries = queries.reshape(B, -1, L_s, H, D_q)  # (b, 5, l_s, h, d_q)
        seg_keys = keys.reshape(B, -1, L_s, H, D_q)  # (b, 3, l_s, h, d_q)
        seg_keys_pre = seg_keys[:, :-1, ...]  # (b, 2, l_s, h, d_q)
        seg_values = values.reshape(B, -1, L_s, H, D_v)  # (b, 3, l_s, h, d_v)
        seg_values_aft = seg_values[:, 1:, ...]  # (b, 2, l_s, h, d_v)

        correlation_scores = torch.einsum("bmlhd,bnlhd->bhmnd", seg_queries, seg_keys_pre)  # (b, h, 5, 2)
        A = torch.softmax(scale * correlation_scores, dim=-2)  # (b, h, 5, 2)
        tmp_V = torch.einsum("bhmnd,bnlhd->bmlhd", A, seg_values_aft)  # (b, 5, l_s, h, d_v)
        V = torch.roll(tmp_V, shifts=1, dims=1)

        V = V.reshape(B, -1, H, D_v)  # (b, l_q, h, d_v)
        if self.head_flag:
            if addition_Q is not None:
                V = torch.cat([addition_Q, V], 1)
        else:
            if addition_Q is not None:
                V = torch.cat([V, addition_Q], 1)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)