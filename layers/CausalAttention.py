import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask
import math


class CausalAttention(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, order=0.2, scale=None, attention_dropout=0.1, output_attention=False):
        super(CausalAttention, self).__init__()
        self.factor = factor
        self.order = order
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)


    def causal(self, corr, order):
        length = corr.shape[-1] * corr.shape[-2]
        # print('length', length)
        w_sequence = [1.0]

        for i in range(1, length):
            w_j = (1 - (order + 1) / i) * w_sequence[-1]
            w_sequence.append(torch.abs(torch.tensor(w_j)))
            
        frac_array = torch.tensor(w_sequence, device=corr.device, dtype=corr.dtype)
        frac_array = frac_array.view(corr.shape[-2], corr.shape[-1]).unsqueeze(0).unsqueeze(0)

        casual_corr = torch.mul(corr, frac_array)

        return casual_corr

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :] 
        q_fft = torch.fft.rfftn(queries.contiguous(), dim=(-2, -1))
        k_fft = torch.fft.rfftn(keys.contiguous(), dim=(-2, -1))
        res = q_fft * torch.conj(k_fft)
        corr = torch.softmax(torch.fft.irfftn(res, s=(H, H), dim=(-2, -1)) * scale, dim=-1)
        
        # Causal Inference
        corr = self.causal(corr, self.order)

        V = corr @ values

        if self.output_attention:
            return (V.contiguous(), corr)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, data, c_in, n_heads, d_keys=None,
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

        if data =='Solar' or data =='PEMS':
            self.lnet = nn.Linear(c_in, 5)
            self.unet = nn.Linear(5, c_in)
        else:
            self.lnet = nn.Linear(c_in + 4, 5)
            self.unet = nn.Linear(5, c_in + 4)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        queries = self.lnet(queries.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        keys = self.lnet(keys.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        values = self.lnet(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )

        out = self.unet(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):         # [bs, nvars, d_model]
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def torch_diff(x, dim, n):
    out = torch.diff(torch.cat((x, x[:, :1]), dim=dim), dim=dim)
    if n > 1:
        for _ in range(n - 1):
            out = torch.diff(torch.cat((out, out[:, :1]), dim=dim), dim=dim)

    return out
