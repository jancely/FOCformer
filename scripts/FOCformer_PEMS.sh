export CUDA_VISIBLE_DEVICES=0

model_name=FOCformer


python -u run.py \
  --is_training 1 \
  --order 0.7 \
  --task_name long_term_forecast \
  --root_path ./data/PEMS/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --use_norm 1 \
  --learning_rate 0.0005 \
  --dropout 0.05

#el 3  lr0.0005  od 0.7

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.5 \
  --root_path ./data/PEMS/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --use_norm 1 \
  --learning_rate 0.0005 \
  --dropout 0.01
#el 3  lr 0.0005 od 0.4

python -u run.py \
  --is_training 1 \
  --order $od \
  --task_name long_term_forecast \
  --root_path ./data/PEMS/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_48 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16\
  --learning_rate 0.001 \
  --itr 1 \
  --use_norm 0 \
  --dropout 0.05
#el 2  lr 0.001 od 0.5

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.5 \
  --root_path /./data/PEMS/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16\
  --learning_rate 0.0005 \
  --itr 1 \
  --use_norm 0 \
  --dropout 0.0004

