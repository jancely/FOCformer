export CUDA_VISIBLE_DEVICES=1

model_name=FOCformer


  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.3 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len\_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.0002

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.5 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.0002 \
  --dropout 0.5


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.2 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --itr 1 \
  --d_model 256 \
  --d_ff 256 \
  --learning_rate 0.0002 \
  --train_epochs 1 \
  --dropout 0.5


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.6 \
  --root_path ./data/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.002 \
  --train_epochs 1 \
  --dropout 0.5


