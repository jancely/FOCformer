export CUDA_VISIBLE_DEVICES=0

model_name=FOCformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.5 \
  --root_path ./data/Flight \
  --data_path Flight.csv \
  --model_id Flight_$seq_len\_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --learning_rate 0.001


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.8 \
  --root_path ./data/Flight \
  --data_path Flight.csv \
  --model_id Flight_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --learning_rate 0.001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.3 \
  --root_path ./data/Flight \
  --data_path Flight.csv \
  --model_id Flight_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --learning_rate 0.001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.4 \
  --root_path ./data/Flight \
  --data_path Flight.csv \
  --model_id Flight_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --learning_rate 0.001

