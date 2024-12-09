export CUDA_VISIBLE_DEVICES=1

model_name=FOCformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 1.0 \
  --root_path ./data/Weather/  \
  --data_path 2020.csv \
  --model_id weather_96_96_ \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.001 \
  --dropout 0.1
#e_layers 2 lr 0.0005


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.5 \
  --root_path ./data/Weather/  \
  --data_path 2020.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.002
#el 1 0.001

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.9 \
  --root_path ./data/Weather/  \
  --data_path 2020.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.0005 \
  --dropout 0.2
#el2 lr0.0005 0.264/0.288


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --order 0.9 \
  --root_path ./data/Weather/  \
  --data_path 2020.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.0005 \
  --dropout 0.3 \
