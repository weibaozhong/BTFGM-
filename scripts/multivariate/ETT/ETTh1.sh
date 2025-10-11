# export CUDA_VISIBLE_DEVICES=1

model_name=BTFGM

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path yss.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data custom\
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 4 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 13 \
  --dec_in 13 \
  --c_out 13 \
  --d_model 512 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path yss.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data custom\
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 8 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 13 \
  --dec_in 13 \
  --c_out 13 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path yss.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data custom\
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 14 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 13 \
  --dec_in 13 \
  --c_out 13 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 32 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path yss.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data custom\
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 30 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 13 \
  --dec_in 13 \
  --c_out 13 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 32 \
  --itr 1
