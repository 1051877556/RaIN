### M
                                                          #从实验结果来看，此处参数应该修改为--seq_len 96 --label_len 48 --pred_len 24,用模型所给的参数效果很差
python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 672 --label_len 96 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

### S

python -u main_informer.py --model informer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

python -u main_informer.py --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5
