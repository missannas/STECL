#!/usr/bin/env bash

python main.py --task aste \
            --dataset rest14 \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --element all \
            --train_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 30 \
            --cl=True \
            --T=0.07 
