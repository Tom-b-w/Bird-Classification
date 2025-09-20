#!/bin/bash

DATASET_ROOT='./birdclef-preprocess/birdclef-preprocess/birdclef2023-dataset'
MODEL_NAME='whisper'
SAVE_MODEL_ROOT='./Training_MODEL_NAME'
TRAINING_MODE='fine-tuning'



python main.py --model_name 'whisper' \
    --save_model_path  './Training_MODEL_NAME'\
    --dataset_root './birdclef-preprocess/birdclef-preprocess/birdclef2023-dataset' \
    --training_mode 'fine-tuning' \
    --augmented_run \
    --spec_aug \
    --n_epochs 20 \
    --start_epoch 0 \
    --batch_size 16 \
    --num_workers 4 \
    --lr 0.01 \
    --seed 42 \
    --do_logging\
    --eval_only False