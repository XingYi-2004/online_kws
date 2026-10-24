#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4

python train_ann.py \
    --world-size 1 \
    --num-epochs 20 \
    --num-class 29 \
    --max-duration 200
