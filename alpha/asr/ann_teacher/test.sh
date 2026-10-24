#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4

# ANN decode
python decode_ann.py \
    --epoch 17 \
    --avg 9 \
    --num-class 29 \
    --max-duration 200 \
    --export True

# SNN decode
# python decode_snn.py \
#     --epoch 25 \
#     --avg 5 \
#     --num-class 29 \
#     --max-duration 200 \
#     --export True