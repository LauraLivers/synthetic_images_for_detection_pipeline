#!/bin/bash

export PYTHONPATH=/Users/laura/Desktop/ba_thesis/Laura:/Users/laura/Desktop/ba_thesis/MobI:/Users/laura/Desktop/ba_thesis/taming-transformers:$PYTHONPATH
# Set accelerator: 'mps' (Apple Silicon) or 'cuda' (NVIDIA)
# 'auto' will detect automatically but explicit is safer
# ACCELERATOR="auto"

uv run --active python /Users/laura/Desktop/ba_thesis/MobI/main.py \
    --logdir /Users/laura/Desktop/ba_thesis/MoBI_outputs/mobi_inpaining1 \
    --pretrained_model /Users/laura/Desktop/ba_thesis/MoBI_outputs/checkpoints/model.ckpt \
    --base /Users/laura/Desktop/ba_thesis/Laura/ladder_dataset_mobi.yaml \
    --scale_lr False \
    --save_top_k 5