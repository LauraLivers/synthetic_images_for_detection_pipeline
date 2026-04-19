#!/bin/bash
BASE=$(cd "$(dirname "$0")/.." && pwd)
MOBI_ROOT=$BASE/MobI
MY_FOLDER=$BASE/Laura
LOG_DIR=$BASE/MoBI_outputs/mobi_inpaining1
PRETRAINED=$BASE/MoBI_outputs/checkpoints/model.ckpt
CONFIG=$MY_FOLDER/ladder_dataset_mobi.yaml

git clone https://github.com/CompVis/taming-transformers.git $BASE/taming-transformers 2>/dev/null || true

export PYTHONPATH=$MOBI_ROOT:$MY_FOLDER:$BASE/taming-transformers:$PYTHONPATH

uv run --active python $MOBI_ROOT/main.py \
  --logdir $LOG_DIR \
  --pretrained_model $PRETRAINED \
  --base $CONFIG \
  --scale_lr False \
  --save_top_k 5