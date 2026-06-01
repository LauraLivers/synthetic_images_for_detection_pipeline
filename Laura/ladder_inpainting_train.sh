#!/bin/bash
set -e

BASE=$(cd "$(dirname "$0")/.." && pwd)
MOBI_ROOT=$BASE/MobI
MY_FOLDER=$BASE/Laura
LOG_DIR=$BASE/MoBI_outputs/mobi_inpainting2
CHECKPOINTS_DIR=$BASE/MoBI_outputs/checkpoints
PBE_REPO="Fantasy-Studio/Paint-by-Example"
PBE_FILENAME="model.ckpt"
PRETRAINED=$CHECKPOINTS_DIR/$PBE_FILENAME

CONFIG=$MY_FOLDER/ladder_dataset_mobi.yaml

if [ ! -f "$PRETRAINED" ]; then
    echo "PbE checkpoint not found at $PRETRAINED"
    echo "Downloading from HuggingFace: $PBE_REPO ..."
    mkdir -p "$CHECKPOINTS_DIR"
    uv run --active python - <<EOF
from huggingface_hub import hf_hub_download
import shutil, os

dest_dir = "$CHECKPOINTS_DIR"
os.makedirs(dest_dir, exist_ok=True)

print("Downloading $PBE_FILENAME from $PBE_REPO ...")
tmp = hf_hub_download(
    repo_id="$PBE_REPO",
    filename="$PBE_FILENAME",
    cache_dir=dest_dir + "/.cache",
)
dest = os.path.join(dest_dir, "$PBE_FILENAME")
shutil.copy2(tmp, dest)
print(f"Saved to {dest}")
EOF
    echo "Download complete."
else
    echo "PbE checkpoint already present at $PRETRAINED — skipping download."
fi

git clone https://github.com/CompVis/taming-transformers.git $BASE/taming-transformers 2>/dev/null || true

export PYTHONPATH=$MOBI_ROOT:$MY_FOLDER:$BASE/taming-transformers:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export PL_TORCH_DISTRIBUTED_BACKEND=gloo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --active python $MOBI_ROOT/main.py \
  --logdir $LOG_DIR \
  --pretrained_model $PRETRAINED \
  --base $CONFIG \
  --scale_lr False \
  --save_top_k 5