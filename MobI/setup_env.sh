#!/usr/bin/env bash
# MobI environment setup with uv
# Usage: bash setup_env.sh
set -e

PYTHON_VERSION="3.8"

echo "=== MobI Environment Setup ==="

# Detect platform
if [[ "$(uname)" == "Darwin" ]]; then
    REQUIREMENTS="requirements-uv.txt"
    PLATFORM="macos"
    echo "Platform: macOS"
elif [[ "$(uname)" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        REQUIREMENTS="requirements-linux-cuda.txt"
        PLATFORM="linux-cuda"
        echo "Platform: Linux with CUDA"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    else
        REQUIREMENTS="requirements-uv.txt"
        PLATFORM="linux"
        echo "Platform: Linux (no CUDA)"
    fi
else
    echo "Unknown platform: $(uname)"
    exit 1
fi

# Create venv
echo ""
echo "Creating venv with Python ${PYTHON_VERSION}..."
uv venv --python "${PYTHON_VERSION}"

# Install dependencies
echo ""
echo "Installing dependencies from ${REQUIREMENTS}..."
uv pip install -r "${REQUIREMENTS}"

# Install taming-transformers
echo ""
echo "Installing taming-transformers..."
uv pip install "git+https://github.com/CompVis/taming-transformers.git@master"

# Install MobI itself (editable)
echo ""
echo "Installing MobI (editable)..."
uv pip install -e .

# BEVFusion (Linux+CUDA only)
if [[ "${PLATFORM}" == "linux-cuda" ]]; then
    echo ""
    echo "=== BEVFusion Setup ==="
    echo "To complete BEVFusion setup, run:"
    echo "  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html --no-cache-dir"
    echo "  cd bevfusion && python setup.py develop"
fi

echo ""
echo "=== Done ==="
echo "Activate with: source .venv/bin/activate"
