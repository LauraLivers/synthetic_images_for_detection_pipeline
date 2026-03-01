#!/usr/bin/env bash
# DiffCL environment setup with uv
# Usage: bash setup_env.sh
set -e

PYTHON_VERSION="3.10"

echo "=== DiffCL Environment Setup ==="

# Detect platform
if [[ "$(uname)" == "Darwin" ]]; then
    REQUIREMENTS="requirements.txt"
    echo "Platform: macOS"
elif [[ "$(uname)" == "Linux" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        REQUIREMENTS="requirements-linux-cuda.txt"
        echo "Platform: Linux with CUDA"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    else
        REQUIREMENTS="requirements.txt"
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

echo ""
echo "=== Done ==="
echo "Activate with: source .venv/bin/activate"
