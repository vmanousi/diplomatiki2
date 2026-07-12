#!/bin/bash
# Creates .venv and installs dependencies, picking the right PyTorch build
# for the machine: CPU wheels if no NVIDIA GPU is detected, CUDA wheels
# otherwise. src/utils/device.py then selects cuda/cpu automatically at
# runtime, so the same codebase runs on a CPU-only laptop or a GPU cluster
# node without any config changes.
#
# Usage: scripts/setup_env.sh

set -euo pipefail

PYTHON=${PYTHON:-python3}

"$PYTHON" -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "GPU detected: installing CUDA-enabled PyTorch"
    pip install torch torchvision
else
    echo "No GPU detected: installing CPU-only PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Environment ready in .venv/"
