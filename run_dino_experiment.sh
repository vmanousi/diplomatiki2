#!/bin/bash

#SBATCH --job-name=dino_gi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err

set -euo pipefail

# Accepted usages:
#
# New experiment:
#   sbatch run_dino_experiment.sh CONFIG.yaml
#
# Resume experiment:
#   sbatch run_dino_experiment.sh CONFIG.yaml CHECKPOINT.pt

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage:"
    echo "  New run:"
    echo "    sbatch run_dino_experiment.sh CONFIG.yaml"
    echo
    echo "  Resume run:"
    echo "    sbatch run_dino_experiment.sh CONFIG.yaml CHECKPOINT.pt"
    exit 1
fi

CONFIG="$1"
RESUME_CHECKPOINT="${2:-}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: configuration file not found:"
    echo "  $CONFIG"
    exit 1
fi

if [ -n "$RESUME_CHECKPOINT" ] && [ ! -f "$RESUME_CHECKPOINT" ]; then
    echo "Error: resume checkpoint not found:"
    echo "  $RESUME_CHECKPOINT"
    exit 1
fi

EXP_NAME="$(basename "$CONFIG" .yaml)"
LOG_DIR="outputs/logs/${EXP_NAME}"

mkdir -p outputs/logs
mkdir -p "$LOG_DIR"

cleanup() {
    mv "outputs/logs/slurm_${SLURM_JOB_ID}.out" \
       "$LOG_DIR/slurm_${SLURM_JOB_ID}.out" \
       2>/dev/null || true

    mv "outputs/logs/slurm_${SLURM_JOB_ID}.err" \
       "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" \
       2>/dev/null || true
}

trap cleanup EXIT

source .venv/bin/activate

echo "================================="
echo "DINO self-supervised pretraining"
echo "Config: $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"
python --version
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi || true

if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Mode: resume training"
    echo "Resume checkpoint: $RESUME_CHECKPOINT"
else
    echo "Mode: new training"
fi

echo "================================="

if [ -n "$RESUME_CHECKPOINT" ]; then
    python -u -m scripts.pretrain_dino \
        --config "$CONFIG" \
        --resume "$RESUME_CHECKPOINT" \
        2>&1 | tee "$LOG_DIR/dino_${SLURM_JOB_ID}.log"
else
    python -u -m scripts.pretrain_dino \
        --config "$CONFIG" \
        2>&1 | tee "$LOG_DIR/dino_${SLURM_JOB_ID}.log"
fi

echo "================================="
echo "End: $(date)"
echo "================================="
