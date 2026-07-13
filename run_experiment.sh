#!/bin/bash
#SBATCH --job-name=gastrodino_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm_%j.out
#SBATCH --error=outputs/logs/slurm_%j.err

set -euo pipefail

CONFIG=$1
EXP_NAME=$(basename "$CONFIG" .yaml)
LOG_DIR="outputs/logs/${EXP_NAME}"

mkdir -p outputs/logs
mkdir -p "$LOG_DIR"

# SBATCH --output/--error above can only use SLURM's %j (job id), not
# $EXP_NAME, since they're resolved at submission time before this script
# runs. So SLURM always writes them flat into outputs/logs/ first; this
# relocates them into the per-experiment folder. Registered as a trap so
# it still runs even if training below fails/crashes.
cleanup() {
    mv "outputs/logs/slurm_${SLURM_JOB_ID}.out" "$LOG_DIR/slurm_${SLURM_JOB_ID}.out" 2>/dev/null || true
    mv "outputs/logs/slurm_${SLURM_JOB_ID}.err" "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" 2>/dev/null || true
}
trap cleanup EXIT

source .venv/bin/activate

echo "================================"
echo "Config: $CONFIG"
echo "Experiment: $EXP_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"
python --version
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true
echo "================================"

python -u -m scripts.train --config "$CONFIG" \
  2>&1 | tee "$LOG_DIR/train_${SLURM_JOB_ID}.log"

echo "================================"
echo "End: $(date)"
echo "================================"
