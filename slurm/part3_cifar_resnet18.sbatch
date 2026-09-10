#!/bin/bash
#SBATCH --job-name=comp3710-cifar
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-cifar-%j.out
#SBATCH --error=slurm-cifar-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name final_amp_channels_last \
  --epochs 80 \
  --batch-size 512 \
  --lr 0.4 \
  --amp \
  --channels-last
