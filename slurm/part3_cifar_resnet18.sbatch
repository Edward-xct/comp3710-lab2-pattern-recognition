#!/bin/bash
#SBATCH --job-name=comp3710-cifar
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-cifar-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load python || true
module load cuda || true

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python scripts/part3_cifar_resnet18.py \
  --download \
  --epochs 80 \
  --batch-size 512 \
  --lr 0.4 \
  --amp \
  --channels-last
