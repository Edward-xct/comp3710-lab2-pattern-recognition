#!/bin/bash
#SBATCH --job-name=comp3710-unet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-unet-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load python || true
module load cuda || true

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

DATA_PATH=${DATA_PATH:-/home/groups/comp3710/keras_png_slices_data}

python scripts/part4_unet.py \
  --data "$DATA_PATH" \
  --epochs 80 \
  --batch-size 32 \
  --image-size 128
