#!/bin/bash
#SBATCH --job-name=comp3710-vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-vae-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load python || true
module load cuda || true

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

DATA_PATH=${DATA_PATH:-/home/groups/comp3710/keras_png_slices_data}

python scripts/part4_vae.py \
  --data "$DATA_PATH" \
  --epochs 50 \
  --batch-size 128 \
  --image-size 128 \
  --latent-dim 2
