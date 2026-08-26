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

DATA_PATH=${DATA_PATH:-}
if [ -z "$DATA_PATH" ]; then
  for candidate in \
    /home/groups/comp3710/OASIS \
    /home/groups/comp3710/keras_png_slices_data \
    /home/groups/comp3710/keras_png_slices_data.zip \
    "$SLURM_SUBMIT_DIR/data/keras_png_slices_data" \
    "$SLURM_SUBMIT_DIR/data/keras_png_slices_data.zip"; do
    if [ -e "$candidate" ]; then
      DATA_PATH="$candidate"
      break
    fi
  done
fi

if [ -z "$DATA_PATH" ]; then
  echo "Could not find OASIS data automatically."
  echo "Check Blackboard Demo 2 folder for the updated FileSender link, or inspect /home/groups/comp3710 on Rangpur."
  ls -la /home/groups/comp3710 || true
  exit 1
fi

echo "Using OASIS data: $DATA_PATH"

python scripts/part4_vae.py \
  --data "$DATA_PATH" \
  --epochs 50 \
  --batch-size 128 \
  --image-size 128 \
  --latent-dim 2
