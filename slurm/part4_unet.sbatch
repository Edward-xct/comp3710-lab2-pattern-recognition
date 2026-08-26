#!/bin/bash
#SBATCH --job-name=comp3710-unet
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
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

python scripts/part4_unet.py \
  --data "$DATA_PATH" \
  --epochs 80 \
  --batch-size 32 \
  --image-size 128
