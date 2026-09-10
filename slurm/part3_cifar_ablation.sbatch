#!/bin/bash
#SBATCH --job-name=comp3710-cifar-ablation
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-cifar-ablation-%j.out
#SBATCH --error=slurm-cifar-ablation-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "HOST:"
hostname
echo "GPU:"
nvidia-smi || true
echo "START:"
date

echo "Stage 1: smaller ResNet, no augmentation, no LR scheduler"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name baseline_no_aug \
  --stage-index 1 \
  --stage-note "Small model, no augmentation, and no learning-rate schedule." \
  --epochs 20 \
  --batch-size 512 \
  --lr 0.1 \
  --width 32 \
  --label-smoothing 0.0 \
  --no-augment \
  --scheduler none

echo "Stage 2: full-width ResNet with crop/flip augmentation and cosine LR"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name add_aug_cosine \
  --stage-index 2 \
  --stage-note "Increase width and add crop/flip augmentation with cosine LR." \
  --epochs 40 \
  --batch-size 512 \
  --lr 0.2 \
  --width 64 \
  --label-smoothing 0.0 \
  --random-erasing-prob 0.0 \
  --scheduler cosine

echo "Stage 3: add label smoothing and random erasing regularisation"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name regularized \
  --stage-index 3 \
  --stage-note "Add label smoothing and random erasing regularisation." \
  --epochs 60 \
  --batch-size 512 \
  --lr 0.3 \
  --width 64 \
  --label-smoothing 0.1 \
  --random-erasing-prob 0.25 \
  --scheduler cosine

echo "Stage 4: final A100 speed run with AMP and channels-last tensors"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name final_amp_channels_last \
  --stage-index 4 \
  --stage-note "Add AMP and channels-last format for faster A100 training." \
  --epochs 80 \
  --batch-size 512 \
  --lr 0.4 \
  --width 64 \
  --label-smoothing 0.1 \
  --random-erasing-prob 0.25 \
  --scheduler cosine \
  --amp \
  --channels-last

python scripts/summarize_part3_experiments.py

echo "END:"
date
