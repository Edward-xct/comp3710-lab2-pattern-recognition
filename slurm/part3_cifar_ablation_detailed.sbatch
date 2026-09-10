#!/bin/bash
#SBATCH --job-name=comp3710-cifar-detail
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-cifar-detail-%j.out
#SBATCH --error=slurm-cifar-detail-%j.err

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

echo "Stage 1: tiny baseline, short training, no augmentation"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name s01_tiny_baseline \
  --stage-index 1 \
  --stage-note "Small model, 10 epochs, no augmentation; establishes a weak but honest baseline." \
  --epochs 10 \
  --batch-size 512 \
  --lr 0.1 \
  --width 32 \
  --label-smoothing 0.0 \
  --no-augment \
  --scheduler none

echo "Stage 2: train the same baseline for longer"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name s02_longer_baseline \
  --stage-index 2 \
  --stage-note "Same small model and no augmentation, but 20 epochs instead of 10." \
  --epochs 20 \
  --batch-size 512 \
  --lr 0.1 \
  --width 32 \
  --label-smoothing 0.0 \
  --no-augment \
  --scheduler none

echo "Stage 3: use a full-width ResNet but still no augmentation"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name s03_wider_no_aug \
  --stage-index 3 \
  --stage-note "Increase model width from 32 to 64 while keeping the data pipeline simple." \
  --epochs 20 \
  --batch-size 512 \
  --lr 0.1 \
  --width 64 \
  --label-smoothing 0.0 \
  --no-augment \
  --scheduler none

echo "Stage 4: add random crop and horizontal flip"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name s04_add_crop_flip \
  --stage-index 4 \
  --stage-note "Add CIFAR10 crop/flip augmentation; no random erasing or label smoothing yet." \
  --epochs 25 \
  --batch-size 512 \
  --lr 0.1 \
  --width 64 \
  --label-smoothing 0.0 \
  --random-erasing-prob 0.0 \
  --scheduler none

echo "Stage 5: add cosine learning-rate schedule"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name s05_add_cosine_lr \
  --stage-index 5 \
  --stage-note "Keep crop/flip augmentation and add cosine LR decay for more stable convergence." \
  --epochs 35 \
  --batch-size 512 \
  --lr 0.2 \
  --width 64 \
  --label-smoothing 0.0 \
  --random-erasing-prob 0.0 \
  --scheduler cosine

echo "Stage 6: add label smoothing and random erasing"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name s06_regularized \
  --stage-index 6 \
  --stage-note "Add label smoothing and random erasing to improve generalisation." \
  --epochs 60 \
  --batch-size 512 \
  --lr 0.3 \
  --width 64 \
  --label-smoothing 0.1 \
  --random-erasing-prob 0.25 \
  --scheduler cosine

echo "Stage 7: final A100 speed run with AMP and channels-last tensors"
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name s07_final_amp_channels_last \
  --stage-index 7 \
  --stage-note "Use the best regularised recipe and add AMP plus channels-last format for A100 speed." \
  --epochs 80 \
  --batch-size 512 \
  --lr 0.4 \
  --width 64 \
  --label-smoothing 0.1 \
  --random-erasing-prob 0.25 \
  --scheduler cosine \
  --amp \
  --channels-last

python scripts/summarize_part3_experiments.py --name-prefix s

echo "END:"
date
