#!/bin/bash
#SBATCH --job-name=comp3710-all
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm-all-%j.out
#SBATCH --error=slurm-all-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
  echo "ERROR: 找不到项目虚拟环境 .venv/bin/activate"
  exit 1
fi
source .venv/bin/activate

export PYTHONUNBUFFERED=1

OASIS_DATA="/home/groups/comp3710/OASIS"
if [ ! -d "$OASIS_DATA" ]; then
  echo "ERROR: 找不到 OASIS 数据集：$OASIS_DATA"
  exit 1
fi

ARCHIVE_DIR="archives/pre_full_run_${SLURM_JOB_ID}"
mkdir -p "$ARCHIVE_DIR"
if [ -d "outputs" ]; then
  cp -a outputs "$ARCHIVE_DIR/"
fi
if [ -d "checkpoints" ]; then
  cp -a checkpoints "$ARCHIVE_DIR/"
fi

FULL_RUN_START=$(date +%s)

echo "============================================================"
echo "COMP3710 Lab 2 full run"
echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
echo "OASIS: $OASIS_DATA"
echo "Previous results archived at: $ARCHIVE_DIR"
echo "============================================================"
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo
echo "[1/7] Part 1: Fourier series, NumPy DFT/FFT and PyTorch DFT"
python scripts/part1_dft.py

echo
echo "[2/7] Part 2: PCA eigenfaces and Random Forest"
python scripts/part2_eigenfaces.py

echo
echo "[3/7] Part 3.1: LFW CNN"
python scripts/part3_lfw_cnn.py

echo
echo "[4/7] Part 3.2: CIFAR10 ResNet-18 final configuration"
python scripts/part3_cifar_resnet18.py \
  --download \
  --epochs 80 \
  --batch-size 512 \
  --lr 0.4 \
  --scheduler cosine \
  --random-erasing-prob 0.25 \
  --label-smoothing 0.1 \
  --amp \
  --channels-last \
  --experiment-name "full_run_${SLURM_JOB_ID}" \
  --stage-note "Two-hour complete Lab 2 verification run"

echo
echo "[5/7] Part 4.1: VAE"
python scripts/part4_vae.py \
  --data "$OASIS_DATA" \
  --epochs 50 \
  --batch-size 128 \
  --image-size 128 \
  --latent-dim 2

echo
echo "[6/7] Part 4.2: UNet segmentation"
python scripts/part4_unet.py \
  --data "$OASIS_DATA" \
  --epochs 80 \
  --batch-size 32 \
  --image-size 128

echo
echo "[7/7] Part 4.3: GAN"
python scripts/part4_gan.py \
  --data "$OASIS_DATA" \
  --epochs 80 \
  --batch-size 128 \
  --image-size 64

FULL_RUN_END=$(date +%s)
FULL_RUN_SECONDS=$((FULL_RUN_END - FULL_RUN_START))

echo
echo "============================================================"
echo "All seven tasks completed successfully."
echo "Finished: $(date)"
echo "Total seconds: $FULL_RUN_SECONDS"
echo "Outputs: $SLURM_SUBMIT_DIR/outputs"
echo "Checkpoints: $SLURM_SUBMIT_DIR/checkpoints"
echo "Individual timestamped logs: $SLURM_SUBMIT_DIR/outputs/*/logs"
echo "Previous results backup: $SLURM_SUBMIT_DIR/$ARCHIVE_DIR"
echo "============================================================"
