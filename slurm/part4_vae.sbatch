#!/bin/bash
#SBATCH --job-name=comp3710-vae
# Part 4.1 / Task 1 VAE 正式训练：在 comp3710 分区申请 1 张 A100 GPU。
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-vae-%j.out
#SBATCH --error=slurm-vae-%j.err

set -euo pipefail

# 从提交 sbatch 的项目根目录运行，确保 outputs/checkpoints 写回本项目。
cd "$SLURM_SUBMIT_DIR"

if [ -d ".venv" ]; then
  # 使用 Rangpur 项目里的 .venv，避免系统 Python 缺少 torch。
  source .venv/bin/activate
fi

DATA_PATH=${DATA_PATH:-}
if [ -z "$DATA_PATH" ]; then
  # 优先使用 Rangpur 共享 OASIS 数据，找不到再尝试项目 data/ 里的本地副本或 zip。
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
  # 如果自动定位失败，就打印共享目录信息，方便现场快速排查路径问题。
  echo "Could not find OASIS data automatically."
  echo "Check Blackboard Demo 2 folder for the updated FileSender link, or inspect /home/groups/comp3710 on Rangpur."
  ls -la /home/groups/comp3710 || true
  exit 1
fi

echo "Using OASIS data: $DATA_PATH"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# 训练 VAE：128x128 输入、2 维 latent space，输出 reconstruction 和 manifold 图片。
python scripts/part4_vae.py \
  --data "$DATA_PATH" \
  --epochs 50 \
  --batch-size 128 \
  --image-size 128 \
  --latent-dim 2
