#!/bin/bash
#SBATCH --job-name=comp3710-unet
# Part 4 Task 2 UNet 正式训练：需要 A100 GPU 处理 OASIS segmentation。
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-unet-%j.out
#SBATCH --error=slurm-unet-%j.err

set -euo pipefail

# 保证所有相对路径都从项目根目录开始。
cd "$SLURM_SUBMIT_DIR"

if [ -d ".venv" ]; then
  # 激活包含 PyTorch 的虚拟环境。
  source .venv/bin/activate
fi

DATA_PATH=${DATA_PATH:-}
if [ -z "$DATA_PATH" ]; then
  # UNet 需要图片和 seg mask 两套目录，所以优先找完整 OASIS 根目录。
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
  # 找不到数据时直接失败，并给出下一步排查位置。
  echo "Could not find OASIS data automatically."
  echo "Check Blackboard Demo 2 folder for the updated FileSender link, or inspect /home/groups/comp3710 on Rangpur."
  ls -la /home/groups/comp3710 || true
  exit 1
fi

echo "Using OASIS data: $DATA_PATH"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# 训练 UNet：输入 MRI，输出 4 类 segmentation mask；正式结果 best mean DSC 已超过 0.9。
python scripts/part4_unet.py \
  --data "$DATA_PATH" \
  --epochs 80 \
  --batch-size 32 \
  --image-size 128
