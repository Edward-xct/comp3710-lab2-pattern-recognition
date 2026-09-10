#!/bin/bash
#SBATCH --job-name=comp3710-gan
# Part 4 Task 3 GAN 正式训练：用 A100 GPU 训练 DCGAN 生成 OASIS MRI。
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-gan-%j.out
#SBATCH --error=slurm-gan-%j.err

set -euo pipefail

# 从 sbatch 提交目录开始运行，输出文件留在当前项目。
cd "$SLURM_SUBMIT_DIR"

if [ -d ".venv" ]; then
  # 激活项目环境，保证 torch/torchvision 版本可用。
  source .venv/bin/activate
fi

DATA_PATH=${DATA_PATH:-}
if [ -z "$DATA_PATH" ]; then
  # GAN 只需要 MRI 图像目录，但完整 OASIS 根目录也能被 Python Dataset 自动解析。
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
  # 自动定位失败时打印共享目录，方便确认老师更新的数据路径是否变化。
  echo "Could not find OASIS data automatically."
  echo "Check Blackboard Demo 2 folder for the updated FileSender link, or inspect /home/groups/comp3710 on Rangpur."
  ls -la /home/groups/comp3710 || true
  exit 1
fi

echo "Using OASIS data: $DATA_PATH"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# 训练 DCGAN：64x64 输入，保存 loss 曲线和每 5 个 epoch 的生成样本。
python scripts/part4_gan.py \
  --data "$DATA_PATH" \
  --epochs 80 \
  --batch-size 128 \
  --image-size 64
