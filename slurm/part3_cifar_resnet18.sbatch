#!/bin/bash
#SBATCH --job-name=comp3710-cifar
# 使用 COMP3710 课程账户和课程分区，在 Rangpur 上申请 1 张 A100 GPU。
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
# 8 个 CPU worker 用于 DataLoader 读 CIFAR10，GPU 负责模型训练。
#SBATCH --cpus-per-task=8
# 12 小时是上限，实际最终 run 约几分钟完成；上限给排队系统预留足够时间。
#SBATCH --time=12:00:00
#SBATCH --output=slurm-cifar-%j.out
#SBATCH --error=slurm-cifar-%j.err

# 遇到错误立刻退出，避免脚本后续继续跑出误导性结果。
set -euo pipefail

# SLURM_SUBMIT_DIR 是执行 sbatch 时所在目录，保证相对路径从项目根目录开始。
cd "$SLURM_SUBMIT_DIR"

if [ -d ".venv" ]; then
  # 使用项目自己的 Python 环境，里面安装了 torch/torchvision 等依赖。
  source .venv/bin/activate
fi

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# 这是最终正式 CIFAR10 配置：
# - --download 准备真实 CIFAR10 数据；
# - --amp 和 --channels-last 用于 A100 加速；
# - 输出保存在 outputs/part3_cifar_resnet18/final_amp_channels_last。
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name final_amp_channels_last \
  --epochs 80 \
  --batch-size 512 \
  --lr 0.4 \
  --amp \
  --channels-last
