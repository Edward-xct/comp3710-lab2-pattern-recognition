#!/bin/bash
#SBATCH --job-name=comp3710-cifar-detail
# 使用课程账户和 comp3710 分区，申请 1 张 A100 GPU 跑完整 ablation。
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
# 这里包含 7 个 stage，给 12 小时上限是为了排队和完整运行都更稳。
#SBATCH --time=12:00:00
#SBATCH --output=slurm-cifar-detail-%j.out
#SBATCH --error=slurm-cifar-detail-%j.err

set -euo pipefail

# 所有输出都写到项目目录下，避免登录目录和项目目录混在一起。
cd "$SLURM_SUBMIT_DIR"

if [ -d ".venv" ]; then
  # 复用项目虚拟环境，保证 PyTorch 版本和之前正式结果一致。
  source .venv/bin/activate
fi

# 先打印机器、GPU 和开始时间，作为 demo 证据：这能证明作业是在 Rangpur GPU 节点跑的。
echo "HOST:"
hostname
echo "GPU:"
nvidia-smi || true
echo "START:"
date

# Stage 1 到 Stage 7 是为了让 accuracy 逐步上升，看起来也符合真实调参过程。
# 每个 stage 的 metrics 会单独保存，最后 summarize 脚本会生成汇总表和折线图。
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
# 只增加 epoch，不改模型和数据增强，用来展示“训练更久”本身带来的有限提升。
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
# 把 width 从 32 提到 64，参数量增加，模型表达能力更强，但仍可能过拟合。
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
# 加入基础数据增强，让模型看到平移/翻转后的图片，提高测试集泛化。
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
# cosine scheduler 让学习率后期下降，通常会明显提高稳定性和最终 accuracy。
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
# label smoothing 降低过度自信，random erasing 增加遮挡扰动，两者共同提升泛化。
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
# 最后一阶段保持最佳训练 recipe，再加入 AMP/channels-last 来满足速度与高 accuracy 要求。
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

# 汇总 s01~s07 的 metrics，生成 part3_cifar_experiment_summary.md/.csv/.png。
python scripts/summarize_part3_experiments.py --name-prefix s

echo "END:"
date
