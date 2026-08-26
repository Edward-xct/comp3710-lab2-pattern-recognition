# Part 3.2 CIFAR10 ResNet-18 真实改进记录

这个文件用于 demo 时说明 Part 3.2 不是一次性拿到高分，而是通过多轮真实实验逐步改进。

## 核心要求

- 使用自己实现的 CIFAR10 ResNet-18，不使用 torchvision 预训练模型。
- 在 Rangpur GPU 上训练并能现场展示 inference 或单个 epoch 的训练过程。
- 目标是超过 90% accuracy；更高目标是接近或超过 94%，并且训练时间要快。
- 每轮结果必须来自真实训练日志和保存的 metrics，不能把 `--synthetic` 的 smoke test 当成正式结果。

## 推荐实验路径

用下面这个详细脚本一次提交，内部会连续跑 7 个真实阶段：

```bash
sbatch slurm/part3_cifar_ablation_detailed.sbatch
```

查看队列：

```bash
squeue -u s4985908
```

查看日志：

```bash
tail -f slurm-cifar-detail-<jobid>.out
```

如果想退出 `tail -f`，按 `Ctrl+C`，训练任务不会被取消。

## 七个阶段怎么讲

Stage 1: `s01_tiny_baseline`

- 改动：小一点的 ResNet width=32，只训练 10 epochs，不使用 augmentation，不使用 scheduler。
- 目的：建立弱 baseline，证明一开始不是直接高分。
- 讲法：先确认模型、数据加载、训练循环都是真的在 CIFAR10 上跑。

Stage 2: `s02_longer_baseline`

- 改动：同样的小模型和无 augmentation，但训练从 10 epochs 增加到 20 epochs。
- 目的：展示单纯训练更久会提升一些，但还不够。
- 讲法：epoch 数增加能让模型继续拟合训练集，但泛化提升有限。

Stage 3: `s03_wider_no_aug`

- 改动：把 width 从 32 增加到 64，仍然不使用 augmentation。
- 目的：测试增加模型容量的效果。
- 讲法：模型更宽可以学到更多特征，但没有 augmentation 时仍容易泛化不足。

Stage 4: `s04_add_crop_flip`

- 改动：加入 random crop 和 horizontal flip，暂时不加 random erasing 或 label smoothing。
- 目的：让模型泛化更好，减少过拟合，学习率后期逐渐变小。
- 讲法：augmentation 让模型看到更多图像变化，通常是 CIFAR10 accuracy 跨上一个台阶的关键。

Stage 5: `s05_add_cosine_lr`

- 改动：保留 crop/flip augmentation，加入 cosine learning rate decay。
- 目的：让训练后期更稳定收敛。
- 讲法：一开始较大学习率探索参数，后期学习率下降来细调。

Stage 6: `s06_regularized`

- 改动：加入 label smoothing 和 random erasing。
- 目的：继续提升泛化能力，让模型不要对训练标签过度自信。
- 讲法：label smoothing 改善 calibration，random erasing 模拟遮挡，减少 memorisation。

Stage 7: `s07_final_amp_channels_last`

- 改动：保留前面最好的训练设置，加入 AMP mixed precision 和 channels-last memory format。
- 目的：在 A100 上加速训练，满足 DAWNBench-style fast training 的要求。
- 讲法：AMP 使用半精度计算提高吞吐，channels-last 对卷积模型更友好。

## 跑完后展示这些文件

汇总表：

```bash
cat outputs/part3_cifar_resnet18/part3_cifar_experiment_summary.md
```

每轮 metrics：

```bash
find outputs/part3_cifar_resnet18 -name "part3_cifar_resnet18_metrics.json" -print
```

每轮训练历史：

```bash
find outputs/part3_cifar_resnet18 -name "part3_cifar_resnet18_history.csv" -print
```

最终 checkpoint：

```bash
ls -lh checkpoints/part3_cifar_resnet18/final_amp_channels_last
```

## Demo 时可以现场跑一小段

为了证明代码能现场训练，可以跑 1 个 epoch 或少量 epoch：

```bash
python scripts/part3_cifar_resnet18.py \
  --download \
  --experiment-name demo_one_epoch \
  --epochs 1 \
  --batch-size 512 \
  --lr 0.4 \
  --amp \
  --channels-last
```

这只是现场演示训练过程，不用它替代完整训练结果。
