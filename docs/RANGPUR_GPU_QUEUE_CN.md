# Rangpur GPU 排队和运行说明

## 为什么要排队

Rangpur 是学校的 HPC cluster。GPU 不是直接独占使用，而是通过 SLURM 队列申请资源。

流程通常是：

1. 登录 Rangpur。
2. 进入项目目录。
3. 准备 Python 环境。
4. 用 `sbatch` 提交任务。
5. 用 `squeue` 查看排队/运行状态。
6. 训练完成后查看 log、metrics、plots、checkpoints。

## 常用命令

查看当前队列：

```bash
squeue -u "$USER"
```

提交 CIFAR10 任务：

```bash
sbatch slurm/part3_cifar_resnet18.sbatch
```

提交 VAE：

```bash
sbatch slurm/part4_vae.sbatch
```

提交 UNet：

```bash
sbatch slurm/part4_unet.sbatch
```

提交 GAN：

```bash
sbatch slurm/part4_gan.sbatch
```

看日志：

```bash
tail -f slurm-cifar-<jobid>.out
```

取消任务：

```bash
scancel <jobid>
```

## 数据路径

Lab sheet 说 OASIS 数据在：

```text
/home/groups/comp3710/
```

本项目 SLURM 默认使用：

```text
/home/groups/comp3710/keras_png_slices_data
```

如果实际路径不同，可以提交时覆盖：

```bash
DATA_PATH=/actual/path/to/keras_png_slices_data sbatch slurm/part4_unet.sbatch
```

## Demo 时可以怎么做

正式训练可能已经提前跑完。Demo 现场可以展示：

1. `squeue -u "$USER"` 说明任务如何排队。
2. `slurm-*.out` 说明训练确实在 GPU 上跑过。
3. `outputs/*/metrics.json` 展示准确率、DSC、训练时间。
4. `checkpoints/*/*.pt` 展示模型权重。
5. 用 `--eval-only --checkpoint ...` 跑 inference。
6. 用 `--fast-dev` 或小 epoch 跑一轮训练，证明代码能训练。
