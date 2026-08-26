# COMP3710 Lab Demonstration 2 中文展示步骤

## 0. 你要先说明什么

这个项目用 PyTorch 完成 Lab 2。代码分成本地快速检查和 Rangpur GPU 正式训练两种模式。

本地主要用于：

- 检查代码能不能 import。
- 检查 OASIS PNG 数据能不能读。
- 小 batch 跑通 forward/loss。
- 生成少量示例图。

Rangpur GPU 主要用于：

- CIFAR10 ResNet-18 的 DAWNBench 任务。
- OASIS VAE/UNet/GAN 的正式训练。
- demo 时展示 inference 和至少一个 epoch 的训练。

## 1. Part 1 - DFT

运行：

```bash
python scripts/part1_dft.py
```

快速检查：

```bash
python scripts/part1_dft.py --fast-dev
```

展示文件：

- `outputs/part1_dft/part1_square_wave_harmonics.png`
- `outputs/part1_dft/part1_dft_spectrum.png`
- `outputs/part1_dft/part1_dft_timings.csv`

讲解要点：

1. 方波可以用奇数次谐波叠加近似。
2. 谐波越多，边缘越 sharp，但会出现 Gibbs phenomenon，也就是跳变处附近有 overshoot。
3. Naive DFT 是直接按公式做双重循环，复杂度是 `O(N^2)`。
4. NumPy FFT 用快速傅里叶变换，复杂度约 `O(N log N)`，所以通常最快。
5. PyTorch matrix DFT 没用内置 FFT，而是显式构造 DFT matrix；GPU 版本可以并行矩阵乘法，但仍然是 `O(N^2)`。

## 2. Part 2 - Eigenfaces

运行：

```bash
python scripts/part2_eigenfaces.py
```

展示文件：

- `outputs/part2_eigenfaces/part2_eigenfaces.png`
- `outputs/part2_eigenfaces/part2_compactness.png`
- `outputs/part2_eigenfaces/part2_classification_report.txt`
- `outputs/part2_eigenfaces/part2_metrics.json`

讲解要点：

1. LFW 图片先 flatten 成向量。
2. 只用 training set 的 mean 做 centering，避免 test contamination。
3. SVD 得到 principal components，这些 reshape 回图片就是 eigenfaces。
4. Compactness plot 显示前多少个 eigenfaces 可以解释多少 variance。
5. Random Forest 用 PCA face space 特征做分类。

## 3. Part 3.1 - LFW CNN

运行：

```bash
python scripts/part3_lfw_cnn.py --epochs 20
```

展示文件：

- `outputs/part3_lfw_cnn/part3_lfw_cnn_history.png`
- `outputs/part3_lfw_cnn/part3_lfw_cnn_classification_report.txt`
- `checkpoints/part3_lfw_cnn/best_lfw_cnn.pt`

讲解要点：

1. CNN 保留了图片的二维空间结构，和 Eigenfaces flatten 成向量不同。
2. 网络有两个 `3x3` convolution layer，每层 32 filters。
3. 后面接 fully connected/dense layer 做分类。
4. Loss 是 categorical classification 的 cross entropy。

## 4. Part 3.2 - CIFAR10 ResNet-18

Rangpur 提交：

```bash
sbatch slurm/part3_cifar_resnet18.sbatch
```

或者直接运行：

```bash
python scripts/part3_cifar_resnet18.py --download --epochs 80 --batch-size 512 --lr 0.4 --amp --channels-last
```

展示文件：

- `outputs/part3_cifar_resnet18/part3_cifar_resnet18_history.csv`
- `outputs/part3_cifar_resnet18/part3_cifar_resnet18_metrics.json`
- `checkpoints/part3_cifar_resnet18/best_cifar_resnet18.pt`
- `slurm-cifar-<jobid>.out`

讲解要点：

1. ResNet-18 是自己在 `src/comp3710_lab2/resnet_cifar.py` 里实现的，不是调用 torchvision 预训练模型。
2. Residual block 用 shortcut 缓解深层网络的梯度传播问题。
3. CIFAR10 用 random crop、horizontal flip、random erasing 做 augmentation。
4. `--amp` 使用 mixed precision，A100/V100 上会更快。
5. Demo 时可以用 checkpoint 做 inference，也可以用 `--fast-dev` 或少量 epoch 现场跑一轮训练。

本地没有 CIFAR10 时，可以只检查训练循环：

```bash
python scripts/part3_cifar_resnet18.py --fast-dev --synthetic
```

注意：`--synthetic` 只是 smoke test，正式 demo 不能用它声称 CIFAR10 accuracy。

## 5. Part 4 Task 1 - VAE

Rangpur 提交：

```bash
sbatch slurm/part4_vae.sbatch
```

本地小检查：

```bash
python scripts/part4_vae.py --data /Users/xct/Downloads/keras_png_slices_data.zip --fast-dev
```

展示文件：

- `outputs/part4_vae/part4_vae_reconstructions.png`
- `outputs/part4_vae/part4_vae_manifold.png`
- `outputs/part4_vae/part4_vae_history.csv`
- `checkpoints/part4_vae/best_vae.pt`

讲解要点：

1. Encoder 输出 `mu` 和 `logvar`。
2. Reparameterization trick 用 `z = mu + eps * sigma` 让采样过程可反向传播。
3. Loss = reconstruction loss + KL divergence。
4. latent dim 默认是 2，所以可以直接画 2D manifold。

## 6. Part 4 Task 2 - UNet

Rangpur 提交：

```bash
sbatch slurm/part4_unet.sbatch
```

本地小检查：

```bash
python scripts/part4_unet.py --data /Users/xct/Downloads/keras_png_slices_data.zip --fast-dev
```

展示文件：

- `outputs/part4_unet/part4_unet_examples.png`
- `outputs/part4_unet/part4_unet_history.csv`
- `outputs/part4_unet/part4_unet_metrics.json`
- `checkpoints/part4_unet/best_unet.pt`

讲解要点：

1. 输入是 MRI slice，target 是 segmentation mask。
2. 原始 mask 像素值是 `0/85/170/255`，代码映射成类别 `0/1/2/3`。
3. UNet 输出是 4 个 channel，对应 categorical/one-hot segmentation。
4. Loss 是 CrossEntropy + soft Dice loss。
5. 指标是每个 label 的 DSC，任务要求每个 label 都要超过 0.9。
6. Skip connections 把 encoder 的空间细节传给 decoder，有利于边界恢复。

## 7. Part 4 Task 3 - GAN

Rangpur 提交：

```bash
sbatch slurm/part4_gan.sbatch
```

本地小检查：

```bash
python scripts/part4_gan.py --data /Users/xct/Downloads/keras_png_slices_data.zip --fast-dev
```

展示文件：

- `outputs/part4_gan/samples/epoch_*.png`
- `outputs/part4_gan/part4_gan_losses.png`
- `outputs/part4_gan/part4_gan_history.csv`

讲解要点：

1. Generator 从 random latent vector 生成 MRI-like image。
2. Discriminator 判断图片是真实数据还是生成数据。
3. 两个网络交替优化，训练过程可能不稳定。
4. 要展示生成图、loss 曲线，并解释有没有 mode collapse。

## 8. GitHub 展示

Part 4 明确要求 GitHub project 和 commit logs。

Demo 时展示：

```bash
git status -sb
git log --oneline -5
```

你需要能说明：

1. 数据没有上传到 GitHub。
2. checkpoint 没有上传到 GitHub。
3. README 说明了怎么运行。
4. commit message 有意义。
5. AI 使用记录在 `AI_USAGE_LOG.md`。
