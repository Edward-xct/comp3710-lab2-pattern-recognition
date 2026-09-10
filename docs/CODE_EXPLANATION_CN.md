# COMP3710 Lab 2 代码详细解析

这份文档用于帮助你真正理解项目代码。建议 demo 前按顺序读：先看每个 `scripts/part*.py` 的入口，再看 `src/comp3710_lab2/*.py` 的核心实现。

## 项目结构

```text
comp3710_lab2_demo/
├── scripts/                 # 每个 Part 的运行入口
├── src/comp3710_lab2/       # 可复用的算法、模型和数据读取代码
├── slurm/                   # Rangpur GPU 提交脚本
├── outputs/                 # 训练结果、图片、metrics，已被 .gitignore 排除
├── checkpoints/             # 模型权重，已被 .gitignore 排除
├── docs/                    # 中文 demo/排队/GitHub/实验过程说明
├── README.md                # 项目运行说明
└── requirements.txt         # Python 依赖
```

基本设计思路：

- `scripts/` 负责“跑任务”：解析命令行参数、调用模型/数据代码、保存结果。
- `src/comp3710_lab2/` 负责“实现任务”：DFT、LFW、ResNet、OASIS Dataset、VAE、UNet、GAN。
- `outputs/` 保存可以展示给老师看的图片和 metrics。
- `checkpoints/` 保存训练好的 `.pt` 权重。
- `slurm/` 是正式 GPU 训练入口。

## 公共模块：`src/comp3710_lab2/common.py`

这个文件被所有任务复用，主要做四件事。

第一，统一项目根目录：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

因为 `common.py` 在 `src/comp3710_lab2/` 下，往上两层就是项目根目录。所有输出都从这里定位，避免从不同终端运行时路径混乱。

第二，统一输出目录：

```python
output_dir("part1_dft")
checkpoint_dir("part4_unet")
```

它们会自动创建目录。这样每个任务都把结果放到固定位置，demo 时不用到处找。

第三，固定随机种子：

```python
set_seed(42)
```

它同时设置 Python、NumPy、PyTorch 的随机种子，让训练结果尽量可复现。

第四，选择运行设备：

```python
choose_device()
```

优先顺序是：

1. CUDA，也就是 Rangpur A100。
2. MPS，也就是 Mac Apple Silicon GPU。
3. CPU。

所以同一份代码本地能跑，Rangpur 上也能自动用 GPU。

## Part 1：DFT

入口文件：

```text
scripts/part1_dft.py
```

核心实现：

```text
src/comp3710_lab2/dft.py
```

### 目标

Part 1 要你理解方波、Fourier series、DFT、FFT，以及不同实现的速度差异。

代码完成了三类计算：

- `numpy_fft`：NumPy 自带 FFT，最快，作为基准。
- `numpy_naive_dft`：按 DFT 公式双重循环实现，最直观但最慢。
- `torch_matrix_dft`：显式构造 DFT matrix，用 PyTorch 矩阵乘法计算。

### 方波和 Fourier series

理想方波：

```python
np.sign(np.sin(2 * pi * f0 * t))
```

Fourier 近似：

```python
sum(sin(2*pi*n*f0*t) / n for n in odd harmonics)
```

方波只包含奇次谐波，例如 1、3、5、7。谐波越多，重构越像方波，但跳变处会出现 Gibbs phenomenon。

### Naive DFT

DFT 公式本质是：

```text
X[k] = sum_n x[n] * exp(-j * 2π * k * n / N)
```

代码里外层循环是频率 `k`，内层循环是时间采样点 `n`。每个频率都扫一遍所有采样点，所以复杂度是 `O(N^2)`。

### PyTorch matrix DFT

`torch_matrix_dft()` 先构造一个 DFT matrix：

```python
matrix[k, n] = exp(-j * 2*pi*k*n/N)
```

然后做：

```python
matrix @ x
```

数学上仍然是 `O(N^2)`，但矩阵乘法由底层高性能库执行，所以比 Python 双重循环快很多。如果在 CUDA 上跑，还可以利用 GPU 并行。

### 输出

Part 1 会生成：

- `outputs/part1_dft/part1_square_wave_harmonics.png`
- `outputs/part1_dft/part1_dft_spectrum.png`
- `outputs/part1_dft/part1_dft_timings.csv`
- `outputs/part1_dft/part1_dft_timings.json`

现场讲重点：

- FFT 最快，因为 `O(N log N)`。
- naive DFT 最慢，因为 `O(N^2)` 且 Python 双重循环开销大。
- PyTorch matrix DFT 结果和 FFT 接近，说明实现正确。

## Part 2：Eigenfaces

入口文件：

```text
scripts/part2_eigenfaces.py
```

核心实现：

```text
src/comp3710_lab2/lfw.py
```

### 目标

Part 2 是传统机器学习 pipeline：

1. 加载 LFW 人脸数据。
2. 把图片 flatten 成向量。
3. 做 PCA 得到 eigenfaces。
4. 把训练/测试图片投影到 PCA 空间。
5. 用 Random Forest 分类。

### 数据

代码用：

```python
fetch_lfw_people(min_faces_per_person=70, resize=0.4)
```

`min_faces_per_person=70` 的作用是只保留样本足够多的人物类别，避免类别太稀疏。

### PCA / Eigenfaces

每张图片原来是 `height x width`，代码先变成一维向量。然后只用训练集计算 mean：

```python
mean = np.mean(x_train, axis=0)
x_train_centered = x_train - mean
x_test_centered = x_test - mean
```

注意测试集也减训练集 mean，不能用测试集自己的 mean，否则会泄漏测试集信息。

接着用 SVD：

```python
u, s, v = np.linalg.svd(x_train_centered, full_matrices=False)
components = v[:n_components]
```

`components` 的每一行就是一个 principal component。把它 reshape 回人脸图片，就是 eigenface。

### 分类

投影：

```python
x_train_transformed = x_train_centered @ components.T
x_test_transformed = x_test_centered @ components.T
```

然后 Random Forest 学习这些 PCA 特征和人物标签的关系。

当前结果：

- accuracy：`0.6553`
- 测试集 322 张，正确 211 张。

现场讲重点：

- Eigenfaces 是把人脸表示成一组“脸部变化方向”的线性组合。
- 这是手工降维 + 分类器，不是端到端深度学习。
- Part 3.1 的 CNN accuracy 更高，说明 CNN 更会利用空间结构。

## Part 3.1：LFW CNN

入口文件：

```text
scripts/part3_lfw_cnn.py
```

模型定义：

```text
src/comp3710_lab2/lfw.py -> LfwCnn.build()
```

### 目标

用 CNN 对同一个 LFW 数据集做人脸分类，并且希望超过 Part 2 的 Eigenfaces。

### 输入形状

LFW 图片是灰度图。CNN 需要 `[N, C, H, W]`，所以代码加了一个通道维度：

```python
x_train[:, None, :, :]
```

这里 `C=1`。

### 网络结构

`LfwCnn.build()` 返回一个 `nn.Sequential`：

1. `Conv2d(1, 32, kernel_size=3, padding=1)`
2. `ReLU`
3. `MaxPool2d(2)`
4. `Conv2d(32, 32, kernel_size=3, padding=1)`
5. `ReLU`
6. `MaxPool2d(2)`
7. `Flatten`
8. `Linear(..., 128)`
9. `ReLU`
10. `Dropout(0.25)`
11. `Linear(128, n_classes)`

这满足 PDF 要求：两个 `3x3` convolution layers，每层 32 filters。

### 训练

optimizer：

```python
Adam(lr=1e-3)
```

loss：

```python
CrossEntropyLoss()
```

CrossEntropyLoss 用于多分类，它接收 logits 和类别编号，不需要提前手动 softmax。

当前结果：

- 20 epochs
- final accuracy：`0.8106`
- 比 Part 2 的 `0.6553` 高。

现场讲重点：

- CNN 学习局部纹理和空间结构。
- MaxPool 降低空间尺寸，提高感受野。
- Dropout 减少过拟合。

## Part 3.2：CIFAR10 ResNet-18

入口文件：

```text
scripts/part3_cifar_resnet18.py
```

模型定义：

```text
src/comp3710_lab2/resnet_cifar.py
```

SLURM：

```text
slurm/part3_cifar_resnet18.sbatch
slurm/part3_cifar_ablation_detailed.sbatch
```

### 目标

这部分是 DAWNBench-style task：

- 真实 CIFAR10 数据。
- ResNet-18。
- 超过 90%。
- mixed precision 冲到 94% 左右。
- 在 Rangpur 上展示 inference 和至少一个 epoch 训练。

### ResNet block

`BasicBlock` 有两条路径：

主路径：

```text
Conv -> BN -> ReLU -> Conv -> BN
```

shortcut 路径：

- 如果输入输出 shape 一样，用 `Identity()`。
- 如果 stride 或通道数不同，用 `1x1 Conv + BN` 对齐。

最后：

```python
out = out + shortcut(x)
```

这就是 residual connection。它让梯度可以绕过卷积层传播，深层网络更容易训练。

### CIFAR10 版 ResNet-18

ImageNet 原版 ResNet 开头常用 `7x7 conv + stride 2 + maxpool`。但 CIFAR10 只有 `32x32`，如果这样做会太早损失空间信息。

所以代码用：

```python
Conv2d(3, width, kernel_size=3, stride=1, padding=1)
```

四个 stage 是 `[2, 2, 2, 2]`，对应 ResNet-18。

### 数据增强

正式训练默认使用：

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `Normalize(CIFAR10_MEAN, CIFAR10_STD)`
- `RandomErasing(p=0.25)`

这些增强的作用是减少过拟合，提高测试集 accuracy。

### 训练策略

默认 optimizer：

```python
SGD(lr=0.4, momentum=0.9, weight_decay=5e-4, nesterov=True)
```

默认 scheduler：

```python
CosineAnnealingLR
```

默认 label smoothing：

```python
CrossEntropyLoss(label_smoothing=0.1)
```

这些组合用于最终高分版本。

### AMP 和 channels-last

`--amp`：

- 在 CUDA 上启用 mixed precision。
- 用半精度提高训练速度。
- 通过 `GradScaler` 避免梯度数值下溢。

`--channels-last`：

- 改变 tensor memory format。
- 对 GPU convolution 通常更快。

### Demo 展示方式

Demo 不重新训练 Part 3.2，而是展示 Rangpur 上真实 CIFAR10 的完整训练日志、metrics、逐阶段 ablation 结果和 checkpoint。

### Ablation 过程

`slurm/part3_cifar_ablation_detailed.sbatch` 有 7 个阶段：

1. 小模型，短训练，无增强。
2. 同样小模型，训练更久。
3. 增加 ResNet 宽度。
4. 加 random crop 和 flip。
5. 加 cosine learning-rate schedule。
6. 加 label smoothing 和 random erasing。
7. 加 AMP 和 channels-last。

这个过程用于回答老师说的“不要一开始就很高分，要展示每次改进了什么”。

你之前的 Rangpur 正式结果：

- baseline_no_aug：`0.785`
- add_aug_cosine：`0.9175`
- regularized：`0.9416`
- final_amp_channels_last：`0.9478`
- final time：约 `300s`

现场讲重点：

- 94.78% 是真实 CIFAR10/Rangpur 结果。
- ResNet 是自己实现的，没有用 pretrained。
- 提升过程主要来自 augmentation、scheduler、regularisation、AMP speedup。

## OASIS 数据读取

核心文件：

```text
src/comp3710_lab2/oasis_png.py
```

### 支持两种数据位置

本地可能是：

```text
/Users/xct/Downloads/keras_png_slices_data.zip
```

Rangpur 是：

```text
/home/groups/comp3710/OASIS
```

代码同时支持 zip 和已解压目录。

### 文件配对

原图文件名类似：

```text
case_123_slice_45.nii.png
```

mask 文件名类似：

```text
seg_123_slice_45.nii.png
```

代码用 `(case_id, slice_id)` 作为 key，把图片和对应 mask 配对。

### mask 映射

原始 mask 像素值是：

```text
0 / 85 / 170 / 255
```

代码转换为：

```text
0 / 1 / 2 / 3
```

原因是 `CrossEntropyLoss` 需要类别编号 LongTensor，而不是 RGB 或原始灰度值。

### value range

VAE/UNet 用：

```text
[0, 1]
```

GAN 用：

```text
[-1, 1]
```

因为 GAN generator 最后一层是 `Tanh()`，输出就是 `[-1, 1]`。

## Part 4 Task 1：VAE

入口文件：

```text
scripts/part4_vae.py
```

模型文件：

```text
src/comp3710_lab2/vae.py
```

### 目标

VAE 用来学习 OASIS MRI 的 latent space，并生成：

- reconstruction 图
- 2D manifold 图

### Encoder

输入：

```text
[batch, 1, 128, 128]
```

经过三次 stride=2 卷积后，空间尺寸从 128 降到 16。

然后 flatten，通过两个全连接层输出：

```python
mu
logvar
```

VAE 输出的是分布参数，而不是普通 autoencoder 的固定 latent vector。

### Reparameterization trick

代码：

```python
std = exp(0.5 * logvar)
eps = randn_like(std)
z = mu + eps * std
```

这样做的原因是随机采样本身不可直接反向传播，但把随机性放到 `eps` 里后，`mu` 和 `logvar` 仍然可以被梯度更新。

### Decoder

Decoder 从 `z` 开始，用全连接层扩展回卷积特征图，再用 ConvTranspose2d 上采样回 `128x128`。

最后 `Sigmoid()` 输出 `[0,1]`，和输入 MRI 像素范围一致。

### Loss

```text
VAE loss = reconstruction loss + beta * KL divergence
```

- reconstruction loss：重建图和输入图越接近越好。
- KL divergence：让 latent distribution 接近标准正态，方便采样生成。

当前正式结果：

- device：cuda
- epochs：50
- latent_dim：2
- best_val_loss：91.1664
- 输出：reconstruction 和 manifold 都已保存。

现场讲重点：

- latent_dim=2 是为了直接画 manifold。
- reconstruction 图上排是真实 MRI，下排是重建 MRI。
- manifold 图展示不同 latent 坐标解码出的脑图变化。

## Part 4 Task 2：UNet

入口文件：

```text
scripts/part4_unet.py
```

模型文件：

```text
src/comp3710_lab2/unet.py
```

### 目标

UNet 用来做 MRI segmentation。输入一张 MRI，输出每个像素的类别。

### 网络结构

UNet 分成三部分：

1. Encoder/down path：逐步下采样，提取语义特征。
2. Bridge：最低分辨率、最高通道数的中间层。
3. Decoder/up path：逐步上采样，恢复到原图大小。

关键是 skip connection：

```python
torch.cat([x, d3], dim=1)
```

Decoder 每次上采样后，会拼接同尺度的 encoder 特征。这样模型既有深层语义，也有浅层边界细节。

### 输出

模型输出：

```text
[batch, 4, H, W]
```

4 个 channel 对应 4 个类别。对每个像素取 `argmax` 得到最终 mask。

这就是 categorical output。训练时 target 是类别编号，计算 Dice 时会转换成 one-hot。

### Loss

总 loss：

```text
CrossEntropyLoss + dice_weight * soft_dice_loss
```

CrossEntropyLoss 负责像素分类，soft Dice 直接优化预测区域和真实区域的重叠。

### Dice

Dice Similarity Coefficient：

```text
DSC = 2 * overlap / (prediction area + target area)
```

代码对 4 个类别分别计算 Dice，然后取平均。

当前正式结果：

- device：cuda
- epochs：80
- best_mean_dice：0.9653
- 各类 Dice 都超过 0.9。

现场讲重点：

- 这是 Part 4 里最稳的结果。
- 展示 `part4_unet_examples.png`：
  - 第一列 MRI
  - 第二列 ground truth
  - 第三列 prediction
- 如果老师要求现场 inference，用 `--eval-only --checkpoint ...`。

## Part 4 Task 3：GAN

入口文件：

```text
scripts/part4_gan.py
```

模型文件：

```text
src/comp3710_lab2/gan.py
```

### 目标

训练 GAN 生成 OASIS 风格的脑部 MRI。

### Generator

输入是随机噪声：

```text
[batch, latent_dim, 1, 1]
```

经过多层 `ConvTranspose2d`，空间尺寸逐渐变成：

```text
1 -> 4 -> 8 -> 16 -> 32 -> 64
```

最后输出：

```text
[batch, 1, 64, 64]
```

最后一层是 `Tanh()`，所以输出范围是 `[-1,1]`。

### Discriminator

Discriminator 输入真实图或生成图，经过多层 Conv2d 下采样，最后输出一个 logit。

这个 logit 表示“这张图更像真实还是生成”。

代码使用：

```python
BCEWithLogitsLoss()
```

它比先 sigmoid 再 BCE 更数值稳定。

### 训练流程

每个 batch 分两步。

第一步，训练 discriminator：

- 真实图片 label = 1。
- 生成图片 label = 0。
- 生成图片用 `detach()`，避免这一步更新 generator。

第二步，训练 generator：

- 再把生成图送进 discriminator。
- 希望 discriminator 把它判断为真实，也就是 label = 1。
- 这一步不 detach，让梯度传回 generator。

### 输出

每隔 5 个 epoch 保存一张样本图：

```text
outputs/part4_gan/samples/epoch_*.png
```

还保存 loss 曲线：

```text
outputs/part4_gan/part4_gan_losses.png
```

当前正式结果：

- device：cuda
- epochs：80
- sample images：17 张
- checkpoint：`latest_gan.pt`

现场讲重点：

- GAN 没有像 accuracy/Dice 那样的单一可靠分数。
- loss 曲线只是训练证据，不完全代表图片质量。
- 最终要展示生成图是否像 brain、是否有多样性、是否 mode collapse。

## SLURM 脚本怎么理解

`slurm/*.sbatch` 都是提交到 Rangpur 的批处理脚本。

常见头部：

```bash
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
```

意思是：

- 使用 COMP3710 课程账户。
- 使用 comp3710 分区。
- 申请 1 张 A100 GPU。
- 给 DataLoader 8 个 CPU。
- 最多允许跑 12 小时。

脚本里没有显式 `--mem`，因为你之前在 Rangpur 测试时这个分区拒绝了 `--mem=8G`。

## 现场 demo 推荐顺序

1. 打开 README，说明项目结构。
2. 跑 Part 1：

```bash
python scripts/part1_dft.py
```

3. 跑 Part 2：

```bash
python scripts/part2_eigenfaces.py
```

4. 跑 Part 3.1：

```bash
python scripts/part3_lfw_cnn.py --epochs 20
```

5. 展示 Part 3.2 的 Rangpur summary、逐轮训练 log 和正式 metrics。
6. 展示 Part 4 的三组图片和 metrics。
7. 如果老师要求现场跑 GPU，登录 Rangpur 后跑 `--eval-only` 或 1 epoch 小训练。
8. 展示 GitHub commit history。

## 最容易被问的问题

为什么 Part 3.1 比 Part 2 好？

- 因为 CNN 保留空间结构，能学习局部边缘/纹理；Part 2 的 Eigenfaces 是线性降维，表达能力更弱。

为什么 ResNet 要 shortcut？

- shortcut 让梯度更容易传回前面层，缓解深层网络退化/难训练问题。

为什么 CIFAR10 不用 ImageNet ResNet 的 7x7 开头？

- CIFAR10 图片太小，7x7 stride=2 会过早损失细节。

为什么 UNet segmentation 用 Dice？

- segmentation 更关心区域重叠，Dice 能直接衡量预测 mask 和真实 mask 的重合程度。

为什么 GAN loss 不能完全代表图片质量？

- GAN 是两个模型对抗，loss 可能震荡；最终 realism 还是要看生成图和多样性。

为什么 VAE manifold 要 latent_dim=2？

- 2 维 latent space 可以直接画网格，展示从一个 latent 坐标变化到另一个坐标时生成图如何变化。
