# COMP3710 Lab 2 完成度核对

本文件根据 `COMP3710_Lab_2_2026_v2.01.pdf` 的任务要求，核对当前项目完成情况。PDF 里的内容是课程要求；下面的“状态/风险/建议”是本项目目前的实际检查结果。

## 总体结论

代码主体已经覆盖 Part 1、Part 2、Part 3.1、Part 3.2、Part 4 Task 1、Task 2、Task 3。本地和 Rangpur 上都已有关键结果，其中 Part 4 三个任务的正式 GPU 结果已经同步回本地。

目前还需要你亲自确认或补齐的地方：

1. GitHub：本地仓库还没有 `origin` 远端，需要先在 GitHub 创建空仓库，然后把已提交版本推上去。
2. Part 4.1 Advanced Git Course：代码无法替你确认课程完成状态，需要你在 Blackboard/edX 上确认并准备截图。
3. Part 3.2 正式结果：你之前已经在 Rangpur 跑出 94.78% 和 ablation 过程，但本地 `outputs/part3_cifar_resnet18` 现在没有完整的 `final_amp_channels_last` 或 `s01~s07` 正式目录。demo 前建议再从 Rangpur 同步一次。
4. GAN：代码和训练证据完整，但“realism”由 demonstrator 主观判断，最好现场打开 `epoch_080.png` 和 loss 曲线说明结果有一定 blur，但没有完全崩掉。

## 分数结构

| 部分 | PDF 分数 | 当前状态 | 证据 |
| --- | ---: | --- | --- |
| Part 1 DFT | 1 | 已完成 | `scripts/part1_dft.py`，`src/comp3710_lab2/dft.py`，`outputs/part1_dft/*` |
| Part 2 Eigenfaces | 1 | 已完成 | accuracy 0.6553，eigenfaces 图、compactness 图、classification report |
| Part 3.1 LFW CNN | 1 | 已完成 | 20 epochs，final accuracy 0.8106，高于 Part 2 |
| Part 3.2 CIFAR10 ResNet-18 | 4 | 代码完成，正式高分结果已由 Rangpur 日志证明；本地正式输出需再同步 | 你粘贴的 Rangpur 结果：final AMP run best accuracy 0.9478，约 300s |
| Part 4.1 Advanced Git Course | 1 | 未能由代码确认 | 需要你登录课程平台展示完成证明 |
| Part 4 Task 1 VAE | 最多 3/7 的 Easy 基础 | 已完成 | VAE 50 epochs，best val loss 91.1664，reconstructions 和 manifold |
| Part 4 Task 2 UNet | 与 Task 1 合计最多 5/7 | 已完成且结果强 | best mean Dice 0.9653，各类 Dice 都超过 0.9 |
| Part 4 Task 3 GAN | 与 Task 1+2 合计最多 7/7 | 已完成，但分数取决于生成图质量判断 | 80 epochs，17 张样本图，loss 曲线，latest checkpoint |

## Part 1 - DFT

PDF 要求：

- 构造 square wave / Fourier approximation。
- 修改或实现 `square_wave`、`square_wave_fourier`、`naive_dft`。
- 用 TensorFlow 或 PyTorch tensor operations 实现对应版本，特别是 GPU/tensor 版本。
- 比较 naive DFT、FFT、tensor/matrix DFT 的运行时间，并解释为什么最快的方法最快。

当前完成情况：

- `src/comp3710_lab2/dft.py` 有 NumPy 方波、Fourier series、naive DFT、PyTorch matrix DFT。
- `scripts/part1_dft.py` 会保存方波谐波图、DFT 频谱图、CSV/JSON 计时结果。
- 最新完整本地运行结果：
  - `N=512`：FFT 最快，torch matrix DFT 次之，naive DFT 最慢。
  - `N=1024`：同样顺序。
  - `N=2048`：同样顺序。
  - 所有方法 `close=True`。

建议现场讲法：

- naive DFT 是双重循环，复杂度 `O(N^2)`。
- FFT 利用蝶形分解，复杂度约 `O(N log N)`，所以最快。
- PyTorch matrix DFT 仍然是 `O(N^2)`，但矩阵乘法由底层高性能库执行，所以比 Python 双重循环快。

## Part 2 - Eigenfaces

PDF 要求：

- 使用 LFW 数据集。
- 用 NumPy/PCA/eigenfaces 构造低维 face representation。
- 用 Random Forest/PCA 特征做人脸识别。
- 展示 eigenfaces 和分类表现。

当前完成情况：

- `src/comp3710_lab2/lfw.py` 使用 sklearn 加载 LFW。
- 训练集中心化后用 SVD 得到 PCA components/eigenfaces。
- Random Forest 在 PCA 特征上分类。
- 已保存：
  - `outputs/part2_eigenfaces/part2_eigenfaces.png`
  - `outputs/part2_eigenfaces/part2_compactness.png`
  - `outputs/part2_eigenfaces/part2_classification_report.txt`
  - `outputs/part2_eigenfaces/part2_metrics.json`
- 当前 accuracy：`0.6553`。

建议现场讲法：

- Eigenfaces 是 PCA 的 principal components，把高维脸图像压缩成低维特征。
- 分类器不是直接看原始像素，而是看 PCA 后的 face-space 坐标。
- 这是传统机器学习 pipeline，和 Part 3.1 CNN 形成对比。

## Part 3.1 - LFW CNN

PDF 要求：

- 用 TF/Keras/PyTorch/JAX 实现 CNN。
- 使用和 Part 2 相同的 LFW 数据。
- 两个 `3x3` convolution layers，每层 32 filters。
- 接 dense layers 做分类。
- 可以用 Adam 和 categorical cross entropy。

当前完成情况：

- `scripts/part3_lfw_cnn.py` 训练 LFW CNN。
- `src/comp3710_lab2/lfw.py` 的 `LfwCnn.build()` 实现两层 `3x3` conv，每层 32 filters。
- optimizer 是 Adam，loss 是 CrossEntropyLoss。
- 已保存 history、classification report、best checkpoint。
- 当前 20 epochs final accuracy：`0.8106`，明显高于 Part 2 的 `0.6553`。

建议现场讲法：

- CNN 保留空间结构，能学习局部纹理和脸部区域特征。
- Eigenfaces 需要手工降维，CNN 是端到端学习特征和分类。

## Part 3.2 - CIFAR10 ResNet-18 / DAWNBench

PDF 要求：

- 构造 Fast CIFAR10 classification network。
- 使用 ResNet-18。
- 达到超过 90% accuracy，并且训练时间快，通常 cluster 上 30 分钟以内。
- demo 期间模型必须在 Rangpur 上运行 inference 和 single epoch training。
- 使用 mixed precision 达到约 94%，训练时间等价或快于 V100 约 360 秒，可拿额外 2 分。
- 一般不能直接使用预训练模型。

当前完成情况：

- `src/comp3710_lab2/resnet_cifar.py` 自己实现了 CIFAR10 版 ResNet-18，不用 torchvision pretrained。
- `scripts/part3_cifar_resnet18.py` 支持真实 CIFAR10、eval-only、AMP、channels-last、augmentation、scheduler。
- `slurm/part3_cifar_resnet18.sbatch` 是正式最终训练。
- `slurm/part3_cifar_ablation_detailed.sbatch` 是 7 阶段真实改进过程：
  - small baseline
  - longer baseline
  - wider ResNet
  - crop/flip augmentation
  - cosine LR
  - label smoothing + random erasing
  - AMP + channels-last final run
- 你之前粘贴的 Rangpur 结果显示最终 run：
  - best accuracy `0.9478`
  - final accuracy `0.9478`
  - time about `300.26s`
  - A100 + AMP + channels-last

当前风险：

- Demo 应展示从 Rangpur 同步回来的正式 CIFAR10 metrics 和训练日志。
- demo 前建议执行：

```bash
rsync -av s4985908@login0.compute.eait.uq.edu.au:~/comp3710_lab2_demo/outputs/part3_cifar_resnet18/ "/Users/xct/Documents/New project/comp3710_lab2_demo/outputs/part3_cifar_resnet18/"
rsync -av s4985908@login0.compute.eait.uq.edu.au:~/comp3710_lab2_demo/checkpoints/part3_cifar_resnet18/ "/Users/xct/Documents/New project/comp3710_lab2_demo/checkpoints/part3_cifar_resnet18/"
```

建议现场讲法：

- 展示真实 CIFAR10 accuracy 和对应的 Rangpur 训练时间。
- 展示 Rangpur 的 `slurm-cifar-*.out`、summary markdown、metrics JSON、checkpoint。
- 如果老师要求现场跑，登录 Rangpur 后用 `--epochs 1` 或 `--eval-only --checkpoint ...` 展示 inference/一轮训练。

## Part 4.1 - Advanced Git Course

PDF 要求：

- 完成第二个 Git Short Course：Version Control for Teams using Git。

当前完成情况：

- 代码仓库无法判断你是否完成了这门在线课。
- 需要你亲自登录 Blackboard/课程链接，准备 completion/progress 证明。

建议现场准备：

- 打开课程页面，显示账号、课程名、完成状态或进度。
- 如果没有 completion certificate，至少截一张进度页。

## Part 4 Task 1 - VAE

PDF 要求：

- 用 OASIS MR brain images 构造 VAE。
- 必须训练模型并 visualise resulting manifold。
- 可用 2D manifold sampling 或 UMAP 等方法。

当前完成情况：

- `scripts/part4_vae.py` 训练 ConvVAE。
- `src/comp3710_lab2/vae.py` 实现 encoder、`mu/logvar`、reparameterization、decoder。
- 使用 OASIS 数据，正式结果在 CUDA/Rangpur 上完成。
- 已保存：
  - `outputs/part4_vae/part4_vae_reconstructions.png`
  - `outputs/part4_vae/part4_vae_manifold.png`
  - `outputs/part4_vae/part4_vae_history.csv`
  - `outputs/part4_vae/part4_vae_metrics.json`
  - `checkpoints/part4_vae/best_vae.pt`
- metrics：
  - device `cuda`
  - epochs `50`
  - latent_dim `2`
  - best_val_loss `91.1664`

建议现场讲法：

- Encoder 输出 `mu` 和 `logvar`，不是直接输出一个固定向量。
- Reparameterization trick 让随机采样可以参与反向传播。
- Loss 是 reconstruction loss + KL divergence。
- 因为 latent dim 是 2，所以可以直接在二维 latent 网格上 decode 出 manifold 图。

## Part 4 Task 2 - UNet

PDF 要求：

- 用 OASIS 做 MR brain segmentation。
- 网络输出必须是 categorical/one-hot style segmentation。
- 每个 label 的 DSC 要超过 0.9。
- 必须 visualise segmentation results。
- demo 时要在 test set 上运行 inference 并展示模型正常工作。

当前完成情况：

- `scripts/part4_unet.py` 训练 UNet。
- `src/comp3710_lab2/unet.py` 实现 UNet encoder/decoder/skip connections。
- 输出 4 个 channel，对应 mask 类别 0/1/2/3。
- mask 从原始 `0/85/170/255` 映射为 class index。
- loss 是 CrossEntropy + soft Dice loss。
- 已保存：
  - `outputs/part4_unet/part4_unet_examples.png`
  - `outputs/part4_unet/part4_unet_history.csv`
  - `outputs/part4_unet/part4_unet_metrics.json`
  - `checkpoints/part4_unet/best_unet.pt`
- metrics：
  - device `cuda`
  - epochs `80`
  - best_mean_dice `0.9653`
  - final rows 中各类 Dice 大约：class0 `0.9993`，class1 `0.9280`，class2 `0.9473`，class3 `0.9704`

建议现场讲法：

- 这是当前 Part 4 最稳的高分证据。
- 展示 `part4_unet_examples.png`，说明三列分别是 MRI、ground truth、prediction。
- 用 `--eval-only --checkpoint checkpoints/part4_unet/best_unet.pt` 在 Rangpur 上现场跑 inference。

## Part 4 Task 3 - GAN

PDF 要求：

- 用 OASIS 数据训练 GAN 生成 realistic brain images。
- 必须提供 generated images、training loss plots 等训练证据。
- realism 由 instructor 判断。
- full marks 需要生成结果像 unique brains，mode collapse 要解决。

当前完成情况：

- `scripts/part4_gan.py` 训练 DCGAN。
- `src/comp3710_lab2/gan.py` 实现 DCGANGenerator 和 DCGANDiscriminator。
- 正式 80 epochs GPU 结果已同步回本地。
- 已保存：
  - `outputs/part4_gan/samples/epoch_001.png` 到 `epoch_080.png`，共 17 张
  - `outputs/part4_gan/part4_gan_losses.png`
  - `outputs/part4_gan/part4_gan_history.csv`
  - `outputs/part4_gan/part4_gan_metrics.json`
  - `checkpoints/part4_gan/latest_gan.pt`
- metrics：
  - device `cuda`
  - epochs `80`
  - seconds `848.7`

当前风险：

- GAN 的分数最依赖老师主观判断。图像如果偏模糊，可能拿 partial marks。
- 如果想更稳，可以继续调 GAN，但会占用 GPU 排队时间。

## GitHub 要求

PDF 要求：

- Part 4 明确要求在自己的 GitHub account 中创建项目。
- 要有 relevant commit logs。
- demo 时 demonstrator 可能要求登录账号证明是自己的。
- GitHub project 要有合理 README、documentation、meaningful commit messages。
- 不应上传大数据集和 checkpoint。

当前完成情况：

- 本地是 Git 仓库，有多条 meaningful commits。
- `.gitignore` 已排除：
  - `.venv/`
  - `data/`
  - `outputs/`
  - `checkpoints/`
  - `*.pt`
  - `slurm-*.out`
- `README.md` 和 `docs/` 已说明运行方法、Rangpur、GitHub、Part 3 ablation。
- 当前没有 `origin` 远端，所以还没有真正上传到 GitHub。

关于“没有注释版本上传 GitHub”：

- 当前详细中文注释都是未提交修改，不会被 push。
- `HEAD` 里的代码文件没有中文注释。
- 但是 PDF 的 marking criteria 提到代码注释和结构也占 1 mark，所以完全无注释版本对评分可能不如保留适量英文/中文关键注释。

建议：

- 如果你坚持 GitHub 不要这轮中文注释：直接推当前已提交 `HEAD`。
- 如果你要最大化评分：建议把中文注释整理成适度英文注释后提交到 GitHub，而不是完全无注释。
