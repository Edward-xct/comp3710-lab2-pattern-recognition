from __future__ import annotations

import torch.nn as nn


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        # Generator 从 1x1 latent vector 开始，用反卷积逐步生成 64x64 MRI-like 图像。
        # 每个 ConvTranspose2d 基本都会把空间尺寸放大 2 倍，通道数逐步减少到 1 个灰度通道。
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 1, 4, 2, 1, bias=False),
            # Tanh 输出范围是 [-1, 1]，和 GAN 数据预处理保持一致。
            nn.Tanh(),
        )

    def forward(self, z):
        # z 的 shape 是 [batch, latent_dim, 1, 1]，输出是 [batch, 1, 64, 64]。
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        # Discriminator 用卷积下采样，把输入图片压成一个真假 logit。
        # LeakyReLU 比普通 ReLU 更适合 GAN 判别器，负半轴仍保留一点梯度。
        self.net = nn.Sequential(
            nn.Conv2d(1, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        # 输出不是概率而是 logit，后面配 BCEWithLogitsLoss 更数值稳定。
        return self.net(x).flatten(1).squeeze(1)
