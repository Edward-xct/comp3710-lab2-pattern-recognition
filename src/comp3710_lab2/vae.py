from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        if image_size % 8 != 0:
            # Encoder 连续下采样 3 次，每次 /2，所以输入尺寸必须能被 8 整除。
            raise ValueError("image_size must be divisible by 8")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        reduced = image_size // 8
        hidden_channels = base_channels * 4
        self.reduced = reduced
        self.hidden_channels = hidden_channels

        # Encoder 用三次 stride=2 卷积把 128x128 MRI 压缩到较小特征图。
        # 通道数从 base -> 2*base -> 4*base 增加，空间尺寸减小但语义特征更丰富。
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        flat = hidden_channels * reduced * reduced
        # VAE 不直接输出 z，而是输出潜变量分布的均值 mu 和方差 logvar。
        # 这样模型学到的是 p(z|x) 的分布，而不是一个固定点，后面才能从 latent space 连续采样。
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)
        # Decoder 用反卷积逐步上采样，把 latent vector 还原成 MRI slice。
        # 最后一层 Sigmoid 输出 [0,1]，和 OASIS 图片预处理后的像素范围一致。
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        # encode 返回 latent Gaussian 的两个参数：mu 和 logvar。
        hidden = self.encoder(x).flatten(1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        # reparameterization trick: z = mu + eps * sigma，使随机采样仍可反向传播。
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 先用全连接层把低维 z 扩回卷积特征图，再通过 decoder 还原图片。
        hidden = self.fc_decode(z)
        hidden = hidden.view(z.shape[0], self.hidden_channels, self.reduced, self.reduced)
        return self.decoder(hidden)

    def forward(self, x):
        # 完整 VAE 前向：图片 -> 分布参数 -> 采样 z -> reconstruction。
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta: float = 1.0):
    # VAE loss = reconstruction loss + beta * KL divergence。
    # reconstruction loss 越低，重建图越接近输入；KL 越低，latent space 越接近标准正态。
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction="sum") / target.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.shape[0]
    return reconstruction_loss + beta * kl, reconstruction_loss, kl
