from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, image_size: int = 128, latent_dim: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        reduced = image_size // 8
        hidden_channels = base_channels * 4
        self.reduced = reduced
        self.hidden_channels = hidden_channels

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
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)
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
        hidden = self.encoder(x).flatten(1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        hidden = self.fc_decode(z)
        hidden = hidden.view(z.shape[0], self.hidden_channels, self.reduced, self.reduced)
        return self.decoder(hidden)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta: float = 1.0):
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction="sum") / target.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.shape[0]
    return reconstruction_loss + beta * kl, reconstruction_loss, kl
