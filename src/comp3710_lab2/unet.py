from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # UNet 每个层级常用两个 3x3 卷积来提取局部纹理和边界信息。
        # BatchNorm 让训练更稳定，ReLU 提供非线性表达能力。
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 返回同分辨率的新特征图，供 encoder/decoder 继续处理。
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        # Encoder 路径逐步下采样，获得更大感受野。
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base_channels * 4, base_channels * 8)

        # Decoder 路径逐步上采样，并与 encoder 特征做 skip connection。
        # skip connection 是 UNet 的关键：深层特征负责语义，浅层特征保留边界位置。
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        # d1/d2/d3 会被保存下来，后面 decoder 上采样时拼接回去。
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bridge = self.bridge(self.pool3(d3))
        x = self.up3(bridge)
        # 拼接同尺度 encoder 特征，帮助恢复 segmentation 边界细节。
        # 拼接后通道数翻倍，所以 dec3/dec2/dec1 的输入通道按两路特征相加设计。
        x = self.dec3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, d1], dim=1))
        return self.out(x)


def dice_per_class(logits, targets, num_classes: int = 4, eps: float = 1e-6):
    # DSC = 2 * overlap / (prediction + target)，按类别分别计算。
    # argmax 把模型输出的 4 个 channel 转成最终类别 mask。
    predictions = logits.argmax(dim=1)
    pred_one_hot = F.one_hot(predictions, num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    # 在 batch、高、宽三个维度上求和，得到每个类别的整体 overlap。
    intersection = (pred_one_hot * target_one_hot).sum(dim=dims)
    denominator = pred_one_hot.sum(dim=dims) + target_one_hot.sum(dim=dims)
    return (2 * intersection + eps) / (denominator + eps)


def soft_dice_loss(logits, targets, num_classes: int = 4, eps: float = 1e-6):
    # soft dice 用概率而不是 argmax，训练时可以反向传播。
    # 这能直接优化分割区域的重叠程度，弥补 CrossEntropy 对类别不平衡的敏感。
    probabilities = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probabilities * target_one_hot).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()
