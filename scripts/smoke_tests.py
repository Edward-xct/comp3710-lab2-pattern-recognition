from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from comp3710_lab2.common import output_dir, set_seed
from comp3710_lab2.dft import benchmark_dft, write_dft_timings
from comp3710_lab2.gan import DCGANDiscriminator, DCGANGenerator
from comp3710_lab2.lfw import LfwCnn
from comp3710_lab2.oasis_png import OasisPngDataset
from comp3710_lab2.resnet_cifar import resnet18_cifar
from comp3710_lab2.unet import UNet, dice_per_class
from comp3710_lab2.vae import ConvVAE, vae_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fast import/DataLoader/model smoke tests.")
    parser.add_argument(
        "--oasis",
        type=Path,
        default=Path("/Users/xct/Downloads/keras_png_slices_data.zip"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    out = output_dir("smoke_tests")

    rows = benchmark_dft([16, 32], n_harmonics=5, max_naive_n=32, include_gpu=False)
    write_dft_timings(out / "smoke_dft_timings.csv", rows)
    print("DFT smoke test passed.")

    lfw_model = LfwCnn.build(50, 37, 7)
    lfw_logits = lfw_model(torch.randn(2, 1, 50, 37))
    assert lfw_logits.shape == (2, 7)
    print("LFW CNN forward pass passed.")

    cifar_model = resnet18_cifar(width=16)
    cifar_logits = cifar_model(torch.randn(2, 3, 32, 32))
    assert cifar_logits.shape == (2, 10)
    print("CIFAR ResNet forward pass passed.")

    vae = ConvVAE(image_size=64, latent_dim=2, base_channels=8)
    images = torch.rand(2, 1, 64, 64)
    reconstruction, mu, logvar = vae(images)
    loss, _, _ = vae_loss(reconstruction, images, mu, logvar)
    assert reconstruction.shape == images.shape and torch.isfinite(loss)
    print("VAE forward/loss passed.")

    unet = UNet(base_channels=8)
    masks = torch.randint(0, 4, (2, 64, 64))
    logits = unet(torch.rand(2, 1, 64, 64))
    dice = dice_per_class(logits, masks)
    assert logits.shape == (2, 4, 64, 64) and dice.shape[0] == 4
    print("UNet forward/DSC passed.")

    gen = DCGANGenerator(latent_dim=16, base_channels=8)
    disc = DCGANDiscriminator(base_channels=8)
    fake = gen(torch.randn(2, 16, 1, 1))
    scores = disc(fake)
    assert fake.shape == (2, 1, 64, 64) and scores.shape == (2,)
    print("GAN forward passes passed.")

    if args.oasis.exists():
        image_ds = OasisPngDataset(args.oasis, split="train", with_masks=False, image_size=64, max_samples=2)
        seg_ds = OasisPngDataset(args.oasis, split="train", with_masks=True, image_size=64, max_samples=2)
        image = image_ds[0]
        image2, mask = seg_ds[0]
        assert image.shape == (1, 64, 64)
        assert image2.shape == (1, 64, 64)
        assert mask.shape == (64, 64)
        assert int(mask.max()) <= 3
        print("OASIS zip dataset passed.")
    else:
        print(f"OASIS zip not found at {args.oasis}; skipped dataset check.")

    print(f"Smoke test outputs: {out}")


if __name__ == "__main__":
    main()
