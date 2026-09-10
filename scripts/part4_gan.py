from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
from torchvision.utils import save_image

from comp3710_lab2.common import (
    checkpoint_dir,
    choose_device,
    output_dir,
    parameter_count,
    set_matplotlib_cache,
    set_seed,
    start_run_log,
    synchronize_device,
    write_csv,
    write_json,
)
from comp3710_lab2.gan import DCGANDiscriminator, DCGANGenerator
from comp3710_lab2.oasis_png import make_oasis_loader


# Part 4 Task 3 的目标：
# 1. 训练 DCGAN 生成 OASIS 风格的脑部 MRI slice；
# 2. 保存每隔几个 epoch 的 samples，用图片展示生成质量逐渐改善；
# 3. 保存 loss 曲线和 checkpoint，作为“确实训练过”的证据。
def parse_args():
    parser = argparse.ArgumentParser(description="Part 4 Task 3: DCGAN for OASIS brain MRI slices.")
    # GAN 只需要真实 MRI 图片，不需要 segmentation mask。
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "keras_png_slices_data")
    # 正式结果用 80 epochs；sample-every=5 会保留多个中间阶段图。
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    # latent-dim 是随机噪声向量维度，generator 从这个向量生成一张图。
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    # DCGAN 论文常用 Adam beta1=0.5，比默认 0.9 更适合 GAN 的不稳定对抗训练。
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    if args.image_size != 64:
        # 当前 DCGAN 上采样层固定为 1x1 -> 4 -> 8 -> 16 -> 32 -> 64。
        raise ValueError("This DCGAN architecture expects --image-size 64.")

    out = output_dir("part4_gan")
    samples_dir = output_dir("part4_gan", "samples")
    ckpt_dir = checkpoint_dir("part4_gan")
    run_log = start_run_log(out, "part4_gan")
    loader = make_oasis_loader(
        args.data,
        split="train",
        with_masks=False,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        value_range="minus_one_one",
    )
    device = choose_device()
    # DCGAN 包含一个 generator 和一个 discriminator，二者交替训练。
    # generator 输出范围是 [-1,1]，所以 DataLoader 也把真实图缩放到 [-1,1]。
    generator = DCGANGenerator(args.latent_dim, args.base_channels).to(device)
    discriminator = DCGANDiscriminator(args.base_channels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # fixed_noise 固定不变，便于比较 epoch_001 到 epoch_080 的生成进步。
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
    history = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        generator.train()
        discriminator.train()
        d_loss_total = 0.0
        g_loss_total = 0.0
        total = 0
        for real_images in loader:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.shape[0]

            # 每个 batch 都重新采样随机 z，让 generator 学会从不同 latent code 生成不同脑图。
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)

            # 先训练 discriminator：真实图标签为 1，生成图标签为 0。
            # fake_images.detach() 阻止 discriminator 更新时梯度传回 generator。
            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images.detach())
            d_loss_real = criterion(real_logits, torch.ones_like(real_logits))
            d_loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            opt_d.step()

            # 再训练 generator：希望生成图被 discriminator 判断为真实。
            # 这次不 detach，因为 generator 需要根据 discriminator 的反馈更新自己。
            opt_g.zero_grad(set_to_none=True)
            fake_logits_for_g = discriminator(fake_images)
            g_loss = criterion(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            g_loss.backward()
            opt_g.step()

            d_loss_total += d_loss.item() * batch_size
            g_loss_total += g_loss.item() * batch_size
            total += batch_size

        row = {
            "epoch": epoch,
            "d_loss": d_loss_total / total,
            "g_loss": g_loss_total / total,
        }
        synchronize_device(device)
        row["epoch_seconds"] = time.perf_counter() - epoch_start
        history.append(row)
        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            generator.eval()
            with torch.no_grad():
                generated = generator(fixed_noise).cpu()
            # 定期保存同一组 latent noise 的生成结果，用来展示 GAN 训练过程。
            # 保存前把 [-1,1] 还原到 [0,1]，否则图片会显示异常。
            save_image((generated + 1.0) / 2.0, samples_dir / f"epoch_{epoch:03d}.png", nrow=8)
            torch.save(
                {
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                },
                ckpt_dir / "latest_gan.pt",
            )
        print(
            f"epoch {epoch:03d} d_loss={row['d_loss']:.4f} "
            f"g_loss={row['g_loss']:.4f} seconds={row['epoch_seconds']:.2f}"
        )

    synchronize_device(device)
    total_seconds = time.perf_counter() - start
    write_csv(out / "part4_gan_history.csv", history)
    write_json(
        out / "part4_gan_metrics.json",
        {
            # GAN 没有像分类 accuracy 那样的单一分数，所以这里记录训练配置、耗时和模型规模。
            "device": str(device),
            "generator_parameters": int(parameter_count(generator)),
            "discriminator_parameters": int(parameter_count(discriminator)),
            "epochs": int(args.epochs),
            "image_size": int(args.image_size),
            "seconds": float(total_seconds),
            "final_discriminator_loss": float(history[-1]["d_loss"]),
            "final_generator_loss": float(history[-1]["g_loss"]),
            "checkpoint": str(ckpt_dir / "latest_gan.pt"),
            "run_log": str(run_log),
        },
    )
    import matplotlib.pyplot as plt

    # loss 曲线是训练证据之一，但 GAN 质量最终还需要人工看生成图片。
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([h["epoch"] for h in history], [h["d_loss"] for h in history], label="D loss")
    ax.plot([h["epoch"] for h in history], [h["g_loss"] for h in history], label="G loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "part4_gan_losses.png", dpi=160)
    plt.close(fig)
    print(f"Part 4 Task 3 complete. Outputs: {out}")
    print(f"Total training time: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
