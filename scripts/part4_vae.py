from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
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
from comp3710_lab2.oasis_png import make_oasis_loader
from comp3710_lab2.vae import ConvVAE, vae_loss


# Part 4 Task 1 的目标：
# 1. 在 OASIS 脑部 MRI slice 上训练 Variational Autoencoder；
# 2. VAE 学到一个连续 latent space，可以从 z 采样生成/重建脑图；
# 3. 输出 reconstruction 图和 2D manifold 图，作为 recognition task 的可视化证据。
def parse_args():
    parser = argparse.ArgumentParser(description="Part 4 Task 1: VAE for OASIS brain MRI slices.")
    # Rangpur 上正式数据路径通常是 /home/groups/comp3710/OASIS；本地默认路径用于下载/解压版本。
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "keras_png_slices_data")
    # 50 epochs 是已经跑过的正式配置。
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    # VAE 使用 128x128 输入，比 GAN 的 64x64 更适合展示重建细节。
    parser.add_argument("--image-size", type=int, default=128)
    # latent_dim=2 是为了能直接画二维 manifold；更高维可能重建更好但不好可视化。
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device, beta: float, train: bool):
    # train=True 时更新参数；train=False 时只计算验证集 loss。
    # 同一个函数处理训练和验证，保证两边 loss 计算方式完全一致。
    model.train(train)
    totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl": 0.0, "count": 0}
    for images in loader:
        images = images.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(images)
        # reconstruction_loss 衡量图像重建误差；KL 让 latent distribution 接近标准正态分布。
        loss, reconstruction_loss, kl = vae_loss(reconstruction, images, mu, logvar, beta=beta)
        if train:
            # VAE 同时优化重建质量和 latent distribution 的 KL regularisation。
            loss.backward()
            optimizer.step()
        batch_size = images.shape[0]
        totals["loss"] += loss.item() * batch_size
        totals["reconstruction_loss"] += reconstruction_loss.item() * batch_size
        totals["kl"] += kl.item() * batch_size
        totals["count"] += batch_size
    return {key: value / totals["count"] for key, value in totals.items() if key != "count"}


def save_reconstructions(model, loader, device, path: Path) -> None:
    model.eval()
    # 取验证集第一个 batch 的前 8 张，避免只展示训练集记忆效果。
    images = next(iter(loader)).to(device)
    with torch.no_grad():
        reconstruction, _, _ = model(images[:8])
    # 上排是真实 MRI，下排是 VAE reconstruction，方便 demo 直观看重建质量。
    comparison = torch.cat([images[:8].cpu(), reconstruction.cpu()], dim=0)
    save_image(comparison, path, nrow=8)


def save_manifold(model, device, path: Path, grid_size: int = 20, span: float = 3.0) -> None:
    if model.latent_dim != 2:
        return
    model.eval()
    # latent_dim=2 时可以在二维网格上采样，展示 latent space manifold。
    # 网格从 -3 到 3 覆盖标准正态的大部分概率质量，能看到不同 z 对生成脑图的影响。
    coords = torch.linspace(-span, span, grid_size, device=device)
    points = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        images = model.decode(points).cpu()
    save_image(images, path, nrow=grid_size)


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()

    out = output_dir("part4_vae")
    ckpt_dir = checkpoint_dir("part4_vae")
    run_log = start_run_log(out, "part4_vae")
    # VAE/GAN 只需要 MRI 图片本身，所以 with_masks=False。
    train_loader = make_oasis_loader(
        args.data,
        split="train",
        with_masks=False,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    val_loader = make_oasis_loader(
        args.data,
        split="validate",
        with_masks=False,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        shuffle=False,
    )

    device = choose_device()
    # OASIS MRI 是单通道灰度图，latent_dim=2 便于画二维 manifold。
    # choose_device 会优先用 CUDA，所以在 Rangpur GPU 节点上会自动跑到 A100。
    model = ConvVAE(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_val = float("inf")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.beta, train=True)
        with torch.no_grad():
            # 验证集不更新权重，只用于选择 best checkpoint。
            val_metrics = run_epoch(model, val_loader, optimizer, device, args.beta, train=False)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "train_kl": train_metrics["kl"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_kl": val_metrics["kl"],
        }
        synchronize_device(device)
        row["epoch_seconds"] = time.perf_counter() - epoch_start
        history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            # 保存验证集 loss 最低的模型，避免最后一轮波动影响展示结果。
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                ckpt_dir / "best_vae.pt",
            )
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
            f"train_recon={row['train_reconstruction_loss']:.4f} "
            f"train_kl={row['train_kl']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_recon={row['val_reconstruction_loss']:.4f} "
            f"val_kl={row['val_kl']:.4f} seconds={row['epoch_seconds']:.2f}"
        )

    # 训练结束后保存重建图、manifold、history 和 metrics，作为 demo 证据。
    save_reconstructions(model, val_loader, device, out / "part4_vae_reconstructions.png")
    save_manifold(model, device, out / "part4_vae_manifold.png")
    synchronize_device(device)
    total_seconds = time.perf_counter() - start
    write_csv(out / "part4_vae_history.csv", history)
    write_json(
        out / "part4_vae_metrics.json",
        {
            # 这些字段可以证明正式结果是在 GPU 上、用多少 epoch 和什么 latent_dim 跑出来的。
            "device": str(device),
            "parameters": int(parameter_count(model)),
            "epochs": int(args.epochs),
            "image_size": int(args.image_size),
            "latent_dim": int(args.latent_dim),
            "seconds": float(total_seconds),
            "best_val_loss": float(best_val),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_train_reconstruction_loss": float(history[-1]["train_reconstruction_loss"]),
            "final_train_kl": float(history[-1]["train_kl"]),
            "final_val_loss": float(history[-1]["val_loss"]),
            "final_val_reconstruction_loss": float(history[-1]["val_reconstruction_loss"]),
            "final_val_kl": float(history[-1]["val_kl"]),
            "checkpoint": str(ckpt_dir / "best_vae.pt"),
            "run_log": str(run_log),
        },
    )
    print(f"Part 4 Task 1 complete. Best val loss: {best_val:.4f}. Outputs: {out}")
    print(f"Total training and output time: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
