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


def parse_args():
    parser = argparse.ArgumentParser(description="Part 4 Task 1: VAE for OASIS brain MRI slices.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "keras_png_slices_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device, beta: float, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl": 0.0, "count": 0}
    for images in loader:
        images = images.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        reconstruction, mu, logvar = model(images)
        loss, reconstruction_loss, kl = vae_loss(reconstruction, images, mu, logvar, beta=beta)
        if train:
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
    images = next(iter(loader)).to(device)
    with torch.no_grad():
        reconstruction, _, _ = model(images[:8])
    comparison = torch.cat([images[:8].cpu(), reconstruction.cpu()], dim=0)
    save_image(comparison, path, nrow=8)


def save_manifold(model, device, path: Path, grid_size: int = 20, span: float = 3.0) -> None:
    if model.latent_dim != 2:
        return
    model.eval()
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

    save_reconstructions(model, val_loader, device, out / "part4_vae_reconstructions.png")
    save_manifold(model, device, out / "part4_vae_manifold.png")
    synchronize_device(device)
    total_seconds = time.perf_counter() - start
    write_csv(out / "part4_vae_history.csv", history)
    write_json(
        out / "part4_vae_metrics.json",
        {
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
