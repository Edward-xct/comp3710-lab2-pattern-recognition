from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn

from comp3710_lab2.common import (
    checkpoint_dir,
    choose_device,
    output_dir,
    parameter_count,
    set_matplotlib_cache,
    set_seed,
    write_csv,
    write_json,
)
from comp3710_lab2.oasis_png import make_oasis_loader
from comp3710_lab2.unet import UNet, dice_per_class, soft_dice_loss


PALETTE = np.array(
    [
        [0, 0, 0],
        [220, 40, 40],
        [40, 180, 90],
        [70, 120, 230],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Part 4 Task 2: UNet OASIS segmentation.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "keras_png_slices_data")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=["validate", "test"], default="test")
    parser.add_argument("--fast-dev", action="store_true")
    return parser.parse_args()


def evaluate(model, loader, device, criterion):
    model.eval()
    total = 0
    loss_total = 0.0
    dice_total = torch.zeros(4, device=device)
    batches = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks) + soft_dice_loss(logits, masks)
            dice = dice_per_class(logits, masks, num_classes=4)
            loss_total += loss.item() * images.shape[0]
            total += images.shape[0]
            dice_total += dice
            batches += 1
    dice_mean = dice_total / max(batches, 1)
    return {"loss": loss_total / total, "dice": dice_mean.detach().cpu().tolist()}


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask.clip(0, 3)]


def save_segmentation_examples(model, loader, device, path: Path) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu().numpy()
    images_np = images.cpu().numpy()[:, 0]
    masks_np = masks.numpy()
    rows = min(4, images_np.shape[0])
    fig, axes = plt.subplots(rows, 3, figsize=(8, 2.8 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i in range(rows):
        axes[i, 0].imshow(images_np[i], cmap="gray")
        axes[i, 0].set_title("MRI")
        axes[i, 1].imshow(colorize(masks_np[i]))
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(colorize(predictions[i]))
        axes[i, 2].set_title("Prediction")
        for ax in axes[i]:
            ax.set_xticks(())
            ax.set_yticks(())
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    if args.fast_dev:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 8)
        args.max_samples = 32
        args.num_workers = 0

    out = output_dir("part4_unet")
    ckpt_dir = checkpoint_dir("part4_unet")
    device = choose_device()
    model = UNet(in_channels=1, num_classes=4, base_channels=args.base_channels).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])

    criterion = nn.CrossEntropyLoss()
    eval_loader = make_oasis_loader(
        args.data,
        split=args.eval_split,
        with_masks=True,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if args.eval_only:
        metrics = evaluate(model, eval_loader, device, criterion)
        save_segmentation_examples(model, eval_loader, device, out / "part4_unet_examples.png")
        write_json(out / "part4_unet_eval_metrics.json", metrics)
        print(f"eval loss={metrics['loss']:.4f}, dice={metrics['dice']}")
        return

    train_loader = make_oasis_loader(
        args.data,
        split="train",
        with_masks=True,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )
    val_loader = make_oasis_loader(
        args.data,
        split="validate",
        with_masks=True,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_mean_dice = 0.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total = 0
        epoch_start = time.perf_counter()
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks) + args.dice_weight * soft_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.shape[0]
            total += images.shape[0]
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, criterion)
        mean_dice = float(np.mean(val_metrics["dice"]))
        row = {
            "epoch": epoch,
            "train_loss": train_loss / total,
            "val_loss": val_metrics["loss"],
            "dice_class_0": val_metrics["dice"][0],
            "dice_class_1": val_metrics["dice"][1],
            "dice_class_2": val_metrics["dice"][2],
            "dice_class_3": val_metrics["dice"][3],
            "mean_dice": mean_dice,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "mean_dice": best_mean_dice,
                },
                ckpt_dir / "best_unet.pt",
            )
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_mean_dice={mean_dice:.4f} dice={val_metrics['dice']}"
        )

    save_segmentation_examples(model, val_loader, device, out / "part4_unet_examples.png")
    write_csv(out / "part4_unet_history.csv", history)
    write_json(
        out / "part4_unet_metrics.json",
        {
            "device": str(device),
            "parameters": int(parameter_count(model)),
            "epochs": int(args.epochs),
            "image_size": int(args.image_size),
            "seconds": float(time.perf_counter() - start),
            "best_mean_dice": float(best_mean_dice),
            "best_checkpoint": str(ckpt_dir / "best_unet.pt"),
        },
    )
    print(f"Part 4 Task 2 complete. Best mean DSC: {best_mean_dice:.4f}. Outputs: {out}")


if __name__ == "__main__":
    main()
