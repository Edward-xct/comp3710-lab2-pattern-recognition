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
    start_run_log,
    synchronize_device,
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


# Part 4 Task 2 的目标：
# 1. 用 UNet 对 OASIS MRI slice 做四分类语义分割；
# 2. 输出每个像素属于哪个脑组织/区域类别；
# 3. 用 Dice Similarity Coefficient 检查 prediction 和 ground truth mask 的重叠程度。
def parse_args():
    parser = argparse.ArgumentParser(description="Part 4 Task 2: UNet OASIS segmentation.")
    # 这里的数据必须包含 keras_png_slices_* 和 keras_png_slices_seg_* 两套目录。
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "keras_png_slices_data")
    # 正式结果用 80 epochs；你的 best_mean_dice 已经超过 0.96，满足 >0.9 的核心要求。
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    # dice-weight 控制 soft Dice loss 在总 loss 里的比例，默认和 CrossEntropy 同等重要。
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=["validate", "test"], default="test")
    return parser.parse_args()


def evaluate(model, loader, device, criterion):
    # 验证/测试时同时计算 loss 和每个类别的 Dice score。
    # 任务要求看 DSC，所以这里不只输出 loss，还按类别保存 Dice。
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
            # CrossEntropy 学习类别分类，soft Dice 直接优化 segmentation overlap。
            # 注意 evaluate 里不乘 args.dice_weight，只用于统一观察一个可比较的验证 loss。
            loss = criterion(logits, masks) + soft_dice_loss(logits, masks)
            dice = dice_per_class(logits, masks, num_classes=4)
            loss_total += loss.item() * images.shape[0]
            total += images.shape[0]
            dice_total += dice
            batches += 1
    dice_mean = dice_total / max(batches, 1)
    return {"loss": loss_total / total, "dice": dice_mean.detach().cpu().tolist()}


def colorize(mask: np.ndarray) -> np.ndarray:
    # 把类别 0/1/2/3 映射成颜色，方便人工检查 segmentation。
    # 黑色是背景，其余颜色分别代表三个非背景类别，保存到 examples.png。
    return PALETTE[mask.clip(0, 3)]


def save_segmentation_examples(model, loader, device, path: Path) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    images, masks = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu().numpy()
    # images_np 是原图，masks_np 是人工标注，predictions 是模型输出的类别图。
    images_np = images.cpu().numpy()[:, 0]
    masks_np = masks.numpy()
    rows = min(4, images_np.shape[0])
    fig, axes = plt.subplots(rows, 3, figsize=(8, 2.8 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i in range(rows):
        # 每一行展示 MRI、ground truth 和 prediction 三列，便于直接对比。
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

    out = output_dir("part4_unet")
    ckpt_dir = checkpoint_dir("part4_unet")
    run_log = start_run_log(out, "part4_unet")
    device = choose_device()
    # 输出 4 个 channel，对应 OASIS mask 的 4 个类别。
    # 对每个像素，logits 中最大的 channel 就是预测类别。
    model = UNet(in_channels=1, num_classes=4, base_channels=args.base_channels).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])

    criterion = nn.CrossEntropyLoss()
    # eval_loader 默认用 test split；训练时还会单独创建 train/validate loader。
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
        # eval-only 用已训练 checkpoint 做推理，适合正式 demo 现场展示“加载模型 -> 分割 -> 输出 Dice”。
        metrics = evaluate(model, eval_loader, device, criterion)
        synchronize_device(device)
        save_segmentation_examples(model, eval_loader, device, out / "part4_unet_examples.png")
        metrics["run_log"] = str(run_log)
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
    # CosineAnnealingLR 让学习率逐渐变小，训练后期更容易稳定在高 Dice。
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
            # 对所有类别的像素分类误差反向传播，更新 UNet 参数。
            # CrossEntropy 负责像素分类，soft Dice 直接鼓励预测区域和标注区域重叠。
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.shape[0]
            total += images.shape[0]
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, criterion)
        synchronize_device(device)
        # mean Dice 是 4 个类别 Dice 的平均值；你的正式结果约 0.965，已经超过任务要求。
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
            # 任务要求 DSC > 0.9，所以 checkpoint 以 mean Dice 最高为准。
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
            f"val_loss={row['val_loss']:.4f} val_mean_dice={mean_dice:.4f} "
            f"dice={val_metrics['dice']} seconds={row['epoch_seconds']:.2f}"
        )

    # 保存预测示例图和指标，demo 时展示模型确实会做 segmentation。
    save_segmentation_examples(model, val_loader, device, out / "part4_unet_examples.png")
    synchronize_device(device)
    total_seconds = time.perf_counter() - start
    write_csv(out / "part4_unet_history.csv", history)
    write_json(
        out / "part4_unet_metrics.json",
        {
            # metrics.json 保存 best_mean_dice 和 checkpoint 路径，方便 demo 前快速确认结果。
            "device": str(device),
            "parameters": int(parameter_count(model)),
            "epochs": int(args.epochs),
            "image_size": int(args.image_size),
            "seconds": float(total_seconds),
            "best_mean_dice": float(best_mean_dice),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_val_loss": float(history[-1]["val_loss"]),
            "final_mean_dice": float(history[-1]["mean_dice"]),
            "final_dice_per_class": [
                float(history[-1][f"dice_class_{index}"]) for index in range(4)
            ],
            "best_checkpoint": str(ckpt_dir / "best_unet.pt"),
            "run_log": str(run_log),
        },
    )
    print(f"Part 4 Task 2 complete. Best mean DSC: {best_mean_dice:.4f}. Outputs: {out}")
    print(f"Total training and output time: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
