from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from comp3710_lab2.common import (
    checkpoint_dir,
    choose_device,
    output_dir,
    parameter_count,
    set_seed,
    write_csv,
    write_json,
)
from comp3710_lab2.resnet_cifar import resnet18_cifar


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    parser = argparse.ArgumentParser(description="Part 3.2: Fast CIFAR10 ResNet-18 training.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "cifar10")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random data for training-loop smoke tests only.",
    )
    parser.add_argument("--fast-dev", action="store_true")
    return parser.parse_args()


def make_loaders(args):
    if args.synthetic:
        train_count = 2048 if args.fast_dev else 8192
        test_count = 1024 if args.fast_dev else 2048
        generator = torch.Generator().manual_seed(42)
        train_set = TensorDataset(
            torch.randn(train_count, 3, 32, 32, generator=generator),
            torch.randint(0, 10, (train_count,), generator=generator),
        )
        test_set = TensorDataset(
            torch.randn(test_count, 3, 32, 32, generator=generator),
            torch.randint(0, 10, (test_count,), generator=generator),
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                transforms.RandomErasing(p=0.25),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        )
        train_set = datasets.CIFAR10(
            root=args.data_dir, train=True, download=args.download, transform=train_transform
        )
        test_set = datasets.CIFAR10(
            root=args.data_dir, train=False, download=args.download, transform=test_transform
        )
        if args.fast_dev:
            train_set = Subset(train_set, range(2048))
            test_set = Subset(test_set, range(1024))
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, test_loader


def evaluate(model, loader, device, use_amp: bool):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    inference_start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            total += targets.numel()
            correct += (predictions == targets).sum().item()
            loss_total += loss.item() * targets.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "loss": loss_total / total,
        "accuracy": correct / total,
        "seconds": time.perf_counter() - inference_start,
    }


def main() -> None:
    args = parse_args()
    set_seed(42)
    if args.fast_dev:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 128)
        args.num_workers = 0
        args.width = min(args.width, 16)

    out = output_dir("part3_cifar_resnet18")
    ckpt_dir = checkpoint_dir("part3_cifar_resnet18")
    train_loader, test_loader = make_loaders(args)
    device = choose_device()
    use_amp = bool(args.amp and device.type == "cuda")

    model = resnet18_cifar(num_classes=10, width=args.width).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, use_amp=use_amp)
        write_json(out / "part3_cifar_eval_metrics.json", metrics)
        print(f"eval accuracy={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}")
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_acc = 0.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.perf_counter()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * targets.shape[0]
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_metrics = evaluate(model, test_loader, device, use_amp=use_amp)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / total,
            "train_accuracy": correct / total,
            "test_loss": eval_metrics["loss"],
            "test_accuracy": eval_metrics["accuracy"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "accuracy": best_acc,
                    "args": vars(args),
                },
                ckpt_dir / "best_cifar_resnet18.pt",
            )
        print(
            f"epoch {epoch:03d} train_acc={row['train_accuracy']:.4f} "
            f"test_acc={row['test_accuracy']:.4f} "
            f"train_loss={row['train_loss']:.4f} seconds={row['epoch_seconds']:.1f}"
        )

    total_seconds = time.perf_counter() - start
    write_csv(out / "part3_cifar_resnet18_history.csv", history)
    write_json(
        out / "part3_cifar_resnet18_metrics.json",
        {
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "parameters": int(parameter_count(model)),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "synthetic": bool(args.synthetic),
            "amp": use_amp,
            "channels_last": bool(args.channels_last),
            "seconds": float(total_seconds),
            "best_accuracy": float(best_acc),
            "final_accuracy": float(history[-1]["test_accuracy"]),
            "checkpoint": str(ckpt_dir / "best_cifar_resnet18.pt"),
        },
    )
    print(f"Part 3.2 complete. Best accuracy: {best_acc:.4f}. Outputs: {out}")


if __name__ == "__main__":
    main()
