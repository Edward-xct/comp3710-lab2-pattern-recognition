from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from comp3710_lab2.common import (
    checkpoint_dir,
    choose_device,
    output_dir,
    parameter_count,
    set_seed,
    start_run_log,
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
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional subdirectory name for preserving separate experiment runs.",
    )
    parser.add_argument(
        "--stage-index",
        type=int,
        default=None,
        help="Optional numeric stage for ordering ablation results in summaries.",
    )
    parser.add_argument(
        "--stage-note",
        type=str,
        default="",
        help="Short note describing the change tested in this experiment.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable CIFAR10 random crop, flip, and random erasing for baseline runs.",
    )
    parser.add_argument("--random-erasing-prob", type=float, default=0.25)
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def make_loaders(args):
    if args.no_augment:
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        )
    else:
        train_steps = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
        if args.random_erasing_prob > 0.0:
            train_steps.append(transforms.RandomErasing(p=args.random_erasing_prob))
        train_transform = transforms.Compose(train_steps)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    train_set = datasets.CIFAR10(
        root=args.data_dir, train=True, download=args.download, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=args.data_dir, train=False, download=args.download, transform=test_transform
    )
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

    output_parts = ["part3_cifar_resnet18"]
    if args.experiment_name:
        output_parts.append(args.experiment_name)
    out = output_dir(*output_parts)
    ckpt_dir = checkpoint_dir(*output_parts)
    run_log = start_run_log(out, "part3_cifar_resnet18")
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
        metrics.update(
            {
                "experiment_name": args.experiment_name or "default",
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "run_log": str(run_log),
            }
        )
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
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        milestones = sorted({max(1, args.epochs // 2), max(1, (3 * args.epochs) // 4)})
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
    else:
        scheduler = None
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
        if scheduler is not None:
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
            f"train_loss={row['train_loss']:.4f} test_loss={row['test_loss']:.4f} "
            f"seconds={row['epoch_seconds']:.1f}"
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
            "experiment_name": args.experiment_name or "default",
            "stage_index": args.stage_index,
            "stage_note": args.stage_note,
            "amp": use_amp,
            "channels_last": bool(args.channels_last),
            "augmentation": not bool(args.no_augment),
            "random_erasing_prob": 0.0
            if args.no_augment
            else float(args.random_erasing_prob),
            "scheduler": args.scheduler,
            "learning_rate": float(args.lr),
            "width": int(args.width),
            "label_smoothing": float(args.label_smoothing),
            "weight_decay": float(args.weight_decay),
            "seconds": float(total_seconds),
            "best_accuracy": float(best_acc),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_test_loss": float(history[-1]["test_loss"]),
            "final_train_accuracy": float(history[-1]["train_accuracy"]),
            "final_accuracy": float(history[-1]["test_accuracy"]),
            "checkpoint": str(ckpt_dir / "best_cifar_resnet18.pt"),
            "run_log": str(run_log),
        },
    )
    print(f"Part 3.2 complete. Best accuracy: {best_acc:.4f}. Outputs: {out}")


if __name__ == "__main__":
    main()
