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


# Part 3.2 的目标：
# 1. 训练一个 CIFAR10 版 ResNet-18，目标是达到 90% 以上 accuracy；
# 2. 进一步用 regularisation、learning-rate schedule、AMP 和 channels-last 冲到约 94%；
# 3. 保存每次 ablation 的 metrics/checkpoint，用来展示“逐步改进”的真实过程。


def parse_args():
    parser = argparse.ArgumentParser(description="Part 3.2: Fast CIFAR10 ResNet-18 training.")
    # data-dir 是 CIFAR10 缓存目录；Rangpur 上使用 --download 自动准备数据。
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "cifar10")
    # 默认 80 epochs 对应最终 A100 版本；ablation 脚本会为不同 stage 覆盖这个参数。
    parser.add_argument("--epochs", type=int, default=80)
    # A100 上 batch 512 能提高吞吐。
    parser.add_argument("--batch-size", type=int, default=512)
    # 最终 recipe 用较大学习率配合 cosine decay；早期 baseline 使用较小 lr。
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
    # AMP 只在 CUDA 上启用，能用半精度加速 A100 训练，是 94%/速度分的重要证据。
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    # channels-last 改变 tensor 内存布局，通常能让 GPU convolution 更快。
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def make_loaders(args):
    if args.no_augment:
        # baseline 关闭 augmentation，用来展示调参前的较弱结果。
        # 这个 stage 容易过拟合训练集，所以测试 accuracy 会明显低一些。
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        )
    else:
        # CIFAR10 常用增强：随机裁剪和左右翻转提升泛化能力。
        # RandomCrop(padding=4) 相当于轻微平移物体，RandomHorizontalFlip 增加左右翻转样本。
        train_steps = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
        if args.random_erasing_prob > 0.0:
            # RandomErasing 模拟局部遮挡，进一步减少过拟合。
            # 它通常在已经有较好 augmentation 和 scheduler 后再加入，否则早期训练可能更难。
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
        # pin_memory=True 时，CUDA 从 CPU 拷贝 batch 到 GPU 会更快；CPU/MPS 下不会强行启用。
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
    # eval 阶段只前向传播，统计 loss、accuracy 和 inference 时间。
    # 这部分对应 demo 里老师可能要求的 inference/run evaluation。
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    inference_start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # 只有 CUDA 上启用 AMP；CPU/MPS 会自动保持普通精度。
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            # CrossEntropyLoss 返回 batch 平均值，所以乘 batch size 后再累计，最后得到全测试集平均。
            total += targets.numel()
            correct += (predictions == targets).sum().item()
            loss_total += loss.item() * targets.shape[0]
    if device.type == "cuda":
        # CUDA kernel 异步执行，统计 inference seconds 前要同步。
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
    # 每个 experiment 单独保存，避免 ablation 多轮结果互相覆盖。
    out = output_dir(*output_parts)
    ckpt_dir = checkpoint_dir(*output_parts)
    run_log = start_run_log(out, "part3_cifar_resnet18")
    train_loader, test_loader = make_loaders(args)
    device = choose_device()
    use_amp = bool(args.amp and device.type == "cuda")

    # width=64 是完整 ResNet-18 宽度；早期 ablation 用 width=32 展示小模型 baseline。
    model = resnet18_cifar(num_classes=10, width=args.width).to(device)
    if args.channels_last:
        # channels-last 对 GPU 卷积更友好，和 AMP 一起用于最终加速版本。
        model = model.to(memory_format=torch.channels_last)
    if args.checkpoint:
        # eval-only 或继续实验时可加载已保存的 best checkpoint。
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])

    if args.eval_only:
        # 只做推理评估，不训练；适合正式 demo 中展示 checkpoint 的 inference accuracy。
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
    # label smoothing 把 one-hot 标签软化，避免模型过度自信，通常能提升 CIFAR10 泛化。
    # SGD + momentum 是 CIFAR/ResNet 的经典组合；nesterov 通常能稍微加速收敛。
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    if args.scheduler == "cosine":
        # cosine scheduler 让学习率从大到小平滑衰减，后期更稳定。
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
                # 前向传播得到 10 个类别 logits；CrossEntropyLoss 内部会处理 softmax。
                logits = model(images)
                loss = criterion(logits, targets)
            # GradScaler 在 AMP 下避免半精度梯度下溢；非 CUDA 时相当于普通 backward。
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * targets.shape[0]
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
        if scheduler is not None:
            # 每个 epoch 后更新学习率；cosine 会逐渐减小 lr，帮助后期细调。
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
            # 只保存当前最好 accuracy 的模型，减少无用 checkpoint。
            # checkpoint 里也保存 args，之后能回看这个结果是由哪组参数跑出来的。
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
            # metrics.json 是 ablation summary 的数据来源，包含速度、精度和关键超参数。
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
