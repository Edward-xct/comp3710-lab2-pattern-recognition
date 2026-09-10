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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
from comp3710_lab2.lfw import LfwCnn, load_lfw


# Part 3.1 的目标：
# 1. 不再像 Part 2 那样手工做 PCA，而是让 CNN 直接从二维人脸图像学习特征；
# 2. 网络结构按任务要求使用两层 3x3 convolution，每层 32 个 filters；
# 3. 每个 epoch 输出测试集 accuracy，并保存 best checkpoint、history 图和分类报告。
def parse_args():
    parser = argparse.ArgumentParser(description="Part 3.1: CNN classifier for the LFW dataset.")
    # data-home 和 Part 2 一样使用 sklearn 的 LFW 缓存目录，避免重复下载。
    parser.add_argument("--data-home", type=Path, default=ROOT / "data" / "lfw")
    # 默认 20 epochs 是完整训练配置。
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    # Adam 对这个较小的人脸分类任务比较稳，1e-3 是常见起点。
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def evaluate(model, loader, device):
    # 评估阶段关闭 dropout/BN 更新，并用 no_grad 节省显存和时间。
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            # DataLoader 给出的是 CPU tensor；to(device) 让同一份代码可在 CPU/MPS/CUDA 上运行。
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            # 这里累计全部 batch 的正确数和样本数，最后得到整个测试集 accuracy。
            total += targets.numel()
            correct += (predictions == targets).sum().item()
            loss_total += loss.item() * targets.shape[0]
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    return loss_total / total, correct / total, np.array(all_targets), np.array(all_predictions)


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    out = output_dir("part3_lfw_cnn")
    ckpt_dir = checkpoint_dir("part3_lfw_cnn")
    run_log = start_run_log(out, "part3_lfw_cnn")

    # Part 3.1 复用 Part 2 的 LFW 数据，但保留二维图像结构给 CNN 使用。
    lfw_people = load_lfw(data_home=args.data_home, min_faces=70, resize=0.4)
    x = lfw_people.images.astype(np.float32)
    # sklearn 有些版本返回 0~255，有些返回 0~1；这里统一缩放到 0~1。
    if x.max() > 1.0:
        x = x / 255.0
    y = lfw_people.target.astype(np.int64)
    n_samples, height, width = x.shape
    # stratify=y 让训练/测试集的人物分布保持一致，避免某个名人在测试集中比例异常。
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    # 卷积层要求输入形状是 N,C,H,W，所以这里增加灰度通道 C=1。
    train_ds = TensorDataset(
        torch.from_numpy(x_train[:, None, :, :]),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(x_test[:, None, :, :]),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    device = choose_device()
    # LfwCnn.build 根据图片尺寸自动计算 flatten 后的维度，避免手写魔法数字。
    model = LfwCnn.build(height, width, len(lfw_people.target_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_acc = 0.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_correct = 0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            # 反向传播更新 CNN 权重；Adam 负责自适应调整每个参数的步长。
            # 训练集只用于更新参数，测试集只在 evaluate 中看泛化效果。
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.shape[0]
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.shape[0]
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
        synchronize_device(device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / seen,
            "train_accuracy": train_correct / seen,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存测试集表现最好的 checkpoint，demo 时可证明模型训练结果已保留。
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "height": height,
                    "width": width,
                    "target_names": [str(name) for name in lfw_people.target_names],
                },
                ckpt_dir / "best_lfw_cnn.pt",
            )
        print(
            f"epoch {epoch:03d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} test_loss={test_loss:.4f} "
            f"test_acc={test_acc:.4f} seconds={row['epoch_seconds']:.2f}"
        )

    final_loss, final_acc, all_targets, all_predictions = evaluate(model, test_loader, device)
    synchronize_device(device)
    total_seconds = time.perf_counter() - start
    # classification report 展示每个人物类别的 precision/recall/F1。
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=lfw_people.target_names,
        zero_division=0,
    )
    (out / "part3_lfw_cnn_classification_report.txt").write_text(report, encoding="utf-8")
    write_csv(out / "part3_lfw_cnn_history.csv", history)
    write_json(
        out / "part3_lfw_cnn_metrics.json",
        {
            "n_samples": int(n_samples),
            "height": int(height),
            "width": int(width),
            "n_classes": int(len(lfw_people.target_names)),
            "parameters": int(parameter_count(model)),
            "device": str(device),
            "epochs": int(args.epochs),
            "seconds": float(total_seconds),
            "final_loss": float(final_loss),
            "final_accuracy": float(final_acc),
            "best_accuracy": float(best_acc),
            "run_log": str(run_log),
        },
    )

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    # history 图把 loss 和 accuracy 放在同一张图里，汇报时可以说明训练是否收敛。
    ax1.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train loss")
    ax1.plot([h["epoch"] for h in history], [h["test_loss"] for h in history], label="test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot([h["epoch"] for h in history], [h["test_accuracy"] for h in history], "g", label="test acc")
    ax2.set_ylabel("Accuracy")
    ax1.grid(True)
    fig.legend(loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "part3_lfw_cnn_history.png", dpi=160)
    plt.close(fig)

    print(f"Part 3.1 complete. Final accuracy: {final_acc:.4f}. Outputs: {out}")
    print(f"Total training and evaluation time: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()
