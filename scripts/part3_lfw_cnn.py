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
    write_csv,
    write_json,
)
from comp3710_lab2.lfw import LfwCnn, load_lfw


def parse_args():
    parser = argparse.ArgumentParser(description="Part 3.1: CNN classifier for the LFW dataset.")
    parser.add_argument("--data-home", type=Path, default=ROOT / "data" / "lfw")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fast-dev", action="store_true")
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
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
    if args.fast_dev:
        args.epochs = 1
        args.batch_size = 16
        args.num_workers = 0

    lfw_people = load_lfw(data_home=args.data_home, min_faces=70, resize=0.4)
    x = lfw_people.images.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    y = lfw_people.target.astype(np.int64)
    n_samples, height, width = x.shape
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    if args.fast_dev:
        x_train = x_train[:128]
        y_train = y_train[:128]
        x_test = x_test[:64]
        y_test = y_test[:64]

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
    model = LfwCnn.build(height, width, len(lfw_people.target_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_acc = 0.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.shape[0]
            seen += targets.shape[0]
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / seen,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        history.append(row)
        if test_acc > best_acc:
            best_acc = test_acc
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
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    final_loss, final_acc, all_targets, all_predictions = evaluate(model, test_loader, device)
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
            "seconds": float(time.perf_counter() - start),
            "final_loss": float(final_loss),
            "final_accuracy": float(final_acc),
            "best_accuracy": float(best_acc),
        },
    )

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
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


if __name__ == "__main__":
    main()
