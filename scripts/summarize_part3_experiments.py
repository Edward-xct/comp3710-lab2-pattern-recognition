from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "part3_cifar_resnet18"
METRICS_NAME = "part3_cifar_resnet18_metrics.json"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


FIELDS = [
    "stage_index",
    "experiment_name",
    "best_accuracy",
    "final_accuracy",
    "seconds",
    "epochs",
    "width",
    "learning_rate",
    "scheduler",
    "augmentation",
    "random_erasing_prob",
    "label_smoothing",
    "amp",
    "channels_last",
    "stage_note",
    "checkpoint",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise saved Part 3.2 CIFAR10 experiment metrics."
    )
    parser.add_argument("root", type=Path, nargs="?", default=DEFAULT_ROOT)
    parser.add_argument(
        "--name-prefix",
        type=str,
        default=None,
        help="Only include experiments whose directory/name starts with this prefix.",
    )
    return parser.parse_args()


def metric_paths(root: Path) -> list[Path]:
    paths = []
    root_metric = root / METRICS_NAME
    if root_metric.exists():
        paths.append(root_metric)
    paths.extend(sorted(root.glob(f"*/{METRICS_NAME}")))
    return paths


def sort_key(row: dict):
    stage = row.get("stage_index", "")
    try:
        stage_number = int(stage)
        has_stage = True
    except (TypeError, ValueError):
        stage_number = 10_000
        has_stage = False
    return (not has_stage, stage_number, str(row.get("experiment_name", "")))


def load_rows(root: Path, name_prefix: str | None = None) -> list[dict]:
    rows = []
    for path in metric_paths(root):
        data = json.loads(path.read_text(encoding="utf-8"))
        experiment_name = data.get("experiment_name")
        if not experiment_name or experiment_name == "default":
            experiment_name = "default" if path.parent == root else path.parent.name
        if name_prefix and not str(experiment_name).startswith(name_prefix):
            continue
        row = {
            "stage_index": data.get("stage_index", ""),
            "experiment_name": experiment_name,
        }
        for field in FIELDS[2:]:
            row[field] = data.get(field, "")
        rows.append(row)
    rows.sort(key=sort_key)
    return rows


def try_write_plot(root: Path, rows: list[dict]) -> Path | None:
    try:
        (ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labeled_rows = [row for row in rows if row.get("best_accuracy") != ""]
    if not labeled_rows:
        return None

    labels = [str(row["experiment_name"]).replace("_", "\n") for row in labeled_rows]
    values = [float(row["best_accuracy"]) * 100.0 for row in labeled_rows]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 5.5))
    ax.plot(range(len(values)), values, marker="o", linewidth=2.5, color="#2457a6")
    ax.bar(range(len(values)), values, alpha=0.2, color="#2457a6")
    ax.axhline(90.0, color="#a63d24", linestyle="--", linewidth=1.5, label="90% target")
    ax.axhline(94.0, color="#2e7d32", linestyle="--", linewidth=1.5, label="94% stretch")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_title("Part 3.2 CIFAR10 ResNet-18 ablation improvement")
    ax.set_ylim(max(0.0, min(values) - 8.0), min(100.0, max(values) + 4.0))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for i, value in enumerate(values):
        ax.annotate(f"{value:.2f}%", (i, value), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    path = root / "part3_cifar_experiment_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_summary(root: Path, rows: list[dict]) -> tuple[Path, Path, Path | None]:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "part3_cifar_experiment_summary.csv"
    md_path = root / "part3_cifar_experiment_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Part 3.2 CIFAR10 Experiment Summary",
        "",
        "| Stage | Experiment | Best Acc | Final Acc | Seconds | Epochs | Key Settings | Note |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        settings = (
            f"width={row['width']}, lr={row['learning_rate']}, "
            f"scheduler={row['scheduler']}, aug={row['augmentation']}, "
            f"erase={row['random_erasing_prob']}, smoothing={row['label_smoothing']}, "
            f"amp={row['amp']}, channels_last={row['channels_last']}"
        )
        lines.append(
            f"| {row['stage_index']} | {row['experiment_name']} | {row['best_accuracy']} | "
            f"{row['final_accuracy']} | {row['seconds']} | {row['epochs']} | "
            f"{settings} | {row['stage_note']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_path = try_write_plot(root, rows)
    return csv_path, md_path, plot_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root, name_prefix=args.name_prefix)
    if not rows:
        raise SystemExit(f"No {METRICS_NAME} files found under {args.root}")
    csv_path, md_path, plot_path = write_summary(args.root, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if plot_path:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
