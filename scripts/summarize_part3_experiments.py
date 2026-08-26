from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs" / "part3_cifar_resnet18"
METRICS_NAME = "part3_cifar_resnet18_metrics.json"


FIELDS = [
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
    "checkpoint",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise saved Part 3.2 CIFAR10 experiment metrics."
    )
    parser.add_argument("root", type=Path, nargs="?", default=DEFAULT_ROOT)
    return parser.parse_args()


def metric_paths(root: Path) -> list[Path]:
    paths = []
    root_metric = root / METRICS_NAME
    if root_metric.exists():
        paths.append(root_metric)
    paths.extend(sorted(root.glob(f"*/{METRICS_NAME}")))
    return paths


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in metric_paths(root):
        data = json.loads(path.read_text(encoding="utf-8"))
        experiment_name = data.get("experiment_name")
        if not experiment_name or experiment_name == "default":
            experiment_name = "default" if path.parent == root else path.parent.name
        row = {"experiment_name": experiment_name}
        for field in FIELDS[1:]:
            row[field] = data.get(field, "")
        rows.append(row)
    return rows


def write_summary(root: Path, rows: list[dict]) -> tuple[Path, Path]:
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
        "| Experiment | Best Acc | Final Acc | Seconds | Epochs | Key Settings |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        settings = (
            f"width={row['width']}, lr={row['learning_rate']}, "
            f"scheduler={row['scheduler']}, aug={row['augmentation']}, "
            f"erase={row['random_erasing_prob']}, smoothing={row['label_smoothing']}, "
            f"amp={row['amp']}, channels_last={row['channels_last']}"
        )
        lines.append(
            f"| {row['experiment_name']} | {row['best_accuracy']} | "
            f"{row['final_accuracy']} | {row['seconds']} | {row['epochs']} | {settings} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root)
    if not rows:
        raise SystemExit(f"No {METRICS_NAME} files found under {args.root}")
    csv_path, md_path = write_summary(args.root, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
