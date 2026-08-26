from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


def set_matplotlib_cache() -> None:
    """Keep Matplotlib cache files inside this project."""
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(*parts: str) -> Path:
    return ensure_dir(PROJECT_ROOT / "outputs" / Path(*parts))


def checkpoint_dir(*parts: str) -> Path:
    return ensure_dir(PROJECT_ROOT / "checkpoints" / Path(*parts))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def choose_device(prefer_gpu: bool = True):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device) -> None:
    try:
        import torch

        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        pass


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_json(path: str | Path, data: Mapping) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: str | Path, rows: Iterable[Mapping]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
