from __future__ import annotations

import csv
import atexit
import json
import os
import random
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO
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


class _TeeStream:
    """Write terminal output to both the terminal and a persistent log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self):
        return self.terminal.encoding


def start_run_log(output: str | Path, name: str) -> Path:
    """Mirror stdout/stderr into a timestamped log under the run output directory."""
    log_dir = ensure_dir(Path(output) / "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{name}_{timestamp}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = _TeeStream(original_stdout, log_file)
    tee_stderr = _TeeStream(original_stderr, log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    def close_log() -> None:
        if sys.stdout is tee_stdout:
            sys.stdout = original_stdout
        if sys.stderr is tee_stderr:
            sys.stderr = original_stderr
        log_file.flush()
        log_file.close()

    atexit.register(close_log)
    print(f"Run started: {datetime.now().astimezone().isoformat()}")
    print(f"Command: {shlex.join(sys.argv)}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Log file: {log_path}")
    return log_path


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
