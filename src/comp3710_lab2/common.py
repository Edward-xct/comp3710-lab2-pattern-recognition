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
# Matplotlib 在服务器/VS Code 终端里不需要弹窗，统一用 Agg 后端直接保存图片。
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


def set_matplotlib_cache() -> None:
    """Keep Matplotlib cache files inside this project."""
    # 把 cache 放进项目目录，避免 Rangpur 或受限环境里写默认 home/cache 出错。
    (PROJECT_ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    # 所有输出目录/checkpoint 目录都通过这里创建，保证父目录不存在时也能正常运行。
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(*parts: str) -> Path:
    # 标准输出位置：outputs/<part_name>/...，便于统一收集 demo 结果。
    return ensure_dir(PROJECT_ROOT / "outputs" / Path(*parts))


def checkpoint_dir(*parts: str) -> Path:
    # 标准模型保存位置：checkpoints/<part_name>/...，和输出图片/metrics 分开。
    return ensure_dir(PROJECT_ROOT / "checkpoints" / Path(*parts))


def set_seed(seed: int = 42) -> None:
    # 固定 Python、NumPy 和 PyTorch 随机种子，让实验结果尽量可复现。
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 对固定输入尺寸的 CNN，cudnn.benchmark=True 往往能选择更快的卷积实现。
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def choose_device(prefer_gpu: bool = True):
    import torch

    # 优先 CUDA：Rangpur A100 会走这里；本地 Mac 如果支持则退到 MPS；最后才是 CPU。
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def synchronize_device(device) -> None:
    try:
        import torch

        # CUDA 默认异步执行；计时前后同步才能得到真实运行时间。
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        pass


def parameter_count(model) -> int:
    # 只统计 requires_grad=True 的参数，表示训练中真正会被更新的模型参数量。
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
    # JSON 用来保存最终指标和配置，demo 时不用重新训练也能核对结果。
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv(path: str | Path, rows: Iterable[Mapping]) -> Path:
    # CSV 用来保存每个 epoch 的 history，方便画 loss/accuracy 曲线。
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
