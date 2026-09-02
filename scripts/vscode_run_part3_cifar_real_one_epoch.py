from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "part3_cifar_resnet18.py"

# One-click VS Code real CIFAR10 demo run.
# This requires the CIFAR10 files to be available locally, or internet access for
# torchvision to download them. Use the synthetic wrapper if you only need a
# guaranteed quick smoke test.
sys.argv = [
    str(SCRIPT),
    "--download",
    "--fast-dev",
    "--experiment-name",
    "vscode_real_cifar_one_epoch",
]

runpy.run_path(str(SCRIPT), run_name="__main__")
