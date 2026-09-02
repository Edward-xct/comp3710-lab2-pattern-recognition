from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "part3_cifar_resnet18.py"

# One-click VS Code smoke demo:
# - uses synthetic CIFAR-shaped tensors so it works even without internet
# - uses a small subset and one epoch so it finishes quickly on a laptop
# - does not replace the full Rangpur result used for final accuracy evidence
sys.argv = [
    str(SCRIPT),
    "--synthetic",
    "--fast-dev",
    "--experiment-name",
    "vscode_smoke_cifar",
]

runpy.run_path(str(SCRIPT), run_name="__main__")
