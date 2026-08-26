from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comp3710_lab2.common import output_dir, set_matplotlib_cache, set_seed
from comp3710_lab2.lfw import run_eigenfaces


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2: Eigenfaces with PCA and Random Forest.")
    parser.add_argument("--data-home", type=Path, default=ROOT / "data" / "lfw")
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--fast-dev", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    out = output_dir("part2_eigenfaces")
    if args.fast_dev:
        args.components = 50
        args.estimators = 50
    metrics = run_eigenfaces(
        out,
        data_home=args.data_home,
        n_components=args.components,
        n_estimators=args.estimators,
        max_depth=args.max_depth,
    )
    print(f"Part 2 complete. Accuracy: {metrics['accuracy']:.4f}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
