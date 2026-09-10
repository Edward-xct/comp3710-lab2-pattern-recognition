from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comp3710_lab2.common import output_dir, set_matplotlib_cache, set_seed, start_run_log, write_json
from comp3710_lab2.lfw import run_eigenfaces


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2: Eigenfaces with PCA and Random Forest.")
    parser.add_argument("--data-home", type=Path, default=ROOT / "data" / "lfw")
    parser.add_argument("--components", type=int, default=150)
    parser.add_argument("--estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    out = output_dir("part2_eigenfaces")
    run_log = start_run_log(out, "part2_eigenfaces")

    metrics = run_eigenfaces(
        out,
        data_home=args.data_home,
        n_components=args.components,
        n_estimators=args.estimators,
        max_depth=args.max_depth,
    )
    metrics["run_log"] = str(run_log)
    write_json(out / "part2_metrics.json", metrics)
    print(f"Part 2 complete. Accuracy: {metrics['accuracy']:.4f}")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
