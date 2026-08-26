from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comp3710_lab2.common import output_dir, set_matplotlib_cache, set_seed, write_json
from comp3710_lab2.dft import benchmark_dft, save_dft_plots, write_dft_timings


def parse_args():
    parser = argparse.ArgumentParser(description="Part 1: DFT and PyTorch tensor comparison.")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--harmonics", type=int, default=50)
    parser.add_argument("--max-naive-n", type=int, default=2048)
    parser.add_argument("--fast-dev", action="store_true", help="Use very small arrays for quick checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    out = output_dir("part1_dft")

    if args.fast_dev:
        args.sample_sizes = [64, 128]
        args.max_naive_n = 128

    save_dft_plots(out, sample_count=max(args.sample_sizes), n_harmonics=args.harmonics)
    rows = benchmark_dft(
        sample_sizes=args.sample_sizes,
        n_harmonics=args.harmonics,
        max_naive_n=args.max_naive_n,
    )
    write_dft_timings(out / "part1_dft_timings.csv", rows)
    write_json(out / "part1_dft_timings.json", {"rows": rows})

    print("Part 1 complete. Timing order by sample count:")
    for sample_count in args.sample_sizes:
        subset = [row for row in rows if row["sample_count"] == sample_count]
        subset.sort(key=lambda row: float(row["seconds"]))
        print(f"N={sample_count}")
        for row in subset:
            print(
                f"  {row['method']} on {row['device']}: "
                f"{float(row['seconds']):.6f}s, close={row['close_to_numpy_fft']}"
            )
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
