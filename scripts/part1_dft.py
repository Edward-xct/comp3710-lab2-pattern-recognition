from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comp3710_lab2.common import output_dir, set_matplotlib_cache, set_seed, start_run_log, write_json
from comp3710_lab2.dft import benchmark_dft, save_dft_plots, write_dft_timings


# Part 1 的目标：
# 1. 用 Fourier series 把方波分解成不同数量的奇次谐波并画出来；
# 2. 对比三种频域计算方式：NumPy FFT、纯 Python/NumPy naive DFT、PyTorch matrix DFT；
# 3. 通过运行时间说明 FFT 的 O(N log N) 优势，以及 tensor/matrix 计算相比双重 for-loop 的差别。
def parse_args():
    # 函数让同一个Python文件可以通过终端参数切换实验规模和谐波数量，不需要每次修改源代码。
    parser = argparse.ArgumentParser(description="Part 1: DFT and PyTorch tensor comparison.")
    # sample-sizes 控制要测试的信号长度；N 越大，naive DFT 的 O(N^2) 劣势越明显。
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    # harmonics 控制用多少个 Fourier series 项重构方波，数值越大越接近理想方波。
    parser.add_argument("--harmonics", type=int, default=50)
    # naive DFT 非常慢，max-naive-n 防止现场或自动运行时卡太久。
    parser.add_argument("--max-naive-n", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    out = output_dir("part1_dft")
    run_log = start_run_log(out, "part1_dft")
    run_start = time.perf_counter()

    # 先保存方波谐波重构图，再对三种 DFT/FFT 方法做计时比较。
    # 图片用于解释“时域方波”和“频域奇次谐波峰值”的关系；CSV/JSON 用于保留可复查的计时数据。
    save_dft_plots(out, sample_count=max(args.sample_sizes), n_harmonics=args.harmonics)
    rows = benchmark_dft(
        sample_sizes=args.sample_sizes,
        n_harmonics=args.harmonics,
        max_naive_n=args.max_naive_n,
    )
    write_dft_timings(out / "part1_dft_timings.csv", rows)
    total_seconds = time.perf_counter() - run_start
    write_json(
        out / "part1_dft_timings.json",
        {
            "sample_sizes": list(args.sample_sizes),
            "harmonics": int(args.harmonics),
            "max_naive_n": int(args.max_naive_n),
            "total_seconds": float(total_seconds),
            "run_log": str(run_log),
            "rows": rows,
        },
    )

    print("Part 1 complete. Timing order by sample count:")
    for sample_count in args.sample_sizes:
        subset = [row for row in rows if row["sample_count"] == sample_count]
        # 按运行时间排序，FFT 为何最快：是因为它利用复指数的周期性和对称性，把长度为N的 DFT 递归拆成更小的 DFT，
        # 并复用中间结果。这样计算复杂度从 \(O(N^2)\) 降低到 \(O(N\log N)\)。
        # close=True 表示该方法的 DFT 结果和 NumPy FFT 基准足够接近，不只是跑得快。
        subset.sort(key=lambda row: float(row["seconds"]))
        print(f"N={sample_count}")
        for row in subset:
            print(
                f"  {row['method']} on {row['device']}: "
                f"{float(row['seconds']):.6f}s, close={row['close_to_numpy_fft']}"
            )
    print(f"Total running time: {total_seconds:.4f}s")
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
