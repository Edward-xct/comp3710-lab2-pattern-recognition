from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comp3710_lab2.common import output_dir, set_matplotlib_cache, set_seed, start_run_log, write_json
from comp3710_lab2.lfw import run_eigenfaces


# Part 2 的目标：
# 1. 从 LFW 人脸数据中提取 PCA 主成分，也就是 eigenfaces；
# 2. 把高维人脸图片压缩到较低维 PCA 特征；
# 3. 用 Random Forest 在这些特征上做身份分类，并输出 accuracy/report/可视化图片。
def parse_args():
    parser = argparse.ArgumentParser(description="Part 2: Eigenfaces with PCA and Random Forest.")
    # data-home 指向 sklearn 缓存 LFW 的目录；第一次运行会自动下载，之后直接复用本地数据。
    parser.add_argument("--data-home", type=Path, default=ROOT / "data" / "lfw")
    # components 是保留的 eigenfaces 数量；越多信息越完整，但维度和训练时间也会增加。
    parser.add_argument("--components", type=int, default=150)
    # estimators 是 Random Forest 中树的数量，较多树通常更稳定但更慢。
    parser.add_argument("--estimators", type=int, default=150)
    # max-depth 限制每棵树深度，避免树在小数据集上过拟合太严重。
    parser.add_argument("--max-depth", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    set_matplotlib_cache()
    out = output_dir("part2_eigenfaces")
    run_log = start_run_log(out, "part2_eigenfaces")

    # 核心流程在 lfw.py：LFW 数据 -> PCA/eigenfaces -> Random Forest 分类。
    # 这里保持脚本很短，方便 demo 时从入口跳到 src/comp3710_lab2/lfw.py 解释细节。
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
