from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comp3710_lab2.oasis_png import inspect_oasis


# 这个脚本用于检查/解压老师给的 keras_png_slices_data.zip。
# 默认只打印每个 split 的数量；加 --extract 才会真正解压到项目 data/ 目录。
def parse_args():
    parser = argparse.ArgumentParser(description="Inspect or extract keras_png_slices_data.zip.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("/Users/xct/Downloads/keras_png_slices_data.zip"),
        help="Path to keras_png_slices_data.zip.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(args.zip)
    # 先统计 zip 内容，确认 train/validate/test 和 segmentation mask 都在。
    print("Archive counts:")
    for name, count in sorted(inspect_oasis(args.zip).items()):
        print(f"  {name}: {count}")

    if args.extract:
        # 只有用户明确加 --extract 时才解压，避免不小心占用本地磁盘空间。
        args.out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(args.out)
        extracted = args.out / "keras_png_slices_data"
        print(f"Extracted to {extracted}")
        print("Extracted counts:")
        # 解压后再统计一次，确认目录结构和 zip 内容一致。
        for name, count in sorted(inspect_oasis(extracted).items()):
            print(f"  {name}: {count}")
    else:
        print("Use --extract to unpack it into this project data directory.")


if __name__ == "__main__":
    main()
