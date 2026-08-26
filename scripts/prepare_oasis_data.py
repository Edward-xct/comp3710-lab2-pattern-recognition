from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comp3710_lab2.oasis_png import inspect_oasis


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
    print("Archive counts:")
    for name, count in sorted(inspect_oasis(args.zip).items()):
        print(f"  {name}: {count}")

    if args.extract:
        args.out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip) as zf:
            zf.extractall(args.out)
        extracted = args.out / "keras_png_slices_data"
        print(f"Extracted to {extracted}")
        print("Extracted counts:")
        for name, count in sorted(inspect_oasis(extracted).items()):
            print(f"  {name}: {count}")
    else:
        print("Use --extract to unpack it into this project data directory.")


if __name__ == "__main__":
    main()
