from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from comp3710_lab2.common import output_dir, set_matplotlib_cache
from comp3710_lab2.oasis_png import OasisPngDataset


PALETTE = np.array(
    [
        [0, 0, 0],
        [220, 40, 40],
        [40, 180, 90],
        [70, 120, 230],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Save a quick OASIS image/mask sample grid.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/Users/xct/Downloads/keras_png_slices_data.zip"),
    )
    parser.add_argument("--split", choices=["train", "validate", "test"], default="train")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_matplotlib_cache()
    import matplotlib.pyplot as plt

    dataset = OasisPngDataset(
        args.data,
        split=args.split,
        with_masks=True,
        image_size=args.image_size,
        max_samples=args.count,
    )
    rows = min(args.count, len(dataset))
    fig, axes = plt.subplots(rows, 2, figsize=(5, 2.5 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i in range(rows):
        image, mask = dataset[i]
        axes[i, 0].imshow(image[0].numpy(), cmap="gray")
        axes[i, 0].set_title("MRI slice")
        axes[i, 1].imshow(PALETTE[mask.numpy()])
        axes[i, 1].set_title("Segmentation mask")
        for ax in axes[i]:
            ax.set_xticks(())
            ax.set_yticks(())
    fig.tight_layout()
    out = output_dir("data_checks") / "oasis_sample_grid.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
