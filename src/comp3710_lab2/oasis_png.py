from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_RE = re.compile(r"case_(\d+)_slice_(\d+)\.nii\.png$")
MASK_RE = re.compile(r"seg_(\d+)_slice_(\d+)\.nii\.png$")


@dataclass(frozen=True)
class OasisCounts:
    train: int
    validate: int
    test: int
    seg_train: int
    seg_validate: int
    seg_test: int


def _key_from_name(name: str, regex: re.Pattern[str]) -> tuple[int, int]:
    match = regex.search(Path(name).name)
    if not match:
        raise ValueError(f"Unexpected OASIS filename: {name}")
    return int(match.group(1)), int(match.group(2))


def resolve_oasis_directory(root: str | Path) -> Path:
    root = Path(root).expanduser()
    if (root / "keras_png_slices_train").is_dir():
        return root
    nested = root / "keras_png_slices_data"
    if (nested / "keras_png_slices_train").is_dir():
        return nested
    raise FileNotFoundError(
        f"Could not find keras_png_slices_* folders under {root}. "
        "Pass the extracted keras_png_slices_data directory or the zip file."
    )


def inspect_oasis(root: str | Path) -> dict[str, int]:
    root = Path(root).expanduser()
    names: Iterable[str]
    if root.suffix == ".zip":
        with zipfile.ZipFile(root) as zf:
            names = list(zf.namelist())
            counts: dict[str, int] = {}
            for split in ("train", "validate", "test"):
                counts[split] = sum(
                    f"/keras_png_slices_{split}/" in n and n.endswith(".png") for n in names
                )
                counts[f"seg_{split}"] = sum(
                    f"/keras_png_slices_seg_{split}/" in n and n.endswith(".png") for n in names
                )
            return counts

    base = resolve_oasis_directory(root)
    counts = {}
    for split in ("train", "validate", "test"):
        counts[split] = len(list((base / f"keras_png_slices_{split}").glob("*.png")))
        counts[f"seg_{split}"] = len(list((base / f"keras_png_slices_seg_{split}").glob("*.png")))
    return counts


class OasisPngDataset(Dataset):
    """OASIS PNG slice dataset for VAE/GAN image modeling or UNet segmentation."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        with_masks: bool = False,
        image_size: int = 128,
        max_samples: int | None = None,
        value_range: str = "zero_one",
    ) -> None:
        if split not in {"train", "validate", "test"}:
            raise ValueError("split must be one of: train, validate, test")
        if value_range not in {"zero_one", "minus_one_one"}:
            raise ValueError("value_range must be zero_one or minus_one_one")

        self.root = Path(root).expanduser()
        self.split = split
        self.with_masks = with_masks
        self.image_size = image_size
        self.value_range = value_range
        self.is_zip = self.root.suffix == ".zip"
        self._zip: zipfile.ZipFile | None = None

        if self.is_zip:
            self.image_items, self.mask_items = self._list_zip_items()
        else:
            self.image_items, self.mask_items = self._list_directory_items()

        if with_masks:
            mask_by_key = {
                _key_from_name(str(item), MASK_RE): item for item in self.mask_items
            }
            paired = []
            for image_item in self.image_items:
                key = _key_from_name(str(image_item), IMAGE_RE)
                if key in mask_by_key:
                    paired.append((key, image_item, mask_by_key[key]))
            paired.sort(key=lambda x: x[0])
            if max_samples is not None:
                paired = paired[:max_samples]
            self.items = paired
        else:
            image_items = sorted(self.image_items, key=lambda item: _key_from_name(str(item), IMAGE_RE))
            if max_samples is not None:
                image_items = image_items[:max_samples]
            self.items = [(None, item, None) for item in image_items]

        if not self.items:
            raise FileNotFoundError(f"No OASIS PNG files found for split={split}, root={root}")

    def __len__(self) -> int:
        return len(self.items)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip"] = None
        return state

    def _get_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.root)
        return self._zip

    def _list_zip_items(self):
        with zipfile.ZipFile(self.root) as zf:
            names = zf.namelist()
        image_prefix = f"keras_png_slices_data/keras_png_slices_{self.split}/"
        mask_prefix = f"keras_png_slices_data/keras_png_slices_seg_{self.split}/"
        image_items = [n for n in names if n.startswith(image_prefix) and n.endswith(".png")]
        mask_items = [n for n in names if n.startswith(mask_prefix) and n.endswith(".png")]
        return image_items, mask_items

    def _list_directory_items(self):
        base = resolve_oasis_directory(self.root)
        image_items = list((base / f"keras_png_slices_{self.split}").glob("*.png"))
        mask_items = list((base / f"keras_png_slices_seg_{self.split}").glob("*.png"))
        return image_items, mask_items

    def _open_gray(self, item, is_mask: bool) -> np.ndarray:
        if self.is_zip:
            with self._get_zip().open(item) as f:
                image = Image.open(f).convert("L")
                image.load()
        else:
            image = Image.open(item).convert("L")

        if self.image_size and image.size != (self.image_size, self.image_size):
            resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
            image = image.resize((self.image_size, self.image_size), resample)
        return np.asarray(image)

    def __getitem__(self, index: int):
        _, image_item, mask_item = self.items[index]
        image = self._open_gray(image_item, is_mask=False).astype(np.float32) / 255.0
        if self.value_range == "minus_one_one":
            image = image * 2.0 - 1.0
        image_tensor = torch.from_numpy(image[None, :, :])

        if not self.with_masks:
            return image_tensor

        mask = self._open_gray(mask_item, is_mask=True).astype(np.float32)
        mask = np.rint(mask / 85.0).clip(0, 3).astype(np.int64)
        return image_tensor, torch.from_numpy(mask)


def make_oasis_loader(
    root: str | Path,
    split: str,
    with_masks: bool,
    batch_size: int,
    image_size: int,
    max_samples: int | None = None,
    num_workers: int = 2,
    shuffle: bool | None = None,
    value_range: str = "zero_one",
) -> DataLoader:
    dataset = OasisPngDataset(
        root=root,
        split=split,
        with_masks=with_masks,
        image_size=image_size,
        max_samples=max_samples,
        value_range=value_range,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
