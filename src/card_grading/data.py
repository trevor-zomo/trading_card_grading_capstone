"""Shared knowledge about the dataset: PSA grades, file names and the train/test layout."""

from __future__ import annotations

import random
import re
from pathlib import Path

# PSA grades in numeric order, so that class index + 1 == grade.
# (Sorting the grade folders as strings would give 1, 10, 2, ...)
GRADES = tuple(range(1, 11))
CLASS_NAMES = tuple(str(grade) for grade in GRADES)

IMAGE_SUFFIX = ".jpg"
FLIPPED_SUFFIX = "_flipped"
CROPPED_SUFFIX = "_cropped"

DEFAULT_SPLIT_RATIO = 0.8

_GRADE_PATTERN = re.compile(r"_PSA(\d+)")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == IMAGE_SUFFIX


def grade_from_filename(file_name: str) -> int:
    """Extract the PSA grade from a file name such as ``card12_PSA8_cropped.jpg``.

    Half grades (``card12_PSA8_5.jpg``) are rounded down to the whole grade.
    """
    match = _GRADE_PATTERN.search(file_name)
    if match is None:
        raise ValueError(f"No PSA grade found in file name: {file_name!r}")
    return int(match.group(1))


def split_into_train_test(base_path: Path, seed: int, split_ratio: float = DEFAULT_SPLIT_RATIO) -> int:
    """Move any loose images in ``base_path`` into ``train/<grade>`` and ``test/<grade>`` folders.

    Returns the number of images that were moved. Images that are already inside the
    train/test folders are left alone, so calling this again is harmless.
    """
    rng = random.Random(seed)
    moved = 0
    for path in sorted(base_path.iterdir()):
        if not is_image(path):
            continue
        subset = "train" if rng.random() < split_ratio else "test"
        target_dir = base_path / subset / str(grade_from_filename(path.name))
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            path.rename(target_dir / path.name)
            moved += 1
        except FileNotFoundError:
            # another training run (e.g. a parallel grid search job) already moved it
            pass
    return moved
