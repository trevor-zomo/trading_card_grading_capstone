"""Shared knowledge about the dataset: PSA grades, file names and the train/test layout."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

# PSA grades in numeric order, so that class index + 1 == grade.
# (Sorting the grade folders as strings would give 1, 10, 2, ...)
GRADES = tuple(range(1, 11))
CLASS_NAMES = tuple(str(grade) for grade in GRADES)

IMAGE_SUFFIX = ".jpg"
FLIPPED_SUFFIX = "_flipped"
CROPPED_SUFFIX = "_cropped"

DEFAULT_SPLIT_RATIO = 0.8

# Folder (next to train/ and test/) that exact duplicate images are set aside in
DUPLICATES_DIR = "duplicates"

# matches card12_PSA8_cropped.jpg as well as psa8_12.jpg
_GRADE_PATTERN = re.compile(r"(?:^|_)PSA(\d+)", re.IGNORECASE)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == IMAGE_SUFFIX


def grade_from_filename(file_name: str) -> int:
    """Extract the PSA grade from a file name such as ``card12_PSA8_cropped.jpg`` or ``psa8_12.jpg``.

    Half grades (``card12_PSA8_5.jpg``) are rounded down to the whole grade.
    """
    match = _GRADE_PATTERN.search(file_name)
    if match is None or int(match.group(1)) not in GRADES:
        raise ValueError(f"No PSA grade from 1 to 10 found in file name: {file_name!r}")
    return int(match.group(1))


def _move(path: Path, target_dir: Path) -> bool:
    """Move a file into a folder. Returns False if the file is already gone."""
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        path.rename(target_dir / path.name)
        return True
    except FileNotFoundError:
        # another training run (e.g. a parallel grid search job) already moved it
        return False


def _group_by_content(paths: list[Path]) -> list[list[Path]]:
    """Group files that are byte-for-byte identical, keeping the order of the paths."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        try:
            groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
        except FileNotFoundError:
            pass
    return list(groups.values())


def split_into_train_test(base_path: Path, seed: int, split_ratio: float = DEFAULT_SPLIT_RATIO) -> int:
    """Move any loose images in ``base_path`` into ``train/<grade>`` and ``test/<grade>`` folders.

    Exact duplicates are moved to ``duplicates/`` so that the same image cannot end up in
    both train and test: one copy is kept, or none when the copies disagree on the grade.

    Returns the number of images that were moved into train/test. Images that are already
    inside the train/test folders are left alone, so calling this again is harmless.
    """
    rng = random.Random(seed)
    images = sorted(path for path in base_path.iterdir() if is_image(path))
    moved = 0
    for copies in _group_by_content(images):
        grades = {grade_from_filename(path.name) for path in copies}
        kept = copies[:1] if len(grades) == 1 else []
        for path in copies[len(kept):]:
            _move(path, base_path / DUPLICATES_DIR)
        for path in kept:
            subset = "train" if rng.random() < split_ratio else "test"
            moved += _move(path, base_path / subset / str(grades.pop()))
    return moved
