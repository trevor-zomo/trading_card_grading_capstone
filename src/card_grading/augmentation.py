"""Augment the training images with horizontally flipped copies.

USAGE:
    card-grading-flip /path/to/data/train
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from card_grading.data import FLIPPED_SUFFIX, IMAGE_SUFFIX, is_image


def flip_images_in_directory(directory: Path) -> int:
    """Save a flipped copy of every image in each subfolder of ``directory``.

    Images that are flipped copies themselves, or that already have a flipped copy,
    are skipped, so running this twice does not add duplicates. Returns the number
    of flipped images that were written.
    """
    written = 0
    for subdir in sorted(path for path in directory.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in subdir.iterdir() if is_image(path)):
            if image_path.stem.endswith(FLIPPED_SUFFIX):
                continue
            flipped_path = image_path.with_name(image_path.stem + FLIPPED_SUFFIX + IMAGE_SUFFIX)
            if flipped_path.exists():
                continue
            with Image.open(image_path) as image:
                image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(flipped_path)
            written += 1
        print(f"All images flipped and saved for directory: {subdir}")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add horizontally flipped copies of the images in each subfolder")
    parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd(),
                        help="Folder whose subfolders contain the images (default: current directory)")
    args = parser.parse_args(argv)
    flip_images_in_directory(args.directory)


if __name__ == "__main__":
    main()
