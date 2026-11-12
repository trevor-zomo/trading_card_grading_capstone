"""Rotate, crop the PSA label off, and resize the scraped card images.

USAGE:
    card-grading-crop --parent_dir /path/to/final_10k --output_dir /path/to/final10k_cropped
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from card_grading.data import CROPPED_SUFFIX, GRADES, IMAGE_SUFFIX, is_image

FOLDER_NAMES = tuple(f"psa{grade}" for grade in GRADES)

# Width, in pixels, that the cropped images are scaled to
TARGET_PIXEL_SIZE = 400

# The PSA label is only looked for in the top and bottom parts of the slab
LABEL_SEARCH_FRACTION = 0.3

# A row counts as part of the red label when this share of its middle half is red
RED_ROW_FRACTION = 0.5


def find_red_rows(im: Image.Image) -> np.ndarray:
    """Return a boolean array, one entry per pixel row, that is True where the row is mostly label-red."""
    pixels = np.asarray(im.convert("RGB")).astype(int)
    width = pixels.shape[1]
    middle = pixels[:, width // 4: max(width - width // 4, width // 4 + 1)]
    r, g, b = middle[..., 0], middle[..., 1], middle[..., 2]
    red = (r > 150) & (g < 90) & (b < 90)
    return red.mean(axis=1) >= RED_ROW_FRACTION


def find_crop_bounds(im: Image.Image) -> tuple[int, int]:
    """Work out which rows of the (portrait) image to keep so that the PSA label is cut off.

    Returns ``(crop_top, crop_bottom)``: the first row to keep, and the row after the last one to keep.
    """
    height = im.size[1]
    band = int(height * LABEL_SEARCH_FRACTION)
    red_rows = find_red_rows(im)
    top_rows = np.flatnonzero(red_rows[:band])
    bottom_rows = np.flatnonzero(red_rows[height - band:]) + (height - band)

    if top_rows.size:
        # the label is at the top: keep everything below its last red row
        return int(top_rows[-1]) + 1, height
    if bottom_rows.size:
        # the slab is upside down: keep everything above the label
        return 0, int(bottom_rows[0])
    # no label found, assume it is in the top 1/5th of the image
    return height // 5, height


def crop_image(im: Image.Image) -> Image.Image:
    """Rotate to portrait, crop off the PSA label and scale to ``TARGET_PIXEL_SIZE`` wide."""
    im = im.convert("RGB")
    width, height = im.size

    # Step 1: rotate landscape images (90 degrees counter-clockwise) so every slab is portrait
    if width > height:
        im = im.rotate(90, expand=True)
        width, height = im.size

    # Step 2: crop the label off
    crop_top, crop_bottom = find_crop_bounds(im)
    cropped = im.crop((0, crop_top, width, crop_bottom))

    # Step 3: scale to the target width while maintaining the aspect ratio
    cropped_width, cropped_height = cropped.size
    if cropped_width > TARGET_PIXEL_SIZE or cropped_height > TARGET_PIXEL_SIZE:
        new_height = int(TARGET_PIXEL_SIZE * (cropped_height / cropped_width))
        cropped = cropped.resize((TARGET_PIXEL_SIZE, new_height))
    return cropped


def crop_folder(folder: Path, output_dir: Path) -> int:
    """Crop every image in ``folder`` into ``output_dir``. Returns the number of images written."""
    written = 0
    for image_path in sorted(path for path in folder.iterdir() if is_image(path)):
        with Image.open(image_path) as im:
            if im.width < TARGET_PIXEL_SIZE or im.height < TARGET_PIXEL_SIZE:
                print(f"Skipping image '{image_path.name}' as its dimensions are smaller than target pixel size.")
                continue
            cropped = crop_image(im)
        cropped.save(output_dir / (image_path.stem + CROPPED_SUFFIX + IMAGE_SUFFIX))
        written += 1
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crop the PSA label off the scraped card images")
    parser.add_argument("--parent_dir", type=Path, required=True,
                        help="Folder containing the psa1 ... psa10 folders of images")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Folder the cropped images are saved to")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for folder_name in FOLDER_NAMES:
        folder = args.parent_dir / folder_name
        if not folder.is_dir():
            print(f"Skipping missing folder: {folder}")
            continue
        crop_folder(folder, args.output_dir)


if __name__ == "__main__":
    main()
