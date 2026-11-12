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
LABEL_SEARCH_FRACTION = 0.35

# Height of a PSA label (from its top red border to its bottom red border), as a share of the image height
MIN_LABEL_FRACTION = 0.08
MAX_LABEL_FRACTION = 0.25

# An upside-down label only counts if its outer border is this close to the bottom edge of the image.
# (Without this, red banners at the bottom of a card are taken for a label.)
MAX_FLIPPED_LABEL_OFFSET = 0.10

# Share of the image that is cropped off the top when no label is found. In slab photos the
# bottom border of the label sits at 19% of the image height on average, and above 25% for most.
FALLBACK_CROP_FRACTION = 0.25

# A row counts as part of a red label border when this share of its middle half is red
RED_ROW_FRACTION = 0.5


def find_red_rows(im: Image.Image) -> np.ndarray:
    """Return a boolean array, one entry per pixel row, that is True where the row is mostly label-red."""
    pixels = np.asarray(im.convert("RGB")).astype(int)
    width = pixels.shape[1]
    middle = pixels[:, width // 4: max(width - width // 4, width // 4 + 1)]
    r, g, b = middle[..., 0], middle[..., 1], middle[..., 2]
    # "clearly more red than green and blue" holds up better than fixed colour limits
    # when the thin border is blurred into its surroundings in small or compressed images
    red = (r > 110) & (r - g > 45) & (r - b > 45)
    return red.mean(axis=1) >= RED_ROW_FRACTION


def find_label_end(red_rows: np.ndarray, height: int, max_offset: float | None = None) -> int | None:
    """Find a label at the start of ``red_rows`` and return the row of the end of its second border.

    A label shows up as two red borders, the second one between ``MIN_LABEL_FRACTION`` and
    ``MAX_LABEL_FRACTION`` of the image height after the first. Returns ``None`` if there is no
    such pair, or if the first border starts more than ``max_offset`` of the image height in.
    """
    rows = np.flatnonzero(red_rows)
    if rows.size == 0:
        return None
    first_border = int(rows[0])
    if max_offset is not None and first_border > max_offset * height:
        return None
    candidates = rows[(rows >= first_border + MIN_LABEL_FRACTION * height)
                      & (rows <= first_border + MAX_LABEL_FRACTION * height)]
    if candidates.size == 0:
        return None
    end = int(candidates[0])
    while end + 1 < len(red_rows) and red_rows[end + 1]:  # include the whole second border
        end += 1
    return end


def find_crop_bounds(im: Image.Image) -> tuple[int, int]:
    """Work out which rows of the (portrait) image to keep so that the PSA label is cut off.

    Returns ``(crop_top, crop_bottom)``: the first row to keep, and the row after the last one to keep.
    """
    height = im.size[1]
    band = int(height * LABEL_SEARCH_FRACTION)
    red_rows = find_red_rows(im)

    # the label is at the top: keep everything below its bottom border
    label_end = find_label_end(red_rows[:band], height)
    if label_end is not None:
        return label_end + 1, height

    # the slab is upside down: keep everything above the label
    label_end = find_label_end(red_rows[::-1][:band], height, max_offset=MAX_FLIPPED_LABEL_OFFSET)
    if label_end is not None:
        return 0, height - (label_end + 1)

    # no label found, assume it is at the top of the image
    return int(height * FALLBACK_CROP_FRACTION), height


def crop_image(im: Image.Image, target_width: int = TARGET_PIXEL_SIZE) -> Image.Image:
    """Rotate to portrait, crop off the PSA label and scale down to ``target_width`` wide."""
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
    if cropped_width > target_width or cropped_height > target_width:
        new_height = int(target_width * (cropped_height / cropped_width))
        cropped = cropped.resize((target_width, new_height))
    return cropped


def crop_folder(folder: Path, output_dir: Path, target_width: int = TARGET_PIXEL_SIZE) -> int:
    """Crop every image in ``folder`` into ``output_dir``. Returns the number of images written."""
    written = 0
    for image_path in sorted(path for path in folder.iterdir() if is_image(path)):
        with Image.open(image_path) as im:
            if im.width < target_width or im.height < target_width:
                print(f"Skipping image '{image_path.name}' as its dimensions are smaller than target pixel size.")
                continue
            cropped = crop_image(im, target_width)
        cropped.save(output_dir / (image_path.stem + CROPPED_SUFFIX + IMAGE_SUFFIX))
        written += 1
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crop the PSA label off the scraped card images")
    parser.add_argument("--parent_dir", type=Path, required=True,
                        help="Folder containing the psa1 ... psa10 folders of images")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Folder the cropped images are saved to")
    parser.add_argument("--target_width", type=int, default=TARGET_PIXEL_SIZE,
                        help="Width in pixels the cropped images are scaled down to; smaller images are skipped")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for folder_name in FOLDER_NAMES:
        folder = args.parent_dir / folder_name
        if not folder.is_dir():
            print(f"Skipping missing folder: {folder}")
            continue
        crop_folder(folder, args.output_dir, args.target_width)


if __name__ == "__main__":
    main()
