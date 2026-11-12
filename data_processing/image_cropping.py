"""Rotate, crop the PSA label off, and resize the scraped card images.

USAGE:
    python image_cropping.py --parent_dir /path/to/final_10k --output_dir /path/to/final10k_cropped
"""
import argparse
import os

import numpy as np
from PIL import Image

FOLDER_NAMES = ["psa1", "psa2", "psa3", "psa4", "psa5", "psa6", "psa7", "psa8", "psa9", "psa10"]

# Define the target pixel size for scaling images
TARGET_PIXEL_SIZE = 400

# The PSA label is only looked for in the top and bottom parts of the slab
LABEL_SEARCH_FRACTION = 0.3

# A row counts as part of the red label when this share of its middle half is red
RED_ROW_FRACTION = 0.5


def find_red_rows(im):
    """Return a boolean array, one entry per pixel row, that is True where the row is mostly label-red."""
    pixels = np.asarray(im.convert("RGB")).astype(int)
    width = pixels.shape[1]
    middle = pixels[:, width // 4: max(width - width // 4, width // 4 + 1)]
    r, g, b = middle[..., 0], middle[..., 1], middle[..., 2]
    red = (r > 150) & (g < 90) & (b < 90)
    return red.mean(axis=1) >= RED_ROW_FRACTION


def find_crop_bounds(im):
    """
    Work out which rows of the (portrait) image to keep so that the PSA label is cut off.
    Args:
        im (PIL.Image): the slab image, already rotated to portrait
    Returns:
        (crop_top, crop_bottom): the first row to keep, and the row after the last one to keep
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
    return int(height / 5), height


def crop_image(im):
    """Rotate to portrait, crop off the PSA label and scale to TARGET_PIXEL_SIZE wide."""
    im = im.convert("RGB")
    width, height = im.size

    # Step 1: Rotate the image if the x-axis is greater than the y-axis
    if width > height:
        im = im.rotate(90, expand=True)  # Rotate the image by 90 degrees counter-clockwise
        width, height = im.size

    # Step 2: Crop the label off
    crop_top, crop_bottom = find_crop_bounds(im)
    cropped_img = im.crop((0, crop_top, width, crop_bottom))

    # Step 3: Scale the image to the target pixel size while maintaining its aspect ratio
    cropped_width, cropped_height = cropped_img.size
    if cropped_width > TARGET_PIXEL_SIZE or cropped_height > TARGET_PIXEL_SIZE:
        new_height = int(TARGET_PIXEL_SIZE * (cropped_height / cropped_width))
        cropped_img = cropped_img.resize((TARGET_PIXEL_SIZE, new_height))
    return cropped_img


def main():
    parser = argparse.ArgumentParser(description='Crop the PSA label off the scraped card images')
    parser.add_argument('--parent_dir', default="/Users/brianmiller/Desktop/trading_card_data/final_10k",
                        help='Folder containing the psa1 ... psa10 folders of images')
    parser.add_argument('--output_dir', default="/Users/brianmiller/Desktop/trading_card_data/final10k_cropped",
                        help='Folder the cropped images are saved to')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for folder_name in FOLDER_NAMES:
        folder_path = os.path.join(args.parent_dir, folder_name)
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith('.jpg'):
                continue
            with Image.open(os.path.join(folder_path, filename)) as im:
                # Skip images smaller than target pixel size
                if im.width < TARGET_PIXEL_SIZE or im.height < TARGET_PIXEL_SIZE:
                    print(f"Skipping image '{filename}' as its dimensions are smaller than target pixel size.")
                    continue
                cropped_img = crop_image(im)

            # Save the cropped and resized image
            file_name = os.path.splitext(filename)[0] + "_cropped.jpg"
            cropped_img.save(os.path.join(args.output_dir, file_name))


if __name__ == '__main__':
    main()
