import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
import image_cropping  # noqa: E402

RED = (200, 30, 40)
GREY = (120, 120, 120)


def make_slab(width=500, height=1000, label_rows=None):
    im = Image.new("RGB", (width, height), GREY)
    if label_rows:
        im.paste(RED, (0, label_rows[0], width, label_rows[1]))
    return im


def test_label_at_top_is_cropped_off():
    im = make_slab(label_rows=(20, 180))
    assert image_cropping.find_crop_bounds(im) == (180, 1000)


def test_label_at_bottom_is_cropped_off():
    im = make_slab(label_rows=(820, 980))
    assert image_cropping.find_crop_bounds(im) == (0, 820)


def test_no_label_falls_back_to_top_fifth():
    assert image_cropping.find_crop_bounds(make_slab()) == (200, 1000)


def test_red_in_the_middle_of_the_card_is_ignored():
    im = make_slab(label_rows=(450, 550))
    assert image_cropping.find_crop_bounds(im) == (200, 1000)


def test_crop_image_rotates_and_scales_to_target_width():
    out = image_cropping.crop_image(make_slab(width=1000, height=500))
    assert out.size == (400, 640)
    assert out.getcolors() == [(400 * 640, GREY)]
