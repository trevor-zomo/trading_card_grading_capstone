from PIL import Image

from card_grading import cropping

RED = (200, 30, 40)
GREY = (120, 120, 120)


def make_slab(width=500, height=1000, label_rows=None):
    im = Image.new("RGB", (width, height), GREY)
    if label_rows:
        im.paste(RED, (0, label_rows[0], width, label_rows[1]))
    return im


def test_label_at_top_is_cropped_off():
    assert cropping.find_crop_bounds(make_slab(label_rows=(20, 180))) == (180, 1000)


def test_label_at_bottom_is_cropped_off():
    assert cropping.find_crop_bounds(make_slab(label_rows=(820, 980))) == (0, 820)


def test_label_with_thin_red_borders_is_cropped_off():
    im = make_slab()
    im.paste(RED, (0, 40, 500, 50))     # top border
    im.paste(RED, (0, 190, 500, 200))   # bottom border
    assert cropping.find_crop_bounds(im) == (200, 1000)


def test_no_label_falls_back_to_top_quarter():
    assert cropping.find_crop_bounds(make_slab()) == (250, 1000)


def test_red_in_the_middle_of_the_card_is_ignored():
    assert cropping.find_crop_bounds(make_slab(label_rows=(450, 550))) == (250, 1000)


def test_single_red_line_is_not_a_label():
    im = make_slab()
    im.paste(RED, (0, 100, 500, 110))
    assert cropping.find_crop_bounds(im) == (250, 1000)


def test_red_banner_at_the_bottom_of_a_card_is_not_an_upside_down_label():
    assert cropping.find_crop_bounds(make_slab(label_rows=(700, 860))) == (250, 1000)


def test_crop_image_rotates_and_scales_to_target_width():
    out = cropping.crop_image(make_slab(width=1000, height=500))
    assert out.size == (400, 600)
    assert out.getcolors() == [(400 * 600, GREY)]


def test_small_images_can_be_cropped_with_a_smaller_target_width(tmp_path):
    parent, output = tmp_path / "raw", tmp_path / "cropped"
    (parent / "psa7").mkdir(parents=True)
    make_slab(width=150, height=200, label_rows=(8, 40)).save(parent / "psa7" / "psa7_1.jpg", quality=95)

    cropping.main(["--parent_dir", str(parent), "--output_dir", str(output), "--target_width", "150"])

    with Image.open(output / "psa7_1_cropped.jpg") as im:
        assert im.size == (150, 160)


def test_main_crops_grade_folders_and_skips_small_images(tmp_path):
    parent, output = tmp_path / "raw", tmp_path / "cropped"
    (parent / "psa3").mkdir(parents=True)
    make_slab(label_rows=(20, 180)).save(parent / "psa3" / "card1_PSA3.jpg")
    make_slab(width=100, height=200).save(parent / "psa3" / "card2_PSA3.jpg")

    cropping.main(["--parent_dir", str(parent), "--output_dir", str(output)])

    assert [path.name for path in output.iterdir()] == ["card1_PSA3_cropped.jpg"]
    with Image.open(output / "card1_PSA3_cropped.jpg") as im:
        assert im.width == 400
