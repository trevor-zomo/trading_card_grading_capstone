import pytest
from PIL import Image

from card_grading import augmentation, data


def save_image(path, color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 4), color).save(path)


@pytest.mark.parametrize("name, grade", [
    ("card12_PSA8_cropped.jpg", 8),
    ("card12_PSA10_cropped_flipped.jpg", 10),
    ("card12_PSA8_5_cropped.jpg", 8),
    ("card12_PSA1.jpg", 1),
])
def test_grade_from_filename(name, grade):
    assert data.grade_from_filename(name) == grade


def test_grade_from_filename_rejects_names_without_grade():
    with pytest.raises(ValueError):
        data.grade_from_filename("holiday.jpg")


def test_class_names_are_in_grade_order():
    assert data.CLASS_NAMES == ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")


def test_split_moves_every_image_and_is_reproducible(tmp_path):
    def run(base):
        for i in range(40):
            save_image(base / f"card{i}_PSA{i % 10 + 1}_cropped.jpg")
        assert data.split_into_train_test(base, seed=7) == 40
        return sorted(path.relative_to(base).as_posix() for path in base.rglob("*.jpg"))

    first, second = run(tmp_path / "a"), run(tmp_path / "b")
    assert first == second
    assert len(first) == 40
    assert all(path.split("/")[0] in ("train", "test") for path in first)
    assert all(int(path.split("/")[1]) == data.grade_from_filename(path) for path in first)
    # nothing left to move the second time
    assert data.split_into_train_test(tmp_path / "a", seed=7) == 0


def test_flip_writes_mirrored_copies_once(tmp_path):
    image_path = tmp_path / "3" / "card1_PSA3_cropped.jpg"
    image_path.parent.mkdir()
    im = Image.new("RGB", (8, 4), (0, 0, 0))
    im.paste((255, 255, 255), (0, 0, 4, 4))  # left half white
    im.save(image_path, quality=100)

    assert augmentation.flip_images_in_directory(tmp_path) == 1
    assert augmentation.flip_images_in_directory(tmp_path) == 0  # running it again adds nothing

    names = sorted(path.name for path in image_path.parent.iterdir())
    assert names == ["card1_PSA3_cropped.jpg", "card1_PSA3_cropped_flipped.jpg"]
    with Image.open(image_path.parent / names[1]) as flipped:
        assert flipped.getpixel((0, 1))[0] < 50   # left is now dark
        assert flipped.getpixel((7, 1))[0] > 200  # right is now white
