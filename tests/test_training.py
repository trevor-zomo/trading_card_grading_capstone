import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from card_grading import training  # noqa: E402
from card_grading.data import CLASS_NAMES  # noqa: E402


def test_second_round_trains_more_resnet_layers_than_the_first():
    base, model = training.build_model((8, 8, 8), droprate=0.1, weights=None)
    assert model.input_shape == (None, 600, 400, 3)
    assert model.output_shape == (None, 10)

    training.set_trainable_layers(base, training.ROUND_1_TRAINABLE_LAYERS)
    round_1 = sum(layer.trainable for layer in base.layers)
    training.set_trainable_layers(base, 50)
    round_2 = sum(layer.trainable for layer in base.layers)

    assert (round_1, round_2) == (20, 50)


def test_dataset_labels_follow_grade_order(tmp_path):
    for grade in CLASS_NAMES:
        (tmp_path / grade).mkdir()
    Image.new("RGB", (40, 60)).save(tmp_path / "10" / "card1_PSA10_cropped.jpg")

    dataset = training.load_dataset(tmp_path, batch_size=4, shuffle=False, seed=0)
    ((images, labels),) = list(dataset)

    assert images.shape == (1, 600, 400, 3)
    assert labels.numpy().argmax(axis=1).tolist() == [9]  # grade 10 is the last class, not the second


def test_lr_finetune_defaults_to_a_tenth_of_lr_dense():
    args = training.parse_arguments(["--base_path", "data", "--lr_dense", "0.001"])
    assert args.lr_finetune == pytest.approx(0.0001)
