import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")

from card_grading import training  # noqa: E402


def make_images(folder, grades=range(1, 11), per_grade=4, size=(40, 60)):
    rng = np.random.default_rng(0)
    for grade in grades:
        (folder / str(grade)).mkdir(parents=True, exist_ok=True)
        for i in range(per_grade):
            pixels = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / str(grade) / f"card{i}_PSA{grade}_cropped.jpg")


def test_list_images_orders_grades_numerically(tmp_path):
    make_images(tmp_path, grades=(2, 10, 9), per_grade=1)
    paths, grades = training.list_images(tmp_path)
    assert grades.tolist() == [2, 9, 10]  # grade 10 comes last, not after grade 1
    assert [path.split("_PSA")[1].split("_")[0] for path in paths] == ["2", "9", "10"]


def test_validation_split_takes_the_same_share_of_every_grade():
    grades = np.repeat(np.arange(1, 11), 20)
    mask = training.split_validation(grades, 0.1, seed=3)
    assert mask.sum() == 20
    assert all(mask[grades == grade].sum() == 2 for grade in range(1, 11))
    assert (mask == training.split_validation(grades, 0.1, seed=3)).all()
    assert (mask != training.split_validation(grades, 0.1, seed=4)).any()


def test_ordinal_targets_spread_weight_over_neighbouring_grades():
    one_hot = training.ordinal_targets(np.array([1, 10]), sigma=0)
    assert one_hot.argmax(axis=1).tolist() == [0, 9]
    assert one_hot.sum() == 2

    soft = training.ordinal_targets(np.array([5]), sigma=1.0)[0]
    assert soft.sum() == pytest.approx(1.0)
    assert soft.argmax() == 4
    assert soft[3] == pytest.approx(soft[5])   # grades 4 and 6 are equally near
    assert soft[3] > soft[2] > soft[0]         # nearer grades get more weight


def test_dataset_yields_raw_images_and_targets(tmp_path):
    make_images(tmp_path, grades=(7,), per_grade=3)
    paths, grades = training.list_images(tmp_path)
    ((images, targets),) = list(training.make_dataset(paths, grades, (96, 64), batch_size=8))
    assert images.shape == (3, 96, 64, 3)
    assert float(images.numpy().max()) > 1.0   # values are 0-255, preprocessing is part of the model
    assert targets.numpy().argmax(axis=1).tolist() == [6, 6, 6]


def test_model_stages_change_which_layers_train():
    base, model = training.build_model((96, 64), dense_units=(8,), droprate=0.1, weights=None)
    assert model.input_shape == (None, 96, 64, 3)
    assert model.output_shape == (None, 10)

    training.set_trainable_layers(base, 0)
    assert not base.trainable_weights
    training.set_trainable_layers(base, 20)
    assert sum(layer.trainable for layer in base.layers) == 20
    training.set_trainable_layers(base, None)
    assert all(layer.trainable for layer in base.layers)


def test_finetune_learning_rate_decays():
    rates = [training.finetune_learning_rate(1e-4, epoch, 10) for epoch in range(10)]
    assert rates[0] == pytest.approx(1e-4)
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] > 1e-4 * training.FINAL_LR_FRACTION


def test_grade_metrics():
    probabilities = np.eye(10)[[0, 1, 4, 9]]          # predicted grades 1, 2, 5, 10
    metrics = training.grade_metrics(probabilities, np.array([1, 3, 8, 10]))
    assert metrics["exact"] == 0.5
    assert metrics["within_1"] == 0.75
    assert metrics["within_2"] == 0.75
    assert metrics["mean_abs_diff"] == 1.0
    assert metrics["confusion_matrix"][7][4] == 1     # true grade 8 predicted as 5


def test_training_run_saves_results_and_can_be_continued(tmp_path):
    make_images(tmp_path / "data" / "train", per_grade=5)
    make_images(tmp_path / "data" / "test", per_grade=1)
    common = ["--base_path", str(tmp_path / "data"), "--log_root", str(tmp_path / "logs"), "--weights", "none",
              "--image_height", "64", "--image_width", "48", "--batch_size", "8", "--val_fraction", "0.2",
              "--head_epochs", "1", "--early_stopping_patience", "0"]

    training.main(common + ["--n_epochs", "1"])

    (run_dir,) = (tmp_path / "logs").iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["test"]["n_images"] == 10 and metrics["validation"]["n_images"] == 10
    assert (run_dir / "best_model.keras").is_file()
    assert len((run_dir / "log.csv").read_text().splitlines()) == 1 + 2   # header, warm-up epoch, fine-tune epoch

    # continuing the run only trains the epochs that are still missing
    training.main(common + ["--n_epochs", "2", "--run_dir", str(run_dir)])
    rows = (run_dir / "log.csv").read_text().splitlines()
    assert [row.split(",")[:2] for row in rows[1:]] == [["warm-up", "1"], ["fine-tune", "1"], ["fine-tune", "2"]]

    # the saved model takes raw images and can be loaded on its own
    from tensorflow import keras
    model = keras.models.load_model(run_dir / "best_model.keras")
    assert model.predict(np.zeros((1, 64, 48, 3), dtype="float32"), verbose=0).shape == (1, 10)


def test_defaults():
    args = training.parse_arguments(["--base_path", "data"])
    assert args.augment and args.dense_units == [] and args.unfreeze_after is None
    assert not training.parse_arguments(["--base_path", "data", "--no-augment"]).augment
