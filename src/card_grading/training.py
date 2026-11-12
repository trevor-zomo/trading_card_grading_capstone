"""Train and evaluate a ResNet-50 based PSA grade classifier.

USAGE:
    card-grading-train --n_epochs 50 --droprate 0.025 --base_path /home/data/data_for_model_3

Training happens in two rounds:
    1. the dense head plus the last 20 ResNet-50 layers are trained
    2. the last ``--unfreeze_after`` ResNet-50 layers are unfrozen and the model
       is fine-tuned with a lower learning rate (``--lr_finetune``)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from card_grading.data import CLASS_NAMES, split_into_train_test

# Default (height, width): the cropped cards are portrait, 400 px wide
IMAGE_SIZE = (600, 400)

# number of ResNet-50 layers (counted from the end) trained in the first round
ROUND_1_TRAINABLE_LAYERS = 20


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the PSA grade classifier.")
    parser.add_argument("-e", "--n_epochs", type=int, default=10,
                        help="Number of epochs per training cycles (positive integer)")
    parser.add_argument("-b", "--base_path", type=Path, required=True,
                        help="The base path for the data folder containing both /train and /test folders")
    parser.add_argument("-d", "--droprate", type=float, default=0.012,
                        help="The droprate for the dropout layers (float between 0 and 1)")
    parser.add_argument("--lr_dense", type=float, default=0.0001,
                        help="Learning rate for the first training round")
    parser.add_argument("--lr_finetune", type=float, default=None,
                        help="Learning rate for the second (fine-tuning) round, defaults to lr_dense / 10")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and validation data")
    parser.add_argument("--dense1_units", type=int, default=1024,
                        help="Number of units in the first additional dense layer")
    parser.add_argument("--dense2_units", type=int, default=512,
                        help="Number of units in the second additional dense layer")
    parser.add_argument("--dense3_units", type=int, default=128,
                        help="Number of units in the third additional dense layer")
    parser.add_argument("--unfreeze_after", type=int, default=50,
                        help="How many ResNet50 layers, counted from the last, are trainable in the second round")
    parser.add_argument("--image_height", type=int, default=IMAGE_SIZE[0],
                        help="Height in pixels the images are resized to for the model")
    parser.add_argument("--image_width", type=int, default=IMAGE_SIZE[1],
                        help="Width in pixels the images are resized to for the model")
    parser.add_argument("--augment", action="store_true",
                        help="Apply random flip/rotation/shift/zoom augmentation to the training images")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Stop a round after this many epochs without val_loss improvement (default: off)")
    parser.add_argument("--weights", type=str, default="imagenet",
                        help="ResNet50 weights: 'imagenet', a path to a weights file, or 'none' for random init")
    parser.add_argument("--log_root", type=Path, default=Path("logs"),
                        help="Folder in which each run gets its own log folder")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/test split, shuffling and weight init")
    args = parser.parse_args(argv)
    if args.lr_finetune is None:
        args.lr_finetune = args.lr_dense / 10
    return args


def load_dataset(path: Path, batch_size: int, *, shuffle: bool, seed: int,
                 image_size: tuple[int, int] = IMAGE_SIZE) -> tf.data.Dataset:
    """Load the images below ``path``, labelled by their grade folder, in grade order."""
    return keras.utils.image_dataset_from_directory(
        str(path),
        labels="inferred",
        label_mode="categorical",
        class_names=list(CLASS_NAMES),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)


def build_augmentation() -> keras.Sequential:
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(20 / 360, fill_mode="nearest"),
        layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
        layers.RandomZoom(0.2, fill_mode="nearest"),
    ])


def prepare(dataset: tf.data.Dataset, *, augment: bool = False) -> tf.data.Dataset:
    """Apply the (optional) augmentation and the ResNet-50 preprocessing the ImageNet weights expect."""
    if augment:
        augmentation = build_augmentation()
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.map(lambda x, y: (preprocess_input(x), y),
                       num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build_model(dense_units: tuple[int, int, int], droprate: float, weights: str | None = "imagenet",
                image_size: tuple[int, int] = IMAGE_SIZE) -> tuple[keras.Model, keras.Model]:
    """Build the classifier. Returns ``(resnet50_base, full_model)``."""
    base = ResNet50(weights=weights, include_top=False, input_shape=tuple(image_size) + (3,))

    # Pool the feature map instead of flattening it: at the default image size, flattening
    # 19x13x2048 features into the first dense layer would need ~500M weights
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(dense_units[0], activation="relu")(x)
    x = layers.Dropout(droprate)(x)
    x = layers.Dense(dense_units[1], activation="relu")(x)
    x = layers.Dropout(droprate)(x)
    x = layers.Dense(dense_units[2], activation="relu")(x)
    predictions = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

    return base, keras.Model(inputs=base.input, outputs=predictions)


def set_trainable_layers(base: keras.Model, n_trainable: int) -> None:
    """Make only the last ``n_trainable`` layers of the ResNet-50 trainable."""
    first_trainable = len(base.layers) - n_trainable
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= first_trainable


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")])


class RunCSVLogger(keras.callbacks.CSVLogger):
    """CSVLogger that also records the learning rate and dropout rate of the run."""

    def __init__(self, filename: str, learning_rate: float, dropout_rate: float):
        super().__init__(filename, append=True)
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = self.learning_rate
        logs["drop_rate"] = self.dropout_rate
        self.keys = sorted(logs.keys())
        super().on_epoch_end(epoch, logs)


def make_callbacks(log_dir: Path, learning_rate: float, droprate: float, early_stopping_patience: int | None,
                   best_val_accuracy: float | None = None) -> list[keras.callbacks.Callback]:
    """Callbacks for one training round.

    ``best_val_accuracy`` is the best validation accuracy of earlier rounds, so that
    a later round only replaces the saved best model when it actually beats it.
    """
    callbacks = [
        RunCSVLogger(str(log_dir / "log.csv"), learning_rate, droprate),
        # keep the best model seen so far, so a run can be evaluated or reused later
        keras.callbacks.ModelCheckpoint(str(log_dir / "best_model.keras"), monitor="val_accuracy",
                                        save_best_only=True, initial_value_threshold=best_val_accuracy),
    ]
    if early_stopping_patience is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True))
    return callbacks


def grade_differential(model: keras.Model, test_ds: tf.data.Dataset) -> tuple[float, float]:
    """Mean and median absolute difference between predicted and true grade, over the whole test set."""
    true_labels = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    predicted_labels = np.argmax(model.predict(test_ds), axis=1)
    # the classes are in grade order, so a difference of class indices is a difference of grades
    absolute_diffs = np.abs(predicted_labels - true_labels)
    return float(np.mean(absolute_diffs)), float(np.median(absolute_diffs))


def create_log_dir(log_root: Path, args: argparse.Namespace) -> Path:
    """Create a log folder for this run and record its parameters in it.

    The pid keeps runs that start in the same second (e.g. a parallel grid search)
    from writing into the same folder.
    """
    run_name = f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    params = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (log_dir / "params.json").write_text(json.dumps(params, indent=2))
    return log_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    if not args.base_path.is_dir():
        raise SystemExit(f"base_path does not exist: {args.base_path}")
    keras.utils.set_random_seed(args.seed)
    log_dir = create_log_dir(args.log_root, args)

    image_size = (args.image_height, args.image_width)
    split_into_train_test(args.base_path, args.seed)
    train_ds = prepare(load_dataset(args.base_path / "train", args.batch_size, shuffle=True, seed=args.seed,
                                    image_size=image_size), augment=args.augment)
    test_ds = prepare(load_dataset(args.base_path / "test", args.batch_size, shuffle=False, seed=args.seed,
                                   image_size=image_size))

    weights = None if args.weights.lower() == "none" else args.weights
    base, model = build_model((args.dense1_units, args.dense2_units, args.dense3_units), args.droprate, weights,
                              image_size)

    # Round 1: train the dense head and the last few ResNet-50 layers
    set_trainable_layers(base, ROUND_1_TRAINABLE_LAYERS)
    compile_model(model, args.lr_dense)
    first_round = model.fit(
        train_ds, epochs=args.n_epochs, validation_data=test_ds,
        callbacks=make_callbacks(log_dir, args.lr_dense, args.droprate, args.early_stopping_patience))

    # Round 2: unfreeze more of the ResNet-50 and fine-tune with a lower learning rate
    set_trainable_layers(base, args.unfreeze_after)
    compile_model(model, args.lr_finetune)
    history = model.fit(
        train_ds, epochs=args.n_epochs, validation_data=test_ds,
        callbacks=make_callbacks(log_dir, args.lr_finetune, args.droprate, args.early_stopping_patience,
                                 best_val_accuracy=max(first_round.history["val_accuracy"])))

    model.save(log_dir / "final_model.keras")
    mean_diff, median_diff = grade_differential(model, test_ds)
    final = {key: values[-1] for key, values in history.history.items()}

    print(f"Model Summary: [ lr = {args.lr_dense} , lr_finetune = {args.lr_finetune} , droprate = {args.droprate} ]")
    print(f"Train Accuracy:  {final['accuracy']}")
    print(f"Train Precision: {final['precision']}")
    print(f"Train Recall:    {final['recall']}")
    print(f"Test Accuracy (Xval):  {final['val_accuracy']}")
    print(f"Test Precision (Xval): {final['val_precision']}")
    print(f"Test Recall (Xval):    {final['val_recall']}")
    print(f"Grade Differential (mean):   {mean_diff}")
    print(f"Grade Differential (median): {median_diff}")
    print(f"Logs and saved models: {log_dir}\n")


if __name__ == "__main__":
    main()
