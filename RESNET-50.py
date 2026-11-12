"""Train and evaluate a ResNet-50 based PSA grade classifier.

USAGE:
    python RESNET-50.py --n_epochs 50 --droprate 0.025 --base_path /home/data/data_for_model_3

Training happens in two rounds:
    1. the dense head plus the last 20 ResNet-50 layers are trained
    2. the last `--unfreeze_after` ResNet-50 layers are unfrozen and the model
       is fine-tuned with a lower learning rate (`--lr_finetune`)
"""
import argparse
import datetime
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# PSA grades, in numeric order, so that class index + 1 == grade.
# (Letting Keras infer the classes would sort the folders as strings: 1, 10, 2, ...)
CLASS_NAMES = [str(grade) for grade in range(1, 11)]

# (height, width): the cropped cards are portrait, 400 px wide
IMAGE_SIZE = (600, 400)

# number of ResNet-50 layers (counted from the end) trained in the first round
ROUND_1_TRAINABLE_LAYERS = 20

SPLIT_RATIO = 0.8


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to train and evaluate a model.')
    parser.add_argument('-e', '--n_epochs', type=int, default=10,
                        help='Number of epochs per training cycles (positive integer)')
    parser.add_argument('-b', '--base_path', type=str, required=True,
                        help='The base path for the data folder containing both /train and /test folders')
    parser.add_argument('-d', '--droprate', type=float, default=0.012,
                        help='The droprate for the dropout layers (float between 0 and 1)')
    parser.add_argument('--lr_dense', type=float, default=0.0001,
                        help='Learning rate for the first training round')
    parser.add_argument('--lr_finetune', type=float, default=None,
                        help='Learning rate for the second (fine-tuning) round, defaults to lr_dense / 10')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and validation data')
    parser.add_argument('--dense1_units', type=int, default=1024,
                        help='Number of units in the first additional dense layer')
    parser.add_argument('--dense2_units', type=int, default=512,
                        help='Number of units in the second additional dense layer')
    parser.add_argument('--dense3_units', type=int, default=128,
                        help='Number of units in the third additional dense layer')
    parser.add_argument('--unfreeze_after', type=int, default=50,
                        help='How many ResNet50 layers, counted from the last, are trainable in the second round')
    parser.add_argument('--augment', action='store_true',
                        help='Apply random flip/rotation/shift/zoom augmentation to the training images')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Stop a round after this many epochs without val_loss improvement (default: off)')
    parser.add_argument('--weights', type=str, default='imagenet',
                        help="ResNet50 weights: 'imagenet', a path to a weights file, or 'none' for random init")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the train/test split, shuffling and weight init')
    return parser.parse_args()


def split_into_train_test(base_path, seed):
    """Move any loose .jpg files in base_path into train/<grade> and test/<grade> folders."""
    rng = random.Random(seed)
    for file_name in sorted(os.listdir(base_path)):
        if not file_name.lower().endswith('.jpg'):
            continue
        label = int(file_name.split('_PSA')[1].split('_')[0])  # extract PSA grade from file name
        dir_name = 'train' if rng.random() < SPLIT_RATIO else 'test'
        dir_path = os.path.join(base_path, dir_name, str(label))
        os.makedirs(dir_path, exist_ok=True)
        try:
            os.rename(os.path.join(base_path, file_name), os.path.join(dir_path, file_name))
        except FileNotFoundError:
            # another training run (e.g. a parallel grid search job) already moved it
            pass


def load_dataset(path, batch_size, shuffle, seed):
    return keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed)


def build_augmentation():
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(20 / 360, fill_mode='nearest'),
        layers.RandomTranslation(0.2, 0.2, fill_mode='nearest'),
        layers.RandomZoom(0.2, fill_mode='nearest'),
    ])


def build_model(args):
    weights = None if args.weights.lower() == 'none' else args.weights
    resnet50_model = ResNet50(weights=weights, include_top=False, input_shape=IMAGE_SIZE + (3,))

    # Pool the feature map instead of flattening it: flattening 19x13x2048 features
    # into the first dense layer would need ~500M weights
    x = layers.GlobalAveragePooling2D()(resnet50_model.output)
    x = layers.Dense(args.dense1_units, activation='relu')(x)
    x = layers.Dropout(args.droprate)(x)
    x = layers.Dense(args.dense2_units, activation='relu')(x)
    x = layers.Dropout(args.droprate)(x)
    x = layers.Dense(args.dense3_units, activation='relu')(x)
    predictions = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    return resnet50_model, keras.Model(inputs=resnet50_model.input, outputs=predictions)


def set_trainable_layers(resnet50_model, n_trainable):
    """Make only the last n_trainable layers of the ResNet-50 trainable."""
    first_trainable = len(resnet50_model.layers) - n_trainable
    for i, layer in enumerate(resnet50_model.layers):
        layer.trainable = i >= first_trainable


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')])


class CustomCSVLogger(keras.callbacks.CSVLogger):
    """CSVLogger that also records the learning rate and dropout rate of the run."""

    def __init__(self, filename, learning_rate, dropout_rate, append=False, separator=','):
        super().__init__(filename, separator=separator, append=append)
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.learning_rate
        logs['drop_rate'] = self.dropout_rate
        self.keys = sorted(logs.keys())
        super().on_epoch_end(epoch, logs)


def make_callbacks(log_dir, learning_rate, args):
    callbacks = [
        CustomCSVLogger(os.path.join(log_dir, 'log.csv'), learning_rate, args.droprate, append=True),
        # keep the best model seen so far, so a run can be evaluated or reused later
        keras.callbacks.ModelCheckpoint(
            os.path.join(log_dir, 'best_model.keras'), monitor='val_accuracy', save_best_only=True),
    ]
    if args.early_stopping_patience is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.early_stopping_patience, restore_best_weights=True))
    return callbacks


def grade_differential(model, test_ds):
    """Mean and median absolute difference between predicted and true grade, over the whole test set."""
    true_labels = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    predicted_labels = np.argmax(model.predict(test_ds), axis=1)
    # the classes are in grade order, so a difference of class indices is a difference of grades
    absolute_diffs = np.abs(predicted_labels - true_labels)
    return np.mean(absolute_diffs), np.median(absolute_diffs)


def main():
    args = parse_arguments()
    lr_finetune = args.lr_finetune if args.lr_finetune is not None else args.lr_dense / 10
    keras.utils.set_random_seed(args.seed)

    train_path = os.path.join(args.base_path, 'train')
    test_path = os.path.join(args.base_path, 'test')
    if not os.path.isdir(args.base_path):
        raise SystemExit("base_path does not exist: {}".format(args.base_path))

    # One log folder per run. The pid keeps runs that start in the same second
    # (e.g. a parallel grid search) from writing into the same folder.
    run_name = "{}-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), os.getpid())
    log_dir = os.path.join(os.getcwd(), 'logs', run_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        json.dump(dict(vars(args), lr_finetune=lr_finetune), f, indent=2)

    split_into_train_test(args.base_path, args.seed)

    train_ds = load_dataset(train_path, args.batch_size, shuffle=True, seed=args.seed)
    test_ds = load_dataset(test_path, args.batch_size, shuffle=False, seed=args.seed)

    # The ImageNet weights expect the ResNet-50 preprocessing, not a 0-1 rescale
    if args.augment:
        augmentation = build_augmentation()
        train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    resnet50_model, model = build_model(args)

    # Round 1: train the dense head and the last few ResNet-50 layers
    set_trainable_layers(resnet50_model, ROUND_1_TRAINABLE_LAYERS)
    compile_model(model, args.lr_dense)
    model.fit(train_ds, epochs=args.n_epochs, validation_data=test_ds,
              callbacks=make_callbacks(log_dir, args.lr_dense, args))

    # Round 2: unfreeze more of the ResNet-50 and fine-tune with a lower learning rate
    set_trainable_layers(resnet50_model, args.unfreeze_after)
    compile_model(model, lr_finetune)
    history = model.fit(train_ds, epochs=args.n_epochs, validation_data=test_ds,
                        callbacks=make_callbacks(log_dir, lr_finetune, args))

    model.save(os.path.join(log_dir, 'final_model.keras'))

    mean_diff, median_diff = grade_differential(model, test_ds)

    print("Model Summary: [ lr = {} , lr_finetune = {} , droprate = {} ]".format(
        args.lr_dense, lr_finetune, args.droprate))
    print("Train Accuracy:  {}".format(history.history['accuracy'][-1]))
    print("Train Precision: {}".format(history.history['precision'][-1]))
    print("Train Recall:    {}".format(history.history['recall'][-1]))
    print("Test Accuracy (Xval):  {}".format(history.history['val_accuracy'][-1]))
    print("Test Precision (Xval): {}".format(history.history['val_precision'][-1]))
    print("Test Recall (Xval):    {}".format(history.history['val_recall'][-1]))
    print("Grade Differential (mean):   {}".format(mean_diff))
    print("Grade Differential (median): {}".format(median_diff))
    print("Logs and saved models: {}\n".format(log_dir))


if __name__ == '__main__':
    main()
