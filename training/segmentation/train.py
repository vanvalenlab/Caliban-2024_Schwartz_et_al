import os
import tempfile

import deepcell
import numpy as np
import tensorflow as tf
import typer
import yaml
from deepcell.losses import weighted_categorical_crossentropy
from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell.utils.train_utils import count_gpus, rate_scheduler
from deepcell_toolbox.processing import histogram_normalization
from typing_extensions import Annotated


def update_data_split(source, data_dir):
    """Match the nuclear segmentation training data split to the tracking split"""
    # Load all data splits
    X, y, meta, original_split = [], [], [], []
    for split in {"train", "test", "val"}:
        with np.load(os.path.join(data_dir, f"{split}.npz"), allow_pickle=True) as data:
            X.append(data["X"])
            y.append(data["y"])
            meta.append(data["meta"][1:])
            original_split.append([split] * data["X"].shape[0])

    X = np.concatenate(X)
    y = np.concatenate(y)
    meta = np.concatenate(meta)
    original_split = np.concatenate(original_split)

    # Check for missed data
    all_source = np.concatenate(list(source.values()))

    missing = []
    for f in np.unique(meta[:, 0]):
        if f not in all_source[:, 0]:
            missing.append(f)

    data = {}
    for split in {"train", "test", "val"}:
        data[split] = {"X": [], "y": []}
        for src in source[split][:, 0]:
            data[split]["X"].append(X[meta[:, 0] == src])
            data[split]["y"].append(y[meta[:, 0] == src])

    for f in missing:
        # Look up original split
        split = original_split[meta[:, 0] == f][0]
        data[split]["X"].append(X[meta[:, 0] == f])
        data[split]["y"].append(y[meta[:, 0] == f])

    for d in data.values():
        d["X"] = np.concatenate(d["X"])
        d["y"] = np.concatenate(d["y"])

    return data


def count_cells(y):
    count = 0
    for frame in y:
        count += len(np.unique(frame)) - 1  # Subtract 1 for the background

    return count


def _load_npz(filepath):
    """Load a npz file"""
    data = np.load(filepath)
    X = data["X"]
    y = data["y"]

    print(
        "Loaded {}: X.shape: {}, y.shape {}".format(
            os.path.basename(filepath), X.shape, y.shape
        )
    )

    return {"X": X, "y": y}


def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes
            )
        return tf.keras.losses.MSE(y_pred, y_true)

    return _semantic_loss


def create_model(
    input_shape,
    backbone="resnet50",
    lr=1e-4,
    location=True,
    pyramid_levels=("P3", "P4", "P5", "P6", "P7"),
):
    """Create a model to train and compile"""
    model = PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        norm_method=None,
        num_semantic_classes=[1, 1, 2],  # inner distance, outer distance, fgbg
        location=location,  # should always be true
        include_top=True,
        backbone_levels=["C1", "C2", "C3", "C4", "C5"],
        pyramid_levels=pyramid_levels,
    )

    loss = {}

    # Give losses for all of the semantic heads
    for layer in model.layers:
        if layer.name.startswith("semantic_"):
            n_classes = layer.output_shape[-1]
            loss[layer.name] = semantic_loss(n_classes)

    optimizer = tf.keras.optimizers.Adam(lr=lr, clipnorm=0.001)

    model.compile(loss=loss, optimizer=optimizer)

    return model


def create_prediction_model(
    input_shape, backbone, weights_path, location, pyramid_levels
):
    """Remove the fgbg head from the model and load weights"""
    prediction_model = PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        norm_method=None,
        num_semantic_heads=2,
        num_semantic_classes=[1, 1],  # inner distance, outer distance
        location=location,  # should always be true
        include_top=True,
        backbone_levels=["C1", "C2", "C3", "C4", "C5"],
        pyramid_levels=pyramid_levels,
    )
    prediction_model.load_weights(weights_path, by_name=True)
    return prediction_model


def create_data_generators(
    train_dict,
    val_dict,
    crop_size=256,
    min_objects=1,
    zoom_min=0.75,
    seed=0,
    batch_size=16,
    outer_erosion_width=1,
    inner_distance_alpha="auto",
    inner_distance_beta=1,
    inner_erosion_width=0,
):
    # data augmentation parameters
    zoom_max = 1 / zoom_min

    # Preprocess the data
    train_dict["X"] = histogram_normalization(train_dict["X"])
    val_dict["X"] = histogram_normalization(val_dict["X"])

    # use augmentation for training but not validation
    datagen = deepcell.image_generators.CroppingDataGenerator(
        rotation_range=180,
        zoom_range=(zoom_min, zoom_max),
        horizontal_flip=True,
        vertical_flip=True,
        crop_size=(crop_size, crop_size),
    )

    datagen_val = deepcell.image_generators.CroppingDataGenerator(
        crop_size=(crop_size, crop_size)
    )

    transforms = ["inner-distance", "outer-distance", "fgbg"]

    transforms_kwargs = {
        "outer-distance": {"erosion_width": outer_erosion_width},
        "inner-distance": {
            "alpha": inner_distance_alpha,
            "beta": inner_distance_beta,
            "erosion_width": inner_erosion_width,
        },
    }

    train_data = datagen.flow(
        train_dict,
        seed=seed,
        min_objects=min_objects,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        batch_size=batch_size,
    )

    print("Created training data generator.")

    val_data = datagen_val.flow(
        val_dict,
        seed=seed,
        min_objects=min_objects,
        transforms=transforms,
        transforms_kwargs=transforms_kwargs,
        batch_size=batch_size,
    )

    print("Created validation data generator.")

    return train_data, val_data


def train(
    train_data,
    val_data,
    model_path,
    crop_size=256,
    backbone="resnet50",
    lr=1e-4,
    location=True,
    batch_size=16,
    epochs=8,
    train_log="train-log.csv",
    pyramid_levels=("P3", "P4", "P5", "P6", "P7"),
):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    monitor = "val_loss"

    input_shape = (crop_size, crop_size, 1)

    # Create model instance.
    model = create_model(input_shape, backbone, lr, location, pyramid_levels)

    csv_logger = tf.keras.callbacks.CSVLogger(train_log)

    # Create callbacks for early stopping and pruning.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            save_weights_only=False,
        ),
        tf.keras.callbacks.LearningRateScheduler(rate_scheduler(lr=lr, decay=0.99)),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=5,
            verbose=1,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        ),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        csv_logger,
    ]

    print(f"Training on {count_gpus()} GPUs.")

    # Train model.
    history = model.fit(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=epochs,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=callbacks,
    )

    print("Final", monitor, ":", history.history[monitor][-1])

    with tempfile.TemporaryDirectory() as tmpdirname:
        weights_path = os.path.join(str(tmpdirname), "model_weights.h5")
        model.save_weights(weights_path, save_format="h5")
        prediction_model = create_prediction_model(
            input_shape, backbone, weights_path, location, pyramid_levels
        )

    return prediction_model, history


def main(
    model_path: Annotated[
        str, typer.Option(help="Path to save segmentation model")
    ] = "NuclearSegmentation",
    metrics_path: Annotated[
        str, typer.Option(help="Path to save training metrics")
    ] = "metrics.yaml",
    train_log: Annotated[
        str, typer.Option(help="Path for csv training logs")
    ] = "train_log.csv",
    data_path: Annotated[
        str, typer.Option(help="Directory where training data is located")
    ] = "../../data/segmentation",
    tracking_data_source: Annotated[
        str, typer.Option(help="Path to tracking data-source.npz")
    ] = "data/tracking/data-source.npz",
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 16,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    min_objects: Annotated[
        int, typer.Option(help="Minimum number of objects in each training image")
    ] = 1,
    zoom_min: Annotated[
        float, typer.Option(help="Smallest zoom value. Zoom max is inverse of zoom min")
    ] = 0.75,
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 16,
    backbone: Annotated[
        str, typer.Option(help="Backbone of the model")
    ] = "efficientnetv2bl",
    crop_size: Annotated[
        int, typer.Option(help="Size of square patches to train on")
    ] = 256,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-4,
    outer_erosion_width: Annotated[
        int,
        typer.Option(help="Erosion_width paramter for the outer-distance transform"),
    ] = 1,
    inner_distance_alpha: Annotated[
        str, typer.Option(help="Alpha parameter for the inner-distance transform")
    ] = "auto",
    inner_distance_beta: Annotated[
        float, typer.Option(help="Beta parameter for inner distance transform")
    ] = 1,
    inner_erosion_width: Annotated[
        int, typer.Option(help="erosion width for inner distance transform")
    ] = 0,
    location: Annotated[bool, typer.Option("Whether to include location layer")] = True,
    pyramid_levels: Annotated[
        str, typer.Option("String of pyramid levels")
    ] = "P1-P2-P3-P4-P5-P6-P7",
):
    # Load data source for tracking
    source = np.load(
        os.path.join(os.path.dirname(__file__), tracking_data_source), allow_pickle=True
    )

    data = update_data_split(source, data_path)

    # Set up data generators with updated data
    train_data, val_data = create_data_generators(
        data["train"],
        data["val"],
        crop_size=crop_size,
        min_objects=min_objects,
        zoom_min=zoom_min,
        seed=seed,
        batch_size=batch_size,
        outer_erosion_width=outer_erosion_width,
        inner_distance_alpha=inner_distance_alpha,
        inner_distance_beta=inner_distance_beta,
        inner_erosion_width=inner_erosion_width,
    )

    # train the model
    model, history = train(
        train_data,
        val_data,
        crop_size=crop_size,
        backbone=backbone,
        model_path=model_path,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        train_log=train_log,
        location=location,
        pyramid_levels=pyramid_levels.split("-"),
    )

    # save the model
    model.save(model_path, include_optimizer=False, overwrite=True)

    all_metrics = {
        "training": {k: str(v[-1]) for k, v in history.history.items()},
        "data": {},
    }

    # Record dataset stats
    for split in data.keys():
        all_metrics["data"][split] = {}
        for size, dim in zip(data["X"].shape, "byxc"):
            all_metrics["data"][split][dim] = size
        all_metrics["data"][split]["annotations"] = count_cells(data["y"])

    # save a metadata.yaml file in the saved model directory
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f)


if __name__ == "__main__":
    typer.run(main)
