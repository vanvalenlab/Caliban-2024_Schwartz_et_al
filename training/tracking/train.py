import os

import deepcell
import tensorflow as tf
import typer
import yaml
from deepcell.data.tracking import random_rotate, random_translate, temporal_slice
from deepcell.model_zoo.tracking import GNNTrackingModel
from deepcell.utils.tfrecord_utils import get_tracking_dataset
from deepcell.utils.train_utils import count_gpus, rate_scheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow_addons.optimizers import RectifiedAdam
from typing_extensions import Annotated


def filter_and_flatten(y_true, y_pred):
    n_classes = tf.shape(y_true)[-1]
    new_shape = [-1, n_classes]
    y_true = tf.reshape(y_true, new_shape)
    y_pred = tf.reshape(y_pred, new_shape)

    # Mask out the padded cells
    y_true_reduced = tf.reduce_sum(y_true, axis=-1)
    good_loc = tf.where(y_true_reduced == 1)[:, 0]

    y_true = tf.gather(y_true, good_loc, axis=0)
    y_pred = tf.gather(y_pred, good_loc, axis=0)
    return y_true, y_pred


class Recall(tf.keras.metrics.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = filter_and_flatten(y_true, y_pred)
        super().update_state(y_true, y_pred, sample_weight)


class Precision(tf.keras.metrics.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = filter_and_flatten(y_true, y_pred)
        super().update_state(y_true, y_pred, sample_weight)


def loss_function(y_true, y_pred):
    y_true, y_pred = filter_and_flatten(y_true, y_pred)
    return deepcell.losses.weighted_categorical_crossentropy(
        y_true, y_pred, n_classes=tf.shape(y_true)[-1], axis=-1
    )


def create_model(
    max_cells,
    strategy,
    n_layers=1,
    graph_layer="gcs",
    track_length=8,
    lr=1e-4,
    n_filters=64,
    embedding_dim=64,
    encoder_dim=64,
    norm_layer="batch",
):
    """Create a model to train and compile"""
    with strategy.scope():
        model = GNNTrackingModel(
            max_cells=max_cells,
            graph_layer=graph_layer,
            track_length=track_length,
            n_filters=n_filters,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim,
            n_layers=n_layers,
            norm_layer=norm_layer,
        )

        loss = {"temporal_adj_matrices": loss_function}

        optimizer = RectifiedAdam(learning_rate=lr, clipnorm=0.001)

        training_metrics = [
            Recall(class_id=0, name="same_recall"),
            Recall(class_id=1, name="different_recall"),
            Recall(class_id=2, name="daughter_recall"),
            Precision(class_id=0, name="same_precision"),
            Precision(class_id=1, name="different_precision"),
            Precision(class_id=2, name="daughter_precision"),
        ]

        model.training_model.compile(
            loss=loss, optimizer=optimizer, metrics=training_metrics
        )

    return model


def train(
    model,
    train_data,
    val_data,
    lr=1e-4,
    epochs=8,
    steps_per_epoch=512,
    validation_steps=100,
    train_log_path=None,
):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    monitor = "val_loss"

    csv_logger = CSVLogger(train_log_path)

    # Create callbacks for early stopping and pruning.
    callbacks = [
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
        csv_logger,
    ]

    print(f"Training on {count_gpus()} GPUs.")

    # Train model.
    history = model.training_model.fit(
        train_data,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    print("Final", monitor, ":", history.history[monitor][-1])
    return history


def main(
    inf_model_path: Annotated[
        str, typer.Option(help="Path to the inference model")
    ] = "NuclearTrackingInf",
    ne_model_path: Annotated[
        str, typer.Option(help="Path to the neighborhood encoder")
    ] = "NuclearTrackingNE",
    metrics_path: Annotated[
        str, typer.Option(help="Destination of recorded metrics from training")
    ] = "train-metrics.yaml",
    log_path: Annotated[
        str,
        typer.Option(help="Destination of training metrics recorded at every epoch"),
    ] = "train_log.csv",
    data_path: Annotated[
        str, typer.Option(help="Path to the tfrecord files")
    ] = ".",
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    batch_size: Annotated[int, typer.Option(help="Number of samples per batch")] = 8,
    track_length: Annotated[
        int, typer.Option(help="Number of frames per track object.")
    ] = 8,
    n_layers: Annotated[
        int, typer.Option(help="Number of graph convolution layers.")
    ] = 1,
    n_filters: Annotated[int, typer.Option(help="Number of filters")] = 64,
    encoder_dim: Annotated[int, typer.Option(help="Length of encoder dimension")] = 64,
    embedding_dim: Annotated[
        int, typer.Option(help="Length of embedding dimension")
    ] = 64,
    graph_layer: Annotated[
        str, typer.Option(help="Type of graph convolutional layer.")
    ] = "gat",
    epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 50,
    steps_per_epoch: Annotated[
        int, typer.Option(help="Number of steps per epoch.")
    ] = 1000,
    validation_steps: Annotated[
        int, typer.Option(help="Number of validation steps per epoch")
    ] = 200,
    rotation_range: Annotated[
        int, typer.Option(help="Rotation range for data augmentation")
    ] = 180,
    translation_range: Annotated[
        int, typer.Option(help="Translation range for data augmentation.")
    ] = 512,
    buffer_size: Annotated[
        int, typer.Option(help="Buffer size for dataset shuffling.")
    ] = 128,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    norm_layer: Annotated[
        str, typer.Option(help="Type of normalization layer")
    ] = "batch",
):
    # Augmentation functions
    def sample(X, y):
        return temporal_slice(X, y, track_length=track_length)

    def rotate(X, y):
        return random_rotate(X, y, rotation_range=rotation_range)

    def translate(X, y):
        return random_translate(X, y, range=translation_range)

    with tf.device("/cpu:0"):
        train_data = get_tracking_dataset(os.path.join(data_path, "train"))
        train_data = train_data.shuffle(buffer_size, seed=seed).repeat()
        train_data = train_data.map(sample, num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.map(rotate, num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.map(translate, num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_data = get_tracking_dataset(os.path.join(data_path, "val"))
        val_data = val_data.shuffle(buffer_size, seed=seed).repeat()
        val_data = val_data.map(sample, num_parallel_calls=tf.data.AUTOTUNE)
        val_data = val_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Create model instance.
    max_cells = list(train_data.take(1))[0][0]["appearances"].shape[2]

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    model = create_model(
        max_cells,
        strategy,
        n_layers=n_layers,
        graph_layer=graph_layer,
        track_length=track_length,
        n_filters=n_filters,
        embedding_dim=embedding_dim,
        encoder_dim=encoder_dim,
        lr=lr,
        norm_layer=norm_layer,
    )

    # train the model
    history = train(
        model,
        train_data,
        val_data,
        lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        train_log_path=log_path,
    )

    # save the model
    model.inference_model.save(inf_model_path, include_optimizer=False, overwrite=True)

    model.neighborhood_encoder.save(
        ne_model_path, include_optimizer=False, overwrite=True
    )

    all_metrics = {
        "metrics": {"training": {k: float(v[-1]) for k, v in history.history.items()}}
    }

    # save a metadata.yaml file in the saved model directory
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f)


if __name__ == "__main__":
    typer.run(main)
