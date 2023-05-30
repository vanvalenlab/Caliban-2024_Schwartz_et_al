import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import typer
import yaml
from deepcell.applications import NuclearSegmentation
from deepcell_toolbox.metrics import Metrics
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from typing_extensions import Annotated


def update_data_split(source, data_dir):
    """Match the nuclear segmentation training data split to the tracking split"""
    # Load all data splits
    X, y, meta = [], [], []
    for split in {"train", "test", "val"}:
        with np.load(os.path.join(data_dir, f"{split}.npz"), allow_pickle=True) as data:
            X.append(data["X"])
            y.append(data["y"])
            meta.append(data["meta"][1:])

    X = np.concatenate(X)
    y = np.concatenate(y)
    meta = np.concatenate(meta)

    data = {}
    for split in {"train", "test", "val"}:
        tmp_x, tmp_y = [], []
        for src in source[split][:, 0]:
            tmp_x.append(X[meta[:, 0] == src])
            tmp_y.append(y[meta[:, 0] == src])

        data[split] = {"X": np.concatenate(tmp_x), "y": np.concatenate(tmp_y)}

    return data


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

    return X, y


def evaluate(app, X_test, y_test, postprocess_kwargs=None):
    y_pred = app.predict(X_test, postprocess_kwargs=postprocess_kwargs)
    y_true = y_test.copy()

    # run the metrics
    m = Metrics("DeepWatershed - Remove no pixels", seg=False)
    metrics = m.calc_object_stats(y_true, y_pred)
    summary = m.summarize_object_metrics_df(metrics)

    valid_keys = {
        "recall",
        "precision",
        "jaccard",
        "n_true",
        "n_pred",
        "gained_detections",
        "missed_detections",
        "split",
        "merge",
        "catastrophe",
    }
    output_data = {}
    for k in valid_keys:
        if k in {"jaccard", "recall", "precision"}:
            output_data[k] = float(summary[k])
        else:
            output_data[k] = int(summary[k])

    return output_data


def create_overlays(x, gt, pred):
    x = np.squeeze(x)
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)

    # Rescale raw data
    percentiles = np.percentile(x[np.nonzero(x)], [5, 95])
    raw = rescale_intensity(
        x, in_range=(percentiles[0], percentiles[1]), out_range="float32"
    )

    # Overlay gt on raw
    gt_overlay = label2rgb(gt, image=raw, bg_label=0)

    # Overlay pred on raw
    pred_overlay = label2rgb(pred, image=raw, bg_label=0)

    return gt_overlay, pred_overlay


def main(
    model_path: Annotated[
        str, typer.Option(help="Path to the trained models")
    ] = "NuclearSegmentation",
    metrics_path: Annotated[
        str, typer.Option(help="Destination of evaluation metrics")
    ] = "evaluate-metrics.yaml",
    predictions_path: Annotated[
        str, typer.Option(help="Path to save sample predictions")
    ] = "predictions.png",
    data_path: Annotated[
        str, typer.Option(help="Path to the training data")
    ] = "data/segmentation",
    tracking_data_source: Annotated[
        str, typer.Option(help="Path to tracking data-source.npz")
    ] = "data/tracking/data-source.npz",
    radius: Annotated[
        int, typer.Option(help="Radius parameter for deep_watershed postprocessing")
    ] = 10,
    maxima_threshold: Annotated[
        float,
        typer.Option(
            help="maxima threshold parameter for deep watershed postprocessing"
        ),
    ] = 0.1,
    interior_threshold: Annotated[
        float,
        typer.Option(
            help="Interior threshold parameter for deep watershed postprocessing"
        ),
    ] = 0.01,
    exclude_border: Annotated[
        bool,
        typer.Option(help="Exclude border parameter for deep watershed postprocessing"),
    ] = False,
    small_objects_threshold: Annotated[
        int,
        typer.Option(help="small objects threshold for deep watershed postprocessing"),
    ] = 0,
    min_distance: Annotated[
        int,
        typer.Option(
            help="Deep watershed parameter for minimum distance between objects"
        ),
    ] = 10,
):
    # Load data source for tracking
    source = np.load(
        os.path.join(os.path.dirname(__file__), tracking_data_source), allow_pickle=True
    )

    data = update_data_split(source, data_path)
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    # Prep postprocess kwargs
    postprocess_kwargs = {
        "radius": radius,
        "maxima_threshold": maxima_threshold,
        "interior_threshold": interior_threshold,
        "exclude_border": exclude_border,
        "small_objects_threshold": small_objects_threshold,
        "min_distance": min_distance,
    }

    # Load model and application
    model = tf.keras.models.load_model(model_path)
    app = NuclearSegmentation(model)

    # evaluate the model
    # TODO: evaluate based on experiment data type
    metrics = evaluate(app, X_test, y_test, postprocess_kwargs=postprocess_kwargs)

    all_metrics = {
        "inference": OrderedDict(sorted(metrics.items())),
    }

    # save a metadata.yaml file in the saved model directory
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f)

    # Plot sample predictions
    n = 10
    # Configure plot
    fig, ax = plt.subplots(n, 2, figsize=(20, 10 * n))
    ax[0, 0].set_title("Ground Truth")
    ax[0, 1].set_title("Prediction")
    plt.tight_layout()

    for j, i in enumerate(np.random.randint(X_test.shape[0], size=(n,))):
        gt, pred = create_overlays(
            X_test[i : i + 1], y_test[i : i + 1], app.predict(X_test[i : i + 1])
        )
        ax[j, 0].imshow(gt)
        ax[j, 0].axis("off")
        ax[j, 1].imshow(pred)
        ax[j, 1].axis("off")

    plt.savefig(predictions_path)


if __name__ == "__main__":
    typer.run(main)
