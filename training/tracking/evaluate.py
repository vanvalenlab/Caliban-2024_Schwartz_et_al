import os

import numpy as np
import tensorflow as tf
import typer
import yaml
from deepcell.applications import NuclearSegmentation
from deepcell_toolbox.metrics import Metrics
from deepcell_tracking import CellTracker
from deepcell_tracking.metrics import (
    benchmark_tracking_performance,
    calculate_summary_stats,
)
from deepcell_tracking.trk_io import load_trks
from deepcell_tracking.utils import is_valid_lineage
from typing_extensions import Annotated


def find_frames_with_objects(y):
    frames = []
    for f in range(y.shape[0]):
        objs = np.unique(y[f])
        objs = np.delete(objs, np.where(objs == 0))
        if len(objs) > 0:
            frames.append(f)
    return frames


def evaluate(
    ne_model,
    inf_model,
    X_test,
    y_test,
    lineages_test,
    meta,
    prediction_dir,
    track_length=8,
    death=0.99,
    birth=0.99,
    division=0.9,
    y_pred=None,
    iou_thresh=1,
):
    # Check that prediction directory exists and make if needed
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    metrics = {}
    exp_metrics = {}
    bad_batches = []
    for b in range(len(X_test)):
        # currently NOT saving any recall/precision information
        gt_path = os.path.join(prediction_dir, f"{b}-gt.trk")
        res_path = os.path.join(prediction_dir, f"{b}-res.trk")

        # Check that lineage is valid before proceeding
        if not is_valid_lineage(y_test[b], lineages_test[b]):
            bad_batches.append(b)
            continue

        frames = find_frames_with_objects(y_test[b])

        # Swap out predicted y for gt if y_pred is available
        if y_pred is not None:
            y = y_pred
        else:
            y = y_test

        tracker = CellTracker(
            movie=X_test[b][frames],
            annotation=y[b][frames],
            track_length=track_length,
            neighborhood_encoder=ne_model,
            tracking_model=inf_model,
            death=death,
            birth=birth,
            division=division,
        )

        try:
            tracker.track_cells()
        except Exception as err:
            print(
                "Failed to track batch {} due to {}: {}".format(
                    b, err.__class__.__name__, err
                )
            )
            bad_batches.append(b)
            continue

        tracker.dump(res_path)

        gt = {
            "X": X_test[b][frames],
            "y_tracked": y_test[b][frames],
            "tracks": lineages_test[b],
        }

        tracker.dump(filename=gt_path, track_review_dict=gt)

        results = benchmark_tracking_performance(
            gt_path, res_path, threshold=iou_thresh
        )

        exp = meta[b, 1]  # Grab the experiment column from metadata
        tmp_exp = exp_metrics.get(exp, {})

        for k in results:
            if k in metrics:
                metrics[k] += results[k]
            else:
                metrics[k] = results[k]

            if k in tmp_exp:
                tmp_exp[k] += results[k]
            else:
                tmp_exp[k] = results[k]

        exp_metrics[exp] = tmp_exp

    # Calculate summary stats for each set of metrics
    tmp_metrics = metrics.copy()
    del tmp_metrics["mismatch_division"]
    summary = calculate_summary_stats(**tmp_metrics, n_digits=3)
    metrics = {**metrics, **summary}

    for exp, m in exp_metrics.items():
        tmp_m = m.copy()
        del tmp_m["mismatch_division"]
        summary = calculate_summary_stats(**tmp_m, n_digits=3)
        exp_metrics[exp] = {**m, **summary}

    print(f"Failed to track {len(bad_batches)} batches: {bad_batches}")
    return metrics, exp_metrics


def prep_segmentations(X, y, model):
    app = NuclearSegmentation(model)

    # The model does not perform well on zero-padded images
    segmentations = []
    for b in range(X.shape[0]):
        # Calculate position of padding based on first frame
        # Assume that padding is in blocks on the edges of image
        good_rows = np.where(X[b, 0].any(axis=0))[0]
        good_cols = np.where(X[b, 0].any(axis=1))[0]

        slc = (
            slice(None),
            slice(good_cols[0], good_cols[-1] + 1),
            slice(good_rows[0], good_rows[-1] + 1),
            slice(None),
        )

        seg_movie = np.zeros_like(y[b])
        seg_movie[slc] = app.predict(X[b][slc])
        segmentations.append(seg_movie)

    y_pred = np.stack(segmentations)

    # Reshape to 4d for evaluation
    new_shape = [y.shape[0] * y.shape[1], *list(y.shape[2:])]

    m = Metrics("Nuclear Segmentation", seg=False)
    metrics = m.calc_object_stats(y.reshape(new_shape), y_pred.reshape(new_shape))
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

    return y_pred, output_data


def main(
    inf_model_path: Annotated[
        str, typer.Option(help="Path to the trained inference model")
    ] = "NuclearTrackingInf",
    ne_model_path: Annotated[
        str, typer.Option(help="Path to the trained neighborhood encoder")
    ] = "NuclearTrackingNE",
    metrics_path: Annotated[
        str, typer.Option(help="Path to save metrics")
    ] = "evaluate-metrics.yaml",
    data_path: Annotated[
        str, typer.Option(help="Path to data directory")
    ] = "../../data/tracking",
    track_length: Annotated[int, typer.Option(help="Number of frames per track")] = 8,
    death: Annotated[
        float, typer.Option(help="Parameter used to fill the death matrix in the LAP")
    ] = 0.99,
    birth: Annotated[
        float, typer.Option(help="Parameter used to fill the birth matrix in the LAP")
    ] = 0.99,
    division: Annotated[
        float, typer.Option(help="Probability threshold for assigning daughter cells")
    ] = 0.01,
    nuc_model_path: Annotated[
        str, typer.Option(help="Path to segmentation model")
    ] = "../segmentation/NuclearSegmentation",
    prediction_gt_dir: Annotated[
        str,
        typer.Option(help="Directory to save tracking predictions on GT segmentations"),
    ] = "predictions-gt",
    prediction_caliban_dir: Annotated[
        str,
        typer.Option(
            help="Directory to save tracking predictions on predicted segmentations"
        ),
    ] = "predictions-caliban",
):
    # Load models
    ne_model = tf.keras.models.load_model(ne_model_path)
    inf_model = tf.keras.models.load_model(inf_model_path)

    # Load data
    test_data = load_trks(os.path.join(data_path, "test.trks"))
    X_test = test_data["X"]
    y_test = test_data["y"]
    lineages_test = test_data["lineages"]

    # Load metadata array
    with np.load(os.path.join(data_path, "data-source.npz"), allow_pickle=True) as data:
        meta = data["test"]

    # # evaluate the model
    metrics, exp_metrics = evaluate(
        ne_model,
        inf_model,
        X_test,
        y_test,
        lineages_test,
        meta,
        prediction_gt_dir,
        track_length=track_length,
        birth=birth,
        death=death,
        division=division,
    )

    # Generate nuclear predictions for end-end testing
    nuc_model = tf.keras.models.load_model(nuc_model_path)
    y_pred, segment_metrics = prep_segmentations(X_test, y_test, nuc_model)

    # Evaluate tracking on segmentation predictions
    caliban_metrics, caliban_exp_metrics = evaluate(
        ne_model,
        inf_model,
        X_test,
        y_test,
        lineages_test,
        meta,
        prediction_caliban_dir,
        track_length=track_length,
        birth=birth,
        death=death,
        division=division,
        y_pred=y_pred,
        iou_thresh=0.6,
    )  # Allow for imperfect matching between segmentations

    all_metrics = {
        "ground_truth_segmentation_metrics": {
            "tracking": metrics,
            "tracking_by_experiment": exp_metrics,
        },
        "caliban_metrics": {
            "segmentation": segment_metrics,
            "tracking": caliban_metrics,
            "tracking_by_experiment": caliban_exp_metrics,
        },
    }

    # save a metadata.yaml file in the saved model directory
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f)

if __name__ == "__main__":
    typer.run(main)
