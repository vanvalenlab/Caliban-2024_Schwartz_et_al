import os

import tensorflow as tf
import typer
from deepcell.data.tracking import Track
from deepcell.datasets import DynamicNuclearNetTracking
from deepcell.utils.tfrecord_utils import write_tracking_dataset_to_tfr
from deepcell_tracking.trk_io import load_trks
from deepcell_tracking.utils import get_max_cells
from typing_extensions import Annotated


def main(
    data_path: Annotated[
        str, typer.Option(help="Path to training data directory")
    ] = None,
    appearance_dim: Annotated[
        int, typer.Option(help="Length of appearance dimension")
    ] = 32,
    distance_threshold: Annotated[int, typer.Option(help="Distance threshold")] = 64,
    crop_mode: Annotated[str, typer.Option(help="Crop mode")] = "resize",
):
    # Download tracking data if path is not provided
    if data_path is None:
        dnn_trk = DynamicNuclearNetTracking()
        data_path = dnn_trk.path
    train_trks = load_trks(os.path.join(data_path, "train.trks"))
    val_trks = load_trks(os.path.join(data_path, "val.trks"))

    max_cells = max([get_max_cells(train_trks["y"]), get_max_cells(val_trks["y"])])

    for split, trks in zip({"train", "val"}, [train_trks, val_trks]):
        print(f"Preparing {split} as tf record")

        with tf.device("/cpu:0"):
            tracks = Track(
                tracked_data=trks,
                appearance_dim=appearance_dim,
                distance_threshold=distance_threshold,
                crop_mode=crop_mode,
            )

            write_tracking_dataset_to_tfr(
                tracks, target_max_cells=max_cells, filename=split
            )


if __name__ == "__main__":
    typer.run(main)
