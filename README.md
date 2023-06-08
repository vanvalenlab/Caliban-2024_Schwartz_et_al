# Caliban-2023_Schwartz_et_al

## Setup a deepcell environment
These instructions are written using `mamba` as a package manager, but `conda` can be substituted for `mamba`.

```bash
mamba create --name deepcell python=3.8 tensorflow=2.8 cudatoolkit=11.8.0 ipykernel -c conda-forge
mamba activate deepcell
pip install -r requirements.txt
```

## Data
Instructions for accessing the data needed to run the notebooks and scripts in this repo are located in the `README.md` files inside `data`.

## Training Calban Models
The scripts for training the nuclear segmentation and tracking models used in Caliban are included in the `training/segmentation` and `training/tracking` folders. The `deepcell` environment created above can be used for all of these scripts. Instructions for running the scripts are located in `training/README.md`.

## Benchmarking
Instructions and code for reproducing model benchmarking are included in the `benchmarking` folder with specific instructions for each model located in each subfolder.
