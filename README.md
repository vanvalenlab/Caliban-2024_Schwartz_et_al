# Caliban-2024_Schwartz_et_al

## Running deepcell

### In a `mamba` environment
These instructions are written using `mamba` as a package manager, but `conda` can be substituted for `mamba`.

```bash
mamba create --name deepcell python=3.8 tensorflow=2.8 cudatoolkit=11.8.0 ipykernel -c conda-forge
mamba activate deepcell
pip install -r requirements.txt
```

### In a docker container
```bash
# Start a GPU enabled container on one GPUs
docker run --gpus '"device=0"' -it --rm \
    -p 8888:8888 \
    -v $PWD/notebooks:/notebooks \
    vanvalenlab/deepcell-tf:0.12.9-gpu
```

### DeepCell API Key
An API key is required to access the DynamicNuclearNet dataset and Caliban models. Please see the [docs](https://deepcell.readthedocs.io/en/master/API-key.html) for more information.

## Data
The DynamicNuclearNet dataset can be accessed through `deepcell.datasets` ([docs](https://deepcell.readthedocs.io/en/master/data-gallery/dynamicnuclearnet.html)). Instructions for accessing additional data needed to run the notebooks and scripts in this repo are located in the `README.md` files inside `data`.

## Training Caliban Models
The scripts for training the nuclear segmentation and tracking models used in Caliban are included in the `training/segmentation` and `training/tracking` folders. The `deepcell` environment created above can be used for all of these scripts. Instructions for running the scripts are located in `training/README.md`.

## Benchmarking
Instructions and code for reproducing model benchmarking are included in the `benchmarking` folder with specific instructions for each model located in each subfolder.
