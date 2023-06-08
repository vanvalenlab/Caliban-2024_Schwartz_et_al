# EmbedTrack

K. Löffler and M. Mikut (2022). EmbedTrack -- Simultaneous Cell Segmentation and Tracking Through Learning Offsets and Clustering Bandwidths. arXiv preprint. DOI: [10.48550/arXiv.2204.10713](https://doi.org/10.48550/arXiv.2204.10713)

## Setup

### Clone repo and checkout the commit used for benchmarking
```bash
git clone https://git.scc.kit.edu/kit-loe-ge/embedtrack.git
git checkout 28af4579a541a91dc4f57c4dee65b24f637278d8
```

### Download model submission for CTC
```bash
wget http://public.celltrackingchallenge.net/participants/KIT-Loe-GE.zip
unzip KIT-LOE-GE.zip
```

### Deepcell test data
Confirm that the deepcell dataset `test.trks` file is placed in the root `data` folder.

### Build the enviornment
```bash
mamba env create -f environment.yml
mamba activate venv_embedtrack
pip install imagecodecs --no-dependencies
pip install jupyterlab "cffi==1.15.0" deepcell-tracking~=0.6.4
```