# /bin/bash

# Clone repo and checkout the commit used for benchmarking
git clone https://git.scc.kit.edu/kit-loe-ge/embedtrack.git
git checkout 28af4579a541a91dc4f57c4dee65b24f637278d8

# Create conda environment
conda env create -f embedtrack/environment.yml
conda activate venv_embedtrack

# Install additional pip requirements
pip install imagecodecs --no-dependencies
pip install cffi=="1.15.0"