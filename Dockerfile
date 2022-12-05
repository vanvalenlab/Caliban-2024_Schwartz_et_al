ARG CUDA_TAG=11.4-base-ubuntu20.04

FROM gpuci/miniconda-cuda:$CUDA_TAG

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install gcc in order to build deepcell-toolbox wheels
RUN apt-get update && apt-get install -y gcc git

# Install compiler for scikit-fmm
RUN apt install -y build-essential

WORKDIR /publication-tracking

# Install requirements for the base environment
RUN conda create -n deepcell python=3.8 nb_conda_kernels jupyterlab

# Activate base environment
SHELL ["conda", "run", "-n", "deepcell", "/bin/bash", "-c"]
COPY requirements.txt .
RUN pip install -r requirements.txt

### EmbedTrack ###
COPY benchmarking/EmbedTrack/embedtrack/environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "venv_embedtrack", "/bin/bash", "-c"]
RUN pip install imagecodecs --no-dependencies \
    && pip install ipykernel "cffi==1.15.0" "git+https://github.com/vanvalenlab/deepcell-tracking.git@master"


### GNN TF ###
RUN conda env create --name gnn-tf python==3.6
SHELL ["conda", "run", "-n", "gnn-tf", "/bin/bash", "-c"]
RUN pip install tensorflow-gpu==2.4.1 scikit-learn scikit-image imagecodecs imageio Pillow \
    scikit-fmm==2022.3.26 seaborn opencv-python-headless ipykernel


### GNN PyTorch ###
COPY benchmarking/CellTrackerGNN/cell-tracker-gnn/requirements-conda.txt .
RUN conda create --name gnn-pytorch --file requirements-conda.txt
SHELL ["conda", "run", "-n", "gnn-pytorch", "/bin/bash", "-c"]
RUN pip install ipykernel opencv-python-headless PyYAML==5.4.1 omegaconf

CMD ["conda", "run", "--no-capture-output", "-n", "deepcell", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]