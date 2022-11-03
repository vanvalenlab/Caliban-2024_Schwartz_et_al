ARG CUDA_TAG=11.4-base-ubuntu20.04

FROM gpuci/miniconda-cuda:$CUDA_TAG

# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install gcc in order to build deepcell-toolbox wheels
RUN apt-get update && apt-get install -y gcc

WORKDIR /EmbedTrack

COPY embedtrack/environment.yml .

RUN conda env create -f environment.yml

# Activate conda environment
SHELL ["conda", "run", "-n", "venv_embedtrack", "/bin/bash", "-c"]

RUN pip install imagecodecs --no-dependencies \
    && pip install jupyterlab "cffi==1.15.0" "deepcell-tracking~=0.6.3"

CMD ["conda", "run", "--no-capture-output", "-n", "venv_embedtrack", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
