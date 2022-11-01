ARG CUDA_TAG=11.4-base-ubuntu20.04

FROM gpuci/miniconda-cuda:$CUDA_TAG

WORKDIR /EmbedTrack

COPY embedtrack/environment.yml .

RUN conda update -n base -c defaults conda

RUN conda env create -f environment.yml

# Activate conda environment
SHELL ["conda", "run", "-n", "venv_embedtrack", "/bin/bash", "-c"]

RUN pip install imagecodecs --no-dependencies \
    && pip install cffi=="1.15.0" jupyterlab
    #"deepcell-tracking==0.6.3"

CMD ["conda", "run", "--no-capture-output", "-n", "venv_embedtrack", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
