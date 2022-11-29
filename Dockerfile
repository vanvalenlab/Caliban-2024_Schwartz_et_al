ARG DOCKER_TAG=0.12.3-gpu

FROM vanvalenlab/deepcell-tf:$DOCKER_TAG

WORKDIR /publication-tracking

RUN apt-get update && apt-get install -y \
    git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt jupyterlab

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
