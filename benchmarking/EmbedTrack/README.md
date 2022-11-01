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
gzip KIT-LOE-GE.zip
```

### Build and run the docker container
```bash
docker build -t $USER/EmbedTrack:latest .
```

```bash
docker run -it --gpus "device=0" \
    -p 8888:8888 \
    -v $PWD:/EmbedTrack
    -v $PWD/../../data:/data \
    $USER/EmbedTrack:latest
```
