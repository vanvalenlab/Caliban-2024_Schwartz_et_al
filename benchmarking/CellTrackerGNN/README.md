# Cell Tracker GNN

## Download model and software submission for CTC
```bash
wget https://github.com/talbenha/cell-tracker-gnn/releases/download/CTC/software.zip
unzip software.zip
```

## Build the environments
```bash
mamba create --name gnn-tf python=3.7
mamba activate gnn-tf
pip install tensorflow-gpu==2.4.1 scikit-learn scikit-image imagecodecs imageio Pillow scikit-fmm==2022.3.26 seaborn opencv-python-headless jupyterlab
```

```bash
mamba create --name gnn-pytorch --file requirements-conda.txt
mamba activate gnn-pytorch
pip install jupyterlab opencv-python-headless PyYAML==5.4.1 omegaconf
```