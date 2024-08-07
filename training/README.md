# README

# Training the segmentation model
Run the training script from `training/segmentation`
```bash
python train.py \
    --model-path NuclearSegmentation \
    --metrics-path metrics.yaml \
    --train-log train_log.csv \
    --data-path ../../data/segmentation \
    --tracking-data-source ../../data/tracking/data-source.npz \
    --epochs 16 \
    --seed 0 \
    --min-objects 1 \
    --zoom-min 0.75 \
    --batch-size 16 \
    --backbone efficientnetv2bl \
    --crop-size 256 \
    --lr "1e-4" \
    --outer-erosion-width 1 \
    --inner-distance-alpha auto \
    --inner-distance-beta 1 \
    --inner-erosion-width 0 \
    --location True \
    --pyramid-levels "P1-P2-P3-P4-P5-P6-P7"
```

Run the evaluation script from `training/segmentation`
```bash
python evaluate.py \
    --model-path NuclearSegmentation \
    --metrics-path evaluate-metrics.yaml \
    --predictions-path predictions.png \
    --data-path ../../data/segmentation \
    --tracking-data-source ../../data/tracking/data-source.npz \
    --radius 10 \
    --maxima-threshold 0.1 \
    --interior-threshold 0.01 \
    --exclude-border False \
    --small-objects-threshold 0 \
    --min-distance 10
```

# Training the tracking model
First, save the training and validation datasets as a TFRecord. Run the following script from `training/tracking`
```bash
python save_tfrecord.py \
    --data-path ../../data/tracking \
    --appearance-dim 16 \
    --distance-threshold 64 \
    --crop-mode fixed \
    --norm True
```

Run the training script from `training/tracking`
```bash
python train.py \
    --inf-model-path NuclearTrackingInf \
    --ne-model-path NuclearTrackingNE \
    --metrics-path train-metrics.yaml \
    --log-path train_log.csv \
    --data-path ../../data/tracking \
    --seed 0 \
    --batch-size 8 \
    --track-length 8 \
    --n-layers 1 \
    --n-filters 64 \
    --encoder-dim 64 \
    --embedding-dim 64 \
    --graph-layer gat \
    --epochs 50 \
    --steps-per-epoch 1000 \
    --validation-steps 200 \
    --rotation-range 180 \
    --translation-range 512 \
    --buffer-size 128 \
    --lr "1e-3" \
    --norm-layer batch
```

Run the evaluation script from `training/tracking`
```bash
python evaluate.py
    --inf-model-path NuclearTrackingInf \
    --ne-model-path NuclearTrackingNE \
    --metrics-path evaluate-metrics.yaml \
    --data-path ../../data/tracking \
    --track-length 8 \
    --death 0.99 \
    --birth 0.99 \
    --division 0.01 \
    --nuc-model-path ../segmentation/NuclearSegmentation \
    --prediction-gt-dir predictions-gt \
    --prediction-caliban-dir predictions-caliban \
    --appearance-dim 16 \
    --distance-threshold 64 \
    --crop-mode fixed \
    --norm True
```