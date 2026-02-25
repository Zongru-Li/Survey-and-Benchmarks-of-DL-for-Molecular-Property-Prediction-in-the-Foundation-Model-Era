#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# adme_rppb dataset experiments with graphkan
# Dataset: adme_rppb (ADME, regression)
# Splits: scaffold, butina, umap, time
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset adme_rppb --split scaffold --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rppb --split butina --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rppb --split umap --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rppb --split time --epochs 501 &
wait
