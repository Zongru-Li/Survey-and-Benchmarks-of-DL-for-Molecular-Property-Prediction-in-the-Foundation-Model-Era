#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# adme_rlm dataset experiments with graphkan
# Dataset: adme_rlm (ADME, regression)
# Splits: scaffold, butina, umap, time
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset adme_rlm --split scaffold --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rlm --split butina --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rlm --split time --epochs 501 &
wait
