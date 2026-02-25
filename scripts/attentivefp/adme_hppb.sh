#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# adme_hppb dataset experiments with attentivefp
# Dataset: adme_hppb (ADME, regression)
# Splits: scaffold, butina, umap, time
# Epochs: 501

python src/run.py --config configs/attentivefp.yaml --dataset adme_hppb --split scaffold --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_hppb --split butina --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_hppb --split time --epochs 501 &
wait
