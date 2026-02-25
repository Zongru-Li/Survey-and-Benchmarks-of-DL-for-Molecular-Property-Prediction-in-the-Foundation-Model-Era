#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# adme_sol dataset experiments with dmpnn
# Dataset: adme_sol (ADME, regression)
# Splits: scaffold, butina, umap, time
# Epochs: 501

python src/run.py --config configs/dmpnn.yaml --dataset adme_sol --split scaffold --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_sol --split butina --epochs 501 &
python src/run.py --config configs/dmpnn.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/dmpnn.yaml --dataset adme_sol --split time --epochs 501 &
wait
