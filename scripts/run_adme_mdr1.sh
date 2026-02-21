#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ADME-MDR1 dataset experiments
# Dataset: adme_mdr1 (1 task, classification)
# Samples: 2,642
# Threshold: 0.301 (ER >= 2 = P-gp substrate)
# Split: UMAP (optimized for ADME datasets)
# Epochs: 301

echo "Running experiments on ADME-MDR1 dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset adme_mdr1 --split umap --epochs 301
python src/run.py --config configs/ka_gat.yaml --dataset adme_mdr1 --split umap --epochs 301

echo "ADME-MDR1 experiments completed."
