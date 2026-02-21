#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ADME-Sol dataset experiments
# Dataset: adme_sol (1 task, classification)
# Samples: 2,173
# Threshold: 1.543 (median, high solubility)
# Split: UMAP (optimized for ADME datasets)
# Epochs: 301

echo "Running experiments on ADME-Sol dataset..."

# python src/run.py --config configs/ka_gnn.yaml --dataset adme_sol --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset adme_sol --split umap --epochs 501

echo "ADME-Sol experiments completed."
