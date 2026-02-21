#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ADME-hPPB dataset experiments
# Dataset: adme_hppb (1 task, classification)
# Samples: 194
# Threshold: 1.000 (unbound <= 10% = PPB >= 90%)
# Split: UMAP (optimized for ADME datasets)
# Epochs: 301
# Note: Small dataset, consider using smaller batch_size

echo "Running experiments on ADME-hPPB dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset adme_hppb --split umap --epochs 301 --batch-size 32
python src/run.py --config configs/ka_gat.yaml --dataset adme_hppb --split umap --epochs 301 --batch-size 32

echo "ADME-hPPB experiments completed."
