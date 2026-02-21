#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ADME-RLM dataset experiments
# Dataset: adme_rlm (1 task, classification)
# Samples: 3,054
# Threshold: 1.699 (CLint < 50 mL/min/kg = high stability)
# Split: UMAP (optimized for ADME datasets)
# Epochs: 301

echo "Running experiments on ADME-RLM dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset adme_rlm --split umap --epochs 301
python src/run.py --config configs/ka_gat.yaml --dataset adme_rlm --split umap --epochs 301

echo "ADME-RLM experiments completed."
