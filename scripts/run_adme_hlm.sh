#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ADME-HLM dataset experiments
# Dataset: adme_hlm (1 task, classification)
# Samples: 3,087
# Threshold: 1.699 (CLint < 50 mL/min/kg = high stability)
# Split: UMAP (optimized for ADME datasets)
# Epochs: 301

echo "Running experiments on ADME-HLM dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset adme_hlm --split umap --epochs 301
python src/run.py --config configs/ka_gat.yaml --dataset adme_hlm --split umap --epochs 301

echo "ADME-HLM experiments completed."
