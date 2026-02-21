#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# HIV dataset experiments
# Dataset: hiv (1 task, classification)
# Split: scaffold (MoleculeNet standard)
# Epochs: 501 (default from config)

echo "Running experiments on HIV dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset hiv --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset hiv --split umap --epochs 501

python src/run.py --config configs/ka_gnn.yaml --dataset hiv --split scaffold --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset hiv --split scaffold --epochs 501

echo "HIV experiments completed."
