#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# BBBP dataset experiments
# Dataset: bbbp (1 task, classification)
# Split: scaffold (MoleculeNet standard)
# Epochs: 501 (default from config)

echo "Running experiments on BBBP dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset bbbp --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset bbbp --split umap --epochs 501
python src/run.py --config configs/ka_gnn.yaml --dataset bbbp --split scaffold --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset bbbp --split scaffold --epochs 501

echo "BBBP experiments completed."
