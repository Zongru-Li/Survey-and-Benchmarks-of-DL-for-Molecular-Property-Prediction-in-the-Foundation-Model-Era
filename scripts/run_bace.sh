#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# BACE dataset experiments
# Dataset: bace (1 task, classification)
# Split: scaffold (default for MoleculeNet)
# Epochs: 501

echo "Running experiments on BACE dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset bace --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset bace --split umap --epochs 501
python src/run.py --config configs/ka_gnn.yaml --dataset bace --split scaffold --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset bace --split scaffold --epochs 501

echo "BACE experiments completed."
