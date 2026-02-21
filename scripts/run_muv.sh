#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# MUV dataset experiments
# Dataset: muv (17 tasks, classification)
# Split: scaffold (MoleculeNet standard)
# Epochs: 501 (default from config)

echo "Running experiments on MUV dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset muv --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset muv --split umap --epochs 501
python src/run.py --config configs/ka_gnn.yaml --dataset muv --split scaffold --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset muv --split scaffold --epochs 501

echo "MUV experiments completed."
