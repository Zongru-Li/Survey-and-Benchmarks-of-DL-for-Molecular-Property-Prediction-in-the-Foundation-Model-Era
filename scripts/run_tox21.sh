#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# Tox21 dataset experiments
# Dataset: tox21 (12 tasks, classification)
# Split: scaffold (MoleculeNet standard)
# Epochs: 501 (default from config)

echo "Running experiments on Tox21 dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset tox21 --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset tox21 --split umap --epochs 501
python src/run.py --config configs/ka_gnn.yaml --dataset tox21 --split scaffold --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset tox21 --split scaffold --epochs 501

echo "Tox21 experiments completed."
