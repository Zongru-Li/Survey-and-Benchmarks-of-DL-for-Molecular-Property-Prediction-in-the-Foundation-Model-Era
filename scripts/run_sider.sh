#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# SIDER dataset experiments
# Dataset: sider (27 tasks, classification)
# Split: scaffold (MoleculeNet standard)
# Epochs: 501 (default from config)

echo "Running experiments on SIDER dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset sider --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset sider --split umap --epochs 501
python src/run.py --config configs/ka_gnn.yaml --dataset sider --split scaffold --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset sider --split scaffold --epochs 501

echo "SIDER experiments completed."
