#!/bin/bash
set -e

cd "$(dirname "$0")/.."

# ClinTox dataset experiments
# Dataset: clintox (2 tasks, classification)
# Split: scaffold (MoleculeNet standard)
# Epochs: 501 (default from config)

echo "Running experiments on ClinTox dataset..."

python src/run.py --config configs/ka_gnn.yaml --dataset clintox --split umap --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset clintox --split umap --epochs 501

python src/run.py --config configs/ka_gnn.yaml --dataset clintox --split scaffold --epochs 501
python src/run.py --config configs/ka_gat.yaml --dataset clintox --split scaffold --epochs 501

echo "ClinTox experiments completed."
