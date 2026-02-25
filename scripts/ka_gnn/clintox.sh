#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# clintox dataset experiments with ka_gnn
# Dataset: clintox (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/ka_gnn.yaml --dataset clintox --split random --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset clintox --split scaffold --epochs 501 &
wait
