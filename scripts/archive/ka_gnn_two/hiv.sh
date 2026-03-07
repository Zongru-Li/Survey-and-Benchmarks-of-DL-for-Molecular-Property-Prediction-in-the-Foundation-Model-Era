#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# hiv dataset experiments with ka_gnn_two
# Dataset: hiv (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/ka_gnn_two.yaml --dataset hiv --split random --epochs 501 &
python src/run.py --config configs/ka_gnn_two.yaml --dataset hiv --split scaffold --epochs 501 &
wait
