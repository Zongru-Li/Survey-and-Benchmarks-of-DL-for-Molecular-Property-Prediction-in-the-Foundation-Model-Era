#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# bbbp dataset experiments with pretrain_gnn
# Dataset: bbbp (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/pretrain_gnn.yaml --dataset bbbp --split random --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset bbbp --split scaffold --epochs 501 &
wait
