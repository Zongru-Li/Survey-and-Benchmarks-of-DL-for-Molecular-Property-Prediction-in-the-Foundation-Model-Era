#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# muv dataset experiments with pretrain_gnn
# Dataset: muv (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/pretrain_gnn.yaml --dataset muv --split random --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset muv --split scaffold --epochs 501 &
wait
