#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# tox21 dataset experiments with kan_gat
# Dataset: tox21 (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/kan_gat.yaml --dataset tox21 --split random --epochs 501 &
python src/run.py --config configs/kan_gat.yaml --dataset tox21 --split scaffold --epochs 501 &
wait
