#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# tox21 dataset experiments with molclr_gcn
# Dataset: tox21 (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/molclr_gcn.yaml --dataset tox21 --split random --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset tox21 --split scaffold --epochs 501 &
wait
