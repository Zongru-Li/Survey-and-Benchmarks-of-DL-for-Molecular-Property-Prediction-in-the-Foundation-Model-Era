#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# bbbp dataset experiments with graphkan
# Dataset: bbbp (MoleculeNet, classification)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset bbbp --split random --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset bbbp --split scaffold --epochs 501 &
wait
