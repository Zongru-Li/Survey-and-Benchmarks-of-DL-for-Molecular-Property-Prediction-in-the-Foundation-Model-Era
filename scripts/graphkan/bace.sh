#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# bace dataset experiments with graphkan
# Dataset: bace (MoleculeNet, classification)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset bace --split random --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset bace --split scaffold --epochs 501 &
wait
