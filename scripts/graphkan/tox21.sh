#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# tox21 dataset experiments with graphkan
# Dataset: tox21 (MoleculeNet, classification, 12 tasks)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset tox21 --split random --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset tox21 --split scaffold --epochs 501 &
wait
