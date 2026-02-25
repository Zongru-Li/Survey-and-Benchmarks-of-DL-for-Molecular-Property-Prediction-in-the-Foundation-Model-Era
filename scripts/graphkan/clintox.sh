#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# clintox dataset experiments with graphkan
# Dataset: clintox (MoleculeNet, classification, 2 tasks)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset clintox --split random --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset clintox --split scaffold --epochs 501 &
wait
