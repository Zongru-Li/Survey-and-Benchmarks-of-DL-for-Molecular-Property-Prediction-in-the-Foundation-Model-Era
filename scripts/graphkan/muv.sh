#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# muv dataset experiments with graphkan
# Dataset: muv (MoleculeNet, classification, 17 tasks)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset muv --split random --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset muv --split scaffold --epochs 501 &
wait
