#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# hiv dataset experiments with graphkan
# Dataset: hiv (MoleculeNet, classification)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset hiv --split random --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset hiv --split scaffold --epochs 501 &
wait
