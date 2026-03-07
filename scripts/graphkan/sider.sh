#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# sider dataset experiments with graphkan
# Dataset: sider (MoleculeNet, classification, 27 tasks)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/graphkan.yaml --dataset sider --split random --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset sider --split scaffold --epochs 501 &
wait
