#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# sider dataset experiments with ngram_xgb
# Dataset: sider (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/ngram_xgb.yaml --dataset sider --split random --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset sider --split scaffold --epochs 501 &
wait
