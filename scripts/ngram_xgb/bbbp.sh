#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# bbbp dataset experiments with ngram_xgb
# Dataset: bbbp (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/ngram_xgb.yaml --dataset bbbp --split random --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset bbbp --split scaffold --epochs 501 &
wait
