#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# bbbp dataset experiments with molclr_gin
# Dataset: bbbp (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/molclr_gin.yaml --dataset bbbp --split random --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset bbbp --split scaffold --epochs 501 &
wait
