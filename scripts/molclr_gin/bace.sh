#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# bace dataset experiments with molclr_gin
# Dataset: bace (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/molclr_gin.yaml --dataset bace --split random --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset bace --split scaffold --epochs 501 &
wait
