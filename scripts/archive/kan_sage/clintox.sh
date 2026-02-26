#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# clintox dataset experiments with kan_sage
# Dataset: clintox (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/kan_sage.yaml --dataset clintox --split random --epochs 501 &
python src/run.py --config configs/kan_sage.yaml --dataset clintox --split scaffold --epochs 501 &
wait
