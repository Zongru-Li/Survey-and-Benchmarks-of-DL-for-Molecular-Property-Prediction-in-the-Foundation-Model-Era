#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# clintox dataset experiments with attentivefp
# Dataset: clintox (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/attentivefp.yaml --dataset clintox --split random --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset clintox --split scaffold --epochs 501 &
wait
