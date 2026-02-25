#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# sider dataset experiments with po_gat
# Dataset: sider (MoleculeNet)
# Splits: random, scaffold
# Epochs: 501

python src/run.py --config configs/po_gat.yaml --dataset sider --split random --epochs 501 &
python src/run.py --config configs/po_gat.yaml --dataset sider --split scaffold --epochs 501 &
wait
