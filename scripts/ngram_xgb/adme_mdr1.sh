#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# adme_mdr1 dataset experiments with ngram_xgb
# Dataset: adme_mdr1 (ADME, regression)
# Splits: scaffold, butina, umap, time
# Epochs: 501

python src/run.py --config configs/ngram_xgb.yaml --dataset adme_mdr1 --split scaffold --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_mdr1 --split butina --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_mdr1 --split time --epochs 501 &
wait
