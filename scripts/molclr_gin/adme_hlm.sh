#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

# adme_hlm dataset experiments with molclr_gin
# Dataset: adme_hlm (ADME, regression)
# Splits: scaffold, butina, umap, time
# Epochs: 501

# python src/run.py --config configs/molclr_gin.yaml --dataset adme_hlm --split scaffold --epochs 501 &
# python src/run.py --config configs/molclr_gin.yaml --dataset adme_hlm --split butina --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_hlm --split time --epochs 501 &
wait
