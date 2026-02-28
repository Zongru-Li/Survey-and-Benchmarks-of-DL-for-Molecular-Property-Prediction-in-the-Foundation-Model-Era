#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --split scaffold --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --split butina --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --split time --epochs 501 &
wait
