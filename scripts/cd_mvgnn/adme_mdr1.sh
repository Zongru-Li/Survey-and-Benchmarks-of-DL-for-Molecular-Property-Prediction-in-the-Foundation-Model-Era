#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_mdr1 --split scaffold --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_mdr1 --split butina --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_mdr1 --split time --epochs 501 &
wait
