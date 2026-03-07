#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

python src/run.py --config configs/gcn.yaml --dataset adme_rlm --split scaffold --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_rlm --split butina --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_rlm --split time --epochs 501 &
wait
