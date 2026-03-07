#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

python src/run.py --config configs/gcn.yaml --dataset adme_hppb --split scaffold --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_hppb --split butina --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_hppb --split time --epochs 501 &
wait
