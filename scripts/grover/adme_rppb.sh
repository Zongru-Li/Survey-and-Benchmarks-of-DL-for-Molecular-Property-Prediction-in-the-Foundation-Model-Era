#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

python src/run.py --config configs/grover.yaml --dataset adme_rppb --split scaffold --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_rppb --split butina --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_rppb --split umap --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_rppb --split time --epochs 501 &
wait
