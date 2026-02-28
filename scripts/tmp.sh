#!/bin/bash
set -e
cd "$(dirname "$0")/.."

python src/run.py --config configs/ngram_xgb.yaml --dataset adme_rppb --split time --epochs 501 &

wait
