#!/bin/bash
# KAN-SAGE experiment script

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/kan_sage.yaml"
DATASETS=("bace" "bbbp" "clintox" "sider" "tox21" "hiv" "muv")

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Running kan_sage on $dataset..."
    echo "=========================================="
    
    python src/run.py \
        --config "$CONFIG" \
        --dataset "$dataset"
    
    echo ""
done

echo "All experiments completed!"
