#!/bin/bash
# PO-GAT experiment script

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/po_gat.yaml"
DATASETS=("bace" "bbbp" "clintox" "sider" "tox21" "hiv" "muv")

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Running po_gat on $dataset..."
    echo "=========================================="
    
    python src/run.py \
        --config "$CONFIG" \
        --dataset "$dataset"
    
    echo ""
done

echo "All experiments completed!"
