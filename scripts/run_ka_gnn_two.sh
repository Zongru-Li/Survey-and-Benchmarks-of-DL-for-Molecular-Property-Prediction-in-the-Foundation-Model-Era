#!/bin/bash
# KA-GNN-Two experiment script

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/ka_gnn_two.yaml"
DATASETS=("bace" "bbbp" "clintox" "sider" "tox21" "hiv" "muv")

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Running ka_gnn_two on $dataset..."
    echo "=========================================="
    
    python src/run.py \
        --config "$CONFIG" \
        --dataset "$dataset"
    
    echo ""
done

echo "All experiments completed!"
