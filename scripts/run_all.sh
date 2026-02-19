#!/bin/bash
# Run all model experiments

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SCRIPTS=(
    "scripts/run_ka_gnn.sh"
    "scripts/run_ka_gnn_two.sh"
    "scripts/run_mlp_sage.sh"
    "scripts/run_kan_sage.sh"
    "scripts/run_ka_gat.sh"
    "scripts/run_kan_gat.sh"
    "scripts/run_mlp_gat.sh"
    "scripts/run_po_gat.sh"
)

for script in "${SCRIPTS[@]}"; do
    echo "=========================================="
    echo "Running $script..."
    echo "=========================================="
    bash "$script"
done

echo "All experiments completed!"
