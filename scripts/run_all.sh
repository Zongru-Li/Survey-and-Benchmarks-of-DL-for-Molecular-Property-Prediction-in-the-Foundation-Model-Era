#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Running all experiments..."

# MoleculeNet datasets
bash scripts/run_bace.sh &
bash scripts/run_bbbp.sh &
bash scripts/run_clintox.sh &
bash scripts/run_sider.sh &
bash scripts/run_tox21.sh &
bash scripts/run_hiv.sh &
bash scripts/run_muv.sh &
wait

# ADME datasets
bash scripts/run_adme_hlm.sh &
bash scripts/run_adme_rlm.sh &
bash scripts/run_adme_mdr1.sh &
bash scripts/run_adme_sol.sh &
bash scripts/run_adme_hppb.sh &
bash scripts/run_adme_rppb.sh &
wait

echo "All experiments completed."
