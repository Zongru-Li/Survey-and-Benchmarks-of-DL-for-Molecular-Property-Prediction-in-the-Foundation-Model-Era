#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# attentivefp
bash scripts/attentivefp/adme_hlm.sh
bash scripts/attentivefp/adme_rlm.sh
bash scripts/attentivefp/adme_mdr1.sh
bash scripts/attentivefp/adme_sol.sh
bash scripts/attentivefp/adme_hppb.sh
bash scripts/attentivefp/adme_rppb.sh


echo "All experiments completed."
