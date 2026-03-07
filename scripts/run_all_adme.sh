#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# ka_gnn
bash scripts/ka_gnn/adme_hlm.sh
bash scripts/ka_gnn/adme_rlm.sh
bash scripts/ka_gnn/adme_mdr1.sh
bash scripts/ka_gnn/adme_sol.sh
bash scripts/ka_gnn/adme_hppb.sh
bash scripts/ka_gnn/adme_rppb.sh


# ka_gat
bash scripts/ka_gat/adme_hlm.sh
bash scripts/ka_gat/adme_rlm.sh
bash scripts/ka_gat/adme_mdr1.sh
bash scripts/ka_gat/adme_sol.sh
bash scripts/ka_gat/adme_hppb.sh
bash scripts/ka_gat/adme_rppb.sh


# attentivefp
bash scripts/attentivefp/adme_hlm.sh
bash scripts/attentivefp/adme_rlm.sh
bash scripts/attentivefp/adme_mdr1.sh
bash scripts/attentivefp/adme_sol.sh
bash scripts/attentivefp/adme_hppb.sh
bash scripts/attentivefp/adme_rppb.sh


# pretrain_gnn
bash scripts/pretrain_gnn/adme_hlm.sh
bash scripts/pretrain_gnn/adme_rlm.sh
bash scripts/pretrain_gnn/adme_mdr1.sh
bash scripts/pretrain_gnn/adme_sol.sh
bash scripts/pretrain_gnn/adme_hppb.sh
bash scripts/pretrain_gnn/adme_rppb.sh


# graphmvp
bash scripts/graphmvp/adme_hlm.sh
bash scripts/graphmvp/adme_rlm.sh
bash scripts/graphmvp/adme_mdr1.sh
bash scripts/graphmvp/adme_sol.sh
bash scripts/graphmvp/adme_hppb.sh
bash scripts/graphmvp/adme_rppb.sh


# molclr_gcn
bash scripts/molclr_gcn/adme_hlm.sh
bash scripts/molclr_gcn/adme_rlm.sh
bash scripts/molclr_gcn/adme_mdr1.sh
bash scripts/molclr_gcn/adme_sol.sh
bash scripts/molclr_gcn/adme_hppb.sh
bash scripts/molclr_gcn/adme_rppb.sh


# molclr_gin
bash scripts/molclr_gin/adme_hlm.sh
bash scripts/molclr_gin/adme_rlm.sh
bash scripts/molclr_gin/adme_mdr1.sh
bash scripts/molclr_gin/adme_sol.sh
bash scripts/molclr_gin/adme_hppb.sh
bash scripts/molclr_gin/adme_rppb.sh


# mol_gdl
bash scripts/mol_gdl/adme_hlm.sh
bash scripts/mol_gdl/adme_rlm.sh
bash scripts/mol_gdl/adme_mdr1.sh
bash scripts/mol_gdl/adme_sol.sh
bash scripts/mol_gdl/adme_hppb.sh
bash scripts/mol_gdl/adme_rppb.sh


# ngram_xgb
bash scripts/ngram_xgb/adme_hlm.sh
bash scripts/ngram_xgb/adme_rlm.sh
bash scripts/ngram_xgb/adme_mdr1.sh
bash scripts/ngram_xgb/adme_sol.sh
bash scripts/ngram_xgb/adme_hppb.sh
bash scripts/ngram_xgb/adme_rppb.sh


# ngram_rf
bash scripts/ngram_rf/adme_hlm.sh
bash scripts/ngram_rf/adme_rlm.sh
bash scripts/ngram_rf/adme_mdr1.sh
bash scripts/ngram_rf/adme_sol.sh
bash scripts/ngram_rf/adme_hppb.sh
bash scripts/ngram_rf/adme_rppb.sh


# dmpnn
bash scripts/dmpnn/adme_hlm.sh
bash scripts/dmpnn/adme_rlm.sh
bash scripts/dmpnn/adme_mdr1.sh
bash scripts/dmpnn/adme_sol.sh
bash scripts/dmpnn/adme_hppb.sh
bash scripts/dmpnn/adme_rppb.sh


# graphkan
bash scripts/graphkan/adme_hlm.sh
bash scripts/graphkan/adme_rlm.sh
bash scripts/graphkan/adme_mdr1.sh
bash scripts/graphkan/adme_sol.sh
bash scripts/graphkan/adme_hppb.sh
bash scripts/graphkan/adme_rppb.sh


# cd-mvgnn
bash scripts/cd_mvgnn/adme_hlm.sh
bash scripts/cd_mvgnn/adme_rlm.sh
bash scripts/cd_mvgnn/adme_mdr1.sh
bash scripts/cd_mvgnn/adme_sol.sh
bash scripts/cd_mvgnn/adme_hppb.sh
bash scripts/cd_mvgnn/adme_rppb.sh

# grover
bash scripts/grover/adme_hlm.sh
bash scripts/grover/adme_rlm.sh
bash scripts/grover/adme_mdr1.sh
bash scripts/grover/adme_sol.sh
bash scripts/grover/adme_hppb.sh
bash scripts/grover/adme_rppb.sh

# gcn
bash scripts/gcn/adme_hlm.sh
bash scripts/gcn/adme_rlm.sh
bash scripts/gcn/adme_mdr1.sh
bash scripts/gcn/adme_sol.sh
bash scripts/gcn/adme_hppb.sh
bash scripts/gcn/adme_rppb.sh

# gin
bash scripts/gin/adme_hlm.sh
bash scripts/gin/adme_rlm.sh
bash scripts/gin/adme_mdr1.sh
bash scripts/gin/adme_sol.sh
bash scripts/gin/adme_hppb.sh
bash scripts/gin/adme_rppb.sh

echo "All experiments completed."
