#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# # ka_gnn
# bash scripts/ka_gnn/adme_hlm.sh
# bash scripts/ka_gnn/adme_rlm.sh
# bash scripts/ka_gnn/adme_mdr1.sh
# bash scripts/ka_gnn/adme_sol.sh
# bash scripts/ka_gnn/adme_hppb.sh
bash scripts/ka_gnn/adme_rppb.sh


# # ka_gat
# bash scripts/ka_gat/adme_hlm.sh
# bash scripts/ka_gat/adme_rlm.sh
# bash scripts/ka_gat/adme_mdr1.sh
# bash scripts/ka_gat/adme_sol.sh
bash scripts/ka_gat/adme_hppb.sh
bash scripts/ka_gat/adme_rppb.sh


# # ka_gnn_two
# bash scripts/ka_gnn_two/adme_hlm.sh
# bash scripts/ka_gnn_two/adme_rlm.sh
# bash scripts/ka_gnn_two/adme_mdr1.sh
# bash scripts/ka_gnn_two/adme_sol.sh
# bash scripts/ka_gnn_two/adme_hppb.sh
# bash scripts/ka_gnn_two/adme_rppb.sh

# # kan_sage
# bash scripts/kan_sage/adme_hlm.sh
# bash scripts/kan_sage/adme_rlm.sh
# bash scripts/kan_sage/adme_mdr1.sh
# bash scripts/kan_sage/adme_sol.sh
# bash scripts/kan_sage/adme_hppb.sh
# bash scripts/kan_sage/adme_rppb.sh

# # kan_gat
# bash scripts/kan_gat/adme_hlm.sh
# bash scripts/kan_gat/adme_rlm.sh
# bash scripts/kan_gat/adme_mdr1.sh
# bash scripts/kan_gat/adme_sol.sh
# bash scripts/kan_gat/adme_hppb.sh
# bash scripts/kan_gat/adme_rppb.sh

# # mlp_sage
# bash scripts/mlp_sage/adme_hlm.sh
# bash scripts/mlp_sage/adme_rlm.sh
# bash scripts/mlp_sage/adme_mdr1.sh
# bash scripts/mlp_sage/adme_sol.sh
# bash scripts/mlp_sage/adme_hppb.sh
# bash scripts/mlp_sage/adme_rppb.sh

# # mlp_gat
# bash scripts/mlp_gat/adme_hlm.sh
# bash scripts/mlp_gat/adme_rlm.sh
# bash scripts/mlp_gat/adme_mdr1.sh
# bash scripts/mlp_gat/adme_sol.sh
# bash scripts/mlp_gat/adme_hppb.sh
# bash scripts/mlp_gat/adme_rppb.sh

# # po_gat
# bash scripts/po_gat/adme_hlm.sh
# bash scripts/po_gat/adme_rlm.sh
# bash scripts/po_gat/adme_mdr1.sh
# bash scripts/po_gat/adme_sol.sh
# bash scripts/po_gat/adme_hppb.sh
# bash scripts/po_gat/adme_rppb.sh


# # attentivefp
# bash scripts/attentivefp/adme_hlm.sh
# bash scripts/attentivefp/adme_rlm.sh
# bash scripts/attentivefp/adme_mdr1.sh
# bash scripts/attentivefp/adme_sol.sh
bash scripts/attentivefp/adme_hppb.sh
bash scripts/attentivefp/adme_rppb.sh


# pretrain_gnn
# bash scripts/pretrain_gnn/adme_hlm.sh
# bash scripts/pretrain_gnn/adme_rlm.sh
# bash scripts/pretrain_gnn/adme_mdr1.sh
# bash scripts/pretrain_gnn/adme_sol.sh
bash scripts/pretrain_gnn/adme_hppb.sh
bash scripts/pretrain_gnn/adme_rppb.sh


# graphmvp
# bash scripts/graphmvp/adme_hlm.sh
# bash scripts/graphmvp/adme_rlm.sh
# bash scripts/graphmvp/adme_mdr1.sh
# bash scripts/graphmvp/adme_sol.sh
bash scripts/graphmvp/adme_hppb.sh
bash scripts/graphmvp/adme_rppb.sh


# molclr_gcn
# bash scripts/molclr_gcn/adme_hlm.sh
# bash scripts/molclr_gcn/adme_rlm.sh
# bash scripts/molclr_gcn/adme_mdr1.sh
# bash scripts/molclr_gcn/adme_sol.sh
bash scripts/molclr_gcn/adme_hppb.sh
bash scripts/molclr_gcn/adme_rppb.sh


# molclr_gin
# bash scripts/molclr_gin/adme_hlm.sh
# bash scripts/molclr_gin/adme_rlm.sh
# bash scripts/molclr_gin/adme_mdr1.sh
# bash scripts/molclr_gin/adme_sol.sh
bash scripts/molclr_gin/adme_hppb.sh
bash scripts/molclr_gin/adme_rppb.sh


# mol_gdl
# bash scripts/mol_gdl/adme_hlm.sh
# bash scripts/mol_gdl/adme_rlm.sh
# bash scripts/mol_gdl/adme_mdr1.sh
# bash scripts/mol_gdl/adme_sol.sh
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
# bash scripts/dmpnn/adme_hlm.sh
# bash scripts/dmpnn/adme_rlm.sh
# bash scripts/dmpnn/adme_mdr1.sh
# bash scripts/dmpnn/adme_sol.sh
# bash scripts/dmpnn/adme_hppb.sh
# bash scripts/dmpnn/adme_rppb.sh


# graphkan
bash scripts/graphkan/adme_hlm.sh
bash scripts/graphkan/adme_rlm.sh
bash scripts/graphkan/adme_mdr1.sh
bash scripts/graphkan/adme_sol.sh
bash scripts/graphkan/adme_hppb.sh
bash scripts/graphkan/adme_rppb.sh


echo "All experiments completed."
