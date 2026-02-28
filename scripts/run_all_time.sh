#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# Run all models on ADME datasets with time split
# ADME datasets: adme_hlm, adme_rlm, adme_mdr1, adme_sol, adme_hppb, adme_rppb

# cd_mvgnn
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# grover
python src/run.py --config configs/grover.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/grover.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# gcn
python src/run.py --config configs/gcn.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/gcn.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# gin
python src/run.py --config configs/gin.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/gin.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/gin.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/gin.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/gin.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/gin.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# ka_gnn
python src/run.py --config configs/ka_gnn.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# ka_gat
python src/run.py --config configs/ka_gat.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# # attentivefp
# python src/run.py --config configs/attentivefp.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/attentivefp.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/attentivefp.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/attentivefp.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/attentivefp.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/attentivefp.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

# pretrain_gnn
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# graphmvp
python src/run.py --config configs/graphmvp.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# molclr_gcn
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# molclr_gin
python src/run.py --config configs/molclr_gin.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# mol_gdl
python src/run.py --config configs/mol_gdl.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# ngram_xgb
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# ngram_rf
python src/run.py --config configs/ngram_rf.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/ngram_rf.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/ngram_rf.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/ngram_rf.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/ngram_rf.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/ngram_rf.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# # dmpnn
# python src/run.py --config configs/dmpnn.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

# graphkan
python src/run.py --config configs/graphkan.yaml --dataset adme_hlm --split time --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rlm --split time --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_mdr1 --split time --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_sol --split time --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_hppb --split time --epochs 501 &
python src/run.py --config configs/graphkan.yaml --dataset adme_rppb --split time --epochs 501 &
wait

# # ka_gnn_two
# python src/run.py --config configs/ka_gnn_two.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/ka_gnn_two.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/ka_gnn_two.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/ka_gnn_two.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/ka_gnn_two.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/ka_gnn_two.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

# # kan_sage
# python src/run.py --config configs/kan_sage.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/kan_sage.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/kan_sage.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/kan_sage.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/kan_sage.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/kan_sage.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

# # kan_gat
# python src/run.py --config configs/kan_gat.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/kan_gat.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/kan_gat.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/kan_gat.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/kan_gat.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/kan_gat.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

# # mlp_sage
# python src/run.py --config configs/mlp_sage.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/mlp_sage.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/mlp_sage.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/mlp_sage.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/mlp_sage.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/mlp_sage.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

# # mlp_gat
# python src/run.py --config configs/mlp_gat.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/mlp_gat.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/mlp_gat.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/mlp_gat.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/mlp_gat.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/mlp_gat.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

# # po_gat
# python src/run.py --config configs/po_gat.yaml --dataset adme_hlm --split time --epochs 501 &
# python src/run.py --config configs/po_gat.yaml --dataset adme_rlm --split time --epochs 501 &
# python src/run.py --config configs/po_gat.yaml --dataset adme_mdr1 --split time --epochs 501 &
# python src/run.py --config configs/po_gat.yaml --dataset adme_sol --split time --epochs 501 &
# python src/run.py --config configs/po_gat.yaml --dataset adme_hppb --split time --epochs 501 &
# python src/run.py --config configs/po_gat.yaml --dataset adme_rppb --split time --epochs 501 &
# wait

echo "All time split experiments completed."
