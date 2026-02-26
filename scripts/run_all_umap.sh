#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# UMAP split experiments for all models and datasets

# Classification datasets: bace, bbbp, clintox, hiv, muv, sider, tox21
# Regression datasets: adme_hlm, adme_rlm, adme_mdr1, adme_sol, adme_hppb, adme_rppb

# ka_gnn
python src/run.py --config configs/ka_gnn.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/ka_gnn.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# ka_gat
python src/run.py --config configs/ka_gat.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/ka_gat.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# attentivefp
python src/run.py --config configs/attentivefp.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/attentivefp.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# # ngram_rf
# python src/run.py --config configs/ngram_rf.yaml --dataset adme_hlm --split umap --epochs 501 &
# python src/run.py --config configs/ngram_rf.yaml --dataset adme_rlm --split umap --epochs 501 &
# python src/run.py --config configs/ngram_rf.yaml --dataset adme_mdr1 --split umap --epochs 501 &
# python src/run.py --config configs/ngram_rf.yaml --dataset adme_sol --split umap --epochs 501 &
# python src/run.py --config configs/ngram_rf.yaml --dataset adme_hppb --split umap --epochs 501 &
# python src/run.py --config configs/ngram_rf.yaml --dataset adme_rppb --split umap --epochs 501 &
# wait

# ngram_xgb
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/ngram_xgb.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# pretrain_gnn
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# graphmvp
python src/run.py --config configs/graphmvp.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/graphmvp.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# molclr-gcn
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# molclr-gin
python src/run.py --config configs/molclr_gin.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/molclr_gin.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# mol_gdl
python src/run.py --config configs/mol_gdl.yaml --dataset adme_hlm --split umap --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_rlm --split umap --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_mdr1 --split umap --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_sol --split umap --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_hppb --split umap --epochs 501 &
python src/run.py --config configs/mol_gdl.yaml --dataset adme_rppb --split umap --epochs 501 &
wait

# # graphkan
# python src/run.py --config configs/graphkan.yaml --dataset adme_hlm --split umap --epochs 501 &
# python src/run.py --config configs/graphkan.yaml --dataset adme_rlm --split umap --epochs 501 &
# python src/run.py --config configs/graphkan.yaml --dataset adme_mdr1 --split umap --epochs 501 &
# python src/run.py --config configs/graphkan.yaml --dataset adme_sol --split umap --epochs 501 &
# python src/run.py --config configs/graphkan.yaml --dataset adme_hppb --split umap --epochs 501 &
# python src/run.py --config configs/graphkan.yaml --dataset adme_rppb --split umap --epochs 501 &
# wait

# # dmpnn
# python src/run.py --config configs/dmpnn.yaml --dataset adme_hlm --split umap --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_rlm --split umap --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_mdr1 --split umap --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_sol --split umap --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_hppb --split umap --epochs 501 &
# python src/run.py --config configs/dmpnn.yaml --dataset adme_rppb --split umap --epochs 501 &
# wait