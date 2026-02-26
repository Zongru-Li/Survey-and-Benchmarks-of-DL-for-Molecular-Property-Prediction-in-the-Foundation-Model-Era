#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# ka_gnn
bash scripts/ka_gnn/bace.sh
bash scripts/ka_gnn/bbbp.sh
bash scripts/ka_gnn/clintox.sh
bash scripts/ka_gnn/hiv.sh
bash scripts/ka_gnn/muv.sh
bash scripts/ka_gnn/sider.sh
bash scripts/ka_gnn/tox21.sh


# ka_gat
bash scripts/ka_gat/bace.sh
bash scripts/ka_gat/bbbp.sh
bash scripts/ka_gat/clintox.sh
bash scripts/ka_gat/hiv.sh
bash scripts/ka_gat/muv.sh
bash scripts/ka_gat/sider.sh
bash scripts/ka_gat/tox21.sh


# ka_gnn_two
bash scripts/ka_gnn_two/bace.sh
bash scripts/ka_gnn_two/bbbp.sh
bash scripts/ka_gnn_two/clintox.sh
bash scripts/ka_gnn_two/hiv.sh
bash scripts/ka_gnn_two/muv.sh
bash scripts/ka_gnn_two/sider.sh
bash scripts/ka_gnn_two/tox21.sh


# kan_sage
bash scripts/kan_sage/bace.sh
bash scripts/kan_sage/bbbp.sh
bash scripts/kan_sage/clintox.sh
bash scripts/kan_sage/hiv.sh
bash scripts/kan_sage/muv.sh
bash scripts/kan_sage/sider.sh
bash scripts/kan_sage/tox21.sh


# kan_gat
bash scripts/kan_gat/bace.sh
bash scripts/kan_gat/bbbp.sh
bash scripts/kan_gat/clintox.sh
bash scripts/kan_gat/hiv.sh
bash scripts/kan_gat/muv.sh
bash scripts/kan_gat/sider.sh
bash scripts/kan_gat/tox21.sh


# mlp_sage
bash scripts/mlp_sage/bace.sh
bash scripts/mlp_sage/bbbp.sh
bash scripts/mlp_sage/clintox.sh
bash scripts/mlp_sage/hiv.sh
bash scripts/mlp_sage/muv.sh
bash scripts/mlp_sage/sider.sh
bash scripts/mlp_sage/tox21.sh

# mlp_gat
bash scripts/mlp_gat/bace.sh
bash scripts/mlp_gat/bbbp.sh
bash scripts/mlp_gat/clintox.sh
bash scripts/mlp_gat/hiv.sh
bash scripts/mlp_gat/muv.sh
bash scripts/mlp_gat/sider.sh
bash scripts/mlp_gat/tox21.sh

# po_gat
bash scripts/po_gat/bace.sh
bash scripts/po_gat/bbbp.sh
bash scripts/po_gat/clintox.sh
bash scripts/po_gat/hiv.sh
bash scripts/po_gat/muv.sh
bash scripts/po_gat/sider.sh
bash scripts/po_gat/tox21.sh


# attentivefp
bash scripts/attentivefp/bace.sh
bash scripts/attentivefp/bbbp.sh
bash scripts/attentivefp/clintox.sh
bash scripts/attentivefp/hiv.sh
bash scripts/attentivefp/muv.sh
bash scripts/attentivefp/sider.sh
bash scripts/attentivefp/tox21.sh


# pretrain_gnn
bash scripts/pretrain_gnn/bace.sh
bash scripts/pretrain_gnn/bbbp.sh
bash scripts/pretrain_gnn/clintox.sh
bash scripts/pretrain_gnn/hiv.sh
bash scripts/pretrain_gnn/muv.sh
bash scripts/pretrain_gnn/sider.sh
bash scripts/pretrain_gnn/tox21.sh


# graphmvp
bash scripts/graphmvp/bace.sh
bash scripts/graphmvp/bbbp.sh
bash scripts/graphmvp/clintox.sh
bash scripts/graphmvp/hiv.sh
bash scripts/graphmvp/muv.sh
bash scripts/graphmvp/sider.sh
bash scripts/graphmvp/tox21.sh


# molclr_gcn
bash scripts/molclr_gcn/bace.sh
bash scripts/molclr_gcn/bbbp.sh
bash scripts/molclr_gcn/clintox.sh
bash scripts/molclr_gcn/hiv.sh
bash scripts/molclr_gcn/muv.sh
bash scripts/molclr_gcn/sider.sh
bash scripts/molclr_gcn/tox21.sh

# molclr_gin
bash scripts/molclr_gin/bace.sh
bash scripts/molclr_gin/bbbp.sh
bash scripts/molclr_gin/clintox.sh
bash scripts/molclr_gin/hiv.sh
bash scripts/molclr_gin/muv.sh
bash scripts/molclr_gin/sider.sh
bash scripts/molclr_gin/tox21.sh


# mol_gdl
bash scripts/mol_gdl/bace.sh
bash scripts/mol_gdl/bbbp.sh
bash scripts/mol_gdl/clintox.sh
bash scripts/mol_gdl/hiv.sh
bash scripts/mol_gdl/muv.sh
bash scripts/mol_gdl/sider.sh
bash scripts/mol_gdl/tox21.sh


# ngram_xgb
bash scripts/ngram_xgb/bace.sh
bash scripts/ngram_xgb/bbbp.sh
bash scripts/ngram_xgb/clintox.sh
bash scripts/ngram_xgb/hiv.sh
bash scripts/ngram_xgb/muv.sh
bash scripts/ngram_xgb/sider.sh
bash scripts/ngram_xgb/tox21.sh


# ngram_rf
bash scripts/ngram_rf/bace.sh
bash scripts/ngram_rf/bbbp.sh
bash scripts/ngram_rf/clintox.sh
bash scripts/ngram_rf/hiv.sh
bash scripts/ngram_rf/muv.sh
bash scripts/ngram_rf/sider.sh
bash scripts/ngram_rf/tox21.sh


# dmpnn
bash scripts/dmpnn/bace.sh
bash scripts/dmpnn/bbbp.sh
bash scripts/dmpnn/clintox.sh
bash scripts/dmpnn/hiv.sh
bash scripts/dmpnn/muv.sh
bash scripts/dmpnn/sider.sh
bash scripts/dmpnn/tox21.sh


# graphkan
bash scripts/graphkan/bace.sh
bash scripts/graphkan/bbbp.sh
bash scripts/graphkan/clintox.sh
bash scripts/graphkan/hiv.sh
bash scripts/graphkan/muv.sh
bash scripts/graphkan/sider.sh
bash scripts/graphkan/tox21.sh


echo "All experiments completed."
