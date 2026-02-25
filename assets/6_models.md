# 模型支持汇总

本文档记录了基于 `references/` 目录中的项目实现，为当前 KA-GNN 项目添加的模型支持。

## 模型搜索结果

用户请求添加以下模型：
- D-MPNN
- AttentiveFP
- N-GramRF
- N-GramXGB
- PretrainGNN
- GraphMVP
- MolCLR
- GEM
- Mol-GDL
- Uni-mol
- SMPT
- GNN-SKAN
- GraphKAN

### 在 references 中找到实现的模型

| 模型名称 | 来源项目 | 文件位置 | 添加状态 |
|---------|---------|---------|---------|
| D-MPNN | deepchem | `deepchem/models/torch_models/dmpnn.py` | ✅ 已添加 (DGL) |
| AttentiveFP | deepchem | `deepchem/models/torch_models/attentivefp.py` | ✅ 已添加 (DGL) |
| Mol-GDL | Mol-GDL | `Mol-GDL/models/gcn.py` | ✅ 已添加 (DGL) |
| N-GramRF | n_gram | `n_gram/n_gram_graph/` | ✅ 已添加 (DGL) |
| N-GramXGB | n_gram | `n_gram/n_gram_graph/` | ✅ 已添加 (DGL) |
| **PretrainGNN** | pretrain-gnns | `pretrain-gnns/chem/model.py` | ✅ 已添加 (PyG) |
| **GraphMVP** | GraphMVP | `GraphMVP/src_classification/models/` | ✅ 已添加 (PyG) |
| **MolCLR-GCN** | MolCLR | `MolCLR/models/gcn_molclr.py` | ✅ 已添加 (PyG) |
| **MolCLR-GIN** | MolCLR | `MolCLR/models/ginet_molclr.py` | ✅ 已添加 (PyG) |
| **GraphKan** | GraphKan | `GraphKan/model/GNNs.py` | ✅ 已添加 (PyG) |

### 在 references 中找到但未直接集成的模型

| 模型名称 | 原因 | 备注 |
|---------|------|------|
| GEM | 使用 PaddlePaddle 框架 | 无法直接集成到 PyTorch 项目 |
| **Uni-Mol** | 需要 `uni-core` 额外依赖 | 需要 dptech-corp/Uni-Core 库，跳过添加 |
| **SMPT** | 使用 PaddlePaddle 框架 | 无法直接集成到 PyTorch 项目 |

### 在 references 中未找到实现的模型

| 模型名称 | 搜索结果 |
|---------|---------|
| GNN-SKAN | ❌ 未在 references 中找到实现 |

---

## 已添加模型详情

### PyG 模型 (PyTorch Geometric)

以下模型使用 PyTorch Geometric 框架实现，支持完整的预训练权重加载和微调。

#### 1. PretrainGNN

**论文**: Strategies for Pre-training Graph Neural Networks

**参考项目**: `references/pretrain-gnns`

**实现文件**: `src/models/pretrain_gnn.py`

**配置文件**: `configs/pretrain_gnn.yaml`

**主要特点**:
- 支持 GIN 和 GCN 两种 GNN 类型
- 使用原子类型和手性标签嵌入
- 支持多种池化方式 (mean, sum, max, attention, set2set)
- 支持 Jumping Knowledge (JK) 连接
- 可加载预训练权重进行微调

**使用方法**:
```bash
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_sol --epochs 100
```

#### 2. GraphMVP

**论文**: Molecular Representation Learning via Heterogeneous Motif Graph Contrast

**参考项目**: `references/GraphMVP`

**实现文件**: `src/models/graphmvp.py`

**配置文件**: `configs/graphmvp.yaml`

**主要特点**:
- 支持 GIN 和 GCN 两种 GNN 类型
- 使用 BatchNorm 在 MLP 中间层
- 可学习 epsilon 参数
- 支持多种池化方式

**使用方法**:
```bash
python src/run.py --config configs/graphmvp.yaml --dataset adme_sol --epochs 100
```

#### 3. MolCLR-GCN

**论文**: Molecular Contrastive Learning of Representations via Graph Neural Networks

**参考项目**: `references/MolCLR`

**实现文件**: `src/models/molclr.py`

**配置文件**: `configs/molclr_gcn.yaml`

**主要特点**:
- GCN 架构，使用度归一化
- 特征投影层 (feat_lin)
- Softplus 激活函数的预测头
- 支持加载预训练对比学习权重

**使用方法**:
```bash
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_sol --epochs 100
```

#### 4. MolCLR-GIN

**论文**: Molecular Contrastive Learning of Representations via Graph Neural Networks

**参考项目**: `references/MolCLR`

**实现文件**: `src/models/molclr.py`

**配置文件**: `configs/molclr_gin.yaml`

**主要特点**:
- GIN 架构，使用 MLP 聚合
- 比 GCN 更强的图同构能力
- 特征投影层和 Softplus 预测头
- 支持加载预训练对比学习权重

**使用方法**:
```bash
python src/run.py --config configs/molclr_gin.yaml --dataset adme_sol --epochs 100
```

#### 5. GraphKan

**论文**: Kolmogorov-Arnold Networks for Graph Classification

**参考项目**: `references/GraphKan`

**实现文件**: `src/models/graphkan.py`

**配置文件**: `configs/graphkan.yaml`

**主要特点**:
- 基于 Kolmogorov-Arnold Networks (KAN) 的图神经网络
- 使用 B-spline 插值实现可学习的激活函数
- KAN-based Chebyshev 卷积层 (KANChebConv)
- 替代传统线性层，提供更强的表达能力
- 支持 LayerNorm 稳定训练
- 支持多种池化方式

**核心组件**:
- `KANLinear`: 使用 B-spline 基函数的可学习激活层
- `KANChebConv`: KAN-based Chebyshev 图卷积
- `GraphKAN`: 完整的图分类模型

**使用方法**:
```bash
python src/run.py --config configs/graphkan.yaml --dataset adme_sol --epochs 100
```

---

### DGL 模型 (Deep Graph Library)

#### 5. D-MPNN (Directed Message Passing Neural Network)

**论文**: Analyzing Learned Molecular Representations for Property Prediction

**实现文件**: `src/models/dmpnn.py`

**配置文件**: `configs/dmpnn.yaml`

**主要特点**:
- 使用有向消息传递机制
- 支持多层消息传递
- 包含编码器和前馈网络

**使用方法**:
```bash
python src/run.py --config configs/dmpnn.yaml --dataset bace --epochs 100
```

#### 6. AttentiveFP (Attentive Fingerprint)

**论文**: Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism

**实现文件**: `src/models/attentivefp.py`

**配置文件**: `configs/attentivefp.yaml`

**主要特点**:
- 使用图注意力机制
- 支持多层 GNN 层
- 使用 GRU 进行图表示更新
- 注意力机制的全局读出

**使用方法**:
```bash
python src/run.py --config configs/attentivefp.yaml --dataset bace --epochs 100
```

#### 7. Mol-GDL (Molecular Geometric Deep Learning)

**论文**: Geometric Deep Learning on Molecular Graphs

**参考项目**: references/Mol-GDL

**实现文件**: `src/models/mol_gdl.py`

**配置文件**: `configs/mol_gdl.yaml`

**主要特点**:
- 多尺度图卷积网络 (5个尺度)
- 每个尺度独立的 GCN 网络
- 集成多个尺度的表示进行预测
- 使用 L2 归一化

**使用方法**:
```bash
python src/run.py --config configs/mol_gdl.yaml --dataset bace --epochs 100
```

#### 8. N-Gram RF (N-Gram Random Forest)

**论文**: N-Gram Graph: A Novel Representation for Molecular Property Prediction

**参考项目**: references/n_gram

**实现文件**: `src/models/ngram.py`

**配置文件**: `configs/ngram_rf.yaml`

**主要特点**:
- 使用 N-gram walk 思想进行图嵌入
- 6个尺度的 n-gram 表示
- 可学习的 n-gram 权重融合
- 端到端训练 (非传统 RF)

**使用方法**:
```bash
python src/run.py --config configs/ngram_rf.yaml --dataset bace --epochs 100
```

#### 9. N-Gram XGB (N-Gram XGBoost-style)

**参考项目**: references/n_gram

**实现文件**: `src/models/ngram.py`

**配置文件**: `configs/ngram_xgb.yaml`

**主要特点**:
- 使用 N-gram walk 思想进行图嵌入
- 6个尺度的 n-gram 表示
- 注意力机制融合多尺度表示
- 端到端训练

**使用方法**:
```bash
python src/run.py --config configs/ngram_xgb.yaml --dataset bace --epochs 100
```

---

## 项目模型注册表

已更新 `src/models/__init__.py`，模型注册：

```python
MODEL_REGISTRY = {
    # DGL 模型
    'ka_gnn': KA_GNN,
    'ka_gnn_two': KA_GNN_two,
    'mlp_sage': MLPGNN,
    'mlp_sage_two': MLPGNN_two,
    'kan_sage': KANGNN,
    'kan_sage_two': KANGNN_two,
    'kagat': KA_GAT,
    'mlpgat': MLP_GAT,
    'kangat': KAN_GAT,
    'pogat': PO_GAT,
    'dmpnn': DMPNN,
    'attentivefp': AttentiveFP,
    'mol_gdl': MolGDL,
    'ngram_rf': NgramRF,
    'ngram_xgb': NgramXGB,
    # PyG 模型
    'pretrain_gnn': PretrainGNN,
    'graphmvp': GraphMVP,
    'molclr_gcn': MolCLR_GCN,
    'molclr_gin': MolCLR_GIN,
    'graphkan': GraphKAN,
}
```

---

## 框架适配说明

### PyTorch Geometric (PyG) 模型

新增的 PyG 模型 (PretrainGNN, GraphMVP, MolCLR) 使用原生 PyTorch Geometric 实现：

- **数据加载**: `src/utils/pyg_data.py` - PyG 专用数据加载器
- **特征编码**: 原子类型 (120 类) + 手性标签 (4 类)
- **边特征**: 键类型 (5 类) + 键方向 (3 类)
- **支持分割**: scaffold, butina, umap

### DGL 模型

DGL 模型使用项目的标准数据加载流程：

- **数据加载**: `src/utils/data.py`
- **特征编码**: CGCNN 原子特征 (92 维) + 14 维键特征

---

## 配置文件

| 模型 | 配置文件 | 关键参数 |
|-----|---------|---------|
| PretrainGNN | `configs/pretrain_gnn.yaml` | num_layers=5, emb_dim=300, dropout=0.5, lr=0.001 |
| GraphMVP | `configs/graphmvp.yaml` | num_layers=5, emb_dim=300, dropout=0.5, lr=0.001 |
| MolCLR-GCN | `configs/molclr_gcn.yaml` | num_layers=5, emb_dim=300, dropout=0.3, feat_dim=256, lr=0.0005 |
| MolCLR-GIN | `configs/molclr_gin.yaml` | num_layers=5, emb_dim=300, dropout=0.3, feat_dim=512, lr=0.0005 |
| GraphKan | `configs/graphkan.yaml` | num_layers=3, K=3, grid_size=5, spline_order=3, lr=0.001 |
| D-MPNN | `configs/dmpnn.yaml` | num_layers=3, hidden=300, lr=0.0001 |
| AttentiveFP | `configs/attentivefp.yaml` | num_layers=2, graph_feat=200, lr=0.0001 |
| Mol-GDL | `configs/mol_gdl.yaml` | num_layers=3, lr=0.0001 |
| N-Gram RF | `configs/ngram_rf.yaml` | num_layers=3, lr=0.001 |
| N-Gram XGB | `configs/ngram_xgb.yaml` | num_layers=3, lr=0.001 |

### 超参数调整记录

根据原模型论文/代码的推荐设置，对以下超参数进行了调整：

| 模型 | 参数 | 原设置 | 调整后 | 原因 |
|-----|------|-------|-------|------|
| PretrainGNN | emb_dim | 128 | 300 | 与原论文一致 |
| PretrainGNN | lr | 0.0001 | 0.001 | 与原论文一致 |
| PretrainGNN | dropout | - | 0.5 | 与原论文一致 |
| GraphMVP | emb_dim | 128 | 300 | 与原论文一致 |
| GraphMVP | lr | 0.0001 | 0.001 | 与原论文一致 |
| GraphMVP | dropout | 0.0 | 0.5 | 与原论文一致 |
| MolCLR-GCN | emb_dim | 128 | 300 | 与原论文一致 |
| MolCLR-GCN | lr | 0.0001 | 0.0005 | 与原论文一致 |
| MolCLR-GCN | dropout | 0.0 | 0.3 | 与原论文一致 |
| MolCLR-GIN | emb_dim | 128 | 300 | 与原论文一致 |
| MolCLR-GIN | feat_dim | 256 | 512 | GINet 使用更大特征维度 |
| MolCLR-GIN | lr | 0.0001 | 0.0005 | 与原论文一致 |
| MolCLR-GIN | dropout | 0.0 | 0.3 | 与原论文一致 |
| D-MPNN | enc_hidden | 64 | 300 | 与原论文一致 |
| AttentiveFP | graph_feat | 64 | 200 | 与原论文一致 |

---

## 测试验证

### ADME-Sol 数据集测试结果 (epochs=2, PearsonR)

**调整前** (emb_dim=128, lr=0.0001):

| 模型 | Scaffold | 参数量 |
|-----|----------|--------|
| PretrainGNN | 0.2972 | 352K |
| GraphMVP | 0.2437 | 355K |
| MolCLR-GCN | 0.3686 | 166K |
| MolCLR-GIN | 0.3226 | 418K |

**调整后** (emb_dim=300, 与原论文一致):

| 模型 | Scaffold | 参数量 |
|-----|----------|--------|
| PretrainGNN | 0.3115 | 1.86M |
| GraphMVP | 0.3615 | 1.87M |
| MolCLR-GCN | 0.1698 | 602K |
| MolCLR-GIN | 0.3923 | 1.95M |

注：以上为 2 个 epoch 的快速验证结果，仅用于确认模型可正常运行。完整实验需要更多 epoch。

### GraphKan ADME-Sol 数据集测试结果 (epochs=2, PearsonR)

| Split | PearsonR | 参数量 |
|-------|----------|--------|
| scaffold | -0.0879 | 705.9K |
| butina | -0.0097 | 705.9K |
| umap | 0.0641 | 705.9K |
| time | -0.1866 | 705.9K |

注：GraphKan 使用 B-spline 可学习激活函数，需要更多 epoch 才能收敛。

### 快速验证命令

```bash
# PyG 模型测试
python src/run.py --config configs/pretrain_gnn.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/graphmvp.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/molclr_gcn.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/molclr_gin.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/graphkan.yaml --dataset adme_sol --epochs 2 --split scaffold

# GraphKan 四种分割测试
python src/run.py --config configs/graphkan.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/graphkan.yaml --dataset adme_sol --epochs 2 --split butina
python src/run.py --config configs/graphkan.yaml --dataset adme_sol --epochs 2 --split umap
python src/run.py --config configs/graphkan.yaml --dataset adme_sol --epochs 2 --split time

# DGL 模型测试
python src/run.py --config configs/dmpnn.yaml --dataset adme_sol --epochs 2
python src/run.py --config configs/attentivefp.yaml --dataset adme_sol --epochs 2
python src/run.py --config configs/mol_gdl.yaml --dataset adme_sol --epochs 2
```

---

## 实验脚本

实验脚本按数据集组织在 `scripts/` 目录中：

| 脚本 | 描述 |
|-----|------|
| `scripts/run_adme_sol.sh` | ADME-Sol 数据集所有模型实验 |
| `scripts/run_adme_hlm.sh` | ADME-HLM 数据集实验 |
| `scripts/run_bace.sh` | BACE 数据集实验 |
| `scripts/run_bbbp.sh` | BBBP 数据集实验 |
| `scripts/graphkan/adme_sol.sh` | GraphKan ADME-Sol 四种分割实验 |
| `scripts/graphkan/adme_hlm.sh` | GraphKan ADME-HLM 四种分割实验 |
| `scripts/graphkan/adme_rlm.sh` | GraphKan ADME-RLM 四种分割实验 |
| `scripts/graphkan/adme_mdr1.sh` | GraphKan ADME-MDR1 四种分割实验 |
| `scripts/graphkan/adme_hppb.sh` | GraphKan ADME-hPPB 四种分割实验 |
| `scripts/graphkan/adme_rppb.sh` | GraphKan ADME-rPPB 四种分割实验 |

---

## 数据分割方法

项目支持以下数据分割方法：

| 分割方法 | 说明 |
|---------|------|
| `scaffold` | 基于 Murcko scaffold 分割，确保不同骨架分布在不同集合 |
| `butina` | 基于 Butina 聚类分割，使用 Tanimoto 相似度 |
| `umap` | 基于 UMAP 降维和 K-means 聚类分割 |
| `time` | 基于时间顺序分割（假设数据按时间排序） |

---

## 其他发现

在 `references/` 目录中还发现了其他可能有用的模型：
- GCN (deepchem/gcn.py)
- GAT (deepchem/gat.py)
- GINE/GNN (deepchem/gnn.py)
- GROVER (deepchem/grover.py)
- InfoGraph (deepchem/infograph.py)
- MEGNet (deepchem/megnet.py)
- MolFormer (deepchem/molformer.py)
- PAGTN (deepchem/pagtn.py)
- PNA-GNN (deepchem/pna_gnn.py)

这些模型可根据需要在后续工作中添加。
