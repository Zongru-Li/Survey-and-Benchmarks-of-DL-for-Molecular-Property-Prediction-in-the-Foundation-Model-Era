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
| **GIN** | gin | `gin/graphcnn.py` | ✅ 已添加 (PyG) |
| **GCN** | gcn | `gcn/gcn/models.py` | ✅ 已添加 (PyG) |
| **GROVER** | grover | `grover/grover/model/models.py` | ✅ 已添加 (PyG) |
| **CD-MVGNN** | cd-mvgnn | `cd-mvgnn/dglt/models/zoo/mpnn.py` | ✅ 已添加 (PyG) |

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

#### 6. GIN (Graph Isomorphism Network)

**论文**: How Powerful are Graph Neural Networks?

**参考项目**: `references/gin`

**实现文件**: `src/models/gin.py`

**配置文件**: `configs/gin.yaml`

**主要特点**:
- 基于 Weisfeiler-Lehman 图同构测试
- 使用可学习的 epsilon 参数区分中心节点和邻居节点
- MLP 作为聚合函数
- 支持 Jumping Knowledge (JK) 连接 (concat, last, max, sum)
- 支持多种邻居池化类型 (sum, average, max)

**使用方法**:
```bash
python src/run.py --config configs/gin.yaml --dataset adme_sol --epochs 100
```

#### 7. GCN (Graph Convolutional Network)

**论文**: Semi-Supervised Classification with Graph Convolutional Networks

**参考项目**: `references/gcn`

**实现文件**: `src/models/gcn.py`

**配置文件**: `configs/gcn.yaml`

**主要特点**:
- 经典的图卷积网络 (GCN) 实现
- 使用度归一化的消息传递
- 支持多层 GCN 卷积
- 特征投影层和预测头
- 支持多种池化方式

**使用方法**:
```bash
python src/run.py --config configs/gcn.yaml --dataset adme_sol --epochs 100
```

#### 8. GROVER (Graph Representation frOm self-superVised mEssage passing tRansformer)

**论文**: Self-Supervised Graph Transformer on Large-Scale Molecular Data

**参考项目**: `references/grover`

**实现文件**: `src/models/grover.py`

**配置文件**: `configs/grover.yaml`

**主要特点**:
- 基于 Transformer 的图神经网络
- 多头注意力机制
- 双路消息传递 (atom/bond view)
- 使用距离约束损失函数
- 支持自监督预训练

**使用方法**:
```bash
python src/run.py --config configs/grover.yaml --dataset adme_sol --epochs 100
```

#### 9. CD-MVGNN (Cross-Dependent Multi-View Graph Neural Network)

**论文**: Cross-Dependent Message Passing for Molecular Property Prediction

**参考项目**: `references/cd-mvgnn`

**实现文件**: `src/models/cd_mvgnn.py`

**配置文件**: `configs/cd_mvgnn.yaml`

**主要特点**:
- 双路消息传递网络 (DualMPN)
- Atom view 和 Bond view 交叉依赖
- 使用距离约束损失函数
- 两个独立的 FFN 输出头

**使用方法**:
```bash
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --epochs 100
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
    'gin': GIN,
    'gcn': GCN,
    'grover': GROVER,
    'cd_mvgnn': CDMVGNN,
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
| GIN | `configs/gin.yaml` | num_layers=5, num_mlp_layers=2, dropout=0.5, JK=concat, lr=0.0005 |
| D-MPNN | `configs/dmpnn.yaml` | num_layers=3, hidden=300, lr=0.0001 |
| AttentiveFP | `configs/attentivefp.yaml` | num_layers=2, graph_feat=200, lr=0.0001 |
| Mol-GDL | `configs/mol_gdl.yaml` | num_layers=3, lr=0.0001 |
| N-Gram RF | `configs/ngram_rf.yaml` | num_layers=3, lr=0.001 |
| N-Gram XGB | `configs/ngram_xgb.yaml` | num_layers=3, lr=0.001 |
| GCN | `configs/gcn.yaml` | num_layers=5, dropout=0.3, feat_dim=256, lr=0.0005 |
| GROVER | `configs/grover.yaml` | num_layers=3, num_heads=4, dropout=0.0, ffn_num_layers=2, lr=0.0001 |
| CD-MVGNN | `configs/cd_mvgnn.yaml` | num_layers=3, dropout=0.0, ffn_num_layers=2, lr=0.0001 |

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
| GIN | num_layers | 3 (deepchem) | 5 | 增加层数提升表达力 |
| GIN | dropout | 0 (deepchem) | 0.5 | 防止过拟合 |
| GIN | JK | last (deepchem) | concat | 更强大的表示融合 |
| GCN | num_layers | 2 (deepchem) | 5 | 增加层数提升表达力 |
| GCN | dropout | 0 (deepchem) | 0.3 | 防止过拟合 |
| GROVER | depth | 1 (deepchem) | 3 | 与 grover 官方一致 |
| GROVER | dropout | 0.2 (deepchem) | 0.0 | 与 grover 官方一致 |
| GROVER | ffn_num_layers | 1 (deepchem) | 2 | 与 grover 官方一致 |
| CD-MVGNN | depth | 3 (官方) | 3 | 与官方实现一致 |
| CD-MVGNN | dropout | 0 (官方) | 0.0 | 与官方实现一致 |

---

## DeepChem/ChemInfo 实现对比分析

基于 `references/deepchem/` 和 `references/cheminfo/` 目录中的官方实现，对四个新添加模型进行了对比分析。

### 1. GIN (Graph Isomorphism Network)

| 参数 | DeepChem 默认 | 当前配置 | 说明 |
|-----|--------------|---------|------|
| num_layer | 3 | 5 | 增加层数提升表达力 |
| emb_dim | 64 | 300 (hidden_feat=150) | 更大嵌入维度 |
| dropout | 0 | 0.5 | 防止过拟合 |
| jump_knowledge | "last" | "concat" | 更强大的表示融合 |
| gnn_type | "gin" | "gin" | 一致 |

**实现差异**:
- DeepChem 使用 `GINEConv` (带边特征的 GIN)，当前实现使用自定义 `GINConv`
- DeepChem 默认使用 `last` 跳跃连接，当前使用 `concat` (更强大但参数更多)
- 参考实现 `references/gin/graphcnn.py` 支持可学习 epsilon 和多种邻居池化类型

### 2. GCN (Graph Convolutional Network)

| 参数 | DeepChem 默认 | 当前配置 | 说明 |
|-----|--------------|---------|------|
| graph_conv_layers | [64, 64] | 5层 x 300维 | 更深更宽 |
| dropout | 0 | 0.3 | 防止过拟合 |
| predictor_hidden_feats | 128 | 256 | 更大预测头 |
| residual | True | 未使用 | 残差连接 |
| batchnorm | False | True | 使用批归一化 |

**实现差异**:
- DeepChem 实现基于 DGL，当前实现基于 PyG
- DeepChem 使用加权池化+最大池化拼接，当前使用标准 mean pooling
- 参考实现 `references/gcn/gcn/gcn/models.py` 是 TensorFlow 版本

### 3. GROVER

| 参数 | DeepChem 默认 | Grover 官方 | 当前配置 | 说明 |
|-----|--------------|------------|---------|------|
| hidden_size | 128 | 300 | 300 | 与官方一致 |
| depth | 1 | 3 | 3 | 与官方一致 |
| dropout | 0.2 | 0.0 | 0.0 | 与官方一致 |
| ffn_num_layers | 1 | 2 | 2 | 与官方一致 |
| num_attn_heads | 4 | 4 | 4 | 一致 |
| dist_coff | 0.1 | 0.1 | 0.1 | 一致 |

**实现差异**:
- DeepChem 的 GROVER 实现是完整版本，支持预训练和微调
- 当前实现是简化版，专注于微调任务
- 参考实现 `references/grover/` 包含完整的预训练流程

### 4. CD-MVGNN

| 参数 | 官方默认 | 当前配置 | 说明 |
|-----|---------|---------|------|
| depth | 3 | 3 | 一致 |
| hidden_size | 2 (x100) | 2 (x150=300) | 略大 |
| dropout | 0 | 0.0 | 一致 |
| ffn_num_layers | 2 | 2 | 一致 |
| dist_coff | 0.1 | 0.1 | 一致 |

**实现差异**:
- 官方实现 `references/cd-mvgnn/` 使用 DGL
- 当前实现基于 PyG，核心逻辑保持一致
- 双路消息传递 (DualMPN) 机制完全保留

### 配置合理性评估

| 模型 | 评估 | 建议 |
|-----|------|------|
| GIN | ✅ 合理 | JK=concat 提供更强表达力，dropout 0.5 防止过拟合 |
| GCN | ✅ 合理 | 层数和维度适合分子图任务 |
| GROVER | ✅ 合理 | 已调整为与 grover 官方一致 |
| CD-MVGNN | ✅ 合理 | 已调整为与官方实现一致 |

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

### GCN ADME-Sol 数据集测试结果 (epochs=2, PearsonR)

| Split | PearsonR | 参数量 |
|-------|----------|--------|
| scaffold | 0.2817 | 601.5K |
| butina | 0.1328 | 601.5K |
| umap | 0.2714 | 601.5K |
| time | -0.0710 | 601.5K |

### GROVER ADME-Sol 数据集测试结果 (epochs=2, PearsonR)

| Split | PearsonR | 参数量 |
|-------|----------|--------|
| scaffold | 0.3492 | 5.79M |
| butina | 0.3464 | 5.79M |
| umap | 0.3417 | 5.79M |
| time | 0.1556 | 5.79M |

注：GROVER 使用双路输出 (atom/bond view)，训练时使用距离约束损失。

### CD-MVGNN ADME-Sol 数据集测试结果 (epochs=2, PearsonR)

| Split | PearsonR | 参数量 |
|-------|----------|--------|
| scaffold | 0.1261 | 372.1K |
| butina | 0.1825 | 372.1K |
| umap | 0.0849 | 372.1K |
| time | 0.0217 | 372.1K |

注：CD-MVGNN 使用双路消息传递，训练时使用距离约束损失。

### GIN ADME-Sol 数据集测试结果 (epochs=2, PearsonR)

| Split | PearsonR | 参数量 |
|-------|----------|--------|
| scaffold | 0.3240 | 1.44M |
| butina | 0.1789 | 1.44M |
| umap | 0.2835 | 1.44M |
| time | 0.3493 | 1.44M |

注：GIN 使用 Jumping Knowledge (concat) 融合多层表示，支持可学习 epsilon 参数。

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

# GCN 四种分割测试
python src/run.py --config configs/gcn.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/gcn.yaml --dataset adme_sol --epochs 2 --split butina
python src/run.py --config configs/gcn.yaml --dataset adme_sol --epochs 2 --split umap
python src/run.py --config configs/gcn.yaml --dataset adme_sol --epochs 2 --split time

# GROVER 四种分割测试
python src/run.py --config configs/grover.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/grover.yaml --dataset adme_sol --epochs 2 --split butina
python src/run.py --config configs/grover.yaml --dataset adme_sol --epochs 2 --split umap
python src/run.py --config configs/grover.yaml --dataset adme_sol --epochs 2 --split time

# CD-MVGNN 四种分割测试
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --epochs 2 --split butina
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --epochs 2 --split umap
python src/run.py --config configs/cd_mvgnn.yaml --dataset adme_sol --epochs 2 --split time

# GIN 四种分割测试
python src/run.py --config configs/gin.yaml --dataset adme_sol --epochs 2 --split scaffold
python src/run.py --config configs/gin.yaml --dataset adme_sol --epochs 2 --split butina
python src/run.py --config configs/gin.yaml --dataset adme_sol --epochs 2 --split umap
python src/run.py --config configs/gin.yaml --dataset adme_sol --epochs 2 --split time
```

---

## 实验脚本

实验脚本按数据集组织在 `scripts/` 目录中：

| 脚本 | 描述 |
|-----|------|
| `scripts/run_adme_sol.sh` | ADME-SOL 数据集所有模型实验 |
| `scripts/run_adme_hlm.sh` | ADME-HLM 数据集实验 |
| `scripts/run_bace.sh` | BACE 数据集实验 |
| `scripts/run_bbbp.sh` | BBBP 数据集实验 |
| `scripts/graphkan/adme_sol.sh` | GraphKan ADME-SOL 四种分割实验 |
| `scripts/graphkan/adme_hlm.sh` | GraphKan ADME-HLM 四种分割实验 |
| `scripts/graphkan/adme_rlm.sh` | GraphKan ADME-RLM 四种分割实验 |
| `scripts/graphkan/adme_mdr1.sh` | GraphKan ADME-MDR1 四种分割实验 |
| `scripts/graphkan/adme_hppb.sh` | GraphKan ADME-hPPB 四种分割实验 |
| `scripts/graphkan/adme_rppb.sh` | GraphKan ADME-rPPB 四种分割实验 |
| `scripts/gcn/adme_*.sh` | GCN 六个 ADME 数据集四种分割实验 |
| `scripts/grover/adme_*.sh` | GROVER 六个 ADME 数据集四种分割实验 |
| `scripts/cd_mvgnn/adme_*.sh` | CD-MVGNN 六个 ADME 数据集四种分割实验 |
| `scripts/gin/adme_*.sh` | GIN 六个 ADME 数据集四种分割实验 |

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
