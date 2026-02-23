# Computational-ADME 数据集

## 概述

Computational-ADME 数据集来源于论文 *"Prospective Validation of Machine Learning Algorithms for ADME Prediction: An Industrial Perspective"*，由分子信息学研究团队公开发布。该数据集包含 **3521** 个多样化化合物，测试了 **6 个 ADME 体外终点**，用于药物发现早期的药代动力学和理化性质预测研究。

**数据来源**: [GitHub - molecularinformatics/Computational-ADME](https://github.com/molecularinformatics/Computational-ADME)

**论文引用**: 
> Prospective Validation of Machine Learning Algorithms for ADME Prediction: An Industrial Perspective. Journal of Chemical Information and Modeling.

---

## 数据集统计

| 项目 | 数值 |
|------|------|
| 总化合物数 | 3,521 |
| ADME 终点数 | 6 |
| 数据格式 | CSV, SDF |
| 分子表示 | SMILES |

### 数据来源分布

| 来源 | 数量 | 占比 |
|------|------|------|
| eMolecules | 3,452 | 98.0% |
| EnamineHTS | 23 | 0.7% |
| LabNetworkBB | 21 | 0.6% |
| Mcule | 18 | 0.5% |
| EnamineBB_pmc | 7 | 0.2% |

---

## ADME 终点详细说明

### 1. HLM - 人肝微粒体稳定性 (Human Liver Microsomal Stability)

| 属性 | 值 |
|------|-----|
| 列名 | `LOG HLM_CLint (mL/min/kg)` |
| 有效样本数 | 3,087 |
| 缺失值 | 434 |
| 最小值 | 0.676 |
| 最大值 | 3.373 |
| 均值 | 1.320 |
| 标准差 | 0.624 |
| 物理意义 | 肝脏代谢是药物清除的主要途径，HLM 清除率 (CLint) 表示药物在人肝微粒体中的代谢稳定性 |
| 应用 | 预测药物在人体内的代谢清除率，评估药物候选物的代谢稳定性 |

### 2. RLM - 大鼠肝微粒体稳定性 (Rat Liver Microsomal Stability)

| 属性 | 值 |
|------|-----|
| 列名 | `LOG RLM_CLint (mL/min/kg)` |
| 有效样本数 | 3,054 |
| 缺失值 | 467 |
| 最小值 | 1.028 |
| 最大值 | 3.970 |
| 均值 | 2.256 |
| 标准差 | 0.750 |
| 物理意义 | 大鼠肝微粒体清除率，用于动物药代动力学研究 |
| 应用 | 与 HLM 对比分析，预测药物在大鼠体内的代谢行为 |

### 3. MDR1-MDCK ER - 外排比率 (Efflux Ratio)

| 属性 | 值 |
|------|-----|
| 列名 | `LOG MDR1-MDCK ER (B-A/A-B)` |
| 有效样本数 | 2,642 |
| 缺失值 | 879 |
| 最小值 | -1.162 |
| 最大值 | 2.725 |
| 均值 | 0.398 |
| 物理意义 | MDR1-MDCK 细胞系表达 P-糖蛋白 (P-gp)，外排比率 (B→A / A→B) 表示底物被 P-gp 外排的程度 |
| 应用 | 评估药物是否为 P-gp 底物，预测血脑屏障渗透性和肠道吸收 |

### 4. Solubility - 溶解度 (pH 6.8)

| 属性 | 值 |
|------|-----|
| 列名 | `LOG SOLUBILITY PH 6.8 (ug/mL)` |
| 有效样本数 | 2,173 |
| 缺失值 | 1,348 |
| 最小值 | -1.000 |
| 最大值 | 2.179 |
| 均值 | 1.260 |
| 物理意义 | 在 pH 6.8 缓冲液中的溶解度 (log scale)，模拟肠道环境 |
| 应用 | 评估药物的口服生物利用度，指导制剂开发 |

### 5. hPPB - 人血浆蛋白结合率 (Human Plasma Protein Binding)

| 属性 | 值 |
|------|-----|
| 列名 | `LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)` |
| 有效样本数 | 194 |
| 缺失值 | 3,327 |
| 最小值 | -1.593 |
| 最大值 | 2.000 |
| 均值 | 0.766 |
| 物理意义 | 人血浆中未结合药物的比例 (log scale)，只有未结合的药物才能发挥药理作用 |
| 应用 | 预测药物在体内的游离浓度，影响给药剂量设计 |

### 6. rPPB - 大鼠血浆蛋白结合率 (Rat Plasma Protein Binding)

| 属性 | 值 |
|------|-----|
| 列名 | `LOG PLASMA PROTEIN BINDING (RAT) (% unbound)` |
| 有效样本数 | 168 |
| 缺失值 | 3,353 |
| 最小值 | -1.638 |
| 最大值 | 2.000 |
| 均值 | 0.764 |
| 物理意义 | 大鼠血浆中未结合药物的比例 (log scale) |
| 应用 | 动物药代动力学研究，与 hPPB 对比进行种属外推 |

---

## 整合到 KA-GNN 基准测试框架

### 分类数据集

基于药理学标准（`assets/4_classification.md`），将原始回归数据转换为二分类标签，保存到 `data/ADME/` 目录：

| 数据集名称 | 源终点 | 样本数 | 阈值 | 分类规则 | label=1 含义 |
|-----------|--------|--------|------|----------|-------------|
| adme_sol | Solubility | 2,173 | 1.543 | logS >= 1.543 | 高水溶性 |
| adme_mdr1 | MDR1-ER | 2,642 | 0.301 | log(ER) >= 0.301 | P-gp 底物/高外排 |
| adme_hlm | HLM | 3,087 | 1.699 | log(CLint) < 1.699 | 高代谢稳定性 |
| adme_rlm | RLM | 3,054 | 1.699 | log(CLint) < 1.699 | 高代谢稳定性 |
| adme_hppb | hPPB | 194 | 1.000 | log(unbound%) <= 1 | 高结合率 (PPB>=90%) |
| adme_rppb | rPPB | 168 | 1.000 | log(unbound%) <= 1 | 高结合率 (PPB>=90%) |

### 分类标准详解

基于 `assets/4_classification.md` 中的药理学标准：

- **Solubility (溶解度)**: 高水溶性有利于药物吸收。标准阈值为 logS >= -3，但由于实际数据范围 [-1, 2.179] 均高于此，使用中位数 1.543 作为分类阈值。
- **MDR1-ER (外排比率)**: 外排比率 ER >= 2 (log(ER) >= 0.301) 表示化合物是 P-gp 底物，可能导致低肠道吸收或血脑屏障渗透性差。
- **HLM/RLM (肝微粒体稳定性)**: 代谢半衰期 t1/2 >= 30 min 为高稳定性，对应清除率 CLint < 50 mL/min/kg (log(CLint) < 1.699)。
- **PPB (血浆蛋白结合率)**: PPB >= 90% (unbound <= 10%, log(unbound%) <= 1) 表示高结合率，可能影响游离药物浓度和药效。

### 数据格式

生成的分类数据集为三列格式：

```csv
smiles,label,LOG HLM_CLint (mL/min/kg)
CNc1cc(Nc2cccn(-c3ccccn3)c2=O)nn2c(C(=O)N[C@@H]3C[C@@H]3F)cnc12,1,0.675686709
CCOc1cc2nn(CCC(C)(C)O)cc2cc1NC(=O)c1cccc(C(F)F)n1,1,0.675686709
...
```

### 数据划分方式

支持三种划分方式，**默认使用 UMAP 划分**：

| 划分方式 | 原理 | 优点 | 缺点 | 适用场景 |
|---------|------|------|------|---------|
| **Random** | 随机打乱后按比例划分 | 简单快速、数据分布均匀 | 可能导致数据泄露 | 快速验证 |
| **Scaffold** | 按 Bemis-Murcko 骨架分组划分 | 避免结构泄露、更具挑战性 | 某些骨架可能过大 | 标准基准测试 |
| **UMAP** (默认) | UMAP 降维 + KMeans 聚类后划分 | 基于化学空间分布、更均匀 | 计算开销较大 | 化学空间分析 |

### 使用方式

```bash
# 使用 UMAP 划分（默认）
python src/run.py --config configs/ka_gnn.yaml --dataset adme_hlm --epochs 200

# 使用 Scaffold 划分
python src/run.py --config configs/ka_gnn.yaml --dataset adme_hlm --split scaffold --epochs 200

# 使用 Random 划分
python src/run.py --config configs/ka_gnn.yaml --dataset adme_hlm --split random --epochs 200

# 运行所有 ADME 终点
for dataset in adme_hlm adme_rlm adme_mdr1 adme_sol adme_hppb adme_rppb; do
    python src/run.py --config configs/ka_gnn.yaml --dataset $dataset --epochs 200
done
```

---

## 代码修改

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `src/utils/splitters.py` | 添加 `UMAPSplitter` 类，支持基于 UMAP 降维 + KMeans 聚类的数据集划分 |
| `src/utils/data.py` | 添加 `ADME_TARGETS`、`is_adme_dataset()`、`get_splitter()`、`get_data_path()`；`create_dataset()` 和 `create_dataloader()` 支持 `split_type` 参数（默认 `umap`） |
| `src/run.py` | 添加 `--split` 命令行参数（choices: random, scaffold, umap） |

### 新增的文件

```
data/ADME/
├── adme_hlm.csv          # HLM 分类数据（阈值 1.699）
├── adme_rlm.csv          # RLM 分类数据（阈值 1.699）
├── adme_mdr1.csv         # MDR1 分类数据（阈值 0.301）
├── adme_sol.csv          # Solubility 分类数据（阈值 1.543）
├── adme_hppb.csv         # hPPB 分类数据（阈值 1.000）
└── adme_rppb.csv         # rPPB 分类数据（阈值 1.000）

scripts/
└── generate_adme_datasets.py  # 数据生成脚本
```

### 依赖

```bash
pip install umap-learn
```

---

## 注意事项

1. **PPB 数据稀疏**: hPPB (194) 和 rPPB (168) 样本量较小，建议使用更小的 batch_size (32)
2. **分类标签不平衡**: 
   - adme_hlm: 70.8% positive (高稳定性化合物较多)
   - adme_rlm: 25.5% positive (大鼠代谢稳定性较低)
3. **划分缓存**: 不同划分方式的数据缓存为 `{dataset}_{split_type}.pth`
4. **可复现性**: 建议固定 `--seed` 参数确保结果可复现

---

## 参考资料

1. 原始论文: *Prospective Validation of Machine Learning Algorithms for ADME Prediction: An Industrial Perspective*
2. GitHub 仓库: https://github.com/molecularinformatics/Computational-ADME
3. MoleculeNet: https://moleculenet.org/
4. DGL 文档: https://docs.dgl.ai/
