# 数据集划分方法 (Splitters)

本项目实现了多种数据集划分方法，用于机器学习模型的训练、验证和测试。

## 支持的划分方法

| 方法 | 参数值 | 描述 |
|------|--------|------|
| Random | `random` | 随机划分数据集 |
| Scaffold | `scaffold` | 基于 Murcko 骨架划分 |
| UMAP | `umap` | 基于 UMAP 降维和聚类划分 |
| Butina | `butina` | 基于 Butina 聚类算法划分 |

## 使用方法

### 命令行参数

```bash
python src/run.py --config configs/ka_gnn.yaml --dataset bace --split scaffold
```

支持的 `--split` 参数值：
- `random` - 随机划分
- `scaffold` - 骨架划分（默认）
- `umap` - UMAP 聚类划分
- `butina` - Butina 聚类划分

### 配置文件

在 YAML 配置文件中设置：

```yaml
training:
  split_type: "scaffold"  # 可选: random, scaffold, umap, butina
```

## 划分方法详解

### 1. Random Splitter（随机划分）

最简单的划分方法，随机打乱数据后按比例划分。

**特点：**
- 实现简单，计算速度快
- 可能导致相似分子分布在训练集和测试集中
- 测试集性能可能被高估

**适用场景：**
- 快速原型验证
- 数据量较小的情况

### 2. Scaffold Splitter（骨架划分）

基于 Bemis-Murcko 骨架进行划分，确保具有相同骨架的分子只出现在一个集合中。

**特点：**
- 按分子骨架分组，保证骨架不重叠
- 更真实地评估模型泛化能力
- 是分子性质预测中的标准划分方法

**适用场景：**
- 分子性质预测任务
- 需要评估模型对新骨架的泛化能力

### 3. UMAP Splitter（UMAP 聚类划分）

基于 UMAP 降维和 K-Means 聚类的划分方法。

**参考文献：**
> "Scaffold Splits Overestimate Virtual Screening Performance" (Guo et al., 2024)
> https://doi.org/10.26434/chemrxiv-2024-xj8r2

**方法流程：**
1. 计算分子的 Morgan 指纹
2. 使用 UMAP 降维到 2D 空间
3. K-Means 聚类
4. 按聚类分配到 train/valid/test

**特点：**
- 基于化学空间的分布进行划分
- 确保不同集合中的分子在化学空间中真正不同
- 可以更准确地评估模型的泛化性能

**适用场景：**
- 虚拟筛选任务
- 需要更严格的模型评估

### 4. Butina Splitter（Butina 聚类划分）

基于 Taylor-Butina 聚类算法和 Tanimoto 相似度的划分方法。

**参考文献：**
> Butina, D. "Unsupervised Data Base Clustering Based on Daylight's Fingerprint 
> and Tanimoto Similarity: A Fast and Automated Way To Cluster Small and Large 
> Data Sets." J. Chem. Inf. Comput. Sci. 1999, 39, 4, 747–750.

**方法流程：**
1. 计算分子的 Morgan 指纹
2. 计算 Tanimoto 相似度矩阵
3. 使用 Butina 算法聚类
4. 按聚类大小分配到 train/valid/test

**参数：**
- `cutoff` (默认 0.6): Tanimoto 相似度阈值，低于此值的分子会被分到不同聚类

**特点：**
- O(N²) 计算复杂度
- 相似分子被分到同一聚类
- 确保测试集包含新颖的化学结构

**适用场景：**
- 中等规模数据集
- 需要基于相似度进行严格划分的任务

## 划分比例

所有划分方法支持自定义比例，默认为：
- 训练集: 80%
- 验证集: 10%
- 测试集: 10%

在配置文件中可调整：

```yaml
training:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

## 代码示例

### 直接使用 Splitter

```python
from src.utils.splitters import ScaffoldSplitter, ButinaSplitter

# 准备数据 (格式: [smiles, label, graph])
data_list = [...]

# 使用骨架划分
splitter = ScaffoldSplitter()
train, valid, test = splitter.split(data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

# 使用 Butina 划分（自定义相似度阈值）
splitter = ButinaSplitter(cutoff=0.5)
train, valid, test = splitter.split(data_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
```

### 通过工厂函数获取 Splitter

```python
from src.utils.data import get_splitter

splitter = get_splitter('butina')  # 返回 ButinaSplitter 实例
train, valid, test = splitter.split(data_list)
```
