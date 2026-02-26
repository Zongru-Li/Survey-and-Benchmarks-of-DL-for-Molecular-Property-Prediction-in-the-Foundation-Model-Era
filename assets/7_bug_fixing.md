# Bug 修复报告

## 问题 1: 测试集 DataLoader drop_last 导致数据丢失

### 问题描述

在使用多个模型进行回归任务时，程序抛出以下错误：

```
ValueError: `x` and `y` must have length at least 2.
```

错误发生在 `src/utils/training.py` 的 `predict_gnn_regression` / `predict_gat_regression` / `predict_pyg_regression` 函数中，当调用 `pearsonr` 计算相关系数时。

### 根本原因

在 `src/utils/data.py` 的 `create_dataloader` 函数中，测试集的 `DataLoader` 使用了 `drop_last=True` 参数。当测试集大小小于 `batch_size`（默认 128）时，所有测试数据都会被丢弃，导致 `test_loader` 产生的批次为空，最终在计算 Pearson 相关系数时因为数据为空而报错。

```python
# 问题代码 (data.py:531-539)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=state["shuffle"],
    num_workers=0,
    pin_memory=False,
    drop_last=True,  # <-- 问题所在
    collate_fn=collate,
)
```

### 受影响的实验

从 `nohup.out` 日志中确认的受影响模型和数据集组合：

| 模型 | 数据集 | 测试集大小 | 是否失败 |
|------|--------|------------|----------|
| ka_gnn | adme_hppb | ~20 | **是** |
| ka_gnn | adme_rppb | ~17 | **是** |
| kagat | adme_hppb | ~20 | **是** |
| kagat | adme_rppb | ~17 | **是** |
| attentivefp | adme_hppb | ~20 | **是** |
| attentivefp | adme_rppb | ~17 | **是** |
| mol_gdl | adme_hppb | ~20 | **是** |
| mol_gdl | adme_rppb | ~17 | **是** |
| graphmvp | adme_hppb | ~20 | **是** |
| graphmvp | adme_rppb | ~17 | **是** |
| molclr_gcn | adme_hppb | ~20 | **是** |
| molclr_gcn | adme_rppb | ~17 | **是** |
| molclr_gin | adme_hppb | ~20 | **是** |
| molclr_gin | adme_rppb | ~17 | **是** |
| pretrain_gnn | adme_hppb | ~20 | **是** |
| pretrain_gnn | adme_rppb | ~17 | **是** |

### 数据集详细信息

| 数据集 | 总数据量 | scaffold 分割测试集 | umap 分割测试集 | 是否受影响 |
|--------|----------|---------------------|-----------------|------------|
| adme_hppb | 194 | 20 | 49 | **是** (scaffold) |
| adme_rppb | 168 | 17 | 49 | **是** (scaffold) |

### 修复方案

将测试集 `DataLoader` 的 `drop_last` 参数从 `True` 改为 `False`：

```python
# 修复后的代码 (data.py:531-539)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=state["shuffle"],
    num_workers=0,
    pin_memory=False,
    drop_last=False,  # <-- 修复：测试集不应丢弃数据
    collate_fn=collate,
)
```

### 修复结果

- **修复文件**: 
  - `src/utils/data.py:537` (DGL DataLoader)
  - `src/utils/pyg_data.py:390` (PyG DataLoader)
- **修改内容**: `drop_last=True` → `drop_last=False`
- **修复日期**: 2026-02-26
- **修复状态**: ✅ 已修复

---

## 问题 2: 数据集缓存文件过时导致训练无进展

### 问题描述

在执行 hppb 数据集实验时，部分分割方式（random、scaffold、umap）导致训练没有进展，且显示以下警告：

```
[05:36:19] UFFTYPER: Unrecognized charge state for atom: 21
```

同时发现这些分割方式的数据量仅为 194 条，而非预期的 1000+ 条。

### 根本原因

1. **UFFTYPER 警告（非关键问题）**：这是 RDKit 在处理某些特殊电荷状态原子时的内部警告。代码使用的是 MMFF 力场（`AllChem.MMFFGetMoleculeForceField`），而非 UFF 力场，因此此警告不影响训练。测试显示 1808 个分子中仅 5 个因 MMFF 属性问题失败，其余均能正常处理。

2. **缓存文件过时（关键问题）**：数据文件 `data/ADME/adme_hppb.csv` 已更新为 1808 条数据，但 `data/processed/` 目录中的缓存文件（`adme_hppb_random.pth`、`adme_hppb_scaffold.pth`、`adme_hppb_umap.pth`）是基于旧数据（194 条）生成的。

   | 数据集 | 当前数据量 | 旧缓存数据量 | time 分割数据量 |
   |--------|------------|--------------|-----------------|
   | adme_hppb | 1808 | 194 | 1808 ✅ |
   | adme_rppb | 885 | ~168 | - |

   `time` 分割的缓存文件（81MB）是基于新数据生成的，而其他分割的缓存文件（8MB）仍是基于旧数据。

### 诊断方法

```bash
# 检查数据文件行数
wc -l data/ADME/adme_hppb.csv data/ADME/adme_rppb.csv

# 检查缓存文件大小（差异明显说明数据量不同）
ls -la data/processed/ | grep -E "hppb|rppb"

# 检查缓存文件中的数据量
python3 -c "
import torch
state = torch.load('data/processed/adme_hppb_scaffold.pth', weights_only=False)
total = len(state['train_label']) + len(state['valid_label']) + len(state['test_label'])
print(f'Total samples in cache: {total}')
"
```

### 修复方案

删除过时的缓存文件，让程序重新处理数据：

```bash
rm data/processed/adme_hppb_random.pth
rm data/processed/adme_hppb_scaffold.pth
rm data/processed/adme_hppb_umap.pth
rm data/processed/adme_rppb_scaffold.pth
rm data/processed/adme_rppb_umap.pth
```

### 修复结果

- **修复日期**: 2026-02-26
- **修复状态**: ✅ 已修复（已删除旧缓存文件）
- **后续操作**: 重新运行实验，程序将自动生成基于新数据的缓存文件

---

## 问题汇总

| # | 问题 | 根本原因 | 受影响模型数 | 受影响数据集 | 状态 |
|---|------|----------|--------------|--------------|------|
| 1 | ValueError: `x` and `y` must have length at least 2 | test_loader 使用 drop_last=True 导致小测试集数据全被丢弃 | 8 个模型 | adme_hppb, adme_rppb | ✅ 已修复 |
| 2 | 训练无进展 + UFFTYPER 警告 + 数据量异常 | `data/processed/` 中缓存文件基于旧数据生成 | 所有模型 | adme_hppb, adme_rppb (random/scaffold/umap 分割) | ✅ 已修复 |
| 3 | UMAP 分割性能异常低 (PearsonR 0.38 vs 0.6+) | UMAPSplitter 分配顺序错误：大 cluster 给 test，小 cluster 给 train | 所有模型 | 所有使用 umap 分割的数据集 | ✅ 已修复 |

## 修复后需要重新运行的实验

以下实验需要重新运行以获取正确结果：

1. **adme_hppb** - random、scaffold、umap 分割（所有模型）
2. **adme_rppb** - scaffold、umap 分割（所有模型）

**注意**：time 分割的缓存文件已是最新，无需重新运行。

---

## 问题 3: UMAPSplitter 分配逻辑错误导致模型性能异常

### 问题描述

在 adme_hppb 数据集上，使用不同分割方式训练 ka_gnn 模型，UMAP 分割的 Pearson 相关系数显著低于其他分割方式：

```
kagnn_data:adme_hppb_split:scaffold_nl:4_lr:0.0001_bs:128_ep:501_iter:1
PearsonR:0.6088

kagnn_data:adme_hppb_split:butina_nl:4_lr:0.0001_bs:128_ep:501_iter:1
PearsonR:0.5404

kagnn_data:adme_hppb_split:umap_nl:4_lr:0.0001_bs:128_ep:501_iter:1
PearsonR:0.3779  ← 显著低于其他分割

kagnn_data:adme_hppb_split:time_nl:4_lr:0.0001_bs:128_ep:501_iter:1
PearsonR:0.6394
```

### 根本原因

在 `src/utils/splitters.py` 的 `UMAPSplitter.split()` 方法中（第 386-398 行），cluster 分配顺序存在逻辑错误：

```python
# 问题代码 (splitters.py:386-398)
clusters_with_size.sort(key=lambda x: -x[1])  # 按大小降序排列

for cluster_id, cluster_size in clusters_with_size:
    if test_ratio < (1.0 - tolerance):
        test_clusters.append(cluster_id)      # ← 最大的 cluster 先给 test
    elif valid_ratio < (1.0 - tolerance):
        valid_clusters.append(cluster_id)     # ← 然后给 valid
    else:
        train_clusters.append(cluster_id)     # ← 剩余小 cluster 给 train
```

**问题分析**：

1. `clusters_with_size` 已按 cluster 大小**降序**排列（从大到小）
2. 分配顺序是 `test → valid → train`
3. 结果：**最大的 clusters 被分配给 test，最小的 clusters 被分配给 train**

**对比其他 splitter 的正确实现**（ScaffoldSplitter、ButinaSplitter）：

```python
# 正确实现 (ScaffoldSplitter, ButinaSplitter)
for scaffold_set in all_scaffold_sets:  # 从大到小
    if len(train_idx) + len(scaffold_set) > train_cutoff:
        if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
            test_idx.extend(scaffold_set)
        else:
            valid_idx.extend(scaffold_set)
    else:
        train_idx.extend(scaffold_set)  # ← 优先填充 train
```

### 影响

1. **训练集多样性不足**：训练集得到的是较小的、剩余的 clusters，化学空间覆盖有限
2. **测试集差异过大**：测试集包含最大的 clusters，与训练集化学空间差异显著
3. **模型泛化能力下降**：训练集学到的模式难以推广到测试集
4. **性能显著下降**：PearsonR 从 0.6+ 降至 0.38

### 修复方案

将分配顺序从 `test → valid → train` 改为 `train → valid → test`：

```python
# 修复后的代码 (splitters.py:386-398)
for cluster_id, cluster_size in clusters_with_size:
    train_ratio = train_count / n_train_target if n_train_target > 0 else 1.0
    valid_ratio = valid_count / n_valid_target if n_valid_target > 0 else 1.0
    
    if train_ratio < (1.0 - tolerance):
        train_clusters.append(cluster_id)
        train_count += cluster_size
    elif valid_ratio < (1.0 - tolerance):
        valid_clusters.append(cluster_id)
        valid_count += cluster_size
    else:
        test_clusters.append(cluster_id)
        test_count += cluster_size
```

### 修复结果

- **修复文件**: `src/utils/splitters.py:386-398`
- **修改内容**: 调整 cluster 分配顺序为 `train → valid → test`
- **修复日期**: 2026-02-26
- **修复状态**: ✅ 已修复

### 其他 Splitter 检查结果

检查了所有其他 splitter 方法，确认无相同问题：

| Splitter | 排序方向 | 分配顺序 | 状态 |
|----------|----------|----------|------|
| ScaffoldSplitter | 降序(大→小) | train → valid → test | ✅ 正确 |
| ButinaSplitter | 降序(大→小) | train → valid → test | ✅ 正确 |
| RandomScaffoldSplitter | 随机 | valid → test → train | ✅ 可接受(随机) |

### 后续操作

1. ~~应用修复~~
2. 删除已生成的 umap 缓存文件：`rm data/processed/*_umap.pth`
3. 重新运行所有使用 umap 分割的实验
