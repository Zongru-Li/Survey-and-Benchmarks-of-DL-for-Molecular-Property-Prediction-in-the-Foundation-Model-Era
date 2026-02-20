# 重构前后实验代码差异分析报告

本文档对比分析当前重构后的 KA-GNN/KA-GAT 实验代码与 `7a84daede621b99bc9affc6fac5a2f9e0aca6194` 提交中的原始代码差异，以及这些差异可能导致实验结果偏差的原因。

**最后更新：** 2026-02-19

---

## 1. DataLoader 参数差异

### 1.1 num_workers 和 pin_memory 参数

| 参数 | 重构前 | 重构后 |
|------|--------|--------|
| `num_workers` | 4 | 0 |
| `pin_memory` | True | False |

**代码位置：**
- 重构前: `KA-GNN/main.py:276-279`, `KA-GAT/main.py:205-208`
- 重构后: `src/utils/data.py:229-239`

**重构前代码：**
```python
loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],
    num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
```

**重构后代码：**
```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=state['shuffle'],
    num_workers=0, pin_memory=False, drop_last=True, collate_fn=collate)
```

**影响分析：**
- `num_workers=0` 表示数据加载在主进程中进行，不使用多进程
- `pin_memory=False` 表示不将数据固定在内存中
- **这些变化不直接影响实验结果，但可能影响训练速度**
- 在某些环境或特定硬件配置下，可能导致轻微的数值差异

---

## 2. CustomDataset 设备分配变化

### 2.1 设备选择逻辑

**重构前代码：**
```python
class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu')
```

**重构后代码 (`src/utils/data.py:93-98`)：**
```python
class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list, use_gnn: bool = True):
        self.labels = label_list
        self.graphs = graph_list
        self.use_gnn = use_gnn
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

**影响分析：**
- 重构后在 `__getitem__` 中会将数据直接移动到GPU（如果可用）
- 但在训练循环中又会调用 `.to(device)`，可能导致重复的数据传输
- **理论上不影响实验结果，但可能影响性能**
- 在某些情况下可能导致内存使用模式不同

---

## 3. 未使用的遗留代码

### 3.1 tensor_nan_inf() 函数

**代码位置：** `src/utils/graph.py:156-176`

```python
def tensor_nan_inf(per_bond_feat):
    nan_exists = any(
        math.isnan(x) if isinstance(x, float) else False for x in per_bond_feat
    )
    inf_exists = any(
        x == float("inf") if isinstance(x, float) else False for x in per_bond_feat
    )
    ninf_exists = any(
        x == float("-inf") if isinstance(x, float) else False for x in per_bond_feat
    )
    if nan_exists or inf_exists or ninf_exists:
        clean_list = [
            0 if isinstance(x, float) and math.isnan(x) else x for x in per_bond_feat
        ]
        per_bond_feat = [
            1 if x == float("inf") else -1 if x == float("-inf") else x
            for x in clean_list
        ]
        return per_bond_feat
    else:
        return per_bond_feat
```

**影响分析：**
- 该函数在代码中定义但**未被调用**
- 在 `atom_to_graph()` 函数中，边特征直接添加到 `edge_features` 列表，未经过 `tensor_nan_inf()` 处理
- **不影响实验结果，但建议清理此遗留代码**

---

## 4. 已复原的差异（已解决）

以下差异在之前的排查中已被识别并修复：

### 4.1 非键边距离条件 ✅ 已复原

**状态：** 已修复，当前代码与原始代码一致

**当前代码 (`src/utils/graph.py:247`)：**
```python
if s_d_dis <= 5:
    if check_common_elements(src_list, dst_list, i, j):
        # 添加非键边
```

**原始代码：** 相同

---

### 4.2 NUM_EPOCHS 参数 ✅ 已复原

**状态：** 已修复，当前配置与原始配置一致

**当前配置 (`configs/ka_gnn.yaml:14`)：**
```yaml
NUM_EPOCHS: 501
```

**原始配置：** 相同

---

## 5. 代码清理（不影响实验结果）

### 5.1 ScaffoldSplitter 重复代码删除

**重构前代码 (`KA-GNN/utils/splitters.py:173-195`)：**
```python
# 第一段 (lines 173-181)
train_cutoff = frac_train * N
valid_cutoff = (frac_train + frac_valid) * N
train_idx, valid_idx, test_idx = [], [], []
for scaffold_set in all_scaffold_sets:
    ...

assert len(set(train_idx).intersection(set(valid_idx))) == 0
assert len(set(test_idx).intersection(set(valid_idx))) == 0

# 第二段 - 完全重复的代码 (lines 187-195)
train_cutoff = frac_train * N
valid_cutoff = (frac_train + frac_valid) * N
train_idx, valid_idx, test_idx = [], [], []
for scaffold_set in all_scaffold_sets:
    ...
```

**重构后代码 (`src/utils/splitters.py:77-90`)：**
```python
# 只保留一份，删除了重复代码
train_cutoff = frac_train * N
valid_cutoff = (frac_train + frac_valid) * N
train_idx, valid_idx, test_idx = [], [], []
for scaffold_set in all_scaffold_sets:
    ...
```

**影响分析：**
- 重构前由于代码重复，第一次计算的 `train_idx, valid_idx, test_idx` 被第二次完全覆盖
- **这意味着重构前后的数据集划分逻辑实际上是一致的**
- 此变化不影响实验结果，属于代码清理

---

## 6. 其他等效变化

### 6.1 DGL图创建方式变化

**重构前代码 (`KA-GNN/utils/graph_path.py:214-216`)：**
```python
g = dgl.DGLGraph()
g.add_nodes(num_atoms)
g.add_edges(src_list, dst_list)
```

**重构后代码 (`src/utils/graph.py:265`)：**
```python
g = dgl.graph((src_list, dst_list), num_nodes=num_atoms)
```

**影响分析：**
- 两种创建方式在功能上等效
- `dgl.graph()` 是推荐的现代API
- **不影响实验结果**

---

### 6.2 打印输出格式差异

**重构前代码：**
```python
print(f"Epoch {epoch} | Train Loss: {total_train_loss:.4f} | Vali Loss: {total_loss_val:.4f}")
```

**重构后代码 (`src/utils/training.py:219`)：**
```python
print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
```

**影响分析：**
- 仅影响打印格式和epoch编号显示（重构后从1开始，重构前从epoch开始）
- **不影响实验结果**

---

## 7. 总结：当前可能导致实验结果差异的因素

| 优先级 | 差异项 | 影响程度 | 当前状态 |
|--------|--------|----------|----------|
| **低** | DataLoader num_workers/pin_memory | 理论上不影响，但可能因环境差异产生轻微影响 | 仍存在 |
| **低** | CustomDataset 设备分配 | 理论上不影响，但可能影响内存使用模式 | 仍存在 |
| **无** | tensor_nan_inf 遗留函数 | 未被调用，不影响结果 | 建议清理 |
| ~~高~~ | ~~非键边距离条件~~ | ~~已复原~~ | ✅ 已解决 |
| ~~中~~ | ~~NUM_EPOCHS~~ | ~~已复原~~ | ✅ 已解决 |

---

## 8. 结论

当前重构后的代码与原始代码在以下方面仍然存在差异，但**这些差异理论上不应该影响实验结果**：

1. **DataLoader 参数**：`num_workers=0, pin_memory=False` - 仅影响性能和环境兼容性
2. **CustomDataset 设备分配**：动态选择设备 - 理论上等效，但可能影响内存使用

**重要发现：**
- 之前报告中识别的两个关键差异（非键边距离条件和NUM_EPOCHS）**已经复原**
- `tensor_nan_inf()` 函数虽然存在但未被调用，不是导致差异的原因

**建议：**
1. 如需完全对齐原始环境，可以将 DataLoader 参数改回 `num_workers=4, pin_memory=True`
2. 建议清理未使用的 `tensor_nan_inf()` 函数
3. 如果实验结果仍然存在差异，可能需要检查：
   - 随机种子设置
   - CUDA/cuDNN 版本差异
   - 其他依赖库版本差异

---

## 9. 历史修改记录

| 日期 | 修改内容 |
|------|----------|
| 2026-02-19 | 复查并更新：移除已复原的差异（非键边距离条件、NUM_EPOCHS），确认 tensor_nan_inf 未被调用，更新总结表格 |
| (原日期) | 初始版本：识别所有重构前后差异 |
