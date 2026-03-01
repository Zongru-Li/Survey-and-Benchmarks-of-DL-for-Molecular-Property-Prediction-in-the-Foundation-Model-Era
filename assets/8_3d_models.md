# 3D Molecular Models Analysis

本项目中所有模型均为2D图神经网络，仅使用分子拓扑结构特征（原子特征、键特征），**不使用3D构象坐标**。

尽管 `src/utils/graph.py` 中的 `atom_to_graph()` 函数会生成3D构象并存储在 `g.ndata["coor"]` 中，但所有模型的 `forward()` 方法均未使用这些3D坐标。

## 模型汇总表

| Model | Conformer Generation (Y/N) | Number of Conformers | Protein-Ligand Standardized (Y/N) |
|-------|---------------------------|---------------------|-----------------------------------|
| AttentiveFP | N | 0 | N |
| N-GramRF | N | 0 | N |
| N-GramXGB | N | 0 | N |
| PretrainGNN | N | 0 | N |
| GraphMVP | N | 0 | N |
| MolCLR-GCN | N | 0 | N |
| MolCLR-GIN | N | 0 | N |
| Mol-GDL | N | 0 | N |
| GraphKAN | N | 0 | N |
| KA-GCN | N | 0 | N |
| KA-GAT | N | 0 | N |
| CD-MVGNN | N | 0 | N |
| GROVER | N | 0 | N |
| GCN | N | 0 | N |
| GIN | N | 0 | N |

## 说明

1. **Conformer Generation**: 所有模型均为2D GNN，不依赖3D构象信息
2. **Number of Conformers**: 不适用（N/A），因为不使用3D信息
3. **Protein-Ligand Standardized**: 本项目仅处理小分子（SMILES字符串），不涉及蛋白质-配体复合物结构

## 数据预处理中的3D构象生成

虽然模型不使用3D信息，但 `src/utils/graph.py:122-146` 中的 `atom_to_graph()` 函数仍会生成3D构象：

```python
AllChem.EmbedMolecule(mol)  # ETKDG方法生成初始3D坐标
AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))  # MMFF力场优化
```

这些3D坐标目前仅用于计算非键相互作用的边特征（距离阈值≤5Å的原子对），但主要模型输入仍为2D拓扑特征。
