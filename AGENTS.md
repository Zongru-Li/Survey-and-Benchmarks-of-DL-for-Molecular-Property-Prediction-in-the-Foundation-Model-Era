# AGENTS.md - KA-GNN Project Guidelines

This document provides guidelines for AI coding agents working on the KA-GNN (Kolmogorov-Arnold Graph Neural Networks) project.

## Project Overview

KA-GNN is a molecular property prediction framework implementing Kolmogorov-Arnold Graph Neural Networks. The project contains two main implementations:
- **KA-GNN/**: Graph Convolutional Network variant (KA-GCN)
- **KA-GAT/**: Graph Attention Network variant (KA-GAT)

## Language

The description markdown files generated to @assets/ should use Simplified Chinese.
Opencode CLI output, like explainations or progress monitoring printed in terminal, should use Simplified Chinese.
ALL other generated file, including but not limited to python inline comments, Jupyter Notebooks, Shell Scripts, Readme.md should be written in English.

## Context7

Always use Context7 MCP when I need library/API documentation, code generation, setup or configuration steps without me having to explicitly ask.

## Build/Lint/Test Commands

### Environment Setup
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate molecule-py311
```

### Running the Code
```bash
# Run KA-GNN (from KA-GNN directory)
cd KA-GNN && python main.py

# Run KA-GAT (from KA-GAT directory)
cd KA-GAT && python main.py
```

### Configuration
Experiments are configured via YAML files:
- `KA-GNN/config/c_path.yaml` - KA-GNN configuration
- `KA-GAT/config/gat_path.yaml` - KA-GAT configuration

Key configuration options:
- `model_select`: Model type (ka_gnn, mlp_sage, kan_sage, kagat, kangat, mlpgat)
- `select_dataset`: Dataset (bace, bbbp, clintox, sider, tox21, hiv, muv)
- `NUM_EPOCHS`: Training epochs
- `batch_size`: Batch size for training
- `LR`: Learning rate

### Testing
No formal test suite exists. To verify code works:
```bash
# Quick smoke test with minimal epochs
cd KA-GNN && python main.py  # Uses config defaults
```

### Linting (Recommended)
```bash
# If adding linting, use ruff or flake8
ruff check .
# Or with auto-fix
ruff check . --fix
```

## Code Style Guidelines

### Import Organization
Organize imports in three groups, separated by blank lines:
1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical by package)
3. Local imports

```python
# Standard library
import argparse
import os
import random
import time

# Third-party
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# Local
from model.ka_gnn import KA_GNN
from utils.graph_path import path_complex_mol
```

### Naming Conventions
- **Functions/Variables**: `snake_case` (e.g., `creat_data`, `train_loss`, `batch_size`)
- **Classes**: `PascalCase` (e.g., `KA_GNN`, `CustomDataset`, `KAN_linear`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `NUM_EPOCHS`, `LR`)
- **Private methods**: Prefix with underscore (e.g., `_internal_method`)

### File Naming
- Module files: `snake_case.py` (e.g., `ka_gnn.py`, `graph_path.py`)
- Config files: `snake_case.yaml`

### Type Hints
Type hints are used sparingly. When adding new code, consider adding type hints for function signatures:

```python
def train(model: nn.Module, device: torch.device, train_loader: DataLoader, 
          valid_loader: DataLoader, optimizer: torch.optim.Optimizer, 
          epoch: int) -> tuple[float, float]:
    ...
```

### Docstrings
Use simple docstrings for public functions:

```python
def get_tox():
    """Get that default sider task names and return the side results for the drug."""
    return ['NR-AR', 'NR-AR-LBD', ...]
```

### PyTorch/Neural Network Conventions
- Models extend `nn.Module`
- Use `super().__init__()` in constructors
- Initialize layers in `__init__`, use them in `forward`
- Move tensors to device explicitly: `.to(device)`
- Use `torch.no_grad()` context for inference

```python
class MyModel(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int):
        super().__init__()
        self.layer = nn.Linear(in_feat, hidden_feat)
    
    def forward(self, g, x):
        return self.layer(x)
```

### Error Handling
- Return `False` for failed graph construction rather than raising exceptions
- Use `try/except` for external operations (file I/O, molecule processing)
- Print informative messages for missing configurations

```python
def atom_to_graph(smiles, encoder_atom, encoder_bond):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ...
```

### Code Formatting
- Maximum line length: ~100 characters
- Use 4 spaces for indentation (no tabs)
- Single space around operators
- Blank lines between logical sections

### Comments
- Add comments for complex logic (especially Fourier transform operations)
- Avoid obvious comments
- Use TODO/FIXME for future work markers

### Data Handling
- Processed data cached in `data/processed/` as `.pth` files
- Use `torch.save()` and `torch.load()` for caching
- Check if processed file exists before regenerating

```python
if is_file_in_directory(directory_path, target_file_name):
    return True
```

### Random Seeds
Always set random seeds for reproducibility:

```python
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
```

## Project-Specific Notes

### Supported Datasets
- BACE, BBBP, ClinTox, SIDER, Tox21, HIV, MUV (from MoleculeNet)

### Model Variants
**KA-GNN models:**
- `ka_gnn`, `ka_gnn_two`: KAN-based GNN
- `mlp_sage`, `mlp_sage_two`: MLP-based GraphSAGE
- `kan_sage`, `kan_sage_two`: KAN GraphSAGE

**KA-GAT models:**
- `kagat`: KA-GAT
- `kangat`: KAN-GAT
- `mlpgat`: MLP-GAT
- `pogat`: PO-GAT

### Key Dependencies
- PyTorch (2.10.0+cu128)
- DGL (2.4.0+cu124)
- RDKit (rdkit-pypi)
- pykan (0.2.8)
- scikit-learn, pandas, numpy

### GPU Support
Code automatically detects and uses CUDA if available:
```python
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

## Common Tasks

### Adding a New Model
1. Create model file in `model/` directory
2. Import in `main.py`
3. Add model selection logic in training loop
4. Update config YAML with new option

### Adding a New Dataset
1. Place CSV in `data/` directory with `smiles` column
2. Add task name getter function (e.g., `get_new_dataset()`)
3. Add dataset mapping in `creat_data()` function
4. Update `target_map` dictionary with task count

### Modifying Graph Features
Edit `utils/graph_path.py`:
- Atom features: `get_node_attributes()` from jarvis-tools
- Bond features: `encode_bond_14()` or `encode_bond_26()`
