# AGENTS.md - KA-GNN Project Guidelines

Guidelines for AI coding agents working on KA-GNN (Kolmogorov-Arnold Graph Neural Networks).

## Project Structure

```
molecule/
├── src/models/         # Neural network implementations (ka_gnn, ka_gat, etc.)
├── src/utils/          # Config, data loading, training, checkpointing
├── src/run.py          # Main entry point
├── configs/            # YAML config files per model + common.py
├── scripts/            # Shell scripts for experiments
└── data/               # Raw datasets and processed cache
```

## Build/Lint/Test Commands

### Environment Setup
```bash
conda env create -f environment.yml
conda activate molecule-py311
```

### Running Experiments
```bash
# Smoke test (minimum 2 epochs)
python src/run.py --config configs/ka_gnn.yaml --dataset bace --epochs 2

# Full experiment with overrides
python src/run.py --config configs/ka_gnn.yaml --dataset bace --epochs 501 --lr 0.0001 --split scaffold
```

### CLI Options
`--config` (required): Path to YAML config
`--dataset` (required): bace, bbbp, clintox, sider, tox21, hiv, muv, adme_*
`--epochs`, `--lr`, `--batch-size`: Override training params
`--device`: cuda or cpu
`--split`: random, scaffold, or umap
`--checkpoint`: Resume from checkpoint path

### Linting
```bash
ruff check .
ruff check . --fix  # Auto-fix issues
```

## Configuration

Config files are YAML in `configs/`. Key settings:
```yaml
model_select: "ka_gnn"
pooling: 'avg'
num_layers: 4
LR: 0.0001
NUM_EPOCHS: 501
batch_size: 128
encoder_atom: "cgcnn"
encoder_bond: "dim_14"
```

Global settings in `configs/common.py`: `device`, `seed`, `cuda` config.

## Code Style Guidelines

### Import Order
```python
# 1. Standard library (alphabetical)
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# 2. Third-party (alphabetical by package)
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# 3. Local imports
from src.models import get_model
from src.utils.config import load_model_config
```

### Naming Conventions
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Type Hints
```python
def load_model_config(config_path: str) -> Dict[str, Any]:
    ...

def create_dataloader(config: Dict[str, Any], model_type: str = 'gnn') -> Tuple[DataLoader, DataLoader, DataLoader]:
    ...
```

### PyTorch Conventions
```python
class MyModel(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int):
        super().__init__()
        self.layer = nn.Linear(in_feat, hidden_feat)
    
    def forward(self, g, x):
        return self.layer(x)
```
- Models extend `nn.Module`; initialize layers in `__init__`, use in `forward`
- Move tensors to device: `.to(device)`; use `torch.no_grad()` for inference

### Error Handling
- Raise `ValueError` for invalid model/config names
- Return `False` for failed graph construction

### Formatting
- Line length: ~100 characters
- 4 spaces indentation (no tabs)

### Random Seeds
```python
def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
```

## Model Registry

Models registered in `src/models/__init__.py`:
- GNN: `ka_gnn`, `ka_gnn_two`, `mlp_sage`, `mlp_sage_two`, `kan_sage`, `kan_sage_two`
- GAT: `kagat`, `mlpgat`, `kangat`, `pogat`

## Supported Datasets

Classification: bace (1), bbbp (1), hiv (1), clintox (2), tox21 (12), muv (17), sider (27)
Regression: adme_hlm, adme_rlm, adme_mdr1, adme_sol, adme_hppb, adme_rppb (all 1 task)

## Common Tasks

### Adding a New Model
1. Create model file in `src/models/`
2. Add import and register in `src/models/__init__.py`
3. Create config YAML in `configs/`

### Adding a New Dataset
1. Place CSV in `data/` with `smiles` column
2. Add label getter in `src/utils/data.py`
3. Update `TARGET_MAP` with task count; add to `REGRESSION_DATASETS` if regression

## Language

- `assets/` markdown: Simplified Chinese
- Opencode CLI output: Simplified Chinese
- Code, comments, scripts, README: English

## Context7

Use Context7 MCP when library/API documentation is needed.
