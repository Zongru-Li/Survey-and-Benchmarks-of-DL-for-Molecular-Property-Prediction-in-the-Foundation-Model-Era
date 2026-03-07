# AGENTS.md - KA-GNN Project Guidelines

Guidelines for AI coding agents working on KA-GNN (Kolmogorov-Arnold Graph Neural Networks).

## Project Structure

```
molecule/
├── src/
│   ├── models/         # Neural network implementations
│   │   ├── __init__.py # Model registry (MODEL_REGISTRY, get_model)
│   │   ├── base.py     # BaseModel abstract class
│   │   ├── ka_gnn.py   # KA-GNN implementations
│   │   └── ...         # Other model files
│   ├── utils/
│   │   ├── config.py   # Config loading, device/seed setup
│   │   ├── data.py     # Dataset loading, DGL graph construction
│   │   ├── pyg_data.py # PyG dataloader for PyG models
│   │   ├── training.py # Training loops, loss functions
│   │   ├── checkpoint.py # Model checkpointing
│   │   ├── output.py   # Results writing
│   │   ├── graph.py    # Molecule to graph conversion
│   │   └── splitters.py # Dataset splitting strategies
│   └── run.py          # Main entry point
├── configs/
│   ├── common.py       # Global settings (device, seed, cuda)
│   └── *.yaml          # Per-model config files
├── scripts/            # Shell scripts for experiments
├── data/               # Raw datasets and processed cache
├── output/             # Training results (CSV)
└── tmp/checkpoints/    # Saved model checkpoints
```

## Build/Lint/Test Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Smoke test (quick validation, 2 epochs)
python src/run.py --config configs/ka_gnn.yaml --dataset bace --epochs 2

# Standard experiment
python src/run.py --config configs/ka_gnn.yaml --dataset bace --epochs 100

# Full experiment with all overrides
python src/run.py --config configs/ka_gnn.yaml --dataset bace --epochs 501 --lr 0.0001 --batch-size 128 --split scaffold --device cuda

# CPU-only run
python src/run.py --config configs/ka_gnn.yaml --dataset bace --epochs 2 --device cpu

# With checkpoint saving
python src/run.py --config configs/ka_gnn.yaml --dataset bace --save-checkpoint

# With standard deviation output
python src/run.py --config configs/ka_gnn.yaml --dataset bace --stddev
```

### CLI Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--config` | Yes | Path to YAML config file |
| `--dataset` | Yes | Dataset name |
| `--epochs` | No | Override training epochs |
| `--lr` | No | Override learning rate |
| `--batch-size` | No | Override batch size |
| `--device` | No | cuda or cpu |
| `--seed` | No | Random seed |
| `--split` | No | Split type: random, scaffold, umap, butina, time |
| `--save-checkpoint` | No | Save best model checkpoint |
| `--stddev` | No | Include std dev in output |

### Linting

```bash
ruff check .
ruff check . --fix  # Auto-fix issues
```

## Datasets

### Classification (ROC-AUC metric)
| Dataset | Tasks | Description |
|---------|-------|-------------|
| bace | 1 | BACE-1 inhibitor |
| bbbp | 1 | Blood-brain barrier penetration |
| hiv | 1 | HIV inhibition |
| clintox | 2 | Clinical toxicity |
| tox21 | 12 | Toxicity targets |
| muv | 17 | Maximum unbiased validation |
| sider | 27 | Side effects |

### Regression (PearsonR metric)
| Dataset | Tasks |
|---------|-------|
| adme_hlm | 1 |
| adme_rlm | 1 |
| adme_mdr1 | 1 |
| adme_sol | 1 |
| adme_hppb | 1 |
| adme_rppb | 1 |

## Model Registry

Models are registered in `src/models/__init__.py`:

```python
from src.models import get_model, MODEL_REGISTRY

model = get_model(config)  # Raises ValueError if unknown
```

### Model Categories

**DGL GNN Models** (use `create_dataloader` with `model_type='gnn'`):
- `ka_gnn`, `ka_gnn_two`, `mlp_sage`, `mlp_sage_two`, `kan_sage`, `kan_sage_two`
- `dmpnn`, `attentivefp`, `mol_gdl`
- `ngram_rf`, `ngram_xgb` (non-neural baselines)

**DGL GAT Models** (use `create_dataloader` with `model_type='gat'`):
- `kagat`, `mlpgat`, `kangat`, `pogat`

**PyG Models** (use `create_pyg_dataloader`):
- `pretrain_gnn`, `graphmvp`, `molclr_gcn`, `molclr_gin`
- `graphkan`, `gin`, `gcn`, `grover`, `cd_mvgnn`

## Configuration

### YAML Config Structure

```yaml
model_select: "ka_gnn"      # Model name (required)
force_field: 'uff'          # Molecular force field
encoder_atom: "cgcnn"       # Atom encoder
encoder_bond: "dim_14"      # Bond encoder
pooling: 'avg'              # Pooling: avg, sum, max, attention
loss_sclect: 'bce'          # Loss: bce, l1, l2, sml1
grid_feat: 1                # Grid features for KAN layers
num_layers: 4               # Number of GNN layers
LR: 0.0001                  # Learning rate
NUM_EPOCHS: 501             # Training epochs
batch_size: 128             # Batch size
train_ratio: 0.8            # Training split
vali_ratio: 0.1             # Validation split
test_ratio: 0.1             # Test split
iter: 1                     # Number of iterations
```

### Common Settings (`configs/common.py`)

```python
device = "cuda"
seed = 42
log_level = "INFO"
cuda = {
    "use_cuda": True,
    "device_id": 0,
    "deterministic": True,
    "benchmark": False,
}
```

## Code Style Guidelines

### Import Order

```python
# 1. Standard library (alphabetical)
import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# 2. Third-party (alphabetical)
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dgl.nn import AvgPooling, SumPooling, MaxPooling
from logzero import logger
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

# 3. Local imports (alphabetical)
from src.models import get_model, is_pyg_model, PYG_MODELS
from src.utils.config import load_model_config, setup_device, setup_seed
from src.utils.data import create_dataloader, get_target_dim, is_regression_dataset
```

### Naming Conventions

- **Functions/Variables**: `snake_case` (e.g., `train_loader`, `get_model`)
- **Classes**: `PascalCase` (e.g., `KA_GNN`, `CustomDataset`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MODEL_REGISTRY`, `TARGET_MAP`)
- **Private methods**: `_leading_underscore` (e.g., `_create_dataset_from_time_split`)
- **Config keys**: Legacy YAML uses `UPPER_CASE` (e.g., `LR`, `NUM_EPOCHS`), internal uses `snake_case`

### Type Hints

```python
from typing import Dict, Any, Optional, Tuple, List, Callable

def load_model_config(config_path: str) -> Dict[str, Any]:
    ...

def create_dataloader(
    config: Dict[str, Any],
    model_type: str = 'gnn'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ...

def train_model(
    model_factory: Callable,
    train_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    task_type: str = 'classification',
) -> Tuple[Optional[Dict], float, float]:
    ...
```

### PyTorch Conventions

```python
class MyModel(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, out_feat: int):
        super().__init__()
        self.layer1 = nn.Linear(in_feat, hidden_feat)
        self.layer2 = nn.Linear(hidden_feat, out_feat)
    
    def forward(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x
            # Message passing
            return self.layer2(F.relu(self.layer1(x)))

# Device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
tensor = tensor.to(device)

# Inference mode
with torch.no_grad():
    output = model(g, x)
```

### Error Handling

```python
# Invalid model/config names
if model_name not in MODEL_REGISTRY:
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

# File not found
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Time-split train file not found: {train_path}")

# Failed graph construction (return False, don't raise)
if Graph_list is False:
    return False
```

### Formatting

- Line length: ~100 characters
- 4 spaces indentation (no tabs)
- Blank lines between import groups
- Trailing commas in multi-line collections

### Logging

```python
print(f"[INFO] Model: {model_name}", flush=True)
print(f"[INFO] Dataset was loaded (split={split_type})!", flush=True)
```

### Random Seeds

```python
def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
```

## Common Tasks

### Adding a New Model

1. Create model file in `src/models/` extending `BaseModel`:
   ```python
   from src.models.base import BaseModel
   
   class MyModel(BaseModel):
       def __init__(self, in_feat: int, hidden_feat: int, ...):
           super().__init__()
           # Define layers
       
       def forward(self, g, x):
           # Forward pass
   ```

2. Register in `src/models/__init__.py`:
   ```python
   from src.models.my_model import MyModel
   
   MODEL_REGISTRY = {
       ...
       'my_model': MyModel,
   }
   ```

3. Add model instantiation logic in `get_model()` function

4. Create config YAML in `configs/my_model.yaml`

### Adding a New Dataset

1. Place CSV in `data/` with `smiles` column and label columns

2. Add label getter in `src/utils/data.py`:
   ```python
   def get_new_dataset_labels():
       return ["label1", "label2"]
   
   def get_dataset_labels(dataset_name: str) -> List[str]:
       ...
       elif dataset_name == "new_dataset":
           return get_new_dataset_labels()
   ```

3. Update `TARGET_MAP` with task count:
   ```python
   TARGET_MAP = {
       ...
       "new_dataset": 2,
   }
   ```

4. Add to `REGRESSION_DATASETS` if regression task

### Adding a New Split Type

1. Create splitter in `src/utils/splitters.py`:
   ```python
   class MySplitter:
       def split(self, data_list, frac_train, frac_valid, frac_test, seed):
           # Implementation
           return train_data, valid_data, test_data
   ```

2. Register in `src/utils/data.py`:
   ```python
   SPLITTER_MAP = {
       ...
       "my_split": MySplitter,
   }
   ```

3. Update CLI choices in `configs/common.py`

## References

Projects under `references/` are open-source references. Explore when developing or modifying models/methods, but **DO NOT directly import them**. Use online search if needed.

- [cheminfo](https://github.com/PatWalters/practical_cheminformatics_posts)
- [deepchem](https://github.com/deepchem/deepchem)

## Language

- `assets/` markdown: Simplified Chinese
- Opencode CLI output: Simplified Chinese
- Code, comments, scripts, README: English

## Context7

Use Context7 MCP when library/API documentation is needed.
