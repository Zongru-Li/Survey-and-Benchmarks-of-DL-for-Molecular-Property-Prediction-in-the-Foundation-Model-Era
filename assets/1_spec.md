# 模块化实验框架重构实施计划

## 1. 目标概述

将现有的 KA-GNN 和 KA-GAT 两个独立项目重构为一个统一的模块化实验框架，实现：
- 统一的数据管理
- 统一的模型管理
- 统一的配置管理（含 CUDA 通用配置）
- 规范的实验执行流程
- 标准化的结果输出与检查点管理

## 2. 目标目录结构

```
ka-gnn/
├── data/                          # 统一数据目录
│   ├── bace.csv
│   ├── bbbp.csv
│   ├── clintox.csv
│   ├── hiv.csv
│   ├── muv.csv
│   ├── sider.csv
│   ├── tox21.csv
│   └── processed/                 # 预处理后的缓存数据
│
├── src/                           # 源代码目录
│   ├── __init__.py
│   ├── run.py                    # 统一执行入口（scripts 调用此文件）
│   ├── models/                    # 所有模型
│   │   ├── __init__.py
│   │   ├── base.py               # 模型基类
│   │   ├── ka_gnn.py             # KA-GNN 模型
│   │   ├── ka_gnn_two.py         # KA-GNN-Two 模型
│   │   ├── mlp_sage.py           # MLP-SAGE 模型
│   │   ├── kan_sage.py           # KAN-SAGE 模型
│   │   ├── ka_gat.py             # KA-GAT 模型
│   │   ├── kan_gat.py            # KAN-GAT 模型
│   │   ├── mlp_gat.py            # MLP-GAT 模型
│   │   └── po_gat.py             # PO-GAT 模型
│   │
│   └── utils/                     # 工具函数（合并去重）
│       ├── __init__.py
│       ├── config.py             # 配置加载工具
│       ├── data.py               # 数据加载与处理
│       ├── graph.py              # 图构建与特征编码
│       ├── splitters.py          # 数据集划分
│       ├── training.py           # 训练与评估流程
│       ├── checkpoint.py         # 检查点保存与加载
│       └── output.py             # 结果输出工具
│
├── scripts/                       # 执行脚本目录
│   ├── run_ka_gnn.sh             # KA-GNN 实验脚本
│   ├── run_ka_gnn_two.sh         # KA-GNN-Two 实验脚本
│   ├── run_mlp_sage.sh           # MLP-SAGE 实验脚本
│   ├── run_kan_sage.sh           # KAN-SAGE 实验脚本
│   ├── run_ka_gat.sh             # KA-GAT 实验脚本
│   ├── run_kan_gat.sh            # KAN-GAT 实验脚本
│   ├── run_mlp_gat.sh            # MLP-GAT 实验脚本
│   ├── run_po_gat.sh             # PO-GAT 实验脚本
│   └── run_all.sh                # 批量运行所有实验
│
├── configs/                       # 配置文件目录
│   ├── common.py                 # CUDA 通用配置（Python 格式）
│   ├── ka_gnn.yaml               # KA-GNN 配置
│   ├── ka_gnn_two.yaml           # KA-GNN-Two 配置
│   ├── mlp_sage.yaml             # MLP-SAGE 配置
│   ├── kan_sage.yaml             # KAN-SAGE 配置
│   ├── ka_gat.yaml               # KA-GAT 配置
│   ├── kan_gat.yaml              # KAN-GAT 配置
│   ├── mlp_gat.yaml              # MLP-GAT 配置
│   └── po_gat.yaml               # PO-GAT 配置
│
├── outputs/                       # 实验结果输出目录
│   └── {model_name}.txt          # 每个模型的结果文件
│
├── tmp/                           # 临时文件目录
│   └── checkpoints/               # 模型检查点目录
│       ├── ka_gnn/               # KA-GNN 检查点
│       │   └── ka_gnn_{timestamp}_{dataset}_{lr}_{layers}.pth
│       ├── ka_gat/               # KA-GAT 检查点
│       │   └── ka_gat_{timestamp}_{dataset}_{lr}_{layers}.pth
│       └── ...                   # 其他模型检查点
│
├── assets/                        # 文档与资源
├── Source_data/                   # 原始数据（保留）
├── environment.yml               # Conda 环境配置
├── AGENTS.md                      # AI 代理指南
└── README.md                      # 项目说明
```

## 3. 实施步骤

### 阶段一：目录结构创建与数据迁移

#### 步骤 1.1：创建新目录结构
```bash
mkdir -p data/processed
mkdir -p src/models src/utils
mkdir -p scripts
mkdir -p configs
mkdir -p outputs
mkdir -p tmp/checkpoints
```

#### 步骤 1.2：迁移数据文件
- 将 `KA-GNN/data/*.csv` 复制到 `data/` 目录
- 创建 `data/processed/` 用于缓存处理后的数据
- 原地保留 `KA-GNN/` 和 `KA-GAT/` 直到重构完成

### 阶段二：工具函数模块化（src/utils/）

#### 步骤 2.1：创建模型基类 `src/models/base.py`
```python
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """所有 GNN 模型的基类"""
    
    @abstractmethod
    def forward(self, g, x):
        """前向传播"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回模型名称"""
        pass
```

#### 步骤 2.2：创建配置加载器 `src/utils/config.py`
- 统一的 YAML 配置加载
- 加载 `configs/common.py` 中的 CUDA 配置
- 参数验证
- 默认值处理
- 配置合并逻辑

```python
# src/utils/config.py
import yaml
from pathlib import Path
from typing import Dict, Any

def load_common_config() -> Dict[str, Any]:
    """加载 CUDA 通用配置"""
    common_path = Path("configs/common.py")
    if common_path.exists():
        config = {}
        with open(common_path, 'r') as f:
            exec(f.read(), {"__builtins__": {}}, config)
        return config
    return {}

def load_model_config(config_path: str) -> Dict[str, Any]:
    """加载模型配置并与通用配置合并"""
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    common_config = load_common_config()
    
    # 合并配置，模型配置优先
    merged = {**common_config, **model_config}
    return merged
```

#### 步骤 2.3：合并数据处理 `src/utils/data.py`
- 数据集加载函数
- 数据集获取函数（get_bace, get_bbbp 等）
- 批处理数据加载器创建

#### 步骤 2.4：合并图构建工具 `src/utils/graph.py`
- 原子特征编码（从 `graph_path.py` 提取）
- 键特征编码
- SMILES 转图
- `path_complex_mol` 函数

#### 步骤 2.5：合并数据集划分 `src/utils/splitters.py`
- `random_splitters`
- `scaffold_splitters`
- 其他划分策略

#### 步骤 2.6：创建训练工具 `src/utils/training.py`
- 统一训练循环
- 统一评估函数
- 早停机制
- 模型保存/加载

#### 步骤 2.7：创建检查点工具 `src/utils/checkpoint.py`
```python
# src/utils/checkpoint.py
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def get_checkpoint_path(model_name: str, params: Dict[str, Any]) -> Path:
    """
    生成检查点文件路径
    格式: tmp/checkpoints/{model_name}/{model_name}_{timestamp}_{param1}_{param2...}.pth
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建参数后缀
    param_parts = []
    for key in ['dataset', 'learning_rate', 'num_layers', 'batch_size']:
        if key in params:
            value = params[key]
            if isinstance(value, float):
                param_parts.append(f"{value:.6f}".rstrip('0').rstrip('.'))
            else:
                param_parts.append(str(value))
    
    param_suffix = "_".join(param_parts)
    filename = f"{model_name}_{timestamp}_{param_suffix}.pth"
    
    checkpoint_dir = Path("tmp/checkpoints") / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    return checkpoint_dir / filename

def save_checkpoint(model, optimizer, epoch: int, metrics: Dict[str, float], 
                    config: Dict[str, Any], path: Path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }, path)

def load_checkpoint(path: Path, model, optimizer=None):
    """加载模型检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics'], checkpoint['config']
```

#### 步骤 2.8：创建输出工具 `src/utils/output.py`
- 结果文件格式化输出
- 增量写入功能
- 格式：
  ```
  {model_name}_{param1}_{param2}...
  ROC-AUC:{roc_auc}, STD_DEV:{std_dev}
  
  ```

### 阶段三：模型迁移（src/models/）

#### 步骤 3.1：迁移 KA-GNN 系列模型
- `ka_gnn.py` → `src/models/ka_gnn.py`
- `kan_sage.py` → `src/models/kan_sage.py`
- `mlp_sage.py` → `src/models/mlp_sage.py`
- 继承 `BaseModel` 基类
- 适配新的数据加载接口

#### 步骤 3.2：迁移 KA-GAT 系列模型
- `ka_gat.py` → `src/models/ka_gat.py`
- `kan_gat.py` → `src/models/kan_gat.py`
- `mlp_gat.py` → `src/models/mlp_gat.py`
- `po_gat.py` → `src/models/po_gat.py`
- 继承 `BaseModel` 基类

### 阶段四：配置文件标准化（configs/）

#### 步骤 4.1：创建 CUDA 通用配置 `configs/common.py`
```python
# configs/common.py
# CUDA 相关配置 - 所有实验共用

# 设备配置
device = "cuda"  # "cuda" or "cpu"

# CUDA 具体设置
cuda = {
    # 是否使用 CUDA
    "use_cuda": True,
    
    # CUDA 设备 ID（多 GPU 时指定）
    "device_id": 0,
    
    # 确定性设置（保证可重复性）
    "deterministic": True,
    "benchmark": False,
    
    # 混合精度训练
    "amp": {
        "enabled": False,
        "dtype": "float16",  # "float16" or "bfloat16"
    },
    
    # 多 GPU 配置
    "distributed": {
        "enabled": False,
        "backend": "nccl",
    },
}

# 随机种子
seed = 42

# 日志级别
log_level = "INFO"
```

#### 步骤 4.2：通用模型配置模板（YAML）
每个模型配置文件 `configs/{model_name}.yaml` 应包含：
```yaml
# 模型配置
model:
  name: "ka_gnn"
  num_layers: 4
  hidden_dim: 128

# 数据配置
data:
  dataset: "bace"
  encoder_atom: "cgcnn"
  encoder_bond: "dim_14"
  force_field: "uff"

# 讓练配置
training:
  epochs: 500
  batch_size: 128
  learning_rate: 0.0001
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  iterations: 1
  
  # 早停配置
  early_stopping:
    enabled: True
    patience: 50
    min_delta: 0.0001

# 检查点配置
checkpoint:
  save_best: True
  save_last: True
  save_every_n_epochs: 0  # 0 表示不定期保存

# 输出配置
output:
  params: ["dataset", "num_layers", "learning_rate"]
```

#### 步骤 4.3：创建各模型配置文件
- 为每个模型创建独立配置文件
- 保持与原配置的兼容性

### 阶段五：统一入口程序（src/run.py）

#### 步骤 5.1：创建主入口 `src/run.py`
```python
# src/run.py
"""
模块化实验框架统一入口
scripts/ 中的 shell 脚本通过此文件执行实验
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.config import load_model_config, setup_device, setup_seed
from src.utils.data import create_dataloader
from src.utils.training import train_model
from src.utils.checkpoint import get_checkpoint_path, save_checkpoint
from src.utils.output import write_results
from src.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='KA-GNN 模块化实验框架')
    
    # 必需参数
    parser.add_argument('--config', type=str, required=True,
                        help='模型配置文件路径 (configs/xxx.yaml)')
    
    # 可选覆盖参数
    parser.add_argument('--dataset', type=str, default=None,
                        help='覆盖配置文件中的数据集')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置文件中的训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖配置文件中的学习率')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='覆盖配置文件中的批大小')
    parser.add_argument('--device', type=str, default=None,
                        help='覆盖 common.py 中的设备配置')
    parser.add_argument('--seed', type=int, default=None,
                        help='覆盖 common.py 中的随机种子')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='从检查点恢复训练')
    
    return parser.parse_args()


def override_config(config, args):
    """用命令行参数覆盖配置"""
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    return config


def main():
    args = parse_args()
    
    # 1. 加载配置（包含 common.py 的 CUDA 配置）
    config = load_model_config(args.config)
    config = override_config(config, args)
    
    # 2. 设置设备和随机种子
    device = setup_device(config)
    setup_seed(config.get('seed', 42))
    
    print(f"[INFO] Model: {config['model']['name']}")
    print(f"[INFO] Dataset: {config['data']['dataset']}")
    print(f"[INFO] Device: {device}")
    
    # 3. 创建模型
    model = get_model(config)
    model = model.to(device)
    
    # 4. 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloader(config)
    
    # 5. 训练模型
    best_model_state, roc_auc, std_dev = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        checkpoint_path=args.checkpoint
    )
    
    # 6. 保存最佳检查点
    if best_model_state is not None:
        checkpoint_path = get_checkpoint_path(
            model_name=config['model']['name'],
            params={
                'dataset': config['data']['dataset'],
                'learning_rate': config['training']['learning_rate'],
                'num_layers': config['model'].get('num_layers', 2),
                'batch_size': config['training']['batch_size'],
            }
        )
        save_checkpoint(
            model=model,
            optimizer=None,
            epoch=config['training']['epochs'],
            metrics={'roc_auc': roc_auc, 'std_dev': std_dev},
            config=config,
            path=checkpoint_path
        )
        print(f"[INFO] Checkpoint saved to: {checkpoint_path}")
    
    # 7. 输出结果
    write_results(
        model_name=config['model']['name'],
        params={
            'dataset': config['data']['dataset'],
            'num_layers': config['model'].get('num_layers', 2),
            'learning_rate': config['training']['learning_rate'],
        },
        roc_auc=roc_auc,
        std_dev=std_dev,
        output_dir=Path('outputs')
    )
    
    print(f"[INFO] ROC-AUC: {roc_auc:.4f}, STD_DEV: {std_dev:.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

### 阶段六：执行脚本（scripts/）

#### 步骤 6.1：创建单模型执行脚本模板
```bash
#!/bin/bash
# scripts/run_ka_gnn.sh
# KA-GNN 实验脚本

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 模型配置文件
CONFIG="configs/ka_gnn.yaml"

# 数据集列表
DATASETS=("bace" "bbbp" "clintox" "sider" "tox21" "hiv" "muv")

# 运行实验
for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Running ka_gnn on $dataset..."
    echo "=========================================="
    
    python src/run.py \
        --config "$CONFIG" \
        --dataset "$dataset"
    
    echo ""
done

echo "All experiments completed!"
```

#### 步骤 6.2：创建批量执行脚本
```bash
#!/bin/bash
# scripts/run_all.sh
# 运行所有模型的所有实验

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 所有模型脚本
SCRIPTS=(
    "scripts/run_ka_gnn.sh"
    "scripts/run_mlp_sage.sh"
    "scripts/run_kan_sage.sh"
    "scripts/run_ka_gat.sh"
    "scripts/run_kan_gat.sh"
    "scripts/run_mlp_gat.sh"
    "scripts/run_po_gat.sh"
)

for script in "${SCRIPTS[@]}"; do
    echo "=========================================="
    echo "Running $script..."
    echo "=========================================="
    bash "$script"
done

echo "All experiments completed!"
```

#### 步骤 6.3：创建带参数覆盖的脚本示例
```bash
#!/bin/bash
# scripts/run_ka_gnn_custom.sh
# 带自定义参数的 KA-GNN 实验脚本

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 可自定义参数
DATASET="${1:-bace}"
EPOCHS="${2:-500}"
LR="${3:-0.0001}"
DEVICE="${4:-cuda}"

echo "Running ka_gnn with:"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  Device: $DEVICE"

python src/run.py \
    --config configs/ka_gnn.yaml \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE"
```

### 阶段七：输出格式规范

#### 步骤 7.1：结果输出文件格式
每个实验结果追加到 `outputs/{model_name}.txt`：
```
ka_gnn_bace_4_0.0001
ROC-AUC:0.8567, STD_DEV:0.0234

ka_gnn_bbbp_4_0.0001
ROC-AUC:0.8912, STD_DEV:0.0187

```

第一行格式：`{model_name}_{param1}_{param2}_{param3}...`
第二行格式：`ROC-AUC:{value}, STD_DEV:{value}`
第三行：空行

#### 步骤 7.2：检查点命名格式
```
tmp/checkpoints/{model_name}/{model_name}_{timestamp}_{dataset}_{lr}_{layers}.pth
```

示例：
```
tmp/checkpoints/ka_gnn/ka_gnn_20260219_143052_bace_0.0001_4.pth
tmp/checkpoints/ka_gat/ka_gat_20260219_150321_bbbp_0.0001_2.pth
```

### 阶段八：清理与测试

#### 步骤 8.1：删除旧目录
```bash
# 确认新框架工作正常后
rm -rf KA-GNN/
rm -rf KA-GAT/
```

#### 步骤 8.2：更新文档
- 更新 `README.md`
- 更新 `AGENTS.md`
- 更新 `environment.yml`（如有需要）

#### 步骤 8.3：验证测试
- 每个模型运行一次完整实验
- 验证输出格式正确
- 验证检查点保存正确
- 验证结果可重复

## 4. 配置加载流程

```
┌─────────────────────────────────────────────────────────┐
│                   src/run.py                            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│         load_model_config(configs/ka_gnn.yaml)          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              加载 configs/common.py                      │
│  - device: "cuda"                                       │
│  - cuda.use_cuda: True                                  │
│  - cuda.deterministic: True                             │
│  - seed: 42                                             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              加载 configs/ka_gnn.yaml                    │
│  - model.name: "ka_gnn"                                 │
│  - data.dataset: "bace"                                 │
│  - training.learning_rate: 0.0001                       │
│  - ...                                                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              命令行参数覆盖                               │
│  --dataset, --epochs, --lr, --device, --seed            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              最终合并配置                                 │
└─────────────────────────────────────────────────────────┘
```

## 5. 代码复用策略

### 5.1 需要合并的重复代码
| 源文件 | 目标位置 | 说明 |
|--------|----------|------|
| KA-GNN/utils/graph_path.py | src/utils/graph.py | 合并两个版本 |
| KA-GAT/utils/graph_path.py | src/utils/graph.py | 合并两个版本 |
| KA-GNN/utils/splitters.py | src/utils/splitters.py | 相同代码 |
| KA-GAT/utils/splitters.py | src/utils/splitters.py | 相同代码 |
| KA-GNN/main.py 训练循环 | src/utils/training.py | 提取公共部分 |
| KA-GAT/main.py 训练循环 | src/utils/training.py | 提取公共部分 |
| CUDA 设备设置 | configs/common.py + src/utils/config.py | 统一管理 |

### 5.2 模型特定保留
- `KAN_linear` 类定义保留在各模型文件中
- 模型特定的超参数在配置文件中定义

## 6. 配置参数映射

### 6.1 旧配置 → 新配置

| 旧参数 | 新路径 | 说明 |
|--------|--------|------|
| model_select | model.name | 模型选择 |
| select_dataset | data.dataset | 数据集 |
| encoder_atom | data.encoder_atom | 原子编码 |
| encoder_bond | data.encoder_bond | 键编码 |
| force_field | data.force_field | 力场 |
| pooling | model.pooling | 池化方式 |
| loss_sclect | training.loss | 损失函数 |
| grid_feat / grid | model.grid_feat | 网格特征 |
| head | model.num_heads | 注意力头数 |
| num_layers | model.num_layers | 层数 |
| LR | training.learning_rate | 学习率 |
| NUM_EPOCHS | training.epochs | 训练轮数 |
| batch_size | training.batch_size | 批大小 |
| train_ratio | training.train_ratio | 训练比例 |
| vali_ratio | training.val_ratio | 验证比例 |
| test_ratio | training.test_ratio | 测试比例 |
| iter | training.iterations | 重复次数 |
| (CUDA 相关) | configs/common.py | CUDA 配置 |

### 6.2 新增配置项

| 配置项 | 位置 | 说明 |
|--------|------|------|
| device | common.py | 设备选择 |
| cuda.use_cuda | common.py | 是否使用 CUDA |
| cuda.deterministic | common.py | 确定性模式 |
| cuda.benchmark | common.py | cuDNN benchmark |
| cuda.amp.enabled | common.py | 混合精度训练 |
| seed | common.py | 随机种子 |
| checkpoint.save_best | {model}.yaml | 保存最佳模型 |
| checkpoint.save_last | {model}.yaml | 保存最后模型 |
| early_stopping.enabled | {model}.yaml | 早停开关 |

## 7. 执行优先级

1. **高优先级**：创建目录结构、迁移数据
2. **高优先级**：创建 configs/common.py CUDA 配置
3. **高优先级**：创建 src/utils/ 核心工具（含配置加载）
4. **高优先级**：创建 src/run.py 统一入口
5. **中优先级**：迁移模型到 src/models/
6. **中优先级**：创建模型配置文件
7. **中优先级**：创建执行脚本
8. **低优先级**：清理旧目录
9. **低优先级**：更新文档

## 8. 风险与注意事项

1. **数据兼容性**：确保数据加载方式不变
2. **随机种子**：保持实验可重复性
3. **GPU 支持**：CUDA 配置从 common.py 读取，确保正确使用
4. **依赖管理**：检查 pykan 等特定依赖
5. **渐进式迁移**：逐步迁移，每步验证
6. **检查点兼容性**：旧检查点格式可能需要转换
7. **配置优先级**：命令行参数 > 模型配置 > 通用配置

## 9. 时间估计

| 阶段 | 预计时间 |
|------|----------|
| 阶段一：目录创建与数据迁移 | 0.5 小时 |
| 阶段二：工具函数模块化 | 2.5 小时 |
| 阶段三：模型迁移 | 1.5 小时 |
| 阶段四：配置文件标准化 | 1 小时 |
| 阶段五：统一入口程序 | 1 小时 |
| 阶段六：执行脚本 | 1 小时 |
| 阶段七：输出格式规范 | 0.5 小时 |
| 阶段八：清理与测试 | 1 小时 |
| **总计** | **9 小时** |

## 10. 快速参考

### 10.1 命令示例
```bash
# 运行单个实验
python src/run.py --config configs/ka_gnn.yaml --dataset bace

# 覆盖参数
python src/run.py --config configs/ka_gnn.yaml --dataset bace --epochs 100 --lr 0.001

# 使用 CPU
python src/run.py --config configs/ka_gnn.yaml --device cpu

# 从检查点恢复
python src/run.py --config configs/ka_gnn.yaml --checkpoint tmp/checkpoints/ka_gnn/xxx.pth

# 批量运行
bash scripts/run_ka_gnn.sh
bash scripts/run_all.sh
```

### 10.2 目录快速说明
```
data/          → 数据文件
src/models/    → 模型定义
src/utils/     → 工具函数
src/run.py     → 统一入口
scripts/       → Shell 执行脚本
configs/       → 配置文件
  common.py    → CUDA 通用配置
  *.yaml       → 模型配置
outputs/       → 实验结果
tmp/checkpoints/ → 模型检查点
```
