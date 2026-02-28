import yaml
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any


def load_common_config() -> Dict[str, Any]:
    common_path = Path("configs/common.py")
    if common_path.exists():
        config = {}
        with open(common_path, 'r') as f:
            exec(f.read(), {"__builtins__": {}}, config)
        config = {k: v for k, v in config.items() if not callable(v)}
        return config
    return {}


def load_model_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    common_config = load_common_config()
    
    merged = {**common_config, **model_config}
    
    if 'model' not in merged:
        merged['model'] = {}
    if 'data' not in merged:
        merged['data'] = {}
    if 'training' not in merged:
        merged['training'] = {}
    
    if 'model_select' in model_config:
        merged['model']['name'] = model_config['model_select']
    if 'encoder_atom' in model_config:
        merged['data']['encoder_atom'] = model_config['encoder_atom']
    if 'encoder_bond' in model_config:
        merged['data']['encoder_bond'] = model_config['encoder_bond']
    if 'force_field' in model_config:
        merged['data']['force_field'] = model_config['force_field']
    if 'pooling' in model_config:
        merged['model']['pooling'] = model_config['pooling']
    if 'loss_sclect' in model_config:
        merged['training']['loss'] = model_config['loss_sclect']
    if 'grid_feat' in model_config:
        merged['model']['grid_feat'] = model_config['grid_feat']
    if 'grid' in model_config:
        merged['model']['grid_feat'] = model_config['grid']
    if 'head' in model_config:
        merged['model']['num_heads'] = model_config['head']
    if 'num_layers' in model_config:
        merged['model']['num_layers'] = model_config['num_layers']
    if 'dropout_ratio' in model_config:
        merged['model']['dropout_ratio'] = model_config['dropout_ratio']
    if 'feat_dim' in model_config:
        merged['model']['feat_dim'] = model_config['feat_dim']
    if 'LR' in model_config:
        merged['training']['learning_rate'] = model_config['LR']
    if 'NUM_EPOCHS' in model_config:
        merged['training']['epochs'] = model_config['NUM_EPOCHS']
    if 'batch_size' in model_config:
        merged['training']['batch_size'] = model_config['batch_size']
    if 'train_ratio' in model_config:
        merged['training']['train_ratio'] = model_config['train_ratio']
    if 'vali_ratio' in model_config:
        merged['training']['val_ratio'] = model_config['vali_ratio']
    if 'test_ratio' in model_config:
        merged['training']['test_ratio'] = model_config['test_ratio']
    if 'iter' in model_config:
        merged['training']['iterations'] = model_config['iter']
    if 'hidden_feat' in model_config:
        merged['model']['hidden_feat'] = model_config['hidden_feat']
    if 'out_feat' in model_config:
        merged['model']['out_feat'] = model_config['out_feat']
    if 'K' in model_config:
        merged['model']['K'] = model_config['K']
    if 'grid_size' in model_config:
        merged['model']['grid_size'] = model_config['grid_size']
    if 'spline_order' in model_config:
        merged['model']['spline_order'] = model_config['spline_order']
    if 'num_mlp_layers' in model_config:
        merged['model']['num_mlp_layers'] = model_config['num_mlp_layers']
    if 'learn_eps' in model_config:
        merged['model']['learn_eps'] = model_config['learn_eps']
    if 'neighbor_pooling_type' in model_config:
        merged['model']['neighbor_pooling_type'] = model_config['neighbor_pooling_type']
    if 'JK' in model_config:
        merged['model']['JK'] = model_config['JK']
    
    return merged


def setup_device(config: Dict[str, Any]) -> torch.device:
    device_str = config.get('device', 'cuda')
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print('[INFO] Using GPU...')
    else:
        device = torch.device('cpu')
        print('[INFO] Using CPU...')
    return device


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
