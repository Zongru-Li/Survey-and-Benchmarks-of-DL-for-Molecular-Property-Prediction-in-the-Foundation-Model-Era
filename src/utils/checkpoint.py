import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def get_checkpoint_path(model_name: str, params: Dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics,
        'config': config,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics'], checkpoint['config']
