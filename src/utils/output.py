from pathlib import Path
from typing import Dict, Any


def format_param_value(value: Any) -> str:
    if isinstance(value, float):
        formatted = f"{value:.6f}".rstrip('0').rstrip('.')
        return formatted if formatted else '0'
    return str(value)


def write_results(
    model_name: str,
    params: Dict[str, Any],
    metric_value: float,
    std_dev: float,
    metric_name: str = 'ROC-AUC',
    output_dir: Path = Path('outputs')
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    key_mapping = {
        'dataset': 'data',
        'split': 'split',
        'num_layers': 'nl',
        'learning_rate': 'lr',
        'batch_size': 'bs',
        'epochs': 'ep',
        'iterations': 'iter',
    }
    
    param_parts = []
    for key in ['dataset', 'split', 'num_layers', 'learning_rate', 'batch_size', 'epochs', 'iterations']:
        if key in params:
            short_key = key_mapping.get(key, key)
            value = format_param_value(params[key])
            param_parts.append(f"{short_key}:{value}")
    
    param_suffix = "_".join(param_parts)
    
    output_file = output_dir / f"{model_name}.txt"
    
    with open(output_file, 'a') as f:
        f.write(f"{model_name}_{param_suffix}\n")
        f.write(f"{metric_name}:{metric_value:.4f}, STD_DEV:{std_dev:.4f}\n")
        f.write("\n")
