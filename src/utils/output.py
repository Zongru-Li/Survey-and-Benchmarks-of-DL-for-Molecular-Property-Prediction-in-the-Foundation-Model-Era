from pathlib import Path
from typing import Dict, Any


def get_output_model_name(model_name: str) -> str:
    name_mapping = {
        'ka_gnn': 'kagnn',
        'ka_gnn_two': 'kagnn_two',
    }
    return name_mapping.get(model_name, model_name)


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
    output_dir: Path = Path('output'),
    include_stddev: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = get_output_model_name(model_name)
    
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
    
    output_file = output_dir / f"{output_name}.txt"
    
    with open(output_file, 'a') as f:
        f.write(f"{output_name}_{param_suffix}\n")
        if include_stddev:
            f.write(f"{metric_name}:{metric_value:.4f}, STD_DEV:{std_dev:.4f}\n")
        else:
            f.write(f"{metric_name}:{metric_value:.4f}\n")
        f.write("\n")
