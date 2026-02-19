from pathlib import Path
from typing import Dict, Any


def write_results(
    model_name: str,
    params: Dict[str, Any],
    roc_auc: float,
    std_dev: float,
    output_dir: Path = Path('outputs')
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    param_parts = []
    for key in ['dataset', 'num_layers', 'learning_rate']:
        if key in params:
            value = params[key]
            if isinstance(value, float):
                param_parts.append(f"{value:.6f}".rstrip('0').rstrip('.'))
            else:
                param_parts.append(str(value))
    
    param_suffix = "_".join(param_parts)
    
    output_file = output_dir / f"{model_name}.txt"
    
    with open(output_file, 'a') as f:
        f.write(f"{model_name}_{param_suffix}\n")
        f.write(f"ROC-AUC:{roc_auc:.4f}, STD_DEV:{std_dev:.4f}\n")
        f.write("\n")
