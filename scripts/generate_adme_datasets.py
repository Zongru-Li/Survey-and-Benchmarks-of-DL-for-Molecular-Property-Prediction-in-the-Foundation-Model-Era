#!/usr/bin/env python3
"""
Generate ADME regression datasets from source ADME.csv.

Output format: smiles, value

Usage:
    python scripts/generate_adme_datasets.py
"""

import pandas as pd
from pathlib import Path

ADME_CONFIG = {
    'adme_sol': {
        'column': 'LOG SOLUBILITY PH 6.8 (ug/mL)',
        'description': 'Log solubility at pH 6.8'
    },
    'adme_mdr1': {
        'column': 'LOG MDR1-MDCK ER (B-A/A-B)',
        'description': 'Log efflux ratio (ER)'
    },
    'adme_hlm': {
        'column': 'LOG HLM_CLint (mL/min/kg)',
        'description': 'Log human liver microsome intrinsic clearance'
    },
    'adme_rlm': {
        'column': 'LOG RLM_CLint (mL/min/kg)',
        'description': 'Log rat liver microsome intrinsic clearance'
    },
    'adme_hppb': {
        'column': 'LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)',
        'description': 'Log human plasma protein binding (% unbound)'
    },
    'adme_rppb': {
        'column': 'LOG PLASMA PROTEIN BINDING (RAT) (% unbound)',
        'description': 'Log rat plasma protein binding (% unbound)'
    }
}


def generate_adme_datasets():
    project_root = Path(__file__).parent.parent
    source_path = project_root / 'data' / 'ADME.csv'
    output_dir = project_root / 'data' / 'ADME'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(source_path)
    
    print(f"[INFO] Loading source data from: {source_path}")
    print(f"[INFO] Total compounds: {len(df)}")
    print()
    print("=" * 70)
    print("ADME Regression Datasets")
    print("=" * 70)
    print()
    
    for dataset_name, config in ADME_CONFIG.items():
        column = config['column']
        
        filtered = df[['SMILES', column]].dropna()
        filtered = filtered[['SMILES', column]]
        filtered.columns = ['smiles', 'value']
        
        output_path = output_dir / f'{dataset_name}.csv'
        filtered.to_csv(output_path, index=False)
        
        value_mean = filtered['value'].mean()
        value_std = filtered['value'].std()
        value_min = filtered['value'].min()
        value_max = filtered['value'].max()
        
        print(f"{dataset_name}:")
        print(f"  Column: {column}")
        print(f"  Description: {config['description']}")
        print(f"  Samples: {len(filtered)}")
        print(f"  Value range: [{value_min:.4f}, {value_max:.4f}]")
        print(f"  Mean: {value_mean:.4f}, Std: {value_std:.4f}")
        print(f"  Output: {output_path}")
        print()
    
    print("[INFO] All ADME datasets generated successfully!")


if __name__ == '__main__':
    generate_adme_datasets()
