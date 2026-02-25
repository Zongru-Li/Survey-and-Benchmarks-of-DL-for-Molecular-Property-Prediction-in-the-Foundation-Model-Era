import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.utils.splitters import (
    RandomSplitter,
    ScaffoldSplitter,
    UMAPSplitter,
    ButinaSplitter,
    TimeSplitter,
)


SPLITTER_MAP = {
    "random": RandomSplitter,
    "scaffold": ScaffoldSplitter,
    "umap": UMAPSplitter,
    "butina": ButinaSplitter,
    "time": TimeSplitter,
}

ADME_TARGETS = {
    "adme_hlm": 1,
    "adme_rlm": 1,
    "adme_mdr1": 1,
    "adme_sol": 1,
    "adme_hppb": 1,
    "adme_rppb": 1,
}

REGRESSION_DATASETS = {
    "adme_hlm",
    "adme_rlm",
    "adme_mdr1",
    "adme_sol",
    "adme_hppb",
    "adme_rppb",
}

TARGET_MAP = {
    "tox21": 12,
    "muv": 17,
    "sider": 27,
    "clintox": 2,
    "bace": 1,
    "bbbp": 1,
    "hiv": 1,
    **ADME_TARGETS,
}

ALLOWABLE_FEATURES = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
}


def is_adme_dataset(dataset_name: str) -> bool:
    return dataset_name in ADME_TARGETS


def is_regression_dataset(dataset_name: str) -> bool:
    return dataset_name in REGRESSION_DATASETS


def get_splitter(split_type: str = "scaffold"):
    if split_type not in SPLITTER_MAP:
        raise ValueError(
            f"Unknown split type: {split_type}. Available: {list(SPLITTER_MAP.keys())}"
        )
    return SPLITTER_MAP[split_type]()


def get_data_path(dataset_name: str, data_dir: str = "data") -> str:
    if is_adme_dataset(dataset_name):
        return os.path.join(data_dir, "ADME", dataset_name + ".csv")
    return os.path.join(data_dir, dataset_name + ".csv")


def get_dataset_labels(dataset_name: str) -> List[str]:
    if is_adme_dataset(dataset_name):
        return ["value"]
    if dataset_name == "tox21":
        return [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ]
    elif dataset_name == "muv":
        return [
            "MUV-466",
            "MUV-548",
            "MUV-600",
            "MUV-644",
            "MUV-652",
            "MUV-689",
            "MUV-692",
            "MUV-712",
            "MUV-713",
            "MUV-733",
            "MUV-737",
            "MUV-810",
            "MUV-832",
            "MUV-846",
            "MUV-852",
            "MUV-858",
            "MUV-859",
        ]
    elif dataset_name == "sider":
        return [
            "Hepatobiliary disorders",
            "Metabolism and nutrition disorders",
            "Product issues",
            "Eye disorders",
            "Investigations",
            "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders",
            "Social circumstances",
            "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions",
            "Endocrine disorders",
            "Surgical and medical procedures",
            "Vascular disorders",
            "Blood and lymphatic system disorders",
            "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders",
            "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders",
            "Psychiatric disorders",
            "Renal and urinary disorders",
            "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders",
            "Cardiac disorders",
            "Nervous system disorders",
            "Injury, poisoning and procedural complications",
        ]
    elif dataset_name == "clintox":
        return ["FDA_APPROVED", "CT_TOX"]
    else:
        return ["label"]


def get_target_dim(dataset_name: str) -> int:
    return TARGET_MAP.get(dataset_name, 1)


def mol_to_pyg_data(mol, labels=None):
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_type_idx = (
            ALLOWABLE_FEATURES["possible_atomic_num_list"].index(atom.GetAtomicNum())
            if atom.GetAtomicNum() in ALLOWABLE_FEATURES["possible_atomic_num_list"]
            else 0
        )
        chirality_idx = ALLOWABLE_FEATURES["possible_chirality_list"].index(
            atom.GetChiralTag()
        )
        atom_features.append([atom_type_idx, chirality_idx])

    if len(atom_features) == 0:
        return None

    x = torch.tensor(atom_features, dtype=torch.long)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()

        bond_type_idx = ALLOWABLE_FEATURES["possible_bonds"].index(bond.GetBondType())
        bond_dir_idx = ALLOWABLE_FEATURES["possible_bond_dirs"].index(bond.GetBondDir())

        edge_index.append([src, dst])
        edge_index.append([dst, src])
        edge_attr.append([bond_type_idx, bond_dir_idx])
        edge_attr.append([bond_type_idx, bond_dir_idx])

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    if labels is not None:
        y = torch.tensor(labels, dtype=torch.float)
    else:
        y = None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class PyGDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def pyg_collate_fn(batch):
    return Batch.from_data_list(batch)


def create_pyg_dataset(
    datafile: str,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    data_dir: str = "data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> bool:
    directory_path = os.path.join(data_dir, "processed_pyg")
    target_file_name = f"{datafile}_{split_type}.pth"

    if os.path.isfile(os.path.join(directory_path, target_file_name)):
        return True

    data_path = get_data_path(datafile, data_dir)
    df = pd.read_csv(data_path)
    label_cols = get_dataset_labels(datafile)
    smiles_list = df["smiles"]
    labels = df[label_cols]

    if datafile in ["tox21", "muv"]:
        labels = labels.fillna(0)

    data_list = []
    for i in range(len(smiles_list)):
        if i % 10000 == 0:
            print(f"[INFO] Processing molecule {i}/{len(smiles_list)}", flush=True)

        smiles = smiles_list[i]
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

        if mol is None:
            continue

        label_values = labels.iloc[i].values.tolist()
        pyg_data = mol_to_pyg_data(mol, label_values)

        if pyg_data is None:
            continue

        data_list.append([smiles, pyg_data.y, pyg_data])

    print("[INFO] Graph list was done!", flush=True)

    splitter = get_splitter(split_type)
    splits = splitter.split(
        data_list,
        frac_train=train_ratio,
        frac_valid=val_ratio,
        frac_test=test_ratio,
        seed=seed,
    )
    print(f"[INFO] Splitter ({split_type}) was done!", flush=True)

    train_data = [item[2] for item in splits[0]]
    train_label = [item[1] for item in splits[0]]

    valid_data = [item[2] for item in splits[1]]
    valid_label = [item[1] for item in splits[1]]

    test_data = [item[2] for item in splits[2]]
    test_label = [item[1] for item in splits[2]]

    os.makedirs(directory_path, exist_ok=True)
    torch.save(
        {
            "train_data": train_data,
            "train_label": train_label,
            "valid_data": valid_data,
            "valid_label": valid_label,
            "test_data": test_data,
            "test_label": test_label,
            "batch_size": batch_size,
            "shuffle": True,
            "split_type": split_type,
        },
        os.path.join(directory_path, target_file_name),
    )

    return True


def create_pyg_dataloader(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    datafile = config["data"]["dataset"]
    batch_size = config["training"]["batch_size"]
    train_ratio = config["training"]["train_ratio"]
    val_ratio = config["training"].get(
        "val_ratio", config["training"].get("vali_ratio", 0.1)
    )
    test_ratio = config["training"]["test_ratio"]
    data_dir = config.get("data_dir", "data")
    split_type = config["training"].get("split_type", "scaffold")
    seed = config.get("seed", 42)

    create_pyg_dataset(
        datafile,
        batch_size,
        train_ratio,
        val_ratio,
        test_ratio,
        data_dir,
        split_type,
        seed,
    )

    state = torch.load(
        os.path.join(data_dir, "processed_pyg", f"{datafile}_{split_type}.pth"),
        weights_only=False,
    )

    train_dataset = PyGDataset(state["train_data"])
    valid_dataset = PyGDataset(state["valid_data"])
    test_dataset = PyGDataset(state["test_data"])

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=state["shuffle"],
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    if val_ratio == 0.0:
        valid_loader = None
    else:
        valid_loader = PyGDataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=state["shuffle"],
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=state["shuffle"],
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    print(f"[INFO] PyG Dataset was loaded (split={split_type})!", flush=True)
    print(f"[INFO] Training set size: {len(train_dataset)}", flush=True)
    print(f"[INFO] Validation set size: {len(valid_dataset)}", flush=True)
    print(f"[INFO] Test set size: {len(test_dataset)}", flush=True)

    return train_loader, valid_loader, test_loader
