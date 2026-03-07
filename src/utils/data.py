import os
import random
import torch
import dgl
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from src.utils.graph import path_complex_mol
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

ADME_TIME_FILE_MAP = {
    "adme_hlm": "ADME_HLM",
    "adme_rlm": "ADME_RLM",
    "adme_mdr1": "ADME_MDR1_ER",
    "adme_sol": "ADME_Sol",
    "adme_hppb": "ADME_hPPB",
    "adme_rppb": "ADME_rPPB",
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


def get_adme_time_paths(dataset_name: str, data_dir: str = "data") -> Tuple[str, str]:
    file_prefix = ADME_TIME_FILE_MAP.get(dataset_name)
    if file_prefix is None:
        raise ValueError(f"Dataset {dataset_name} does not have time-split files")
    time_dir = os.path.join(data_dir, "ADME_Time")
    train_path = os.path.join(time_dir, f"{file_prefix}_train.csv")
    test_path = os.path.join(time_dir, f"{file_prefix}_test.csv")
    return train_path, test_path


def get_label():
    return ["label"]


def get_tox():
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


def get_clintox():
    return ["FDA_APPROVED", "CT_TOX"]


def get_sider():
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


def get_muv():
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


def get_dataset_labels(dataset_name: str) -> List[str]:
    if is_adme_dataset(dataset_name):
        return ["value"]
    if dataset_name == "tox21":
        return get_tox()
    elif dataset_name == "muv":
        return get_muv()
    elif dataset_name == "sider":
        return get_sider()
    elif dataset_name == "clintox":
        return get_clintox()
    else:
        return get_label()


def get_target_dim(dataset_name: str) -> int:
    return TARGET_MAP.get(dataset_name, 1)


def has_node_with_zero_in_degree(graph):
    if (graph.in_degrees() == 0).any():
        return True
    return False


def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list, use_gnn: bool = True):
        self.labels = label_list
        self.graphs = graph_list
        self.use_gnn = use_gnn
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index].to(self.device)
        graph = self.graphs[index].to(self.device)
        return label, graph


def collate_fn(batch):
    labels, graphs = zip(*batch)
    labels = torch.stack(labels)
    batched_graph = dgl.batch(graphs)
    return labels, batched_graph


def collate_fn_gat(batch):
    labels, graphs = zip(*batch)
    labels = torch.stack(labels)
    batched_graph = dgl.batch(graphs)
    return labels, batched_graph


def update_node_features(g):
    def message_func(edges):
        return {"feat": edges.data["feat"]}

    def reduce_func(nodes):
        num_edges = nodes.mailbox["feat"].size(1)
        agg_feats = torch.sum(nodes.mailbox["feat"], dim=1) / num_edges
        return {"agg_feats": agg_feats}

    g.send_and_recv(g.edges(), message_func, reduce_func)
    g.ndata["feat"] = torch.cat((g.ndata["feat"], g.ndata["agg_feats"]), dim=1)
    return g


def create_dataset(
    datafile: str,
    encoder_atom: str,
    encoder_bond: str,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    data_dir: str = "data",
    split_type: str = "scaffold",
    seed: int = 42,
) -> bool:
    directory_path = os.path.join(data_dir, "processed")
    target_file_name = f"{datafile}_{split_type}.pth"

    if is_file_in_directory(directory_path, target_file_name):
        return True

    if split_type == "time":
        return _create_dataset_from_time_split(
            datafile,
            encoder_atom,
            encoder_bond,
            batch_size,
            val_ratio,
            data_dir,
            directory_path,
            target_file_name,
        )

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
        Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond)

        if Graph_list is False:
            continue

        if has_node_with_zero_in_degree(Graph_list):
            continue

        data_list.append(
            [
                smiles,
                torch.tensor(labels.iloc[i].values, dtype=torch.float32),
                Graph_list,
            ]
        )

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

    train_label = [item[1] for item in splits[0]]
    train_graph_list = [item[2] for item in splits[0]]

    valid_label = [item[1] for item in splits[1]]
    valid_graph_list = [item[2] for item in splits[1]]

    test_label = [item[1] for item in splits[2]]
    test_graph_list = [item[2] for item in splits[2]]

    os.makedirs(directory_path, exist_ok=True)
    torch.save(
        {
            "train_label": train_label,
            "train_graph_list": train_graph_list,
            "valid_label": valid_label,
            "valid_graph_list": valid_graph_list,
            "test_label": test_label,
            "test_graph_list": test_graph_list,
            "batch_size": batch_size,
            "shuffle": True,
            "split_type": split_type,
        },
        os.path.join(directory_path, target_file_name),
    )

    return True


def _process_dataframe_to_data_list(
    df: pd.DataFrame,
    label_col: str,
    encoder_atom: str,
    encoder_bond: str,
    desc: str = "Processing",
) -> List:
    data_list = []
    smiles_list = df["smiles"]
    labels = df[label_col]

    for i in range(len(smiles_list)):
        if i % 1000 == 0:
            print(f"[INFO] {desc} molecule {i}/{len(smiles_list)}", flush=True)

        smiles = smiles_list.iloc[i]
        Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond)

        if Graph_list is False:
            continue

        if has_node_with_zero_in_degree(Graph_list):
            continue

        label_val = labels.iloc[i]
        if pd.isna(label_val):
            continue

        data_list.append(
            [smiles, torch.tensor([label_val], dtype=torch.float32), Graph_list]
        )

    return data_list


def _create_dataset_from_time_split(
    datafile: str,
    encoder_atom: str,
    encoder_bond: str,
    batch_size: int,
    val_ratio: float,
    data_dir: str,
    directory_path: str,
    target_file_name: str,
) -> bool:
    train_path, test_path = get_adme_time_paths(datafile, data_dir)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Time-split train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Time-split test file not found: {test_path}")

    print(
        f"[INFO] Loading time-split data from {train_path} and {test_path}", flush=True
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = "activity"

    print("[INFO] Processing training set...", flush=True)
    train_data = _process_dataframe_to_data_list(
        train_df, label_col, encoder_atom, encoder_bond, "Training set"
    )

    print("[INFO] Processing test set...", flush=True)
    test_data = _process_dataframe_to_data_list(
        test_df, label_col, encoder_atom, encoder_bond, "Test set"
    )

    print("[INFO] Graph list was done!", flush=True)

    if val_ratio > 0 and len(train_data) > 0:
        import random

        random.seed(42)
        random.shuffle(train_data)
        val_size = int(len(train_data) * val_ratio)
        valid_data = train_data[:val_size]
        train_data = train_data[val_size:]
    else:
        valid_data = []

    train_label = [item[1] for item in train_data]
    train_graph_list = [item[2] for item in train_data]

    valid_label = [item[1] for item in valid_data]
    valid_graph_list = [item[2] for item in valid_data]

    test_label = [item[1] for item in test_data]
    test_graph_list = [item[2] for item in test_data]

    os.makedirs(directory_path, exist_ok=True)
    torch.save(
        {
            "train_label": train_label,
            "train_graph_list": train_graph_list,
            "valid_label": valid_label,
            "valid_graph_list": valid_graph_list,
            "test_label": test_label,
            "test_graph_list": test_graph_list,
            "batch_size": batch_size,
            "shuffle": True,
            "split_type": "time",
        },
        os.path.join(directory_path, target_file_name),
    )

    return True


def create_dataloader(
    config: Dict[str, Any], model_type: str = "gnn"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    datafile = config["data"]["dataset"]
    encoder_atom = config["data"]["encoder_atom"]
    encoder_bond = config["data"]["encoder_bond"]
    batch_size = config["training"]["batch_size"]
    train_ratio = config["training"]["train_ratio"]
    val_ratio = config["training"].get(
        "val_ratio", config["training"].get("vali_ratio", 0.1)
    )
    test_ratio = config["training"]["test_ratio"]
    data_dir = config.get("data_dir", "data")
    split_type = config["training"].get("split_type", "scaffold")
    seed = config.get("seed", 42)

    create_dataset(
        datafile,
        encoder_atom,
        encoder_bond,
        batch_size,
        train_ratio,
        val_ratio,
        test_ratio,
        data_dir,
        split_type,
        seed,
    )

    state = torch.load(
        os.path.join(data_dir, "processed", f"{datafile}_{split_type}.pth"),
        weights_only=False,
    )

    collate = collate_fn if model_type == "gnn" else collate_fn_gat

    train_dataset = CustomDataset(state["train_label"], state["train_graph_list"])
    valid_dataset = CustomDataset(state["valid_label"], state["valid_graph_list"])
    test_dataset = CustomDataset(state["test_label"], state["test_graph_list"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=state["shuffle"],
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    if val_ratio == 0.0:
        valid_loader = None
    else:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=state["shuffle"],
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=state["shuffle"],
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate,
    )

    print(f"[INFO] Dataset was loaded (split={split_type})!", flush=True)
    print(f"[INFO] Training set size: {len(train_dataset)}", flush=True)
    print(f"[INFO] Validation set size: {len(valid_dataset)}", flush=True)
    print(f"[INFO] Test set size: {len(test_dataset)}", flush=True)

    return train_loader, valid_loader, test_loader
