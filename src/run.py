import argparse
import sys
import torch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_model_config, setup_device, setup_seed
from src.utils.data import create_dataloader, get_target_dim, is_regression_dataset
from src.utils.training import train_model
from src.utils.checkpoint import get_checkpoint_path, save_checkpoint
from src.utils.output import write_results
from src.models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="KA-GNN Modular Experiment Framework")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config file path (configs/xxx.yaml)",
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (required)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs from config"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate from config"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size from config"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device from common.py"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override seed from common.py"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["random", "scaffold", "umap"],
        help="Dataset split type (random, scaffold, umap)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume training from checkpoint"
    )

    return parser.parse_args()


def override_config(config, args):
    if "data" not in config:
        config["data"] = {}
    config["data"]["dataset"] = args.dataset
    if args.epochs:
        if "training" not in config:
            config["training"] = {}
        config["training"]["epochs"] = args.epochs
    if args.lr:
        if "training" not in config:
            config["training"] = {}
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        if "training" not in config:
            config["training"] = {}
        config["training"]["batch_size"] = args.batch_size
    if args.device:
        config["device"] = args.device
    if args.seed:
        config["seed"] = args.seed
    if args.split:
        if "training" not in config:
            config["training"] = {}
        config["training"]["split_type"] = args.split
    return config


def setup_model_config(config):
    model_name = config["model"]["name"]
    target_dim = get_target_dim(config["data"]["dataset"])

    config["model"]["out_dim"] = target_dim
    config["model"]["hidden_feat"] = 64
    config["model"]["out_feat"] = 32

    if model_name in [
        "ka_gnn",
        "ka_gnn_two",
        "mlp_sage",
        "mlp_sage_two",
        "kan_sage",
        "kan_sage_two",
    ]:
        config["model"]["in_feat"] = 113
        config["model"]["in_node_dim"] = None
        config["model"]["in_edge_dim"] = None
    else:
        config["model"]["in_feat"] = 113
        config["model"]["in_node_dim"] = 92
        config["model"]["in_edge_dim"] = 21

    return config


def main():
    args = parse_args()

    config = load_model_config(args.config)
    config = override_config(config, args)

    device = setup_device(config)
    setup_seed(config.get("seed", 42))

    model_name = config["model"]["name"]
    dataset = config["data"]["dataset"]
    task_type = 'regression' if is_regression_dataset(dataset) else 'classification'

    print(f"[INFO] Model: {model_name}", flush=True)
    print(f"[INFO] Dataset: {dataset}", flush=True)
    print(f"[INFO] Task type: {task_type}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)

    config = setup_model_config(config)

    is_gnn_model = model_name in [
        "ka_gnn",
        "ka_gnn_two",
        "mlp_sage",
        "mlp_sage_two",
        "kan_sage",
        "kan_sage_two",
    ]
    model_type = "gnn" if is_gnn_model else "gat"

    for key, value in config.items():
        if key not in ['model', 'data', 'training', 'cuda'] and key != 'config':
            print(f"{key}: {value}", flush=True)

    train_loader, val_loader, test_loader = create_dataloader(config, model_type)

    def model_factory():
        return get_model(config)

    model = model_factory()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters: {total_params}", flush=True)
    sys.stdout.flush()

    best_model_state, metric_value, std_dev = train_model(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        model_type=model_type,
        task_type=task_type,
        checkpoint_path=args.checkpoint,
    )

    metric_name = 'PearsonR' if task_type == 'regression' else 'ROC-AUC'

    if best_model_state is not None:
        checkpoint_path = get_checkpoint_path(
            model_name=model_name,
            params={
                "dataset": dataset,
                "split": config["training"].get("split_type", "scaffold"),
                "learning_rate": config["training"]["learning_rate"],
                "num_layers": config["model"].get("num_layers", 2),
                "batch_size": config["training"]["batch_size"],
                "epochs": config["training"]["epochs"],
                "iterations": config["training"].get("iterations", 1),
            },
        )
        model_for_save = model_factory()
        model_for_save.load_state_dict(best_model_state)
        save_checkpoint(
            model=model_for_save,
            optimizer=None,
            epoch=config["training"]["epochs"],
            metrics={f"{metric_name.lower()}": metric_value, "std_dev": std_dev},
            config=config,
            path=checkpoint_path,
        )
        print(f"[INFO] Checkpoint saved to: {checkpoint_path}", flush=True)

    write_results(
        model_name=model_name,
        params={
            "dataset": dataset,
            "split": config["training"].get("split_type", "scaffold"),
            "num_layers": config["model"].get("num_layers", 2),
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "iterations": config["training"].get("iterations", 1),
        },
        metric_value=metric_value,
        std_dev=std_dev,
        metric_name=metric_name,
        output_dir=Path("outputs"),
    )

    print(f"[INFO] {metric_name}: {metric_value:.4f}, STD_DEV: {std_dev:.4f}", flush=True)
    print(f"mean: {metric_value:.4f}", flush=True)
    print(f"std: {std_dev:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
