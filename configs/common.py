device = "cuda"

cuda = {
    "use_cuda": True,
    "device_id": 0,
    "deterministic": True,
    "benchmark": False,
    "amp": {
        "enabled": False,
        "dtype": "float16",
    },
    "distributed": {
        "enabled": False,
        "backend": "nccl",
    },
}

seed = 42

log_level = "INFO"


def add_common_args(parser):
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model config file path (configs/xxx.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (required)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from common.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from common.py",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["random", "scaffold", "umap", "butina"],
        help="Dataset split type (random, scaffold, umap, butina)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--stddev",
        action="store_true",
        default=False,
        help="Include standard deviation in output results",
    )
    return parser
