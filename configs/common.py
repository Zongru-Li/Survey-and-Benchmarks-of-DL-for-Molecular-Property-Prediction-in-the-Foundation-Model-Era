device = "cuda"
# device = "cpu"

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
