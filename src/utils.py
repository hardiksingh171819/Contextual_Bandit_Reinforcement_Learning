import yaml
import torch

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_device(cfg_device):
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
