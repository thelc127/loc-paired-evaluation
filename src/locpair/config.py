"""Configuration loading and small shared helpers."""

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path=None):
    """Load the YAML config. Relative paths in `paths` resolve against the repo root."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    cfg["paths"] = {k: str(resolve(v)) for k, v in cfg["paths"].items()}
    return cfg


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
