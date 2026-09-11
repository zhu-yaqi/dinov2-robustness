from __future__ import annotations
import json, os, random
from pathlib import Path
from typing import Any
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def get_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)

def ensure_parent(path: str | os.PathLike[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_json(data: Any, path: str | os.PathLike[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def total_parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)
