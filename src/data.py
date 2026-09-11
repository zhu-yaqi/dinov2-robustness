from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

@dataclass
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader

def make_transform(image_size: int, train: bool) -> transforms.Compose:
    ops = [transforms.Resize(
        (image_size, image_size),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    # Keep pixel values in [0,1]; normalization happens inside the model.
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)

def _split_indices(n: int, val_fraction: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_size = int(round(n * val_fraction))
    return perm[val_size:], perm[:val_size]

def _truncate(indices, limit: Optional[int]):
    if limit is None or limit <= 0:
        return indices
    return indices[:min(limit, len(indices))]

def build_loaders(
    data_dir: str = "data",
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction: float = 0.10,
    seed: int = 42,
    train_subset: Optional[int] = None,
    val_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    download: bool = True,
) -> LoaderBundle:
    data_dir = str(Path(data_dir))
    train_aug = datasets.CIFAR10(
        root=data_dir, train=True,
        transform=make_transform(image_size, True),
        download=download,
    )
    train_eval = datasets.CIFAR10(
        root=data_dir, train=True,
        transform=make_transform(image_size, False),
        download=False,
    )
    test_base = datasets.CIFAR10(
        root=data_dir, train=False,
        transform=make_transform(image_size, False),
        download=download,
    )

    train_idx, val_idx = _split_indices(len(train_aug), val_fraction, seed)
    train_idx = _truncate(train_idx, train_subset)
    val_idx = _truncate(val_idx, val_subset)
    test_idx = list(range(len(test_base)))
    test_idx = _truncate(test_idx, test_subset)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return LoaderBundle(
        train=DataLoader(Subset(train_aug, train_idx), shuffle=True, **kwargs),
        val=DataLoader(Subset(train_eval, val_idx), shuffle=False, **kwargs),
        test=DataLoader(Subset(test_base, test_idx), shuffle=False, **kwargs),
    )
