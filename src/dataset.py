from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def default_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, eval_tfms


def _build_label_map(class_names: List[str]) -> Dict[str, int]:
    normalized = {name.lower().strip(): idx for idx, name in enumerate(class_names)}
    return normalized


def make_stratified_splits(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_tfms, eval_tfms = default_transforms(image_size=image_size)

    base_for_split = datasets.ImageFolder(root=str(data_dir))
    class_names = base_for_split.classes
    targets = np.array(base_for_split.targets)
    indices = np.arange(len(targets))

    if (val_size + test_size) >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=targets,
    )

    holdout_targets = targets[holdout_idx]
    val_ratio_within_holdout = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=(1.0 - val_ratio_within_holdout),
        random_state=seed,
        stratify=holdout_targets,
    )

    train_ds_full = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    eval_ds_full = datasets.ImageFolder(root=str(data_dir), transform=eval_tfms)

    train_ds = Subset(train_ds_full, train_idx.tolist())
    val_ds = Subset(eval_ds_full, val_idx.tolist())
    test_ds = Subset(eval_ds_full, test_idx.tolist())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    split_info = {
        "num_total": len(indices),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "class_names": class_names,
        "label_map": _build_label_map(class_names),
    }

    return train_loader, val_loader, test_loader, split_info
