from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from device_utils import dataloader_pin_memory
from patient_id import extract_subject_id


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
    return {name.lower().strip(): idx for idx, name in enumerate(class_names)}


def _train_test_split_maybe_stratify(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratify when each class has enough samples for sklearn; else shuffle split."""
    if len(x) < 4:
        return train_test_split(x, test_size=test_size, random_state=random_state)
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return train_test_split(x, test_size=test_size, random_state=random_state)
    try:
        return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)[:2]
    except ValueError:
        return train_test_split(x, test_size=test_size, random_state=random_state)


def _indices_for_patients(patient_to_indices: Dict[str, List[int]], patients: np.ndarray) -> np.ndarray:
    out: List[int] = []
    for p in patients:
        out.extend(patient_to_indices[str(p)])
    return np.array(out, dtype=int)


def make_stratified_splits(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    *,
    split_mode: str = "subject",
    device: torch.device | None = None,
):
    """
    split_mode:
      - "subject": split by OASIS subject ID so all slices of one patient stay in one split (recommended).
      - "slice": original random stratified split at image level (can leak patient across train/test).
    """
    if split_mode not in ("subject", "slice"):
        raise ValueError('split_mode must be "subject" or "slice"')

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_tfms, eval_tfms = default_transforms(image_size=image_size)
    base_for_split = datasets.ImageFolder(root=str(data_dir))
    class_names = base_for_split.classes
    targets = np.array(base_for_split.targets)

    if (val_size + test_size) >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    hold_frac = val_size + test_size
    val_ratio_within_holdout = val_size / hold_frac

    if split_mode == "slice":
        indices = np.arange(len(targets))
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=hold_frac,
            random_state=seed,
            stratify=targets,
        )
        holdout_targets = targets[holdout_idx]
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=(1.0 - val_ratio_within_holdout),
            random_state=seed,
            stratify=holdout_targets,
        )
        num_patients_train = num_patients_val = num_patients_test = None
    else:
        patient_to_indices: Dict[str, List[int]] = defaultdict(list)
        patient_label: Dict[str, int] = {}
        for i, (path, y) in enumerate(base_for_split.samples):
            pid = extract_subject_id(path)
            patient_to_indices[pid].append(i)
            if pid in patient_label and patient_label[pid] != y:
                raise ValueError(
                    f"Subject {pid} appears with conflicting labels ({patient_label[pid]} vs {y}). "
                    "Check data layout."
                )
            patient_label[pid] = y

        patients = np.array(sorted(patient_to_indices.keys()))
        y_patient = np.array([patient_label[p] for p in patients], dtype=int)

        train_patients, hold_patients = _train_test_split_maybe_stratify(
            patients, y_patient, test_size=hold_frac, random_state=seed
        )
        y_hold = np.array([patient_label[p] for p in hold_patients], dtype=int)
        val_patients, test_patients = _train_test_split_maybe_stratify(
            hold_patients,
            y_hold,
            test_size=(1.0 - val_ratio_within_holdout),
            random_state=seed + 1,
        )

        train_idx = _indices_for_patients(patient_to_indices, train_patients)
        val_idx = _indices_for_patients(patient_to_indices, val_patients)
        test_idx = _indices_for_patients(patient_to_indices, test_patients)

        num_patients_train = len(train_patients)
        num_patients_val = len(val_patients)
        num_patients_test = len(test_patients)

    train_ds_full = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    eval_ds_full = datasets.ImageFolder(root=str(data_dir), transform=eval_tfms)

    train_ds = Subset(train_ds_full, train_idx.tolist())
    val_ds = Subset(eval_ds_full, val_idx.tolist())
    test_ds = Subset(eval_ds_full, test_idx.tolist())

    pin = dataloader_pin_memory(device) if device is not None else torch.cuda.is_available()
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

    train_targets = targets[train_idx]
    train_class_counts = np.bincount(train_targets, minlength=len(class_names)).tolist()

    split_info: Dict = {
        "num_total": len(targets),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test": len(test_idx),
        "class_names": class_names,
        "label_map": _build_label_map(class_names),
        "split_mode": split_mode,
        "split_seed": seed,
        "train_class_counts": train_class_counts,
    }
    if split_mode == "subject":
        split_info["num_subjects_total"] = len(patients)
        split_info["num_subjects_train"] = num_patients_train
        split_info["num_subjects_val"] = num_patients_val
        split_info["num_subjects_test"] = num_patients_test

    return train_loader, val_loader, test_loader, split_info
