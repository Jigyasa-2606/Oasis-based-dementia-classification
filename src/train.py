from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torchvision.models import ResNet18_Weights, resnet18

from dataset import make_stratified_splits
from device_utils import pick_device


def _class_weights_from_counts(counts: list[int], num_classes: int) -> torch.Tensor:
    c = np.array(counts, dtype=np.float64)
    c = np.maximum(c, 1.0)
    n = float(c.sum())
    w = n / (num_classes * c)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)
    return epoch_loss, epoch_acc, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=("subject", "slice"),
        default="subject",
        help='subject = split by OASIS subject ID (no patient leakage); slice = random image split.',
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable inverse-frequency class weights in the loss (default: weights ON).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"Using device: {device}")
    print(f"Split mode: {args.split_mode}")

    train_loader, val_loader, test_loader, split_info = make_stratified_splits(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_mode=args.split_mode,
        device=device,
    )
    class_names = split_info["class_names"]
    num_classes = len(class_names)

    if split_info.get("num_subjects_total"):
        print(
            f"Subjects — total={split_info['num_subjects_total']}, "
            f"train={split_info['num_subjects_train']}, val={split_info['num_subjects_val']}, "
            f"test={split_info['num_subjects_test']}"
        )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("Class weights: off")
    else:
        w = _class_weights_from_counts(split_info["train_class_counts"], num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        print(f"Class weights (train rebalanced): {w.detach().cpu().numpy().round(4).tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved best model to: {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    artifact = {
        "class_names": class_names,
        "split_info": split_info,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "split_mode": args.split_mode,
        "class_weights": not args.no_class_weights,
    }
    with (output_dir / "training_artifacts.json").open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": args.image_size,
            "binary_non_demented_class": "non_demented",
            "split_mode": args.split_mode,
            "class_weights": not args.no_class_weights,
        },
        output_dir / "model_bundle.pt",
    )
    print(f"Saved model bundle to: {output_dir / 'model_bundle.pt'}")


if __name__ == "__main__":
    main()
