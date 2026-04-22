from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import resnet18

from dataset import make_stratified_splits


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(labels.tolist())
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_bundle", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = Path(args.model_bundle).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found: {bundle_path}\n"
            "If you only have best_model.pt, create the bundle first:\n"
            "  python src/pack_model_bundle.py --weights outputs/best_model.pt "
            '--data_dir "/path/to/Data-3" --output outputs/model_bundle.pt'
        )

    bundle = torch.load(bundle_path, map_location="cpu")
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    _, _, test_loader, _ = make_stratified_splits(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)

    y_true, y_pred = run_eval(model, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig_path = output_dir / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=200)
    print(f"Saved confusion matrix to: {fig_path}")


if __name__ == "__main__":
    main()
