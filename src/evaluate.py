from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision.models import resnet18

from dataset import make_stratified_splits
from device_utils import pick_device


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
    parser.add_argument(
        "--split",
        type=str,
        choices=("test", "val", "train"),
        default="test",
        help="Which split to score (default: test). Must match split_mode stored in the bundle when training.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = Path(args.model_bundle).expanduser().resolve()
    if not bundle_path.is_file():
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"Model bundle not found: {bundle_path}\n\n"
            "Paths are resolved from your current directory.\n\n"
            "Fix one of these ways:\n"
            "  1) Use an absolute path to model_bundle.pt\n"
            "  2) Copy model_bundle.pt into this repo’s outputs/.\n"
            "  3) python src/pack_model_bundle.py --weights outputs/best_model.pt \\\n"
            '       --data_dir "/Users/jigyasaverma/Downloads/Data-3" \\\n'
            "       --output outputs/model_bundle.pt\n"
            f"\n(Current working directory: {cwd})"
        )

    bundle = torch.load(bundle_path, map_location="cpu")
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]
    split_mode = bundle.get("split_mode", "slice")

    device = pick_device()
    train_loader, val_loader, test_loader, split_info = make_stratified_splits(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_mode=split_mode,
        device=device,
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[args.split]

    n = len(loader.dataset)
    print(
        f"Split mode (from bundle, default slice for old bundles): {split_mode}\n"
        f"Evaluating {args.split.upper()} split: {n} images "
        f"(train={split_info['num_train']}, val={split_info['num_val']}, test={split_info['num_test']}, total={split_info['num_total']})"
    )
    if split_info.get("num_subjects_total"):
        print(
            f"Subjects — total={split_info['num_subjects_total']}, "
            f"train={split_info['num_subjects_train']}, val={split_info['num_subjects_val']}, "
            f"test={split_info['num_subjects_test']}"
        )

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device)

    y_true, y_pred = run_eval(model, loader, device)
    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)

    acc = float((y_true_a == y_pred_a).mean())
    bal = balanced_accuracy_score(y_true_a, y_pred_a)
    macro_f1 = f1_score(y_true_a, y_pred_a, average="macro")

    maj = Counter(y_true).most_common(1)[0][0]
    maj_acc = float((y_true_a == maj).mean())

    print("\n--- Summary (report macro / balanced — not accuracy alone) ---")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced accuracy:  {bal:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Majority baseline: {maj_acc:.4f} (always predict class index {maj} = {class_names[maj]})")
    print("--- Per-class report ---\n")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    report_path = output_dir / f"classification_report_{args.split}.txt"
    header = (
        f"split={args.split} split_mode={split_mode}\n"
        f"accuracy={acc:.6f} balanced_accuracy={bal:.6f} macro_f1={macro_f1:.6f} majority_baseline={maj_acc:.6f}\n\n"
    )
    report_path.write_text(header + report, encoding="utf-8")
    print(f"Saved metrics to: {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({args.split}, {split_mode})")
    plt.tight_layout()
    fig_path = output_dir / f"confusion_matrix_{args.split}.png"
    plt.savefig(fig_path, dpi=200)
    print(f"Saved confusion matrix to: {fig_path}")


if __name__ == "__main__":
    main()
