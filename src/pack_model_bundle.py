"""Build model_bundle.pt from best_model.pt (weights only) + dataset folder for class order."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create model_bundle.pt for evaluate.py / predict.py when training "
            "stopped early and only best_model.pt exists. Class order matches "
            "torchvision.datasets.ImageFolder (sorted folder names)."
        )
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to best_model.pt (state_dict only).")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Same --data_dir used in training (must contain class subfolders).",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output", type=str, default="outputs/model_bundle.pt")
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=("subject", "slice"),
        default="slice",
        help="Must match training. Old models = slice; new default training = subject.",
    )
    parser.add_argument(
        "--no_class_weights_in_training",
        action="store_true",
        help="Set if the run did not use inverse-frequency class weights (metadata only).",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    ds = datasets.ImageFolder(root=str(data_dir))
    class_names = ds.classes

    state = torch.load(weights_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Expected best_model.pt to contain a state_dict (plain dict).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "class_names": class_names,
            "image_size": args.image_size,
            "binary_non_demented_class": "non_demented",
            "split_mode": args.split_mode,
            "class_weights": not args.no_class_weights_in_training,
        },
        out_path,
    )
    print(f"Wrote bundle with {len(class_names)} classes: {class_names}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
