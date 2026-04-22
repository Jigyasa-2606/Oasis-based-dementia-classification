from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


def normalize_class_name(name: str) -> str:
    return name.lower().strip().replace(" ", "_")


def is_dementia_likely(predicted_label: str) -> bool:
    normalized = normalize_class_name(predicted_label)
    return normalized != "non_demented"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_bundle", type=str, required=True)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    bundle = torch.load(args.model_bundle, map_location="cpu")
    class_names = bundle["class_names"]
    image_size = bundle["image_size"]

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

    predicted_label = class_names[pred_idx]
    confidence = float(probs[pred_idx].item())
    likely = is_dementia_likely(predicted_label)

    print(f"Predicted class: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Binary output: {'Dementia likely' if likely else 'Not likely'}")


if __name__ == "__main__":
    main()
