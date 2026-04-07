from __future__ import annotations

import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Code and Data" / "Output" / "card_classifier.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
IMAGE_SIZE = 128


class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes: int = 53) -> None:
        super().__init__()
        base_model = timm.create_model("efficientnet_b0", pretrained=False)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_class_names() -> list[str]:
    return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def load_model(device: torch.device | str | None = None) -> tuple[nn.Module, list[str], transforms.Compose]:
    class_names = load_class_names()
    model = SimpleCardClassifier(num_classes=len(class_names))
    target_device = torch.device(device or "cpu")
    state_dict = torch.load(MODEL_PATH, map_location=target_device)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model, class_names, build_transform()


def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: list[str],
    transform: transforms.Compose,
    device: torch.device | str | None = None,
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    target_device = torch.device(device or "cpu")
    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(target_device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    results: list[dict[str, float | str]] = []

    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
        results.append(
            {
                "label": class_names[index],
                "probability": float(probability),
            }
        )

    return results

