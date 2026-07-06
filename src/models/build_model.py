import torch.nn as nn
from torchvision import models
import timm


def build_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "vit_tiny":
        return timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model: {model_name}")

