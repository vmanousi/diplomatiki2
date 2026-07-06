import torch.nn as nn
from torchvision import models
import timm


def build_model(model_name, num_classes, pretrained=False):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name in ["vit_tiny", "vit_small", "vit_base"]:
        timm_name = {
            "vit_tiny": "vit_tiny_patch16_224",
            "vit_small": "vit_small_patch16_224",
            "vit_base": "vit_base_patch16_224",
        }[model_name]

        return timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model: {model_name}")