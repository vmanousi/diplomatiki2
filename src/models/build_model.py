import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model name: {model_name}")