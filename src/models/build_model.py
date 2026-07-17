from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
import timm


def _load_backbone_checkpoint(
    model,
    checkpoint_path,
):
    """
    Load a backbone-only checkpoint.

    The DINO checkpoint contains ViT backbone parameters but
    no supervised classification head.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The backbone checkpoint must contain a state_dict."
        )

    incompatible_keys = model.load_state_dict(
        checkpoint,
        strict=False,
    )

    unexpected_keys = list(
        incompatible_keys.unexpected_keys
    )

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected parameters while loading the "
            f"backbone checkpoint: {unexpected_keys}"
        )

    print(
        "Loaded backbone checkpoint:",
        checkpoint_path,
    )

    if incompatible_keys.missing_keys:
        print(
            "Parameters not supplied by the backbone checkpoint:",
            list(incompatible_keys.missing_keys),
        )


def _freeze_vit_backbone(model):
    """
    Freeze the complete ViT encoder and leave only the
    classification head trainable.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.head.parameters():
        parameter.requires_grad = True


def build_model(
    model_name,
    num_classes,
    pretrained=False,
    backbone_checkpoint=None,
    freeze_backbone=False,
):
    """
    Build a supervised image-classification model.

    Parameters
    ----------
    model_name:
        Model identifier such as vit_tiny or resnet18.

    num_classes:
        Number of supervised output classes.

    pretrained:
        Whether to use the standard ImageNet pretrained model.

    backbone_checkpoint:
        Optional path to a DINO backbone state_dict.

    freeze_backbone:
        If True, train only the classification head.
    """

    if pretrained and backbone_checkpoint is not None:
        raise ValueError(
            "Use either pretrained=True or a custom "
            "backbone_checkpoint, not both."
        )

    if model_name == "resnet18":
        model = models.resnet18(
            weights=(
                models.ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        if backbone_checkpoint is not None:
            raise ValueError(
                "The current DINO backbone checkpoint is a ViT "
                "checkpoint and cannot be loaded into ResNet18."
            )

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

            for parameter in model.fc.parameters():
                parameter.requires_grad = True

        return model

    if model_name == "resnet50":
        model = models.resnet50(
            weights=(
                models.ResNet50_Weights.DEFAULT
                if pretrained
                else None
            )
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

        if backbone_checkpoint is not None:
            raise ValueError(
                "The current DINO backbone checkpoint is a ViT "
                "checkpoint and cannot be loaded into ResNet50."
            )

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

            for parameter in model.fc.parameters():
                parameter.requires_grad = True

        return model

    if model_name in {
        "vit_tiny",
        "vit_small",
        "vit_base",
    }:
        timm_name = {
            "vit_tiny": "vit_tiny_patch16_224",
            "vit_small": "vit_small_patch16_224",
            "vit_base": "vit_base_patch16_224",
        }[model_name]

        if backbone_checkpoint is not None:
            # Build the encoder without a classification head so
            # the DINO backbone checkpoint matches directly.
            model = timm.create_model(
                timm_name,
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
            )

            _load_backbone_checkpoint(
                model=model,
                checkpoint_path=backbone_checkpoint,
            )

            # Add a new randomly initialized supervised head.
            model.reset_classifier(
                num_classes=num_classes,
            )
        else:
            model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=num_classes,
                dynamic_img_size=True,
            )

        if freeze_backbone:
            _freeze_vit_backbone(model)

        return model

    raise ValueError(
        f"Unknown model: {model_name}"
    )