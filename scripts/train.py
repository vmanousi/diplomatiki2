import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.build_dataset import build_dataset
from src.data.build_transform import build_transform
from src.evaluation.metrics import evaluate_model
from src.evaluation.plots import plot_training_curves
from src.models.build_model import build_model, get_param_groups
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.seed import set_seed


def load_config(config_path):
    """
    Load an experiment configuration from a YAML file.
    """

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_checkpoint(
    checkpoint_path,
    device,
):
    """
    Load either:

    1. a raw model state_dict, or
    2. a checkpoint dictionary containing model_state_dict.
    """

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        checkpoint = checkpoint[
            "model_state_dict"
        ]

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate an "
            "image-classification experiment."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to the YAML experiment "
            "configuration."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Configuration and reproducibility
    # --------------------------------------------------------------
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    experiment_name = config["experiment_name"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    # --------------------------------------------------------------
    # Experiment directories
    # --------------------------------------------------------------
    experiment_dir = (
        Path("outputs")
        / "experiments"
        / experiment_name
    )

    checkpoint_dir = (
        experiment_dir
        / "checkpoints"
    )

    tensorboard_dir = (
        experiment_dir
        / "tensorboard"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tensorboard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the exact configuration used.
    shutil.copy2(
        args.config,
        experiment_dir / "config.yaml",
    )

    # --------------------------------------------------------------
    # Device
    # --------------------------------------------------------------
    device = get_device()

    print("Device:", device)
    print("Experiment:", experiment_name)
    print("Dataset:", dataset_cfg["name"])
    print("Model:", model_cfg["name"])

    print(
        "ImageNet pretrained:",
        model_cfg.get(
            "pretrained",
            False,
        ),
    )

    print(
        "Backbone checkpoint:",
        model_cfg.get(
            "backbone_checkpoint",
        ),
    )

    print(
        "Freeze backbone:",
        model_cfg.get(
            "freeze_backbone",
            False,
        ),
    )

    # --------------------------------------------------------------
    # Image transformations
    # --------------------------------------------------------------
    train_transform = build_transform(config, split="train")
    eval_transform = build_transform(config, split="val")

    # --------------------------------------------------------------
    # Datasets
    # --------------------------------------------------------------
    train_dataset = build_dataset(
        config=config,
        split="train",
        transform=train_transform,
    )

    val_dataset = build_dataset(
        config=config,
        split="val",
        transform=eval_transform,
    )

    # WebDataset is iterable and must not be shuffled by
    # the standard PyTorch DataLoader.
    is_webdataset = (
        dataset_cfg.get("name")
        == "gastrovision_webdataset"
    )

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=(
            False
            if is_webdataset
            else True
        ),
        num_workers=training_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model = build_model(
        model_name=model_cfg["name"],
        num_classes=dataset_cfg["num_classes"],
        pretrained=model_cfg.get(
            "pretrained",
            False,
        ),
        backbone_checkpoint=model_cfg.get(
            "backbone_checkpoint",
        ),
        freeze_backbone=model_cfg.get(
            "freeze_backbone",
            False,
        ),
    )

    model = model.to(device)

    # --------------------------------------------------------------
    # Parameter diagnostics
    # --------------------------------------------------------------
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    trainable_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    print(
        "Total parameters:",
        total_parameters,
    )

    print(
        "Trainable parameters:",
        trainable_parameter_count,
    )

    print(
        "Trainable parameter names:",
        trainable_parameter_names,
    )

    # For frozen linear evaluation with ViT-Tiny and 23
    # classes, only head.weight and head.bias should appear.
    if model_cfg.get(
        "freeze_backbone",
        False,
    ):
        allowed_trainable_names = {
            "head.weight",
            "head.bias",
        }

        unexpected_trainable_names = (
            set(trainable_parameter_names)
            - allowed_trainable_names
        )

        if unexpected_trainable_names:
            raise RuntimeError(
                "Frozen-backbone mode contains unexpected "
                "trainable parameters: "
                f"{sorted(unexpected_trainable_names)}"
            )

    # --------------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "The model contains no trainable parameters."
        )

    # Discriminative learning rates: a smaller LR for the pretrained
    # backbone and a larger LR for the freshly initialized head, so full
    # fine-tuning doesn't wreck the pretrained features. Optional — set
    # both training.backbone_learning_rate and training.head_learning_rate
    # together to enable it; otherwise every trainable parameter uses the
    # single training.learning_rate as before.
    backbone_learning_rate = training_cfg.get("backbone_learning_rate")
    head_learning_rate = training_cfg.get("head_learning_rate")

    use_discriminative_lr = (
        backbone_learning_rate is not None
        or head_learning_rate is not None
    )

    if use_discriminative_lr:
        if backbone_learning_rate is None or head_learning_rate is None:
            raise ValueError(
                "Set both backbone_learning_rate and head_learning_rate "
                "together to use discriminative learning rates."
            )

        optimizer_params = get_param_groups(
            model,
            backbone_lr=backbone_learning_rate,
            head_lr=head_learning_rate,
        )
        default_learning_rate = head_learning_rate
    else:
        optimizer_params = trainable_parameters
        default_learning_rate = training_cfg["learning_rate"]

    optimizer_name = training_cfg.get(
        "optimizer",
        "adam",
    ).lower()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=default_learning_rate,
            weight_decay=training_cfg.get(
                "weight_decay",
                0.0,
            ),
        )

    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=default_learning_rate,
            weight_decay=training_cfg.get(
                "weight_decay",
                0.0,
            ),
        )

    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=default_learning_rate,
            momentum=training_cfg.get(
                "momentum",
                0.9,
            ),
            weight_decay=training_cfg.get(
                "weight_decay",
                0.0,
            ),
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

    print(
        "Optimizer:",
        optimizer.__class__.__name__,
    )

    if use_discriminative_lr:
        print(
            "Learning rate (backbone / head):",
            backbone_learning_rate,
            "/",
            head_learning_rate,
        )
    else:
        print(
            "Learning rate:",
            default_learning_rate,
        )

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
    )

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------
    history = trainer.fit(
        epochs=training_cfg["epochs"],
        early_stopping_patience=training_cfg.get(
            "early_stopping_patience"
        ),
    )

    # --------------------------------------------------------------
    # Save full epoch history
    # --------------------------------------------------------------
    history_path = (
        experiment_dir
        / "history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Generate training curves
    # --------------------------------------------------------------
    plot_training_curves(
        history_csv=history_path,
        output_dir=experiment_dir,
    )

    # --------------------------------------------------------------
    # Load the best checkpoint before final evaluation
    # --------------------------------------------------------------
    best_checkpoint_path = (
        checkpoint_dir
        / "best_model.pt"
    )

    if not best_checkpoint_path.exists():
        raise FileNotFoundError(
            "The best checkpoint was not found at: "
            f"{best_checkpoint_path}"
        )

    best_state_dict = load_checkpoint(
        checkpoint_path=best_checkpoint_path,
        device=device,
    )

    model.load_state_dict(
        best_state_dict
    )

    model = model.to(device)

    print(
        "Loaded best checkpoint:",
        best_checkpoint_path,
    )

    # --------------------------------------------------------------
    # Final evaluation of the best model
    # --------------------------------------------------------------
    idx_to_label = getattr(train_dataset, "idx_to_label", None)

    class_names = (
        [idx_to_label[i] for i in range(len(idx_to_label))]
        if idx_to_label is not None
        else None
    )

    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=experiment_dir,
        class_names=class_names,
    )

    print(
        "Final metrics:",
        metrics,
    )

    print(
        "Saved history:",
        history_path,
    )

    print(
        "Training finished."
    )


if __name__ == "__main__":
    main()