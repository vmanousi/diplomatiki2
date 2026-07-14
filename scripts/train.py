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
from src.models.build_model import build_model
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.seed import set_seed


def load_config(config_path):
    """Load an experiment configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_checkpoint(checkpoint_path, device):
    """
    Load either:
    1. a raw model state_dict, or
    2. a checkpoint dictionary containing 'model_state_dict'.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate an image-classification experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment configuration.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Configuration and reproducibility
    # ------------------------------------------------------------------
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    experiment_name = config["experiment_name"]
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    # ------------------------------------------------------------------
    # Experiment directories
    # ------------------------------------------------------------------
    experiment_dir = (
        Path("outputs")
        / "experiments"
        / experiment_name
    )
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact configuration used for reproducibility.
    shutil.copy2(
        args.config,
        experiment_dir / "config.yaml",
    )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = get_device()

    print("Device:", device)
    print("Experiment:", experiment_name)
    print("Dataset:", dataset_cfg["name"])
    print("Model:", config["model"]["name"])
    print(
        "Pretrained:",
        config["model"].get("pretrained", False),
    )

    # ------------------------------------------------------------------
    # Image transformations
    # ------------------------------------------------------------------
    transform = build_transform(config)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset = build_dataset(
        config=config,
        split="train",
        transform=transform,
    )

    val_dataset = build_dataset(
        config=config,
        split="val",
        transform=transform,
    )

    # WebDataset is iterable, so PyTorch DataLoader must not shuffle it.
    is_webdataset = (
        dataset_cfg.get("name")
        == "gastrovision_webdataset"
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False if is_webdataset else True,
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

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(
        model_name=config["model"]["name"],
        num_classes=dataset_cfg["num_classes"],
        pretrained=config["model"].get(
            "pretrained",
            False,
        ),
    )

    model = model.to(device)

    # ------------------------------------------------------------------
    # Loss and optimizer
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    history = trainer.fit(
        epochs=training_cfg["epochs"]
    )

    # ------------------------------------------------------------------
    # Save full epoch history
    # ------------------------------------------------------------------
    history_path = experiment_dir / "history.csv"

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Generate loss, accuracy and metric curves
    # ------------------------------------------------------------------
    plot_training_curves(
        history_csv=history_path,
        output_dir=experiment_dir,
    )

    # ------------------------------------------------------------------
    # Load the BEST checkpoint before final evaluation
    # ------------------------------------------------------------------
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

    model.load_state_dict(best_state_dict)
    model = model.to(device)

    print(
        "Loaded best checkpoint:",
        best_checkpoint_path,
    )

    # ------------------------------------------------------------------
    # Final evaluation of the best model
    # ------------------------------------------------------------------
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=experiment_dir,
    )

    print("Final metrics:", metrics)
    print("Training finished.")


if __name__ == "__main__":
    main()
