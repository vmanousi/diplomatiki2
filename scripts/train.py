import argparse
from pathlib import Path
import shutil

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from src.data.build_transform import build_transform

from src.data.build_dataset import build_dataset
from src.models.build_model import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.evaluation.plots import plot_training_curves
from src.evaluation.metrics import evaluate_model


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    experiment_name = config["experiment_name"]
    experiment_dir = Path("outputs") / "experiments" / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    tensorboard_dir = experiment_dir / "tensorboard"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, experiment_dir / "config.yaml")

    device = get_device()
    print("Device:", device)
    print("Experiment:", experiment_name)

    transform = build_transform(config)

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    train_ds = build_dataset(
        config,
        split="train",
        transform=transform,
    )

    val_ds = build_dataset(
        config,
        split="val",
        transform=transform,
    )

    is_webdataset = dataset_cfg.get("name") == "gastrovision_webdataset"

    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False if is_webdataset else True,
        num_workers=training_cfg["num_workers"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
    )

    model = build_model(
        model_name=config["model"]["name"],
        num_classes=dataset_cfg["num_classes"],
        pretrained=config["model"].get("pretrained", False),
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
    )

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

    history = trainer.fit(epochs=training_cfg["epochs"])

    history_path = experiment_dir / "history.csv"

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    plot_training_curves(
        history_csv=history_path,
        output_dir=experiment_dir,
    )

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
