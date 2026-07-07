import argparse
from pathlib import Path
import shutil

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.gastrohun_dataset import GastroHUNDataset
from src.models.build_model import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.device import get_device

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
    experiment_dir = Path("experiments") / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, experiment_dir / "config.yaml")

    device = get_device()
    print("Device:", device)
    print("Experiment:", experiment_name)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    train_ds = GastroHUNDataset(
        images_root=dataset_cfg["images_root"],
        csv_path=dataset_cfg["csv_path"],
        split="Train",
        label_column=dataset_cfg["label_column"],
        transform=transform,
    )

    val_ds = GastroHUNDataset(
        images_root=dataset_cfg["images_root"],
        csv_path=dataset_cfg["csv_path"],
        split="Validation",
        label_column=dataset_cfg["label_column"],
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
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
        pretrained=config["model"].get("pretrained", False)
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
    )

    history = trainer.fit(epochs=training_cfg["epochs"])

    pd.DataFrame(history).to_csv(
        experiment_dir / "history.csv",
        index=False,
    )

    print("Training finished.")


if __name__ == "__main__":
    main()