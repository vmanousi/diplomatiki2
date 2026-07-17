import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.build_dataset import build_dataset
from src.data.build_transform import build_transform
from src.evaluation.metrics import evaluate_model
from src.models.build_model import build_model
from src.utils.device import get_device


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    experiment_name = config["experiment_name"]

    device = get_device()
    print("Device:", device)

    transform = build_transform(config, split="val")

    val_dataset = build_dataset(
        config=config,
        split="val",
        transform=transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
    )

    model = build_model(
        model_name=config["model"]["name"],
        num_classes=dataset_cfg["num_classes"],
        pretrained=config["model"].get("pretrained", False),
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model = model.to(device)

    print("Loaded checkpoint:", args.checkpoint)

    experiment_dir = Path("outputs") / "experiments" / experiment_name

    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        output_dir=experiment_dir,
    )

    print("Best-checkpoint evaluation finished.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
