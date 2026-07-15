import argparse
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.build_dataset import build_dataset
from src.data.build_transform import build_transform
from src.losses.dino_loss import DINOLoss
from src.models.dino import build_dino_student_teacher
from src.training.dino_trainer import DINOTrainer
from src.training.dino_schedules import cosine_schedule
from src.utils.device import get_device
from src.utils.seed import set_seed


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description="DINO self-supervised pretraining on GI images."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the DINO YAML configuration.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    experiment_name = config["experiment_name"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    transform_cfg = config["transforms"]

    experiment_dir = (
        Path("outputs")
        / "experiments"
        / experiment_name
    )
    checkpoint_dir = experiment_dir / "checkpoints"

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        args.config,
        experiment_dir / "config.yaml",
    )

    device = get_device()

    print("Device:", device)
    print("Experiment:", experiment_name)
    print("Dataset:", dataset_cfg["name"])
    print("Backbone:", model_cfg["name"])
    print(
        "Student views:",
        2 + transform_cfg["num_local_crops"],
    )
    print("Teacher views: 2")

    transform = build_transform(config)

    train_dataset = build_dataset(
        config=config,
        split="train",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    student, teacher = build_dino_student_teacher(
        model_name=model_cfg["name"],
        out_dim=model_cfg["out_dim"],
        hidden_dim=model_cfg.get(
            "hidden_dim",
            2048,
        ),
        bottleneck_dim=model_cfg.get(
            "bottleneck_dim",
            256,
        ),
    )
    
    
    steps_per_epoch = training_cfg["steps_per_epoch"]
    total_steps = steps_per_epoch * training_cfg["epochs"]

    warmup_epochs = training_cfg.get(
        "warmup_epochs",
        0,
    )
    warmup_steps = steps_per_epoch * warmup_epochs

    if total_steps <= 0:
        raise RuntimeError(
            "The DINO training configuration produced "
            "zero total training steps."
        )

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_epochs cannot exceed the total "
            "number of training epochs."
        )

    learning_rate_schedule = cosine_schedule(
        start_value=training_cfg["learning_rate"],
        final_value=training_cfg.get(
            "final_learning_rate",
            1e-6,
        ),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_start_value=training_cfg.get(
            "warmup_start_learning_rate",
            1e-6,
        ),
    )

    teacher_momentum_schedule = cosine_schedule(
        start_value=training_cfg.get(
            "teacher_momentum",
            0.996,
        ),
        final_value=training_cfg.get(
            "final_teacher_momentum",
            1.0,
        ),
        total_steps=total_steps,
    )

    print("Steps per epoch:", steps_per_epoch)
    print("Total training steps:", total_steps)
    print("Warm-up steps:", warmup_steps)

    print(
        "Learning-rate schedule:",
        learning_rate_schedule[0],
        "->",
        learning_rate_schedule[-1],
    )

    print(
        "Teacher-momentum schedule:",
        teacher_momentum_schedule[0],
        "->",
        teacher_momentum_schedule[-1],
    )
    
    

    num_student_views = (
        2
        + transform_cfg["num_local_crops"]
    )

    criterion = DINOLoss(
        out_dim=model_cfg["out_dim"],
        num_student_views=num_student_views,
        teacher_temperature=training_cfg.get(
            "teacher_temperature",
            0.04,
        ),
        student_temperature=training_cfg.get(
            "student_temperature",
            0.1,
        ),
        center_momentum=training_cfg.get(
            "center_momentum",
            0.9,
        ),
    ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get(
            "weight_decay",
            0.04,
        ),
    )

    trainer = DINOTrainer(
        student=student,
        teacher=teacher,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        teacher_momentum=training_cfg.get(
            "teacher_momentum",
            0.996,
        ),
        gradient_clip=training_cfg.get(
            "gradient_clip",
            3.0,
        ),
        
        learning_rate_schedule=learning_rate_schedule,
        teacher_momentum_schedule=teacher_momentum_schedule,
    )

    history = trainer.fit(
        epochs=training_cfg["epochs"]
    )

    history_path = experiment_dir / "history.csv"

    pd.DataFrame(history).to_csv(
        history_path,
        index=False,
    )

    # Save the encoders separately for downstream GastroHUN experiments.
    torch.save(
        student.backbone.state_dict(),
        experiment_dir / "student_backbone_final.pt",
    )

    torch.save(
        teacher.backbone.state_dict(),
        experiment_dir / "teacher_backbone_final.pt",
    )

    print("Saved history:", history_path)
    print(
        "Saved student backbone:",
        experiment_dir / "student_backbone_final.pt",
    )
    print(
        "Saved teacher backbone:",
        experiment_dir / "teacher_backbone_final.pt",
    )
    print("DINO pretraining finished.")


if __name__ == "__main__":
    main()
