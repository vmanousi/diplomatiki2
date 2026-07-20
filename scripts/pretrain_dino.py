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
from src.training.dino_schedules import (
    cosine_schedule,
    linear_warmup_schedule,
)
from src.training.dino_trainer import DINOTrainer
from src.utils.device import get_device
from src.utils.seed import set_seed


def load_config(config_path):
    """
    Load a YAML configuration file.
    """

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "DINO self-supervised pretraining "
            "on gastrointestinal images."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the DINO YAML configuration.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Optional path to a DINO checkpoint "
            "from which training will resume."
        ),
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

    checkpoint_dir = (
        experiment_dir
        / "checkpoints"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save an exact copy of the configuration used.
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
        "Pretrained backbone init:",
        model_cfg.get(
            "pretrained_backbone",
            False,
        ),
    )

    print(
        "Student views:",
        2 + transform_cfg["num_local_crops"],
    )

    print("Teacher views: 2")

    # Build the DINO multi-crop transformation.
    transform = build_transform(config)

    # Build the GastroVision training dataset.
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

    # Build initially identical student and teacher models. Defaults to
    # a from-scratch backbone exactly as before — set
    # model.pretrained_backbone: true in the config to instead start
    # from that backbone's pretrained weights (e.g. a DINOv2 checkpoint
    # tag), for continued pretraining rather than training from scratch.
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
        pretrained_backbone=model_cfg.get(
            "pretrained_backbone",
            False,
        ),
    )

    # GastroVision is an iterable WebDataset and therefore
    # does not expose a normal Python length.
    #
    # The number of optimizer steps in one epoch is defined
    # explicitly in the YAML configuration.
    steps_per_epoch = int(
        training_cfg["steps_per_epoch"]
    )

    epochs = int(
        training_cfg["epochs"]
    )

    # The schedules (LR / teacher momentum / teacher temperature) must be
    # sized to the FINAL target epoch count, fixed across every resume
    # segment of the same run — not to however far *this* invocation
    # trains. `epochs` still means "run through this epoch this
    # invocation" (as before, for a single-shot run these are the same
    # number). `total_epochs` defaults to `epochs`, so anything that
    # doesn't split a run across resume segments is unaffected.
    total_epochs = int(
        training_cfg.get(
            "total_epochs",
            epochs,
        )
    )

    warmup_epochs = int(
        training_cfg.get(
            "warmup_epochs",
            0,
        )
    )

    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch must be greater than zero."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if total_epochs < epochs:
        raise ValueError(
            "total_epochs cannot be smaller than epochs — "
            "total_epochs is the final schedule target across every "
            "resume segment, epochs is how far this invocation runs."
        )

    if warmup_epochs < 0:
        raise ValueError(
            "warmup_epochs cannot be negative."
        )

    total_steps = (
        steps_per_epoch
        * total_epochs
    )

    warmup_steps = (
        steps_per_epoch
        * warmup_epochs
    )

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_epochs cannot exceed the total "
            "number of training epochs."
        )

    # Learning-rate schedule:
    # linear warm-up followed by cosine decay.
    learning_rate_schedule = cosine_schedule(
        start_value=training_cfg[
            "learning_rate"
        ],
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

    # Teacher momentum:
    # cosine increase from its initial value toward 1.0.
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

    # Teacher temperature:
    # linear warm-up followed by a constant final value.
    teacher_temperature_schedule = (
        linear_warmup_schedule(
            start_value=training_cfg.get(
                "warmup_teacher_temperature",
                0.04,
            ),
            final_value=training_cfg.get(
                "teacher_temperature",
                0.07,
            ),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
    )

    print(
        "Steps per epoch:",
        steps_per_epoch,
    )

    print(
        "Total training steps:",
        total_steps,
    )

    print(
        "Warm-up steps:",
        warmup_steps,
    )

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

    print(
        "Teacher-temperature schedule:",
        teacher_temperature_schedule[0],
        "->",
        teacher_temperature_schedule[-1],
    )

    num_student_views = (
        2
        + transform_cfg["num_local_crops"]
    )

    criterion = DINOLoss(
        out_dim=model_cfg["out_dim"],
        num_student_views=num_student_views,
        teacher_temperature=training_cfg.get(
            "warmup_teacher_temperature",
            training_cfg.get(
                "teacher_temperature",
                0.04,
            ),
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
        learning_rate_schedule=(
            learning_rate_schedule
        ),
        teacher_momentum_schedule=(
            teacher_momentum_schedule
        ),
        teacher_temperature_schedule=(
            teacher_temperature_schedule
        ),
        mixed_precision=training_cfg.get(
            "mixed_precision",
            False,
        ),
        total_steps=total_steps,
    )

    start_epoch = 1

    if args.resume is not None:
        checkpoint_path = Path(
            args.resume
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Resume checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        # These values are required for every resumable
        # DINO checkpoint.
        required_keys = {
            "epoch",
            "global_step",
            "student_state_dict",
            "teacher_state_dict",
            "optimizer_state_dict",
            "criterion_state_dict",
        }

        missing_keys = required_keys.difference(
            checkpoint.keys()
        )

        if missing_keys:
            raise KeyError(
                "Resume checkpoint is missing keys: "
                f"{sorted(missing_keys)}"
            )

        student.load_state_dict(
            checkpoint[
                "student_state_dict"
            ]
        )

        teacher.load_state_dict(
            checkpoint[
                "teacher_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        # Restore the DINO criterion state.
        # This includes the registered center buffer.
        criterion.load_state_dict(
            checkpoint[
                "criterion_state_dict"
            ]
        )

        # New AMP checkpoints contain the GradScaler state.
        # Older non-AMP checkpoints remain usable.
        if "scaler_state_dict" in checkpoint:
            trainer.scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

            print(
                "Restored AMP scaler state."
            )
        else:
            print(
                "Checkpoint has no AMP scaler state; "
                "using a newly initialized scaler."
            )

        # The LR/teacher-momentum/teacher-temperature schedules are
        # recomputed fresh from this run's config, sized to this run's
        # total_steps (steps_per_epoch * total_epochs). If a resume
        # segment uses a different `total_epochs` (or steps_per_epoch)
        # than the run that produced this checkpoint, the recomputed
        # schedule has a different shape than the one actually used for
        # the steps already completed, and every already-executed step's
        # schedule value silently stops matching what the checkpoint
        # assumes. Older checkpoints (predating this check) have no
        # recorded total_steps, so there's nothing to verify against —
        # proceed with a warning instead of failing on existing
        # checkpoints.
        checkpoint_total_steps = checkpoint.get(
            "total_steps"
        )

        if checkpoint_total_steps is None:
            print(
                "Warning: resume checkpoint has no recorded "
                "total_steps (predates this check) — cannot verify "
                "the schedule shape matches. Proceeding anyway."
            )
        elif checkpoint_total_steps != total_steps:
            raise ValueError(
                "Resume schedule mismatch: this checkpoint was "
                f"produced with total_steps={checkpoint_total_steps}, "
                f"but this config computes total_steps={total_steps} "
                f"(steps_per_epoch={steps_per_epoch} * "
                f"total_epochs={total_epochs}). Keep `total_epochs` "
                "(and steps_per_epoch) identical across every resume "
                "segment of the same run — only `epochs` (how far this "
                "invocation runs) and start_epoch should change."
            )

        trainer.global_step = int(
            checkpoint["global_step"]
        )

        if trainer.global_step < 0:
            raise ValueError(
                "Checkpoint global_step "
                "cannot be negative."
            )

        if trainer.global_step > total_steps:
            raise ValueError(
                "Checkpoint global_step exceeds "
                "the total schedule length. "
                "Check that the resume YAML matches "
                "the original experiment."
            )

        start_epoch = (
            int(checkpoint["epoch"])
            + 1
        )

        if start_epoch <= epochs:
            if trainer.global_step >= total_steps:
                raise ValueError(
                    "The checkpoint has exhausted the "
                    "configured schedules, but the YAML "
                    "still requests additional epochs."
                )

        print(
            "Resumed checkpoint:",
            checkpoint_path,
        )

        print(
            "Resume epoch:",
            start_epoch,
        )

        print(
            "Resume global step:",
            trainer.global_step,
        )

    history_path = (
        experiment_dir
        / "history.csv"
    )

    previous_history = []

    # When resuming into the same experiment folder,
    # preserve all previously recorded epochs.
    if (
        args.resume is not None
        and history_path.is_file()
    ):
        previous_history = (
            pd.read_csv(history_path)
            .to_dict(orient="records")
        )

    new_history = trainer.fit(
        epochs=epochs,
        start_epoch=start_epoch,
    )

    history = (
        previous_history
        + new_history
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    # Save student and teacher encoders separately for
    # downstream GastroHUN classification experiments.
    student_backbone_path = (
        experiment_dir
        / "student_backbone_final.pt"
    )

    teacher_backbone_path = (
        experiment_dir
        / "teacher_backbone_final.pt"
    )

    torch.save(
        student.backbone.state_dict(),
        student_backbone_path,
    )

    torch.save(
        teacher.backbone.state_dict(),
        teacher_backbone_path,
    )

    print(
        "Saved history:",
        history_path,
    )

    print(
        "Saved student backbone:",
        student_backbone_path,
    )

    print(
        "Saved teacher backbone:",
        teacher_backbone_path,
    )

    print(
        "DINO pretraining finished."
    )


if __name__ == "__main__":
    main()