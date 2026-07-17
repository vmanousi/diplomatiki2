from pathlib import Path

import torch
from tqdm import tqdm

from src.training.dino_utils import update_teacher


class DINOTrainer:
    def __init__(
        self,
        student,
        teacher,
        dataloader,
        criterion,
        optimizer,
        device,
        checkpoint_dir,
        teacher_momentum=0.996,
        gradient_clip=3.0,
        learning_rate_schedule=None,
        teacher_momentum_schedule=None,
        teacher_temperature_schedule=None,
        mixed_precision=False,
    ):
        self.student = student
        self.teacher = teacher
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)

        self.teacher_momentum = float(
            teacher_momentum
        )
        self.gradient_clip = gradient_clip

        self.learning_rate_schedule = (
            learning_rate_schedule
        )
        self.teacher_momentum_schedule = (
            teacher_momentum_schedule
        )
        self.teacher_temperature_schedule = (
            teacher_temperature_schedule
        )

        self.mixed_precision = bool(
            mixed_precision
        )

        self.amp_enabled = (
            self.mixed_precision
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )

        print(
            "Mixed precision:",
            "enabled"
            if self.amp_enabled
            else "disabled",
        )

        # Number of completed training iterations.
        # It is restored when resuming from a checkpoint.
        self.global_step = 0

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.student = self.student.to(
            self.device
        )
        self.teacher = self.teacher.to(
            self.device
        )
        self.criterion = self.criterion.to(
            self.device
        )

    def _validate_schedule_position(
        self,
        schedule,
        schedule_name,
    ):
        """
        Verify that global_step is a valid schedule index.
        """

        if self.global_step < 0:
            raise ValueError(
                "global_step cannot be negative."
            )

        if self.global_step >= len(schedule):
            raise IndexError(
                f"{schedule_name} is shorter than the "
                "number of executed training steps. "
                f"global_step={self.global_step}, "
                f"schedule_length={len(schedule)}"
            )

    def _get_learning_rate(self):
        """
        Return the learning rate for the current step.
        """

        if self.learning_rate_schedule is None:
            return float(
                self.optimizer.param_groups[0]["lr"]
            )

        self._validate_schedule_position(
            self.learning_rate_schedule,
            "Learning-rate schedule",
        )

        return float(
            self.learning_rate_schedule[
                self.global_step
            ]
        )

    def _set_learning_rate(
        self,
        learning_rate,
    ):
        """
        Apply the learning rate to every optimizer group.
        """

        for parameter_group in (
            self.optimizer.param_groups
        ):
            parameter_group["lr"] = (
                learning_rate
            )

    def _get_teacher_momentum(self):
        """
        Return teacher EMA momentum for the current step.
        """

        if self.teacher_momentum_schedule is None:
            return float(
                self.teacher_momentum
            )

        self._validate_schedule_position(
            self.teacher_momentum_schedule,
            "Teacher-momentum schedule",
        )

        return float(
            self.teacher_momentum_schedule[
                self.global_step
            ]
        )

    def _get_teacher_temperature(self):
        """
        Return teacher temperature for the current step.
        """

        if (
            self.teacher_temperature_schedule
            is None
        ):
            return float(
                self.criterion.teacher_temperature
            )

        self._validate_schedule_position(
            self.teacher_temperature_schedule,
            "Teacher-temperature schedule",
        )

        return float(
            self.teacher_temperature_schedule[
                self.global_step
            ]
        )

    def train_one_epoch(
        self,
        epoch,
    ):
        self.student.train()
        self.teacher.eval()

        running_loss = 0.0
        number_of_batches = 0
        skipped_optimizer_steps = 0

        last_learning_rate = None
        last_teacher_momentum = None
        last_teacher_temperature = None

        progress_bar = tqdm(
            self.dataloader,
            desc=f"DINO epoch {epoch}",
        )

        for views, _ in progress_bar:
            views = [
                view.to(
                    self.device,
                    non_blocking=True,
                )
                for view in views
            ]

            if len(views) < 3:
                raise ValueError(
                    "DINO requires at least two "
                    "global views and one local view."
                )

            # Teacher sees only the two global views.
            global_views = views[:2]

            current_learning_rate = (
                self._get_learning_rate()
            )
            current_teacher_momentum = (
                self._get_teacher_momentum()
            )
            current_teacher_temperature = (
                self._get_teacher_temperature()
            )

            self._set_learning_rate(
                current_learning_rate
            )

            self.criterion.set_teacher_temperature(
                current_teacher_temperature
            )

            self.optimizer.zero_grad(
                set_to_none=True,
            )

            # Run neural networks in mixed precision.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                student_outputs = [
                    self.student(view)
                    for view in views
                ]

                student_output = torch.cat(
                    student_outputs,
                    dim=0,
                )

                with torch.no_grad():
                    teacher_outputs = [
                        self.teacher(view)
                        for view in global_views
                    ]

                    teacher_output = torch.cat(
                        teacher_outputs,
                        dim=0,
                    )

            # Compute the DINO probability loss in float32
            # for improved numerical stability.
            loss = self.criterion(
                student_output=(
                    student_output.float()
                ),
                teacher_output=(
                    teacher_output.float()
                ),
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite DINO loss detected: "
                    f"{loss.detach().item()}"
                )

            self.scaler.scale(
                loss
            ).backward()

            if self.gradient_clip is not None:
                # Convert scaled gradients back to their
                # real values before clipping.
                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    max_norm=float(
                        self.gradient_clip
                    ),
                )

            scale_before_step = (
                self.scaler.get_scale()
            )

            self.scaler.step(
                self.optimizer
            )
            self.scaler.update()

            scale_after_step = (
                self.scaler.get_scale()
            )

            # A lower scale means the optimizer step was
            # skipped because AMP detected overflow.
            optimizer_step_skipped = (
                self.amp_enabled
                and scale_after_step
                < scale_before_step
            )

            if optimizer_step_skipped:
                skipped_optimizer_steps += 1
            else:
                # Update teacher only after a successful
                # student optimizer step.
                update_teacher(
                    student=self.student,
                    teacher=self.teacher,
                    momentum=(
                        current_teacher_momentum
                    ),
                )

            loss_value = float(
                loss.detach().item()
            )

            running_loss += loss_value
            number_of_batches += 1

            last_learning_rate = (
                current_learning_rate
            )
            last_teacher_momentum = (
                current_teacher_momentum
            )
            last_teacher_temperature = (
                current_teacher_temperature
            )

            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                lr=(
                    f"{current_learning_rate:.2e}"
                ),
                momentum=(
                    f"{current_teacher_momentum:.6f}"
                ),
                temperature=(
                    f"{current_teacher_temperature:.4f}"
                ),
                amp_scale=(
                    f"{scale_after_step:.0f}"
                    if self.amp_enabled
                    else "off"
                ),
            )

            # Advance all per-step schedules.
            self.global_step += 1

        if number_of_batches == 0:
            raise RuntimeError(
                "The DINO dataloader produced "
                "no batches."
            )

        average_loss = (
            running_loss / number_of_batches
        )

        return {
            "dino_loss": average_loss,
            "learning_rate": (
                last_learning_rate
            ),
            "teacher_momentum": (
                last_teacher_momentum
            ),
            "teacher_temperature": (
                last_teacher_temperature
            ),
            "skipped_optimizer_steps": (
                skipped_optimizer_steps
            ),
        }

    def save_checkpoint(
        self,
        epoch,
        epoch_metrics,
    ):
        checkpoint_path = (
            self.checkpoint_dir
            / f"dino_epoch_{epoch:03d}.pt"
        )

        torch.save(
            {
                "epoch": int(epoch),
                "global_step": int(
                    self.global_step
                ),
                "student_state_dict": (
                    self.student.state_dict()
                ),
                "teacher_state_dict": (
                    self.teacher.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "scaler_state_dict": (
                    self.scaler.state_dict()
                ),
                # Includes the DINO center buffer.
                "criterion_state_dict": (
                    self.criterion.state_dict()
                ),
                "average_loss": float(
                    epoch_metrics["dino_loss"]
                ),
                "learning_rate": float(
                    epoch_metrics[
                        "learning_rate"
                    ]
                ),
                "teacher_momentum": float(
                    epoch_metrics[
                        "teacher_momentum"
                    ]
                ),
                "teacher_temperature": float(
                    epoch_metrics[
                        "teacher_temperature"
                    ]
                ),
                "skipped_optimizer_steps": int(
                    epoch_metrics[
                        "skipped_optimizer_steps"
                    ]
                ),
                "amp_enabled": bool(
                    self.amp_enabled
                ),
            },
            checkpoint_path,
        )

        return checkpoint_path

    def fit(
        self,
        epochs,
        start_epoch=1,
    ):
        """
        Train from start_epoch through epochs, inclusive.
        """

        epochs = int(epochs)
        start_epoch = int(start_epoch)

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        if start_epoch < 1:
            raise ValueError(
                "start_epoch must be at least 1."
            )

        history = []

        if start_epoch > epochs:
            print(
                "Training is already complete. "
                f"start_epoch={start_epoch}, "
                f"epochs={epochs}"
            )
            return history

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):
            epoch_metrics = (
                self.train_one_epoch(
                    epoch=epoch
                )
            )

            checkpoint_path = (
                self.save_checkpoint(
                    epoch=epoch,
                    epoch_metrics=epoch_metrics,
                )
            )

            history.append(
                {
                    "epoch": epoch,
                    "dino_loss": (
                        epoch_metrics[
                            "dino_loss"
                        ]
                    ),
                    "learning_rate": (
                        epoch_metrics[
                            "learning_rate"
                        ]
                    ),
                    "teacher_momentum": (
                        epoch_metrics[
                            "teacher_momentum"
                        ]
                    ),
                    "teacher_temperature": (
                        epoch_metrics[
                            "teacher_temperature"
                        ]
                    ),
                    "skipped_optimizer_steps": (
                        epoch_metrics[
                            "skipped_optimizer_steps"
                        ]
                    ),
                }
            )

            print(
                f"Epoch {epoch}/{epochs} "
                f"| DINO loss: "
                f"{epoch_metrics['dino_loss']:.6f} "
                f"| LR: "
                f"{epoch_metrics['learning_rate']:.8f} "
                f"| Teacher momentum: "
                f"{epoch_metrics['teacher_momentum']:.6f} "
                f"| Teacher temperature: "
                f"{epoch_metrics['teacher_temperature']:.4f} "
                f"| AMP skipped steps: "
                f"{epoch_metrics['skipped_optimizer_steps']}"
            )

            print(
                "Saved checkpoint:",
                checkpoint_path,
            )

        return history