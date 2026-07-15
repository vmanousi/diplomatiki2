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
    ):
        self.student = student
        self.teacher = teacher
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)

        self.teacher_momentum = teacher_momentum
        self.gradient_clip = gradient_clip

        self.learning_rate_schedule = learning_rate_schedule
        self.teacher_momentum_schedule = teacher_momentum_schedule

        # Counts the total number of processed batches.
        self.global_step = 0

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.student.to(self.device)
        self.teacher.to(self.device)
        self.criterion.to(self.device)

    def _get_learning_rate(self):
        """
        Return the learning rate for the current training step.
        """

        if self.learning_rate_schedule is None:
            return self.optimizer.param_groups[0]["lr"]

        if self.global_step >= len(self.learning_rate_schedule):
            raise IndexError(
                "The learning-rate schedule is shorter than "
                "the number of training steps."
            )

        return float(
            self.learning_rate_schedule[self.global_step]
        )

    def _set_learning_rate(self, learning_rate):
        """
        Apply the selected learning rate to every optimizer group.
        """

        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def _get_teacher_momentum(self):
        """
        Return the teacher momentum for the current training step.
        """

        if self.teacher_momentum_schedule is None:
            return self.teacher_momentum

        if self.global_step >= len(
            self.teacher_momentum_schedule
        ):
            raise IndexError(
                "The teacher-momentum schedule is shorter than "
                "the number of training steps."
            )

        return float(
            self.teacher_momentum_schedule[self.global_step]
        )

    def train_one_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()

        running_loss = 0.0
        number_of_batches = 0

        last_learning_rate = None
        last_teacher_momentum = None

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
                    "DINO requires at least two global views "
                    "and one local view."
                )

            global_views = views[:2]

            # Read and apply the values for this batch.
            current_learning_rate = (
                self._get_learning_rate()
            )
            current_teacher_momentum = (
                self._get_teacher_momentum()
            )

            self._set_learning_rate(
                current_learning_rate
            )

            self.optimizer.zero_grad(
                set_to_none=True,
            )

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

            loss = self.criterion(
                student_output=student_output,
                teacher_output=teacher_output,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite DINO loss detected: "
                    f"{loss.detach().item()}"
                )

            loss.backward()

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    max_norm=self.gradient_clip,
                )

            self.optimizer.step()

            update_teacher(
                student=self.student,
                teacher=self.teacher,
                momentum=current_teacher_momentum,
            )

            running_loss += loss.detach().item()
            number_of_batches += 1

            last_learning_rate = current_learning_rate
            last_teacher_momentum = (
                current_teacher_momentum
            )

            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                lr=f"{current_learning_rate:.2e}",
                momentum=(
                    f"{current_teacher_momentum:.6f}"
                ),
            )

            # Advance only after the current step is complete.
            self.global_step += 1

        if number_of_batches == 0:
            raise RuntimeError(
                "The DINO dataloader produced no batches."
            )

        average_loss = (
            running_loss / number_of_batches
        )

        return {
            "dino_loss": average_loss,
            "learning_rate": last_learning_rate,
            "teacher_momentum": (
                last_teacher_momentum
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
                "epoch": epoch,
                "global_step": self.global_step,
                "student_state_dict": (
                    self.student.state_dict()
                ),
                "teacher_state_dict": (
                    self.teacher.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "criterion_state_dict": (
                    self.criterion.state_dict()
                ),
                "average_loss": (
                    epoch_metrics["dino_loss"]
                ),
                "learning_rate": (
                    epoch_metrics["learning_rate"]
                ),
                "teacher_momentum": (
                    epoch_metrics[
                        "teacher_momentum"
                    ]
                ),
            },
            checkpoint_path,
        )

        return checkpoint_path

    def fit(self, epochs):
        history = []

        for epoch in range(1, epochs + 1):
            epoch_metrics = self.train_one_epoch(
                epoch=epoch
            )

            checkpoint_path = self.save_checkpoint(
                epoch=epoch,
                epoch_metrics=epoch_metrics,
            )

            history.append(
                {
                    "epoch": epoch,
                    "dino_loss": (
                        epoch_metrics["dino_loss"]
                    ),
                    "learning_rate": (
                        epoch_metrics["learning_rate"]
                    ),
                    "teacher_momentum": (
                        epoch_metrics[
                            "teacher_momentum"
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
                f"{epoch_metrics['teacher_momentum']:.6f}"
            )

            print(
                "Saved checkpoint:",
                checkpoint_path,
            )

        return history
