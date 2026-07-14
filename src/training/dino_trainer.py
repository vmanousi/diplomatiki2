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

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.student.to(self.device)
        self.teacher.to(self.device)

    def train_one_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()

        running_loss = 0.0
        number_of_batches = 0

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

            self.optimizer.zero_grad(
                set_to_none=True,
            )

            student_outputs = []

            for view in views:
                student_outputs.append(
                    self.student(view)
                )

            student_output = torch.cat(
                student_outputs,
                dim=0,
            )

            with torch.no_grad():
                teacher_outputs = []

                for view in global_views:
                    teacher_outputs.append(
                        self.teacher(view)
                    )

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
                    f"Non-finite DINO loss detected: {loss.item()}"
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
                momentum=self.teacher_momentum,
            )

            running_loss += loss.detach().item()
            number_of_batches += 1

            progress_bar.set_postfix(
                loss=f"{loss.detach().item():.4f}"
            )

        if number_of_batches == 0:
            raise RuntimeError(
                "The DINO dataloader produced no batches."
            )

        return running_loss / number_of_batches

    def save_checkpoint(self, epoch, average_loss):
        checkpoint_path = (
            self.checkpoint_dir
            / f"dino_epoch_{epoch:03d}.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": (
                    self.student.state_dict()
                ),
                "teacher_state_dict": (
                    self.teacher.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "average_loss": average_loss,
            },
            checkpoint_path,
        )

        return checkpoint_path

    def fit(self, epochs):
        history = []

        for epoch in range(1, epochs + 1):
            average_loss = self.train_one_epoch(
                epoch=epoch
            )

            checkpoint_path = self.save_checkpoint(
                epoch=epoch,
                average_loss=average_loss,
            )

            history.append(
                {
                    "epoch": epoch,
                    "dino_loss": average_loss,
                }
            )

            print(
                f"Epoch {epoch}/{epochs} "
                f"| DINO loss: {average_loss:.6f}"
            )
            print(
                "Saved checkpoint:",
                checkpoint_path,
            )

        return history
