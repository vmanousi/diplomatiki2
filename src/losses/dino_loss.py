import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """
    DINO cross-view self-distillation loss.

    The teacher receives the two global views.
    The student receives all global and local views.
    Matching teacher/student views with the same global index are skipped.
    """

    def __init__(
        self,
        out_dim,
        num_student_views,
        teacher_temperature=0.04,
        student_temperature=0.1,
        center_momentum=0.9,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_student_views = num_student_views
        self.teacher_temperature = teacher_temperature
        self.student_temperature = student_temperature
        self.center_momentum = center_momentum

        self.register_buffer(
            "center",
            torch.zeros(1, out_dim),
        )

    def forward(
        self,
        student_output,
        teacher_output,
    ):
        """
        Parameters
        ----------
        student_output:
            Tensor of shape:
            [num_student_views * batch_size, out_dim]

        teacher_output:
            Tensor of shape:
            [2 * batch_size, out_dim]
        """

        student_chunks = (
            student_output
            / self.student_temperature
        ).chunk(self.num_student_views)

        teacher_probs = F.softmax(
            (
                teacher_output
                - self.center
            )
            / self.teacher_temperature,
            dim=-1,
        ).detach()

        teacher_chunks = teacher_probs.chunk(2)

        total_loss = 0.0
        number_of_terms = 0

        for teacher_index, teacher_view in enumerate(
            teacher_chunks
        ):
            for student_index, student_view in enumerate(
                student_chunks
            ):
                # Skip the matching global view:
                # teacher G1 vs student G1
                # teacher G2 vs student G2
                if student_index == teacher_index:
                    continue

                loss = torch.sum(
                    -teacher_view
                    * F.log_softmax(
                        student_view,
                        dim=-1,
                    ),
                    dim=-1,
                )

                total_loss += loss.mean()
                number_of_terms += 1

        total_loss /= number_of_terms

        self.update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(
            teacher_output,
            dim=0,
            keepdim=True,
        )

        self.center.mul_(
            self.center_momentum
        ).add_(
            batch_center
            * (1.0 - self.center_momentum)
        )
