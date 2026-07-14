import torch


@torch.no_grad()
def update_teacher(
    student,
    teacher,
    momentum,
):
    """
    Update teacher parameters using an exponential moving average
    of the student parameters.

    teacher = momentum * teacher
              + (1 - momentum) * student
    """

    for student_parameter, teacher_parameter in zip(
        student.parameters(),
        teacher.parameters(),
    ):
        teacher_parameter.data.mul_(momentum).add_(
            student_parameter.data,
            alpha=1.0 - momentum,
        )
