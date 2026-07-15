import math

import numpy as np


def cosine_schedule(
    start_value,
    final_value,
    total_steps,
    warmup_steps=0,
    warmup_start_value=0.0,
):
    """
    Create a per-training-step schedule.

    The optional warm-up increases linearly from
    warmup_start_value to start_value.

    After warm-up, the value follows cosine decay
    from start_value to final_value.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    if warmup_steps > total_steps:
        raise ValueError(
            "warmup_steps cannot exceed total_steps."
        )

    schedule = np.empty(total_steps, dtype=np.float64)

    if warmup_steps > 0:
        schedule[:warmup_steps] = np.linspace(
            warmup_start_value,
            start_value,
            warmup_steps,
            endpoint=False,
        )

    remaining_steps = total_steps - warmup_steps

    if remaining_steps > 0:
        for index in range(remaining_steps):
            progress = index / max(remaining_steps - 1, 1)

            schedule[warmup_steps + index] = (
                final_value
                + 0.5
                * (start_value - final_value)
                * (1.0 + math.cos(math.pi * progress))
            )

    return schedule
