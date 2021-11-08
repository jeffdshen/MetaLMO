# Copyright (c) Jeffrey Shen

"""Learning rate schedulers."""

import torch.optim.lr_scheduler as sched


def get_linear_warmup_power_decay_scheduler(
    optimizer, warmup_steps, max_num_steps, end_multiplier=0.0, power=1
):
    """Uses a power function a * x^power + b, such that it equals 1.0 at start_step=1
    and the end_multiplier at end_step. Afterwards, returns the end_multiplier forever.
    For the first warmup_steps, linearly increase the learning rate until it hits the power
    learning rate.
    """

    # a = end_lr - start_lr / (end_step ** power - start_step ** power)
    start_multiplier = 1.0
    start_step = 1
    scale = (end_multiplier - start_multiplier) / (
        max_num_steps ** power - start_step ** power
    )
    # b = start_lr - scale * start_step ** power
    constant = start_multiplier - scale * (start_step ** power)

    def lr_lambda(step):
        step = start_step + step
        if step < warmup_steps:
            warmup_multiplier = scale * (warmup_steps ** power) + constant
            return float(step) / float(max(1, warmup_steps)) * warmup_multiplier
        elif step >= max_num_steps:
            return end_multiplier
        else:
            return scale * (step ** power) + constant

    return sched.LambdaLR(optimizer, lr_lambda)
