# Copyright (c) Jeffrey Shen

"""Custom optimizers."""

import torch
import torch.optim as optim


class NoopOptimizer(optim.Optimizer):
    """An optimizer that does nothing. Used to make use of scaler without optimizing."""

    def __init__(self, params):
        super().__init__(params, {})

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        return loss
