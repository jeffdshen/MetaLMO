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


class GradSaver:
    def __init__(self):
        super().__init__()

    def apply(self, params, scale=1.0):
        with torch.no_grad():
            for param, grad in zip(params, self.grads):
                if grad is None:
                    continue
                if param.grad is None:
                    param.grad = grad * scale
                else:
                    param.grad.add_(grad * scale)
            self.grads = None

    def __call__(self, grads):
        self.grads = [
            grad.detach().clone() if grad is not None else None for grad in grads
        ]
        return grads
