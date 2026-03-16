"""Optimizers for training neural networks.

All optimizers follow: step() applies gradients, zero_grad() clears them.
"""

from __future__ import annotations

from .tensor import Tensor


class SGD:
    """Stochastic gradient descent with optional momentum."""

    __slots__ = ("params", "lr", "momentum", "velocities")

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities: list[list[float]] = (
            [[0.0] * len(p.data) for p in params] if momentum > 0 else []
        )

    def step(self) -> None:
        lr = self.lr
        if self.momentum > 0:
            mu = self.momentum
            for idx, p in enumerate(self.params):
                if p.grad is None:
                    continue
                v = self.velocities[idx]
                gd = p.grad.data
                pd = p.data
                for i in range(len(pd)):
                    v[i] = mu * v[i] - lr * gd[i]
                    pd[i] += v[i]
        else:
            for p in self.params:
                if p.grad is None:
                    continue
                gd = p.grad.data
                pd = p.data
                for i in range(len(pd)):
                    pd[i] -= lr * gd[i]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class Adam:
    """Adam optimizer (Kingma & Ba, 2014)."""

    __slots__ = ("params", "lr", "beta1", "beta2", "eps", "t", "m", "v")

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: list[list[float]] = [[0.0] * len(p.data) for p in params]
        self.v: list[list[float]] = [[0.0] * len(p.data) for p in params]

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        # Bias-corrected learning rate
        lr_t = self.lr * (1.0 - b2**self.t) ** 0.5 / (1.0 - b1**self.t)
        eps = self.eps

        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            m, v = self.m[idx], self.v[idx]
            gd = p.grad.data
            pd = p.data
            for i in range(len(pd)):
                g = gd[i]
                m[i] = b1 * m[i] + (1.0 - b1) * g
                v[i] = b2 * v[i] + (1.0 - b2) * g * g
                pd[i] -= lr_t * m[i] / (v[i] ** 0.5 + eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()


class RMSProp:
    """RMSProp optimizer."""

    __slots__ = ("params", "lr", "alpha", "eps", "v")

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v: list[list[float]] = [[0.0] * len(p.data) for p in params]

    def step(self) -> None:
        alpha = self.alpha
        lr = self.lr
        eps = self.eps
        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue
            v = self.v[idx]
            gd = p.grad.data
            pd = p.data
            for i in range(len(pd)):
                g = gd[i]
                v[i] = alpha * v[i] + (1.0 - alpha) * g * g
                pd[i] -= lr * g / (v[i] ** 0.5 + eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
