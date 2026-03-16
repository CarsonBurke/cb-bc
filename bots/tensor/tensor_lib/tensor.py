"""Lightweight autograd tensor library — pure stdlib Python.

Uses flat lists + operator.mul/map/sum for C-backed inner loops.
Supports reverse-mode automatic differentiation for backpropagation.
"""

from __future__ import annotations

import math
import random
from operator import mul as _mul


class Tensor:
    """N-dimensional tensor with optional autograd support.

    Data is stored as a flat list of floats. Shape is a tuple of ints.
    When requires_grad=True, operations build a computation graph
    for reverse-mode autodiff via .backward().
    """

    __slots__ = ("data", "shape", "grad", "_backward", "requires_grad", "_prev")

    def __init__(
        self,
        data: list[float] | list[list[float]] | float,
        shape: tuple[int, ...] | None = None,
        requires_grad: bool = False,
    ):
        if isinstance(data, (int, float)):
            data = [float(data)]
            shape = shape or (1,)
        elif shape is None:
            if data and isinstance(data[0], (list, tuple)):
                rows = len(data)
                cols = len(data[0])
                shape = (rows, cols)
                data = [float(x) for row in data for x in row]
            else:
                shape = (len(data),)
        self.data: list[float] = data
        self.shape: tuple[int, ...] = shape
        self.grad: Tensor | None = None
        self._backward = _noop
        self.requires_grad: bool = requires_grad
        self._prev: tuple[Tensor, ...] = ()

    # ---- Factories ----

    @staticmethod
    def zeros(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
        n = 1
        for s in shape:
            n *= s
        return Tensor([0.0] * n, shape, requires_grad)

    @staticmethod
    def ones(shape: tuple[int, ...], requires_grad: bool = False) -> Tensor:
        n = 1
        for s in shape:
            n *= s
        return Tensor([1.0] * n, shape, requires_grad)

    @staticmethod
    def randn(
        shape: tuple[int, ...], requires_grad: bool = False, scale: float = 0.01
    ) -> Tensor:
        n = 1
        for s in shape:
            n *= s
        return Tensor(
            [random.gauss(0.0, scale) for _ in range(n)], shape, requires_grad
        )

    @staticmethod
    def xavier(fan_in: int, fan_out: int) -> Tensor:
        """Xavier/Glorot uniform initialization."""
        s = math.sqrt(6.0 / (fan_in + fan_out))
        data = [random.uniform(-s, s) for _ in range(fan_in * fan_out)]
        return Tensor(data, (fan_in, fan_out), requires_grad=True)

    @staticmethod
    def from_list(lst: list[float], shape: tuple[int, ...] | None = None) -> Tensor:
        return Tensor(list(lst), shape)

    # ---- Properties ----

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __repr__(self) -> str:
        preview = self.data[:6]
        suffix = ", ..." if len(self.data) > 6 else ""
        return f"Tensor(shape={self.shape}, data=[{', '.join(f'{x:.4f}' for x in preview)}{suffix}])"

    def __len__(self) -> int:
        return self.shape[0]

    def item(self) -> float:
        return self.data[0]

    # ---- Indexing ----

    def __getitem__(self, idx: int) -> Tensor | float:
        if len(self.shape) == 1:
            return self.data[idx]
        stride = 1
        for s in self.shape[1:]:
            stride *= s
        start = idx * stride
        return Tensor(self.data[start : start + stride], self.shape[1:])

    # ---- Shape ops ----

    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        return Tensor(list(self.data), shape, self.requires_grad)

    def flatten(self) -> Tensor:
        return Tensor(list(self.data), (len(self.data),), self.requires_grad)

    def T(self) -> Tensor:
        """Transpose a 2D tensor. Backward-aware."""
        assert len(self.shape) == 2
        r, c = self.shape
        d = self.data
        out_data = [0.0] * len(d)
        for i in range(r):
            base = i * c
            for j in range(c):
                out_data[j * r + i] = d[base + j]
        out = Tensor(out_data, (c, r))
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                og = out.grad.data
                for i in range(c):
                    base = i * r
                    for j in range(r):
                        self.grad.data[j * c + i] += og[base + j]

        out._backward = _backward
        return out

    # ---- Elementwise arithmetic ----

    def __add__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, (int, float)):
            out = Tensor([x + other for x in self.data], self.shape)
            out.requires_grad = self.requires_grad
            out._prev = (self,)

            def _backward():
                if self.requires_grad:
                    _ensure_grad(self)
                    for i in range(len(self.data)):
                        self.grad.data[i] += out.grad.data[i]

            out._backward = _backward
            return out

        if self.shape == other.shape:
            out = Tensor(
                [a + b for a, b in zip(self.data, other.data)], self.shape
            )
        elif (
            len(other.shape) == 1
            and len(self.shape) == 2
            and other.shape[0] == self.shape[1]
        ):
            # (M, N) + (N,) broadcast
            d = list(self.data)
            od = other.data
            N = self.shape[1]
            for i in range(self.shape[0]):
                base = i * N
                for j in range(N):
                    d[base + j] += od[j]
            out = Tensor(d, self.shape)
        elif (
            len(self.shape) == 1
            and len(other.shape) == 2
            and self.shape[0] == other.shape[1]
        ):
            # (N,) + (M, N) broadcast
            d = list(other.data)
            sd = self.data
            N = other.shape[1]
            for i in range(other.shape[0]):
                base = i * N
                for j in range(N):
                    d[base + j] += sd[j]
            out = Tensor(d, other.shape)
        else:
            raise ValueError(f"Cannot broadcast {self.shape} + {other.shape}")

        out.requires_grad = self.requires_grad or other.requires_grad
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                _accumulate_grad(self, out.grad)
            if other.requires_grad:
                _accumulate_grad(other, out.grad)

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> Tensor:
        return self.__add__(other)

    def __sub__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, (int, float)):
            return self.__add__(-other)
        return self.__add__(other * -1.0)

    def __rsub__(self, other: float) -> Tensor:
        return (self * -1.0).__add__(other)

    def __mul__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, (int, float)):
            out = Tensor([x * other for x in self.data], self.shape)
            out.requires_grad = self.requires_grad
            out._prev = (self,)

            c = other  # capture scalar

            def _backward():
                if self.requires_grad:
                    _ensure_grad(self)
                    for i in range(len(self.data)):
                        self.grad.data[i] += out.grad.data[i] * c

            out._backward = _backward
            return out

        assert self.shape == other.shape, f"Shape mismatch: {self.shape} vs {other.shape}"
        out = Tensor(list(map(_mul, self.data, other.data)), self.shape)
        out.requires_grad = self.requires_grad or other.requires_grad
        out._prev = (self, other)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                for i in range(len(self.data)):
                    self.grad.data[i] += out.grad.data[i] * other.data[i]
            if other.requires_grad:
                _ensure_grad(other)
                for i in range(len(other.data)):
                    other.grad.data[i] += out.grad.data[i] * self.data[i]

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> Tensor:
        return self.__mul__(other)

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __truediv__(self, other: float) -> Tensor:
        return self * (1.0 / other)

    # ---- Matrix multiply ----

    def matmul(self, other: Tensor) -> Tensor:
        """(M, K) @ (K, N) -> (M, N).

        Uses strided slicing (od[j::N]) to extract columns of B,
        then C-backed sum(map(operator.mul, ...)) for the dot product.
        """
        assert len(self.shape) == 2 and len(other.shape) == 2
        M, K = self.shape
        K2, N = other.shape
        assert K == K2, f"matmul dim mismatch: {self.shape} @ {other.shape}"

        od = other.data
        sd = self.data
        out_data = [0.0] * (M * N)

        for i in range(M):
            rs = i * K
            a_row = sd[rs : rs + K]
            ob = i * N
            for j in range(N):
                out_data[ob + j] = sum(map(_mul, a_row, od[j::N]))

        out = Tensor(out_data, (M, N))
        out.requires_grad = self.requires_grad or other.requires_grad
        out._prev = (self, other)

        def _backward():
            og = out.grad.data
            if self.requires_grad:
                _ensure_grad(self)
                # dA = dOut @ B^T
                for i in range(M):
                    for k in range(K):
                        s = 0.0
                        ob = i * N
                        for j in range(N):
                            s += og[ob + j] * od[k * N + j]
                        self.grad.data[i * K + k] += s
            if other.requires_grad:
                _ensure_grad(other)
                # dB = A^T @ dOut
                for k in range(K):
                    for j in range(N):
                        s = 0.0
                        for i in range(M):
                            s += sd[i * K + k] * og[i * N + j]
                        other.grad.data[k * N + j] += s

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    # ---- Activations ----

    def relu(self) -> Tensor:
        out = Tensor([x if x > 0.0 else 0.0 for x in self.data], self.shape)
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                for i in range(len(self.data)):
                    if self.data[i] > 0.0:
                        self.grad.data[i] += out.grad.data[i]

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        out = Tensor([math.tanh(x) for x in self.data], self.shape)
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                for i in range(len(out.data)):
                    self.grad.data[i] += out.grad.data[i] * (1.0 - out.data[i] ** 2)

        out._backward = _backward
        return out

    def silu(self) -> Tensor:
        """SiLU / Swish activation: x * sigmoid(x). Numerically stable."""
        out_data = []
        sig_data = []
        for x in self.data:
            if x >= 0:
                s = 1.0 / (1.0 + math.exp(-x))
            else:
                e = math.exp(x)
                s = e / (1.0 + e)
            sig_data.append(s)
            out_data.append(x * s)
        out = Tensor(out_data, self.shape)
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                # d/dx[x * sig(x)] = sig(x) + x * sig(x) * (1 - sig(x))
                #                   = sig(x) * (1 + x * (1 - sig(x)))
                for i in range(len(self.data)):
                    si = sig_data[i]
                    self.grad.data[i] += out.grad.data[i] * (
                        si + self.data[i] * si * (1.0 - si)
                    )

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        out_data = []
        for x in self.data:
            if x >= 0:
                s = 1.0 / (1.0 + math.exp(-x))
            else:
                e = math.exp(x)
                s = e / (1.0 + e)
            out_data.append(s)
        out = Tensor(out_data, self.shape)
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                for i in range(len(out.data)):
                    self.grad.data[i] += (
                        out.grad.data[i] * out.data[i] * (1.0 - out.data[i])
                    )

        out._backward = _backward
        return out

    def softmax(self) -> Tensor:
        """Row-wise softmax (last axis). Numerically stable."""
        if len(self.shape) == 1:
            mx = max(self.data)
            exps = [math.exp(x - mx) for x in self.data]
            s = sum(exps)
            inv_s = 1.0 / s
            out = Tensor([e * inv_s for e in exps], self.shape)
        else:
            M, N = self.shape
            out_data = [0.0] * (M * N)
            for i in range(M):
                base = i * N
                row = self.data[base : base + N]
                mx = max(row)
                exps = [math.exp(x - mx) for x in row]
                s = sum(exps)
                inv_s = 1.0 / s
                for j in range(N):
                    out_data[base + j] = exps[j] * inv_s
            out = Tensor(out_data, self.shape)

        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                if len(self.shape) == 1:
                    N = self.shape[0]
                    og = out.grad.data
                    od = out.data
                    # Efficient: dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
                    dot = sum(map(_mul, og, od))
                    for i in range(N):
                        self.grad.data[i] += od[i] * (og[i] - dot)
                else:
                    M, N = self.shape
                    og = out.grad.data
                    od = out.data
                    for m in range(M):
                        base = m * N
                        dot = sum(
                            og[base + j] * od[base + j] for j in range(N)
                        )
                        for i in range(N):
                            self.grad.data[base + i] += od[base + i] * (
                                og[base + i] - dot
                            )

        out._backward = _backward
        return out

    def log(self) -> Tensor:
        """Element-wise natural log. Clamped to avoid -inf."""
        out = Tensor([math.log(max(x, 1e-12)) for x in self.data], self.shape)
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                for i in range(len(self.data)):
                    self.grad.data[i] += out.grad.data[i] / max(self.data[i], 1e-12)

        out._backward = _backward
        return out

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        out = Tensor(
            [max(lo, min(hi, x)) for x in self.data], self.shape
        )
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                for i in range(len(self.data)):
                    if lo < self.data[i] < hi:
                        self.grad.data[i] += out.grad.data[i]

        out._backward = _backward
        return out

    # ---- Reductions ----

    def sum(self) -> Tensor:
        out = Tensor([sum(self.data)], (1,))
        out.requires_grad = self.requires_grad
        out._prev = (self,)

        def _backward():
            if self.requires_grad:
                _ensure_grad(self)
                g = out.grad.data[0]
                for i in range(len(self.data)):
                    self.grad.data[i] += g

        out._backward = _backward
        return out

    def mean(self) -> Tensor:
        return self.sum() * (1.0 / len(self.data))

    # ---- Autograd ----

    def backward(self) -> None:
        """Reverse-mode autodiff via topological sort."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build(t: Tensor) -> None:
            tid = id(t)
            if tid not in visited:
                visited.add(tid)
                for p in t._prev:
                    _build(p)
                topo.append(t)

        _build(self)

        self.grad = Tensor([1.0] * len(self.data), self.shape)
        for t in reversed(topo):
            t._backward()

    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> Tensor:
        """Return a copy with no grad tracking."""
        return Tensor(list(self.data), self.shape, requires_grad=False)


# ---- Module helpers ----


def _noop() -> None:
    pass


def _ensure_grad(t: Tensor) -> None:
    if t.grad is None:
        t.grad = Tensor.zeros(t.shape)


def _accumulate_grad(tensor: Tensor, grad: Tensor) -> None:
    """Accumulate gradient with broadcasting reduction."""
    _ensure_grad(tensor)
    if tensor.shape == grad.shape:
        tg = tensor.grad.data
        gg = grad.data
        for i in range(len(tg)):
            tg[i] += gg[i]
    elif len(tensor.shape) == 1 and len(grad.shape) == 2:
        # (N,) from (M, N) — sum over axis 0
        N = tensor.shape[0]
        M = grad.shape[0]
        tg = tensor.grad.data
        gg = grad.data
        for j in range(N):
            s = 0.0
            for i in range(M):
                s += gg[i * N + j]
            tg[j] += s
    elif len(tensor.shape) == 2 and len(grad.shape) == 1:
        # Rare: (M, N) from (N,) — tile grad across rows
        M, N = tensor.shape
        tg = tensor.grad.data
        gg = grad.data
        for i in range(M):
            base = i * N
            for j in range(N):
                tg[base + j] += gg[j]
    else:
        raise ValueError(
            f"Cannot reduce grad {grad.shape} to {tensor.shape}"
        )
