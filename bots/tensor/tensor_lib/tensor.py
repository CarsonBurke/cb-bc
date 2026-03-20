"""Lightweight tensor library — pure stdlib Python.

Uses flat lists + operator.mul/map/sum for C-backed inner loops.
Inference only — training is done in PyTorch.
"""

from __future__ import annotations

import math
import random
from operator import mul as _mul


class Tensor:
    """N-dimensional tensor for forward-pass computation.

    Data is stored as a flat list of floats. Shape is a tuple of ints.
    """

    __slots__ = ("data", "shape")

    def __init__(
        self,
        data: list[float] | list[list[float]] | float,
        shape: tuple[int, ...] | None = None,
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

    # ---- Factories ----

    @staticmethod
    def zeros(shape: tuple[int, ...]) -> Tensor:
        n = 1
        for s in shape:
            n *= s
        return Tensor([0.0] * n, shape)

    @staticmethod
    def ones(shape: tuple[int, ...]) -> Tensor:
        n = 1
        for s in shape:
            n *= s
        return Tensor([1.0] * n, shape)

    @staticmethod
    def randn(shape: tuple[int, ...], scale: float = 0.01) -> Tensor:
        n = 1
        for s in shape:
            n *= s
        return Tensor(
            [random.gauss(0.0, scale) for _ in range(n)], shape
        )

    @staticmethod
    def xavier(fan_in: int, fan_out: int) -> Tensor:
        """Xavier/Glorot uniform initialization."""
        s = math.sqrt(6.0 / (fan_in + fan_out))
        data = [random.uniform(-s, s) for _ in range(fan_in * fan_out)]
        return Tensor(data, (fan_in, fan_out))

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
        return Tensor(list(self.data), shape)

    def flatten(self) -> Tensor:
        return Tensor(list(self.data), (len(self.data),))

    def T(self) -> Tensor:
        """Transpose a 2D tensor."""
        assert len(self.shape) == 2
        r, c = self.shape
        d = self.data
        out_data = [0.0] * len(d)
        for i in range(r):
            base = i * c
            for j in range(c):
                out_data[j * r + i] = d[base + j]
        return Tensor(out_data, (c, r))

    # ---- Elementwise arithmetic ----

    def __add__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, (int, float)):
            return Tensor([x + other for x in self.data], self.shape)

        if self.shape == other.shape:
            return Tensor(
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
            return Tensor(d, self.shape)
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
            return Tensor(d, other.shape)
        else:
            raise ValueError(f"Cannot broadcast {self.shape} + {other.shape}")

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
            return Tensor([x * other for x in self.data], self.shape)

        assert self.shape == other.shape, f"Shape mismatch: {self.shape} vs {other.shape}"
        return Tensor(list(map(_mul, self.data, other.data)), self.shape)

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

        return Tensor(out_data, (M, N))

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    # ---- Activations ----

    def relu(self) -> Tensor:
        return Tensor([x if x > 0.0 else 0.0 for x in self.data], self.shape)

    def tanh(self) -> Tensor:
        return Tensor([math.tanh(x) for x in self.data], self.shape)

    def silu(self) -> Tensor:
        """SiLU / Swish activation: x * sigmoid(x). Numerically stable."""
        out_data = []
        for x in self.data:
            if x >= 0:
                s = 1.0 / (1.0 + math.exp(-x))
            else:
                e = math.exp(x)
                s = e / (1.0 + e)
            out_data.append(x * s)
        return Tensor(out_data, self.shape)

    def sigmoid(self) -> Tensor:
        out_data = []
        for x in self.data:
            if x >= 0:
                s = 1.0 / (1.0 + math.exp(-x))
            else:
                e = math.exp(x)
                s = e / (1.0 + e)
            out_data.append(s)
        return Tensor(out_data, self.shape)

    def softmax(self) -> Tensor:
        """Row-wise softmax (last axis). Numerically stable."""
        if len(self.shape) == 1:
            mx = max(self.data)
            exps = [math.exp(x - mx) for x in self.data]
            s = sum(exps)
            inv_s = 1.0 / s
            return Tensor([e * inv_s for e in exps], self.shape)
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
            return Tensor(out_data, self.shape)

    def log(self) -> Tensor:
        """Element-wise natural log. Clamped to avoid -inf."""
        return Tensor([math.log(max(x, 1e-12)) for x in self.data], self.shape)

    def clamp(self, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        return Tensor(
            [max(lo, min(hi, x)) for x in self.data], self.shape
        )

    # ---- Reductions ----

    def sum(self) -> Tensor:
        return Tensor([sum(self.data)], (1,))

    def mean(self) -> Tensor:
        return self.sum() * (1.0 / len(self.data))
