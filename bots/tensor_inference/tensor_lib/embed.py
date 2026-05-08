"""Positional embedding for spatial input encoding."""

from __future__ import annotations

import math

from .tensor import Tensor


class SinusoidalPositionEmbedding:
    """Fixed sinusoidal position encoding (Vaswani et al.).

    Encodes (x, y) coordinates into a d_model-dimensional vector using
    interleaved sin/cos at geometrically spaced frequencies.
    First half encodes x, second half encodes y.
    """

    __slots__ = ("d_model", "freqs", "_quarter", "_cache")

    def __init__(self, d_model: int, base: float = 10000.0):
        self.d_model = d_model
        self._quarter = d_model // 4
        # Precompute frequency bands (C-backed math.exp)
        self.freqs: list[float] = [
            1.0 / (base ** (2.0 * i / d_model))
            for i in range(self._quarter)
        ]
        self._cache: dict[tuple[int, int], tuple[float, ...]] = {}

    def encode(self, x: int, y: int) -> tuple[float, ...]:
        """Encode a single (x, y) position -> d_model floats."""
        key = (x, y)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        q = self._quarter
        freqs = self.freqs
        vec = [0.0] * self.d_model
        half = q * 2

        # X encoding: sin/cos pairs
        for i in range(q):
            angle = x * freqs[i]
            vec[2 * i] = math.sin(angle)
            vec[2 * i + 1] = math.cos(angle)

        # Y encoding: sin/cos pairs
        for i in range(q):
            angle = y * freqs[i]
            vec[half + 2 * i] = math.sin(angle)
            vec[half + 2 * i + 1] = math.cos(angle)

        cached = tuple(vec)
        self._cache[key] = cached
        return cached

    def encode_batch(self, positions: list[tuple[int, int]]) -> Tensor:
        """Encode multiple positions -> (N, d_model) tensor."""
        data: list[float] = []
        for x, y in positions:
            data.extend(self.encode(x, y))
        return Tensor(data, (len(positions), self.d_model))
