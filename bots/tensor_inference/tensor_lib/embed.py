"""Positional embedding schemes for spatial input encoding.

Provides sinusoidal (fixed) and learned embeddings for encoding
(x, y) positions on the game grid.
"""

from __future__ import annotations

import math

from .tensor import Tensor


class SinusoidalPositionEmbedding:
    """Fixed sinusoidal position encoding (Vaswani et al.).

    Encodes (x, y) coordinates into a d_model-dimensional vector using
    interleaved sin/cos at geometrically spaced frequencies.
    First half encodes x, second half encodes y.
    """

    __slots__ = ("d_model", "freqs", "_quarter")

    def __init__(self, d_model: int, base: float = 10000.0):
        self.d_model = d_model
        self._quarter = d_model // 4
        # Precompute frequency bands (C-backed math.exp)
        self.freqs: list[float] = [
            1.0 / (base ** (2.0 * i / d_model))
            for i in range(self._quarter)
        ]

    def encode(self, x: int, y: int) -> list[float]:
        """Encode a single (x, y) position -> d_model floats."""
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

        return vec

    def encode_batch(self, positions: list[tuple[int, int]]) -> Tensor:
        """Encode multiple positions -> (N, d_model) tensor."""
        data: list[float] = []
        for x, y in positions:
            data.extend(self.encode(x, y))
        return Tensor(data, (len(positions), self.d_model))


class LearnedPositionEmbedding:
    """Learnable embedding table mapping integer indices to vectors.

    Useful for encoding relative offsets in a local grid (e.g., 5x5 around unit).
    """

    __slots__ = ("num_positions", "d_model", "weight")

    def __init__(self, num_positions: int, d_model: int):
        self.num_positions = num_positions
        self.d_model = d_model
        self.weight = Tensor.randn(
            (num_positions, d_model), scale=0.1
        )

    def __call__(self, indices: list[int]) -> Tensor:
        """Look up embeddings for indices. Returns (len(indices), d_model)."""
        data: list[float] = []
        w = self.weight.data
        d = self.d_model
        for idx in indices:
            start = idx * d
            data.extend(w[start : start + d])
        return Tensor(data, (len(indices), d))

    def parameters(self) -> list[Tensor]:
        return [self.weight]


class RelativeGridEmbedding:
    """Encodes a local NxN grid relative to center position.

    Combines learned offset embeddings with sinusoidal absolute
    position encodings. Output: (N*N, d_model) tensor.

    The learned component captures relative spatial patterns,
    while the sinusoidal component provides absolute position context.
    """

    __slots__ = ("grid_size", "d_model", "d_learned", "d_sin", "offset_embed", "sin_embed")

    def __init__(self, grid_size: int, d_model: int):
        self.grid_size = grid_size
        self.d_model = d_model
        self.d_learned = d_model // 2
        self.d_sin = d_model - self.d_learned
        self.offset_embed = LearnedPositionEmbedding(
            grid_size * grid_size, self.d_learned
        )
        self.sin_embed = SinusoidalPositionEmbedding(self.d_sin)

    def encode(self, center_x: int, center_y: int) -> Tensor:
        """Encode full grid centered at (center_x, center_y).

        Returns (grid_size * grid_size, d_model) tensor.
        """
        half = self.grid_size // 2
        n = self.grid_size * self.grid_size
        dl = self.d_learned
        ds = self.d_sin
        dm = self.d_model

        # Learned offset embeddings
        indices = list(range(n))
        offset_emb = self.offset_embed(indices)

        # Sinusoidal absolute position embeddings
        sin_data: list[float] = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                sin_data.extend(
                    self.sin_embed.encode(center_x + dx, center_y + dy)
                )

        # Concatenate: [learned | sinusoidal]
        out_data = [0.0] * (n * dm)
        for i in range(n):
            base = i * dm
            o_base = i * dl
            s_base = i * ds
            for j in range(dl):
                out_data[base + j] = offset_emb.data[o_base + j]
            for j in range(ds):
                out_data[base + dl + j] = sin_data[s_base + j]

        return Tensor(out_data, (n, dm))

    def parameters(self) -> list[Tensor]:
        return self.offset_embed.parameters()
