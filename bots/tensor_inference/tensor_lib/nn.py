"""Neural network layers for inference.

All layers follow the pattern: __call__(x) -> Tensor, parameters() -> list[Tensor].
Training is done in PyTorch — these layers are forward-pass only.
"""

from __future__ import annotations

import math
from operator import mul as _mul

from .tensor import Tensor

_DOT = math.sumprod if hasattr(math, "sumprod") else None


class Linear:
    """Fully connected layer: y = x @ W + b.

    Weight shape: (in_features, out_features) — Xavier initialized.
    Bias shape: (out_features,) — zero initialized.
    """

    __slots__ = ("weight", "bias", "in_features", "out_features")

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.xavier(in_features, out_features)
        self.bias = (
            Tensor.zeros((out_features,)) if bias else None
        )

    def __call__(self, x: Tensor) -> Tensor:
        # Ensure 2D for matmul
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[0]))
        out = x.matmul(self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self) -> list[Tensor]:
        ps = [self.weight]
        if self.bias is not None:
            ps.append(self.bias)
        return ps


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self) -> list[Tensor]:
        return []


class Tanh:
    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()

    def parameters(self) -> list[Tensor]:
        return []


class SiLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.silu()

    def parameters(self) -> list[Tensor]:
        return []


class Sigmoid:
    def __call__(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def parameters(self) -> list[Tensor]:
        return []


class LayerNorm:
    """Layer normalization over the last dimension.

    Learnable affine parameters gamma (scale) and beta (shift).
    """

    __slots__ = ("dim", "gamma", "beta", "eps")

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = Tensor([1.0] * dim, (dim,))
        self.beta = Tensor([0.0] * dim, (dim,))

    def __call__(self, x: Tensor) -> Tensor:
        # Normalize last dim. Supports 1D (N,) and 2D (M, N).
        if len(x.shape) == 1:
            N = x.shape[0]
            mu = sum(x.data) / N
            var = sum((v - mu) ** 2 for v in x.data) / N
            inv_std = 1.0 / (var + self.eps) ** 0.5
            out_data = [
                (v - mu) * inv_std * self.gamma.data[i] + self.beta.data[i]
                for i, v in enumerate(x.data)
            ]
            return Tensor(out_data, x.shape)
        else:
            M, N = x.shape
            out_data = [0.0] * (M * N)
            for m in range(M):
                base = m * N
                row = x.data[base : base + N]
                mu = sum(row) / N
                var = sum((v - mu) ** 2 for v in row) / N
                inv_std = 1.0 / (var + self.eps) ** 0.5
                for j in range(N):
                    normed = (row[j] - mu) * inv_std
                    out_data[base + j] = (
                        normed * self.gamma.data[j] + self.beta.data[j]
                    )
            return Tensor(out_data, x.shape)

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]


class RMSNorm:
    """RMS normalization over the last dimension with learnable scale only."""

    __slots__ = ("dim", "gamma", "eps")

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.gamma = Tensor([1.0] * dim, (dim,))

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape) == 1:
            N = x.shape[0]
            mean_sq = sum(v * v for v in x.data) / N
            inv_rms = 1.0 / (mean_sq + self.eps) ** 0.5
            out_data = [
                x.data[i] * inv_rms * self.gamma.data[i]
                for i in range(N)
            ]
            return Tensor(out_data, x.shape)
        else:
            M, N = x.shape
            out_data = [0.0] * (M * N)
            for m in range(M):
                base = m * N
                row = x.data[base : base + N]
                mean_sq = sum(v * v for v in row) / N
                inv_rms = 1.0 / (mean_sq + self.eps) ** 0.5
                for j in range(N):
                    out_data[base + j] = row[j] * inv_rms * self.gamma.data[j]
            return Tensor(out_data, x.shape)

    def parameters(self) -> list[Tensor]:
        return [self.gamma]


class Dropout:
    """Inverted dropout. Only active during training."""

    __slots__ = ("p", "training")

    def __init__(self, p: float = 0.1):
        self.p = p
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        import random

        scale = 1.0 / (1.0 - self.p)
        mask = [scale if random.random() > self.p else 0.0 for _ in x.data]
        return Tensor(
            [a * m for a, m in zip(x.data, mask)], x.shape
        )

    def parameters(self) -> list[Tensor]:
        return []


class Sequential:
    """Chain of layers executed in order."""

    __slots__ = ("layers",)

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train(self, mode: bool = True) -> None:
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = mode

    def eval(self) -> None:
        self.train(False)


class MLP:
    """Multi-layer perceptron with configurable hidden sizes and output heads.

    Architecture: input -> [Linear -> Activation] * N -> {head_name: Linear}
    Each head produces independent logits for a different decision.
    """

    __slots__ = ("trunk", "heads", "_infer_cache")

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_heads: dict[str, int],
        activation: str = "relu",
        use_layernorm: bool = False,
        rmsnorm_layers: list[bool] | tuple[bool, ...] | None = None,
        dropout: float = 0.0,
    ):
        act_cls = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid, "silu": SiLU}[activation]
        if rmsnorm_layers is None:
            rmsnorm_layers = [False] * len(hidden_sizes)
        else:
            assert len(rmsnorm_layers) == len(hidden_sizes), (
                f"rmsnorm_layers len {len(rmsnorm_layers)} != hidden sizes len {len(hidden_sizes)}"
            )
        assert not (use_layernorm and any(rmsnorm_layers)), (
            "LayerNorm and RMSNorm should not be enabled on the same trunk layer set"
        )

        layers: list = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append(Linear(prev, h))
            if rmsnorm_layers[i]:
                layers.append(RMSNorm(h))
            if use_layernorm:
                layers.append(LayerNorm(h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(Dropout(dropout))
            prev = h
        self.trunk = Sequential(*layers)

        self.heads: dict[str, Linear] = {}
        for name, size in output_heads.items():
            self.heads[name] = Linear(prev, size)

        self._infer_cache: list | None = None

    def __call__(self, x: Tensor) -> dict[str, Tensor]:
        h = self.trunk(x)
        return {name: head(h) for name, head in self.heads.items()}

    def parameters(self) -> list[Tensor]:
        params = list(self.trunk.parameters())
        for head in self.heads.values():
            params.extend(head.parameters())
        return params

    def train(self, mode: bool = True) -> None:
        self.trunk.train(mode)

    def eval(self) -> None:
        self.train(False)

    def _build_infer_cache(self) -> list:
        """Precompute transposed weight rows for fast inference.

        Converts column-strided weight access (w[j::N]) into contiguous
        pre-sliced rows, and records which activation follows each linear
        layer so infer() can fuse them in a single pass.
        """
        _exp = math.exp
        _tanh = math.tanh

        # Build fused ops: ('linear_silu', w_rows, bias, N), ('linear_relu', ...), etc.
        ops: list = []
        trunk_layers = self.trunk.layers
        i = 0
        while i < len(trunk_layers):
            layer = trunk_layers[i]
            if isinstance(layer, Linear):
                w = layer.weight.data
                N = layer.out_features
                w_rows = [w[j::N] for j in range(N)]
                b = list(layer.bias.data) if layer.bias is not None else None
                # Check if following layers are a norm + activation we can fuse
                act = None
                if i + 1 < len(trunk_layers):
                    nxt = trunk_layers[i + 1]
                    if isinstance(nxt, RMSNorm):
                        rms_gamma = list(nxt.gamma.data)
                        if i + 2 < len(trunk_layers):
                            nxt2 = trunk_layers[i + 2]
                            if isinstance(nxt2, (ReLU, SiLU, Tanh, Sigmoid)):
                                act = type(nxt2).__name__
                        ops.append(('linear_rmsnorm', act, w_rows, b, N, rms_gamma, nxt.eps))
                        i += 3 if act is not None else 2
                        continue
                    if isinstance(nxt, (ReLU, SiLU, Tanh, Sigmoid)):
                        act = type(nxt).__name__
                        i += 2
                        ops.append(('linear', act, w_rows, b, N))
                        continue
                ops.append(('linear', act, w_rows, b, N))
                i += 1
                continue
            elif isinstance(layer, LayerNorm):
                ops.append(('layernorm', layer.dim, layer.gamma.data,
                            layer.beta.data, layer.eps))
            elif isinstance(layer, RMSNorm):
                ops.append(('rmsnorm', layer.dim, layer.gamma.data, layer.eps))
            # Standalone activations (not fused) — only hit if not preceded by Linear
            elif isinstance(layer, ReLU):
                ops.append(('relu',))
            elif isinstance(layer, SiLU):
                ops.append(('silu',))
            elif isinstance(layer, Tanh):
                ops.append(('tanh',))
            elif isinstance(layer, Sigmoid):
                ops.append(('sigmoid',))
            i += 1

        # Fuse all output heads into one cached linear pass, then slice back into
        # per-head outputs. This keeps the public infer() API unchanged while
        # reducing Python loop overhead in the 2ms engine hot path.
        head_names: list[str] = []
        head_slices: list[tuple[int, int]] = []
        fused_w_rows: list[list[float]] = []
        fused_bias: list[float] | None = []
        offset = 0
        for name, head in self.heads.items():
            w = head.weight.data
            N = head.out_features
            head_names.append(name)
            head_slices.append((offset, offset + N))
            fused_w_rows.extend(w[j::N] for j in range(N))
            if fused_bias is not None:
                if head.bias is not None:
                    fused_bias.extend(head.bias.data)
                else:
                    fused_bias.extend([0.0] * N)
            offset += N

        return [ops, (head_names, head_slices, fused_w_rows, fused_bias, offset)]

    def infer(self, features: list[float]) -> dict[str, list[float]]:
        """Fast inference for a single input — bypasses Tensor overhead.

        Takes a flat list of floats (length == input_size), returns
        raw logits as {head_name: list[float]}. No Tensor allocation.
        Uses precomputed transposed weight rows and fused
        linear+activation for ~15% speedup over naive path.
        """
        if self._infer_cache is None:
            self._infer_cache = self._build_infer_cache()

        _exp = math.exp
        _dot = _DOT
        ops, fused_heads = self._infer_cache
        h = features

        for op in ops:
            tag = op[0]
            if tag == 'linear':
                _, act, w_rows, b, N = op
                if act == 'SiLU':
                    out = [0.0] * N
                    for j in range(N):
                        x = _dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))
                        if b is not None:
                            x += b[j]
                        if x >= 0:
                            out[j] = x / (1.0 + _exp(-x))
                        else:
                            e = _exp(x)
                            out[j] = x * e / (1.0 + e)
                    h = out
                elif act == 'ReLU':
                    if b is not None:
                        h = [max(0.0, (_dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))) + b[j]) for j in range(N)]
                    else:
                        h = [max(0.0, _dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))) for j in range(N)]
                elif act == 'Tanh':
                    _tanh = math.tanh
                    if b is not None:
                        h = [_tanh((_dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))) + b[j]) for j in range(N)]
                    else:
                        h = [_tanh(_dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))) for j in range(N)]
                elif act == 'Sigmoid':
                    out = [0.0] * N
                    for j in range(N):
                        x = _dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))
                        if b is not None:
                            x += b[j]
                        if x >= 0:
                            out[j] = 1.0 / (1.0 + _exp(-x))
                        else:
                            e = _exp(x)
                            out[j] = e / (1.0 + e)
                    h = out
                else:
                    # No activation to fuse
                    if b is not None:
                        h = [(_dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))) + b[j] for j in range(N)]
                    else:
                        h = [_dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j])) for j in range(N)]
            elif tag == 'linear_rmsnorm':
                _, act, w_rows, b, N, gamma, eps = op
                out = [0.0] * N
                mean_sq = 0.0
                for j in range(N):
                    x = _dot(h, w_rows[j]) if _dot is not None else sum(map(_mul, h, w_rows[j]))
                    if b is not None:
                        x += b[j]
                    out[j] = x
                    mean_sq += x * x
                inv_rms = 1.0 / (mean_sq / N + eps) ** 0.5
                if act == 'SiLU':
                    for j in range(N):
                        x = out[j] * inv_rms * gamma[j]
                        if x >= 0:
                            out[j] = x / (1.0 + _exp(-x))
                        else:
                            e = _exp(x)
                            out[j] = x * e / (1.0 + e)
                elif act == 'ReLU':
                    for j in range(N):
                        x = out[j] * inv_rms * gamma[j]
                        out[j] = x if x > 0.0 else 0.0
                elif act == 'Tanh':
                    for j in range(N):
                        out[j] = math.tanh(out[j] * inv_rms * gamma[j])
                elif act == 'Sigmoid':
                    for j in range(N):
                        x = out[j] * inv_rms * gamma[j]
                        if x >= 0:
                            out[j] = 1.0 / (1.0 + _exp(-x))
                        else:
                            e = _exp(x)
                            out[j] = e / (1.0 + e)
                else:
                    for j in range(N):
                        out[j] = out[j] * inv_rms * gamma[j]
                h = out
            elif tag == 'layernorm':
                _, dim, g, lb, eps = op
                mu = sum(h) / dim
                var = sum((v - mu) ** 2 for v in h) / dim
                inv_std = 1.0 / (var + eps) ** 0.5
                h = [(h[i] - mu) * inv_std * g[i] + lb[i] for i in range(dim)]
            elif tag == 'rmsnorm':
                _, dim, g, eps = op
                mean_sq = sum(v * v for v in h) / dim
                inv_rms = 1.0 / (mean_sq + eps) ** 0.5
                h = [h[i] * inv_rms * g[i] for i in range(dim)]
            elif tag == 'relu':
                h = [x if x > 0.0 else 0.0 for x in h]
            elif tag == 'silu':
                out = [0.0] * len(h)
                for i, x in enumerate(h):
                    if x >= 0:
                        out[i] = x / (1.0 + _exp(-x))
                    else:
                        e = _exp(x)
                        out[i] = x * e / (1.0 + e)
                h = out
            elif tag == 'tanh':
                h = [math.tanh(x) for x in h]
            elif tag == 'sigmoid':
                out = [0.0] * len(h)
                for i, x in enumerate(h):
                    if x >= 0:
                        out[i] = 1.0 / (1.0 + _exp(-x))
                    else:
                        e = _exp(x)
                        out[i] = e / (1.0 + e)
                h = out

        head_names, head_slices, fused_w_rows, fused_bias, total_out = fused_heads
        if fused_bias is not None:
            fused_out = [(_dot(h, fused_w_rows[j]) if _dot is not None else sum(map(_mul, h, fused_w_rows[j]))) + fused_bias[j] for j in range(total_out)]
        else:
            fused_out = [_dot(h, fused_w_rows[j]) if _dot is not None else sum(map(_mul, h, fused_w_rows[j])) for j in range(total_out)]

        results: dict[str, list[float]] = {}
        for name, (start, end) in zip(head_names, head_slices):
            results[name] = fused_out[start:end]
        return results

    def save(self) -> list[list[float]]:
        """Serialize all parameters to nested lists."""
        return [list(p.data) for p in self.parameters()]

    def load(self, state: list[list[float]]) -> None:
        """Load parameters from nested lists.

        Supports legacy MLP states saved before RMSNorm parameters were added
        and before optional newer heads such as `conv_dir` existed.
        """
        full_params = self.parameters()
        no_conv_params: list[Tensor] = []
        for layer in self.trunk.layers:
            no_conv_params.extend(layer.parameters())
        for name, head in self.heads.items():
            if name != "conv_dir":
                no_conv_params.extend(head.parameters())

        legacy_params: list[Tensor] = []
        legacy_no_conv_params: list[Tensor] = []
        for layer in self.trunk.layers:
            if isinstance(layer, Linear):
                legacy_params.extend(layer.parameters())
                legacy_no_conv_params.extend(layer.parameters())
            elif isinstance(layer, LayerNorm):
                legacy_params.extend(layer.parameters())
                legacy_no_conv_params.extend(layer.parameters())
        for name, head in self.heads.items():
            if name != "conv_dir":
                legacy_no_conv_params.extend(head.parameters())
            legacy_params.extend(head.parameters())

        if len(state) == len(full_params):
            params = full_params
        elif len(state) == len(no_conv_params):
            params = no_conv_params
        elif len(state) == len(legacy_params):
            params = legacy_params
        elif len(state) == len(legacy_no_conv_params):
            params = legacy_no_conv_params
        else:
            raise AssertionError(
                f"State has {len(state)} entries, model has {len(full_params)} params "
                f"({len(legacy_params)} legacy without RMSNorm)"
            )

        for p, s in zip(params, state):
            assert len(p.data) == len(s), (
                f"Param size mismatch: {len(p.data)} vs {len(s)}"
            )
            p.data[:] = s
        self._infer_cache = None  # invalidate precomputed weights
