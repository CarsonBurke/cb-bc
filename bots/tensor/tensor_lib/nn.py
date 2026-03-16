"""Neural network layers built on the Tensor autograd engine.

All layers follow the pattern: __call__(x) -> Tensor, parameters() -> list[Tensor].
"""

from __future__ import annotations

import math
from operator import mul as _mul

from .tensor import Tensor


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
            Tensor.zeros((out_features,), requires_grad=True) if bias else None
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
        self.gamma = Tensor([1.0] * dim, (dim,), requires_grad=True)
        self.beta = Tensor([0.0] * dim, (dim,), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        # Normalize last dim. Supports 1D (N,) and 2D (M, N).
        if len(x.shape) == 1:
            N = x.shape[0]
            mu = sum(x.data) / N
            var = sum((v - mu) ** 2 for v in x.data) / N
            inv_std = 1.0 / (var + self.eps) ** 0.5
            normed = [(v - mu) * inv_std for v in x.data]
            out_data = [
                normed[i] * self.gamma.data[i] + self.beta.data[i]
                for i in range(N)
            ]
            out = Tensor(out_data, x.shape)
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
            out = Tensor(out_data, x.shape)

        out.requires_grad = (
            x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        )
        out._prev = (x, self.gamma, self.beta)

        # Store for backward
        _x = x
        _gamma = self.gamma
        _beta = self.beta
        _eps = self.eps

        def _backward():
            from .tensor import _ensure_grad

            if not any(
                t.requires_grad for t in (_x, _gamma, _beta)
            ):
                return

            og = out.grad.data

            if len(_x.shape) == 1:
                N = _x.shape[0]
                mu = sum(_x.data) / N
                var = sum((v - mu) ** 2 for v in _x.data) / N
                inv_std = 1.0 / (var + _eps) ** 0.5
                normed = [(v - mu) * inv_std for v in _x.data]

                if _gamma.requires_grad:
                    _ensure_grad(_gamma)
                    for i in range(N):
                        _gamma.grad.data[i] += og[i] * normed[i]
                if _beta.requires_grad:
                    _ensure_grad(_beta)
                    for i in range(N):
                        _beta.grad.data[i] += og[i]
                if _x.requires_grad:
                    _ensure_grad(_x)
                    # dx = gamma * inv_std * (dout - mean(dout) - normed * mean(dout * normed)) / 1
                    dout_g = [og[i] * _gamma.data[i] for i in range(N)]
                    mean_dg = sum(dout_g) / N
                    mean_dgn = sum(dout_g[i] * normed[i] for i in range(N)) / N
                    for i in range(N):
                        _x.grad.data[i] += inv_std * (
                            dout_g[i] - mean_dg - normed[i] * mean_dgn
                        )
            else:
                M, N = _x.shape
                for m in range(M):
                    base = m * N
                    row = _x.data[base : base + N]
                    mu = sum(row) / N
                    var = sum((v - mu) ** 2 for v in row) / N
                    inv_std = 1.0 / (var + _eps) ** 0.5
                    normed = [(v - mu) * inv_std for v in row]

                    if _gamma.requires_grad:
                        _ensure_grad(_gamma)
                        for j in range(N):
                            _gamma.grad.data[j] += og[base + j] * normed[j]
                    if _beta.requires_grad:
                        _ensure_grad(_beta)
                        for j in range(N):
                            _beta.grad.data[j] += og[base + j]
                    if _x.requires_grad:
                        _ensure_grad(_x)
                        dout_g = [
                            og[base + j] * _gamma.data[j] for j in range(N)
                        ]
                        mean_dg = sum(dout_g) / N
                        mean_dgn = (
                            sum(dout_g[j] * normed[j] for j in range(N)) / N
                        )
                        for j in range(N):
                            _x.grad.data[base + j] += inv_std * (
                                dout_g[j] - mean_dg - normed[j] * mean_dgn
                            )

        out._backward = _backward
        return out

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]


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
        out = Tensor(
            [a * m for a, m in zip(x.data, mask)], x.shape
        )
        out.requires_grad = x.requires_grad
        out._prev = (x,)

        def _backward():
            if x.requires_grad:
                from .tensor import _ensure_grad

                _ensure_grad(x)
                for i in range(len(x.data)):
                    x.grad.data[i] += out.grad.data[i] * mask[i]

        out._backward = _backward
        return out

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
        dropout: float = 0.0,
    ):
        act_cls = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid, "silu": SiLU}[activation]

        layers: list = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(Linear(prev, h))
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
                # Check if next layer is an activation we can fuse
                act = None
                if i + 1 < len(trunk_layers):
                    nxt = trunk_layers[i + 1]
                    if isinstance(nxt, (ReLU, SiLU, Tanh, Sigmoid)):
                        act = type(nxt).__name__
                        i += 1  # skip the activation layer
                ops.append(('linear', act, w_rows, b, N))
            elif isinstance(layer, LayerNorm):
                ops.append(('layernorm', layer.dim, layer.gamma.data,
                            layer.beta.data, layer.eps))
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

        head_ops: list = []
        for name, head in self.heads.items():
            w = head.weight.data
            N = head.out_features
            w_rows = [w[j::N] for j in range(N)]
            b = list(head.bias.data) if head.bias is not None else None
            head_ops.append((name, w_rows, b, N))

        return [ops, head_ops]

    def infer(self, features: list[float]) -> dict[str, list[float]]:
        """Fast inference for a single input — bypasses Tensor overhead.

        Takes a flat list of floats (length == input_size), returns
        raw logits as {head_name: list[float]}. No autograd, no Tensor
        allocation. Uses precomputed transposed weight rows and fused
        linear+activation for ~15% speedup over naive path.
        """
        if self._infer_cache is None:
            self._infer_cache = self._build_infer_cache()

        _exp = math.exp
        ops, head_ops = self._infer_cache
        h = features

        for op in ops:
            tag = op[0]
            if tag == 'linear':
                _, act, w_rows, b, N = op
                if act == 'SiLU':
                    out = [0.0] * N
                    for j in range(N):
                        x = sum(map(_mul, h, w_rows[j]))
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
                        h = [max(0.0, sum(map(_mul, h, w_rows[j])) + b[j]) for j in range(N)]
                    else:
                        h = [max(0.0, sum(map(_mul, h, w_rows[j]))) for j in range(N)]
                elif act == 'Tanh':
                    _tanh = math.tanh
                    if b is not None:
                        h = [_tanh(sum(map(_mul, h, w_rows[j])) + b[j]) for j in range(N)]
                    else:
                        h = [_tanh(sum(map(_mul, h, w_rows[j]))) for j in range(N)]
                elif act == 'Sigmoid':
                    out = [0.0] * N
                    for j in range(N):
                        x = sum(map(_mul, h, w_rows[j]))
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
                        h = [sum(map(_mul, h, w_rows[j])) + b[j] for j in range(N)]
                    else:
                        h = [sum(map(_mul, h, w_rows[j])) for j in range(N)]
            elif tag == 'layernorm':
                _, dim, g, lb, eps = op
                mu = sum(h) / dim
                var = sum((v - mu) ** 2 for v in h) / dim
                inv_std = 1.0 / (var + eps) ** 0.5
                h = [(h[i] - mu) * inv_std * g[i] + lb[i] for i in range(dim)]
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

        results: dict[str, list[float]] = {}
        for name, w_rows, b, N in head_ops:
            if b is not None:
                results[name] = [sum(map(_mul, h, w_rows[j])) + b[j] for j in range(N)]
            else:
                results[name] = [sum(map(_mul, h, w_rows[j])) for j in range(N)]
        return results

    def save(self) -> list[list[float]]:
        """Serialize all parameters to nested lists."""
        return [list(p.data) for p in self.parameters()]

    def load(self, state: list[list[float]]) -> None:
        """Load parameters from nested lists."""
        params = self.parameters()
        assert len(state) == len(params), (
            f"State has {len(state)} entries, model has {len(params)} params"
        )
        for p, s in zip(params, state):
            assert len(p.data) == len(s), (
                f"Param size mismatch: {len(p.data)} vs {len(s)}"
            )
            p.data[:] = s
        self._infer_cache = None  # invalidate precomputed weights
