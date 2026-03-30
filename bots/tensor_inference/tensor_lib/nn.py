"""Neural network layers for inference.

All layers follow the pattern: __call__(x) -> Tensor, parameters() -> list[Tensor].
Training is done in PyTorch — these layers are forward-pass only.
"""

from __future__ import annotations

import math

from .tensor import Tensor

_DOT = math.sumprod
_EXP = math.exp
_MIN = min
_SILU_CLAMP = 500.0
_TEMP_HEADS = ("build_type", "build_dir", "conv_dir", "move_dir")

# Op tags for infer cache (integer dispatch is faster than string comparison)
_OP_LINEAR = 0
_OP_LINEAR_RMSNORM = 1
_OP_RMSNORM = 2
_OP_SILU = 3
_OP_TANH = 4
_OP_SIGMOID = 5

# Activation tags
_ACT_NONE = 0
_ACT_SILU = 1
_ACT_TANH = 2
_ACT_SIGMOID = 3
_ACT_MAP = {"SiLU": _ACT_SILU, "Tanh": _ACT_TANH, "Sigmoid": _ACT_SIGMOID}


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

class MLP:
    """Multi-layer perceptron with configurable hidden sizes and output heads.

    Architecture: input -> [Linear -> Activation] * N -> {head_name: Linear}
    Each head produces independent logits for a different decision.
    """

    __slots__ = (
        "trunk", "heads", "_infer_cache",
        "tile_encoder", "tile_grid_n", "tile_raw_features", "tile_embed_dim", "_tile_dxdy",
        "tile_conv_w", "tile_conv_b", "tile_pointwise", "tile_post_rms",
        "tile_compress", "tile_compress_size",
    )

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_heads: dict[str, int],
        activation: str = "silu",
        rmsnorm_layers: list[bool] | tuple[bool, ...] | None = None,
        tile_encoder_sizes: tuple[int, int] | None = None,
        tile_grid_n: int = 25,
        tile_conv: bool = False,
        tile_compress_size: int | None = None,
    ):
        act_cls = {"tanh": Tanh, "sigmoid": Sigmoid, "silu": SiLU}[activation]
        if rmsnorm_layers is None:
            rmsnorm_layers = [False] * len(hidden_sizes)
        else:
            assert len(rmsnorm_layers) == len(hidden_sizes), (
                f"rmsnorm_layers len {len(rmsnorm_layers)} != hidden sizes len {len(hidden_sizes)}"
            )

        layers: list = []
        prev = input_size
        for i, h in enumerate(hidden_sizes):
            layers.append(Linear(prev, h))
            if rmsnorm_layers[i]:
                layers.append(RMSNorm(h))
            layers.append(act_cls())
            prev = h
        self.trunk = Sequential(*layers)

        self.heads: dict[str, Linear] = {}
        for name, size in output_heads.items():
            self.heads[name] = Linear(prev, size)

        if tile_encoder_sizes is not None:
            te_in, te_out = tile_encoder_sizes
            self.tile_encoder = Sequential(Linear(te_in, te_out), SiLU())
            self.tile_grid_n = tile_grid_n
            self.tile_raw_features = te_in - 2
            self.tile_embed_dim = te_out
            self._tile_dxdy = [(float((i % 5) - 2) / 2.0, float((i // 5) - 2) / 2.0) for i in range(tile_grid_n)]
        else:
            self.tile_encoder = None
            self.tile_grid_n = 0
            self.tile_raw_features = 0
            self.tile_embed_dim = 0
            self._tile_dxdy = None

        # Depthwise 3x3 conv for spatial mixing between adjacent tiles
        if tile_conv and tile_encoder_sizes is not None:
            te_out = tile_encoder_sizes[1]
            # Conv weights: flat (channels * 9) tensor, zero-init (residual starts as identity)
            self.tile_conv_w = Tensor([0.0] * (te_out * 9), (te_out * 9,))
            self.tile_conv_b = Tensor([0.0] * te_out, (te_out,))
            self.tile_pointwise = Linear(te_out, te_out)
            self.tile_post_rms = RMSNorm(te_out)
            # Compress: Linear(n_tiles * embed_dim, compress_size) + RMSNorm + SiLU
            assert tile_compress_size is not None
            self.tile_compress_size = tile_compress_size
            compress_in = tile_grid_n * te_out
            self.tile_compress = Sequential(
                Linear(compress_in, tile_compress_size),
                RMSNorm(tile_compress_size),
                SiLU(),
            )
        else:
            self.tile_conv_w = None
            self.tile_conv_b = None
            self.tile_pointwise = None
            self.tile_post_rms = None
            self.tile_compress = None
            self.tile_compress_size = None

        self._infer_cache: list | None = None

    def eval(self):
        """No-op for compatibility with torch-style callers."""
        return self

    def __call__(self, x: Tensor) -> dict[str, Tensor]:
        h = self.trunk(x)
        return {name: head(h) for name, head in self.heads.items()}

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        if self.tile_encoder is not None:
            params.extend(self.tile_encoder.parameters())
        if self.tile_conv_w is not None:
            params.append(self.tile_conv_w)
            params.append(self.tile_conv_b)
        if self.tile_pointwise is not None:
            params.extend(self.tile_pointwise.parameters())
        if self.tile_post_rms is not None:
            params.extend(self.tile_post_rms.parameters())
        if self.tile_compress is not None:
            params.extend(self.tile_compress.parameters())
        params.extend(self.trunk.parameters())
        for head in self.heads.values():
            params.extend(head.parameters())
        return params

    def _build_infer_cache(self) -> list:
        """Precompute transposed weight rows for fast inference.

        Converts column-strided weight access (w[j::N]) into contiguous
        pre-sliced rows, and records which activation follows each linear
        layer so infer() can fuse them in a single pass.
        """
        ops: list = []
        trunk_layers = self.trunk.layers
        i = 0
        while i < len(trunk_layers):
            layer = trunk_layers[i]
            if isinstance(layer, Linear):
                w = layer.weight.data
                N = layer.out_features
                w_rows = [w[j::N] for j in range(N)]
                b = list(layer.bias.data) if layer.bias is not None else [0.0] * N
                act = _ACT_NONE
                if i + 1 < len(trunk_layers):
                    nxt = trunk_layers[i + 1]
                    if isinstance(nxt, RMSNorm):
                        rms_gamma = list(nxt.gamma.data)
                        if i + 2 < len(trunk_layers):
                            nxt2 = trunk_layers[i + 2]
                            if isinstance(nxt2, (SiLU, Tanh, Sigmoid)):
                                act = _ACT_MAP[type(nxt2).__name__]
                        ops.append((_OP_LINEAR_RMSNORM, act, w_rows, b, N, rms_gamma, nxt.eps))
                        i += 3 if act != _ACT_NONE else 2
                        continue
                    if isinstance(nxt, (SiLU, Tanh, Sigmoid)):
                        act = _ACT_MAP[type(nxt).__name__]
                        i += 2
                        ops.append((_OP_LINEAR, act, w_rows, b, N))
                        continue
                ops.append((_OP_LINEAR, act, w_rows, b, N))
                i += 1
                continue
            elif isinstance(layer, RMSNorm):
                ops.append((_OP_RMSNORM, layer.dim, layer.gamma.data, layer.eps))
            elif isinstance(layer, SiLU):
                ops.append((_OP_SILU,))
            elif isinstance(layer, Tanh):
                ops.append((_OP_TANH,))
            elif isinstance(layer, Sigmoid):
                ops.append((_OP_SIGMOID,))
            i += 1

        # Fuse all output heads into one cached linear pass, then slice back into
        # per-head outputs. This keeps the public infer() API unchanged while
        # reducing Python loop overhead in the 2ms engine hot path.
        head_names: list[str] = []
        head_slices: list[tuple[int, int]] = []
        fused_w_rows: list[list[float]] = []
        fused_bias: list[float] = []
        offset = 0
        for name, head in self.heads.items():
            if name == "logit_scale":
                continue  # not needed for argmax inference
            w = head.weight.data
            N = head.out_features
            b = list(head.bias.data) if head.bias is not None else [0.0] * N
            head_names.append(name)
            head_slices.append((offset, offset + N))
            fused_w_rows.extend(w[j::N] for j in range(N))
            fused_bias.extend(b)
            offset += N

        # Build tile encoder cache
        tile_cache = None
        if self.tile_encoder is not None:
            te_layers = self.tile_encoder.layers
            te_linear = te_layers[0]  # Linear layer
            te_w = te_linear.weight.data
            te_out_dim = te_linear.out_features
            te_w_rows_full = [te_w[j::te_out_dim] for j in range(te_out_dim)]
            te_dx_w = [row[0] for row in te_w_rows_full]
            te_dy_w = [row[1] for row in te_w_rows_full]
            te_w_rows = [row[2:] for row in te_w_rows_full]
            te_bias = list(te_linear.bias.data) if te_linear.bias is not None else [0.0] * te_out_dim
            # Pre-bake dx*w_dx + dy*w_dy + bias per (tile, channel) so infer() just adds a scalar
            dxdy = self._tile_dxdy
            n_tiles = self.tile_grid_n
            te_offsets = []  # flat list: n_tiles * te_out_dim floats
            for t in range(n_tiles):
                dx, dy = dxdy[t]
                te_offsets.append([
                    dx * te_dx_w[j] + dy * te_dy_w[j] + te_bias[j]
                    for j in range(te_out_dim)
                ])
            tile_cache = (te_w_rows, te_out_dim, te_offsets)

        # Build conv cache: branchless neighbors + depthwise + pointwise + post RMSNorm
        conv_cache = None
        if self.tile_conv_w is not None:
            n_tiles = self.tile_grid_n
            grid_w = int(n_tiles ** 0.5)  # 5 for 25 tiles
            te_out = self.tile_embed_dim
            # Precompute per-tile-per-channel gather indices and kernel weights
            # so the hot loop is just: [encoded[i] for i in idx] + one sumprod
            flat_w = self.tile_conv_w.data
            conv_w = [flat_w[c * 9:(c + 1) * 9] for c in range(te_out)]
            conv_b = list(self.tile_conv_b.data)
            conv_gather: list[list[tuple[list[int], list[float]]]] = []
            for t in range(n_tiles):
                row, col = t // grid_w, t % grid_w
                base_indices = []
                k_indices = []
                for kr in range(-1, 2):
                    for kc in range(-1, 2):
                        nr, nc = row + kr, col + kc
                        if 0 <= nr < grid_w and 0 <= nc < grid_w:
                            base_indices.append((nr * grid_w + nc) * te_out)
                            k_indices.append((kr + 1) * 3 + (kc + 1))
                per_ch = [
                    ([bi + c for bi in base_indices], [conv_w[c][k] for k in k_indices])
                    for c in range(te_out)
                ]
                conv_gather.append(per_ch)
            pw_w = self.tile_pointwise.weight.data
            pw_b = list(self.tile_pointwise.bias.data) if self.tile_pointwise.bias is not None else [0.0] * te_out
            pw_w_rows = [pw_w[j::te_out] for j in range(te_out)]
            post_gamma = list(self.tile_post_rms.gamma.data)
            post_eps = self.tile_post_rms.eps
            conv_cache = (conv_b, conv_gather, pw_w_rows, pw_b, post_gamma, post_eps)

        # Build compress cache: Linear + RMSNorm + SiLU fused op
        compress_cache = None
        if self.tile_compress is not None:
            comp_layers = self.tile_compress.layers
            comp_linear = comp_layers[0]  # Linear
            comp_w = comp_linear.weight.data
            comp_N = comp_linear.out_features
            comp_b = list(comp_linear.bias.data) if comp_linear.bias is not None else [0.0] * comp_N
            comp_w_rows = [comp_w[j::comp_N] for j in range(comp_N)]
            comp_rms = comp_layers[1]  # RMSNorm
            comp_gamma = list(comp_rms.gamma.data)
            comp_eps = comp_rms.eps
            compress_cache = (comp_w_rows, comp_b, comp_N, comp_gamma, comp_eps)

        return [ops, (head_names, head_slices, fused_w_rows, fused_bias, offset), tile_cache, conv_cache, compress_cache]

    def infer(self, features: list[float]) -> dict[str, list[float]]:
        """Fast inference for a single input — bypasses Tensor overhead.

        Takes a flat list of floats (length == input_size), returns
        raw logits as {head_name: list[float]}. No Tensor allocation.
        Uses C-accelerated path when _native.so is available, otherwise
        falls back to pure-Python with sumprod.
        """
        if self._infer_cache is None:
            self._infer_cache = self._build_infer_cache()

        _exp = _EXP
        _mn = _MIN
        _sc = _SILU_CLAMP
        _dot = _DOT
        ops, fused_heads, tile_cache, conv_cache, compress_cache = self._infer_cache

        if tile_cache is not None:
            te_w_rows, te_out, te_offsets = tile_cache
            n_tiles = self.tile_grid_n
            raw_feat = self.tile_raw_features
            tile_end = n_tiles * raw_feat

            # Tile encoder: linear + SiLU via branchless list comp per tile
            encoded = [0.0] * (n_tiles * te_out)
            idx = 0
            for t in range(n_tiles):
                base = t * raw_feat
                tile_raw = features[base:base + raw_feat]
                offsets_t = te_offsets[t]
                for j in range(te_out):
                    x = _dot(tile_raw, te_w_rows[j]) + offsets_t[j]
                    encoded[idx] = x * (e := _exp(_mn(x, _sc))) / (1.0 + e)
                    idx += 1

            # Depthwise 3x3 + pointwise 1x1 + residual + RMSNorm + SiLU
            if conv_cache is not None:
                conv_b, conv_gather, pw_w_rows, pw_b, post_gamma, post_eps = conv_cache
                conv_tiles = [
                    [_dot([encoded[i] for i in pc[0]], pc[1]) + conv_b[c] for c, pc in enumerate(per_ch)]
                    for per_ch in conv_gather
                ]
                for t in range(n_tiles):
                    b = t * te_out
                    src = conv_tiles[t]
                    proj = [_dot(src, pw_w_rows[j]) + pw_b[j] + encoded[b + j] for j in range(te_out)]
                    ir = 1.0 / (_dot(proj, proj) / te_out + post_eps) ** 0.5
                    encoded[b:b + te_out] = [
                        (x := p * ir * g) * (e := _exp(_mn(x, _sc))) / (1.0 + e)
                        for p, g in zip(proj, post_gamma)
                    ]

            # Compress + RMSNorm + SiLU, then concat globals
            if compress_cache is not None:
                comp_w_rows, comp_b, comp_N, comp_gamma, comp_eps = compress_cache
                compressed = [_dot(encoded, comp_w_rows[j]) + comp_b[j] for j in range(comp_N)]
                inv_rms = 1.0 / (_dot(compressed, compressed) / comp_N + comp_eps) ** 0.5
                h = [
                    (x := c * inv_rms * g) * (e := _exp(_mn(x, _sc))) / (1.0 + e)
                    for c, g in zip(compressed, comp_gamma)
                ] + features[tile_end:]
            else:
                h = encoded + features[tile_end:]
        else:
            h = features

        for op in ops:
            tag = op[0]
            if tag == _OP_LINEAR:
                _, act, w_rows, b, N = op
                if act == _ACT_SILU:
                    h = [
                        (x := _dot(h, w_rows[j]) + b[j]) * (e := _exp(_mn(x, _sc))) / (1.0 + e)
                        for j in range(N)
                    ]
                elif act == _ACT_TANH:
                    _tanh = math.tanh
                    h = [_tanh(_dot(h, w_rows[j]) + b[j]) for j in range(N)]
                elif act == _ACT_SIGMOID:
                    h = [
                        (e := _exp(_mn(_dot(h, w_rows[j]) + b[j], _sc))) / (1.0 + e)
                        for j in range(N)
                    ]
                else:
                    h = [_dot(h, w_rows[j]) + b[j] for j in range(N)]
            elif tag == _OP_LINEAR_RMSNORM:
                _, act, w_rows, b, N, gamma, eps = op
                out = [_dot(h, w_rows[j]) + b[j] for j in range(N)]
                inv_rms = 1.0 / (_dot(out, out) / N + eps) ** 0.5
                if act == _ACT_SILU:
                    h = [
                        (x := o * inv_rms * g) * (e := _exp(_mn(x, _sc))) / (1.0 + e)
                        for o, g in zip(out, gamma)
                    ]
                elif act == _ACT_TANH:
                    h = [math.tanh(o * inv_rms * g) for o, g in zip(out, gamma)]
                elif act == _ACT_SIGMOID:
                    h = [
                        (e := _exp(_mn(o * inv_rms * g, _sc))) / (1.0 + e)
                        for o, g in zip(out, gamma)
                    ]
                else:
                    h = [o * inv_rms * g for o, g in zip(out, gamma)]
                del out
            elif tag == _OP_RMSNORM:
                _, dim, g, eps = op
                inv_rms = 1.0 / (_dot(h, h) / dim + eps) ** 0.5
                h = [v * inv_rms * gv for v, gv in zip(h, g)]
            elif tag == _OP_SILU:
                h = [x * (e := _exp(_mn(x, _sc))) / (1.0 + e) for x in h]
            elif tag == _OP_TANH:
                h = [math.tanh(x) for x in h]
            elif tag == _OP_SIGMOID:
                h = [(e := _exp(_mn(x, _sc))) / (1.0 + e) for x in h]

        head_names, head_slices, fused_w_rows, fused_bias, total_out = fused_heads
        fused_out = [_dot(h, fused_w_rows[j]) + fused_bias[j] for j in range(total_out)]

        results: dict[str, list[float]] = {}
        for name, (start, end) in zip(head_names, head_slices):
            results[name] = fused_out[start:end]
        return results

    def save(self) -> list[list[float]]:
        """Serialize all parameters to nested lists."""
        return [list(p.data) for p in self.parameters()]

    def load(self, state: list[list[float]]) -> None:
        """Load parameters from nested lists.

        Handles states saved with different subsets of heads or without
        tile encoder parameters.
        """
        def _head_params(skip: set[str] | None = None) -> list[Tensor]:
            skip = skip or set()
            params: list[Tensor] = []
            for name, head in self.heads.items():
                if name in skip:
                    continue
                params.extend(head.parameters())
            return params

        full_params = self.parameters()

        # Params without tile encoder (trunk + heads only)
        no_tile_params: list[Tensor] = list(self.trunk.parameters())
        no_tile_params.extend(_head_params())

        no_tile_no_logit_params: list[Tensor] = list(self.trunk.parameters())
        no_tile_no_logit_params.extend(_head_params({"logit_scale"}))

        # Without tile encoder AND without conv_dir head
        no_tile_no_conv_params: list[Tensor] = list(self.trunk.parameters())
        no_tile_no_conv_params.extend(_head_params({"conv_dir"}))

        no_tile_no_conv_no_logit_params: list[Tensor] = list(self.trunk.parameters())
        no_tile_no_conv_no_logit_params.extend(_head_params({"conv_dir", "logit_scale"}))

        no_conv_params: list[Tensor] = []
        if self.tile_encoder is not None:
            no_conv_params.extend(self.tile_encoder.parameters())
        if self.tile_conv_w is not None:
            no_conv_params.append(self.tile_conv_w)
            no_conv_params.append(self.tile_conv_b)
        if self.tile_pointwise is not None:
            no_conv_params.extend(self.tile_pointwise.parameters())
        if self.tile_post_rms is not None:
            no_conv_params.extend(self.tile_post_rms.parameters())
        if self.tile_compress is not None:
            no_conv_params.extend(self.tile_compress.parameters())
        for layer in self.trunk.layers:
            no_conv_params.extend(layer.parameters())
        no_conv_params.extend(_head_params({"conv_dir"}))

        no_logit_params: list[Tensor] = []
        if self.tile_encoder is not None:
            no_logit_params.extend(self.tile_encoder.parameters())
        if self.tile_conv_w is not None:
            no_logit_params.append(self.tile_conv_w)
            no_logit_params.append(self.tile_conv_b)
        if self.tile_pointwise is not None:
            no_logit_params.extend(self.tile_pointwise.parameters())
        if self.tile_post_rms is not None:
            no_logit_params.extend(self.tile_post_rms.parameters())
        if self.tile_compress is not None:
            no_logit_params.extend(self.tile_compress.parameters())
        no_logit_params.extend(self.trunk.parameters())
        no_logit_params.extend(_head_params({"logit_scale"}))

        no_conv_no_logit_params: list[Tensor] = []
        if self.tile_encoder is not None:
            no_conv_no_logit_params.extend(self.tile_encoder.parameters())
        if self.tile_conv_w is not None:
            no_conv_no_logit_params.append(self.tile_conv_w)
            no_conv_no_logit_params.append(self.tile_conv_b)
        if self.tile_pointwise is not None:
            no_conv_no_logit_params.extend(self.tile_pointwise.parameters())
        if self.tile_post_rms is not None:
            no_conv_no_logit_params.extend(self.tile_post_rms.parameters())
        if self.tile_compress is not None:
            no_conv_no_logit_params.extend(self.tile_compress.parameters())
        for layer in self.trunk.layers:
            no_conv_no_logit_params.extend(layer.parameters())
        no_conv_no_logit_params.extend(_head_params({"conv_dir", "logit_scale"}))

        # When tile_encoder is present, no_conv_params and no_tile_params
        # can have the same count (tile_enc adds 2 params, conv_dir adds 2).
        # Disambiguate by checking the first entry's size: tile encoder weight
        # is much smaller than the trunk's first Linear weight.
        _first_sz = len(state[0]) if state else 0
        _has_tile_first = (
            self.tile_encoder is not None
            and len(self.tile_encoder.parameters()) > 0
            and _first_sz == len(self.tile_encoder.parameters()[0].data)
        )

        if len(state) == len(full_params):
            params = full_params
        elif len(state) == len(no_logit_params):
            params = no_logit_params
        elif len(state) == len(no_conv_params) and _has_tile_first:
            params = no_conv_params
        elif len(state) == len(no_conv_no_logit_params) and _has_tile_first:
            params = no_conv_no_logit_params
        elif len(state) == len(no_tile_params):
            params = no_tile_params
        elif len(state) == len(no_tile_no_logit_params):
            params = no_tile_no_logit_params
        elif len(state) == len(no_tile_no_conv_params):
            params = no_tile_no_conv_params
        elif len(state) == len(no_tile_no_conv_no_logit_params):
            params = no_tile_no_conv_no_logit_params
        elif len(state) == len(no_conv_params):
            params = no_conv_params
        else:
            raise AssertionError(
                f"State has {len(state)} entries, model has {len(full_params)} params"
            )

        for p, s in zip(params, state):
            assert len(p.data) == len(s), (
                f"Param size mismatch: {len(p.data)} vs {len(s)}"
            )
            p.data[:] = s
        self._infer_cache = None  # invalidate precomputed weights
