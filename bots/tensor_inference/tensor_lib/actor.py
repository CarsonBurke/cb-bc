"""Dedicated stdlib actor used by tensor deployment and opponent bots."""

from __future__ import annotations

from array import array
import math

from .nn import Linear, RMSNorm, _DOT
from .tensor import Tensor

try:
    from ..model_constants import (
        ACTOR_CONTEXT_CHUNKS,
        ACTOR_CONTEXT_DIM,
        ACTOR_CLS_TOKENS,
        ACTOR_GRU_INPUT_DIM,
        ACTOR_LATENT_DIM,
        GLOBAL_FEATURES,
        GRID_SIZE,
        H_DIM,
        N_BLD_TYPES,
        TILE_EMBED_DIM,
        TILE_FEATURES,
        TILE_GRID_N,
        TILE_INPUT_DIM,
    )
except ImportError:
    from model_constants import (
        ACTOR_CONTEXT_CHUNKS,
        ACTOR_CONTEXT_DIM,
        ACTOR_CLS_TOKENS,
        ACTOR_GRU_INPUT_DIM,
        ACTOR_LATENT_DIM,
        GLOBAL_FEATURES,
        GRID_SIZE,
        H_DIM,
        N_BLD_TYPES,
        TILE_EMBED_DIM,
        TILE_FEATURES,
        TILE_GRID_N,
        TILE_INPUT_DIM,
    )


_SQRT_TILE_DIM = TILE_EMBED_DIM ** 0.5
_EXP = math.exp


def _sigmoid_scalar(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + _EXP(-x))
    e = _EXP(x)
    return e / (1.0 + e)


def _silu_scalar(x: float) -> float:
    return x * _sigmoid_scalar(x)


def _linear(vec: list[float], rows: list[list[float]], bias: list[float] | None) -> list[float]:
    out = [_DOT(vec, rows[j]) for j in range(len(rows))]
    if bias is not None:
        for j in range(len(rows)):
            out[j] += bias[j]
    return out


def _silu_vec(values: list[float]) -> list[float]:
    return [_silu_scalar(v) for v in values]


def _rmsnorm(vec: list[float], gamma: list[float], eps: float) -> list[float]:
    inv_rms = 1.0 / (sum(x * x for x in vec) / len(vec) + eps) ** 0.5
    return [vec[i] * inv_rms * gamma[i] for i in range(len(vec))]


def _l2_normalize(vec: list[float], eps: float = 1e-6) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    inv = 1.0 / max(norm, eps)
    return [x * inv for x in vec]


def _norm_inverse(vec: list[float], eps: float = 1e-6) -> float:
    norm_sq = 0.0
    for x in vec:
        norm_sq += x * x
    return 1.0 / max(math.sqrt(norm_sq), eps)

def _apply_rotary_inplace(buf: array, off: int, cos: list[float], sin: list[float]) -> None:
    half = len(cos)
    for i in range(half):
        x1 = buf[off + i]
        x2 = buf[off + half + i]
        buf[off + i] = x1 * cos[i] - x2 * sin[i]
        buf[off + half + i] = x1 * sin[i] + x2 * cos[i]


def _flat_zeros(size: int) -> array:
    return array("d", [0.0]) * size


def _flat_vecs(vecs: list[list[float]]) -> array:
    out = _flat_zeros(len(vecs) * TILE_EMBED_DIM)
    for idx, vec in enumerate(vecs):
        off = idx * TILE_EMBED_DIM
        for j in range(TILE_EMBED_DIM):
            out[off + j] = vec[j]
    return out


def _norm_inverse_flat(buf: array, off: int, dim: int, eps: float = 1e-6) -> float:
    norm_sq = 0.0
    for i in range(dim):
        x = buf[off + i]
        norm_sq += x * x
    return 1.0 / max(math.sqrt(norm_sq), eps)


def _residual_rmsnorm_silu_flat(
    base: array,
    base_off: int,
    update,
    update_off: int,
    gamma: list[float],
    eps: float,
    out: array,
    out_off: int,
    dim: int,
) -> None:
    mean_sq = 0.0
    for i in range(dim):
        x = base[base_off + i] + update[update_off + i]
        out[out_off + i] = x
        mean_sq += x * x
    inv_rms = 1.0 / (mean_sq / dim + eps) ** 0.5
    for i in range(dim):
        out[out_off + i] = _silu_scalar(out[out_off + i] * inv_rms * gamma[i])


def _residual_rmsnorm_silu(
    base: list[float],
    update: list[float],
    gamma: list[float],
    eps: float,
) -> list[float]:
    out = [0.0] * len(base)
    mean_sq = 0.0
    for i in range(len(base)):
        x = base[i] + update[i]
        out[i] = x
        mean_sq += x * x
    inv_rms = 1.0 / (mean_sq / len(base) + eps) ** 0.5
    for i in range(len(base)):
        out[i] = _silu_scalar(out[i] * inv_rms * gamma[i])
    return out


def _attention_update_inplace(
    q: list[float],
    q_scale: float,
    keys: list[list[float]],
    key_scales: list[float],
    values: list[list[float]],
    scale: float,
    scores: list[float],
    out: list[float],
) -> list[float]:
    n_keys = len(keys)
    best = float("-inf")
    for i in range(n_keys):
        score = scale * q_scale * key_scales[i] * _DOT(q, keys[i])
        scores[i] = score
        if score > best:
            best = score

    out_dim = len(values[0])
    for j in range(out_dim):
        out[j] = 0.0
    z = 0.0
    for i in range(n_keys):
        weight = _EXP(scores[i] - best)
        z += weight
        value = values[i]
        for j in range(out_dim):
            out[j] += weight * value[j]

    inv_z = 1.0 / z
    for j in range(out_dim):
        out[j] *= inv_z
    return out


def _attention_prescaled(
    scaled_q,
    scaled_keys,
    values_cols: list[list[float]],
    scores: list[float],
    out: list[float],
    n_keys: int,
    out_dim: int,
) -> list[float]:
    """Attention with pre-scaled Q/K and column-major values."""
    best = -1e30
    for i in range(n_keys):
        score = _DOT(scaled_q, scaled_keys[i])
        scores[i] = score
        if score > best:
            best = score
    z = 0.0
    for i in range(n_keys):
        w = _EXP(scores[i] - best)
        scores[i] = w
        z += w
    inv_z = 1.0 / z
    if n_keys == len(scores):
        for j in range(out_dim):
            out[j] = _DOT(scores, values_cols[j]) * inv_z
    else:
        w = scores[:n_keys]
        for j in range(out_dim):
            out[j] = _DOT(w, values_cols[j]) * inv_z
    return out


class ActorModel:
    """Stdlib-only actor matching the PyTorch CLS-attention actor on deploy obs."""

    def __init__(self, output_heads: dict[str, int]):
        self._output_heads = dict(output_heads)
        self.tile_linear = Linear(TILE_INPUT_DIM, TILE_EMBED_DIM)
        self.context_proj = Linear(GLOBAL_FEATURES, ACTOR_CONTEXT_DIM)
        self.context_k_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)
        self.context_v_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)
        self.tile_sa_q_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)
        self.tile_sa_k_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)
        self.tile_sa_v_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)
        self.cls_ca_q_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)
        self.cls_ca_k_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)
        self.cls_ca_v_proj = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM)

        self.context_ca_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_sa_norm = RMSNorm(TILE_EMBED_DIM)
        self.cls_ca_norm = RMSNorm(TILE_EMBED_DIM)
        self.latent_norm = RMSNorm(ACTOR_LATENT_DIM)
        self.context_ca_input_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_sa_out_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_ffn_out_norm = RMSNorm(TILE_EMBED_DIM)
        self.cls_ca_input_norm = RMSNorm(TILE_EMBED_DIM)
        self.latent_ffn_norm = RMSNorm(ACTOR_LATENT_DIM)
        self.latent_ffn_out_norm = RMSNorm(ACTOR_LATENT_DIM)

        self.cls_tokens = Tensor.zeros((ACTOR_CLS_TOKENS, TILE_EMBED_DIM))
        self.context_tau = Tensor.zeros((ACTOR_CLS_TOKENS,))
        self.tile_sa_gamma = Tensor.zeros((1,))
        self.cls_ca_tau = Tensor.zeros((ACTOR_CLS_TOKENS,))

        self.latent_ffn_up = Linear(ACTOR_LATENT_DIM, ACTOR_LATENT_DIM * 2)
        self.latent_ffn_down = Linear(ACTOR_LATENT_DIM * 2, ACTOR_LATENT_DIM)

        self.tile_ffn_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_ffn_up = Linear(TILE_EMBED_DIM, TILE_EMBED_DIM * 2)
        self.tile_ffn_down = Linear(TILE_EMBED_DIM * 2, TILE_EMBED_DIM)

        self.head_build_type = Linear(ACTOR_LATENT_DIM, output_heads["build_type"])
        self.head_build_dir = Linear(ACTOR_LATENT_DIM, output_heads["build_dir"])
        self.head_conv_dir = Linear(ACTOR_LATENT_DIM, output_heads["conv_dir"])
        self.head_bridge_target = Linear(ACTOR_LATENT_DIM, output_heads["bridge_target"])
        self.head_move_dir = Linear(ACTOR_LATENT_DIM, output_heads["move_dir"])
        self.head_temperature = Linear(ACTOR_LATENT_DIM, output_heads["temperature"])
        self.head_h_gate = Linear(ACTOR_GRU_INPUT_DIM, output_heads["h_gate"])
        self.head_h_cand = Linear(ACTOR_GRU_INPUT_DIM, output_heads["h_cand"])

        self._tile_dxdy = [
            (float((i % GRID_SIZE) - GRID_SIZE // 2) / float(GRID_SIZE // 2),
             float((i // GRID_SIZE) - GRID_SIZE // 2) / float(GRID_SIZE // 2))
            for i in range(TILE_GRID_N)
        ]
        self._infer_cache: dict | None = None
        self._attn_scores = [0.0] * TILE_GRID_N
        self._attn_update = [0.0] * TILE_EMBED_DIM
        self._tile_ffn_hidden = [0.0] * (TILE_EMBED_DIM * 2)
        self._tile_ffn_normed = [0.0] * TILE_EMBED_DIM
        self._cls_q_buf = [0.0] * TILE_EMBED_DIM
        self._tmp_token = [0.0] * TILE_EMBED_DIM
        self._latent_input = [0.0] * ACTOR_LATENT_DIM
        self._global_ctx = [0.0] * ACTOR_CONTEXT_DIM
        self._cls_state = [[0.0] * TILE_EMBED_DIM for _ in range(ACTOR_CLS_TOKENS)]
        n_ctx_tokens = ACTOR_CONTEXT_CHUNKS * 2
        self._projected_ctx_k = [[0.0] * TILE_EMBED_DIM for _ in range(n_ctx_tokens)]
        self._projected_ctx_k_scales = [0.0] * n_ctx_tokens
        self._scaled_ctx_k = [[0.0] * TILE_EMBED_DIM for _ in range(n_ctx_tokens)]
        self._ctx_attn_scores = [0.0] * n_ctx_tokens
        self._ctx_v_cols = [[0.0] * n_ctx_tokens for _ in range(TILE_EMBED_DIM)]
        flat_size = TILE_GRID_N * TILE_EMBED_DIM
        self._encoded_flat = _flat_zeros(flat_size)
        self._tile_q_flat = _flat_zeros(flat_size)
        self._tile_k_flat = _flat_zeros(flat_size)
        self._tile_v_flat = _flat_zeros(flat_size)
        self._enriched_flat = _flat_zeros(flat_size)
        self._encoded_views = [
            memoryview(self._encoded_flat)[i * TILE_EMBED_DIM:(i + 1) * TILE_EMBED_DIM]
            for i in range(TILE_GRID_N)
        ]
        self._tile_q_views = [
            memoryview(self._tile_q_flat)[i * TILE_EMBED_DIM:(i + 1) * TILE_EMBED_DIM]
            for i in range(TILE_GRID_N)
        ]
        self._tile_k_views = [
            memoryview(self._tile_k_flat)[i * TILE_EMBED_DIM:(i + 1) * TILE_EMBED_DIM]
            for i in range(TILE_GRID_N)
        ]
        self._enriched_views = [
            memoryview(self._enriched_flat)[i * TILE_EMBED_DIM:(i + 1) * TILE_EMBED_DIM]
            for i in range(TILE_GRID_N)
        ]
        self._tile_k_norm_sq = [0.0] * TILE_GRID_N
        self._cls_v_cache = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
        self._tile_v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
        self._cls_v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
        self._scaled_cls_k = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]

    def parameters(self) -> list[Tensor]:
        params: list[Tensor] = []
        for layer in (
            self.tile_linear,
            self.context_proj,
            self.context_k_proj,
            self.context_v_proj,
            self.tile_sa_k_proj,
            self.tile_sa_v_proj,
            self.cls_ca_k_proj,
            self.cls_ca_v_proj,
        ):
            params.extend(layer.parameters())
        params.append(self.cls_tokens)
        params.append(self.context_tau)
        for layer in (
            self.context_ca_norm,
            self.tile_sa_norm,
            self.cls_ca_norm,
            self.latent_norm,
            self.head_build_type,
            self.head_build_dir,
            self.head_conv_dir,
            self.head_bridge_target,
            self.head_move_dir,
            self.head_temperature,
            self.head_h_gate,
            self.head_h_cand,
        ):
            params.extend(layer.parameters())
        params.extend(self.tile_sa_q_proj.parameters())
        params.extend(self.cls_ca_q_proj.parameters())
        params.append(self.tile_sa_gamma)
        params.append(self.cls_ca_tau)
        params.extend(self.latent_ffn_norm.parameters())
        params.extend(self.latent_ffn_up.parameters())
        params.extend(self.latent_ffn_down.parameters())
        params.extend(self.tile_ffn_norm.parameters())
        params.extend(self.tile_ffn_up.parameters())
        params.extend(self.tile_ffn_down.parameters())
        params.extend(self.context_ca_input_norm.parameters())
        params.extend(self.tile_sa_out_norm.parameters())
        params.extend(self.tile_ffn_out_norm.parameters())
        params.extend(self.cls_ca_input_norm.parameters())
        params.extend(self.latent_ffn_out_norm.parameters())
        return params

    def train(self, mode: bool = True) -> None:
        return None

    def eval(self) -> None:
        return None

    def _build_infer_cache(self) -> dict:
        def _linear_rows(layer: Linear) -> tuple[list[list[float]], list[float] | None]:
            w = layer.weight.data
            rows = [w[j::layer.out_features] for j in range(layer.out_features)]
            b = list(layer.bias.data) if layer.bias is not None else None
            return rows, b

        def _scale_input_rows(
            rows: list[list[float]],
            input_scales: list[float],
        ) -> list[list[float]]:
            return [
                [row[i] * input_scales[i] for i in range(len(input_scales))]
                for row in rows
            ]

        action_heads = (
            ("build_type", self.head_build_type),
            ("build_dir", self.head_build_dir),
            ("conv_dir", self.head_conv_dir),
            ("bridge_target", self.head_bridge_target),
            ("move_dir", self.head_move_dir),
        )
        action_names: list[str] = []
        action_slices: list[tuple[int, int]] = []
        action_rows: list[list[float]] = []
        action_bias: list[float] = []
        offset = 0
        for name, head in action_heads:
            rows, bias = _linear_rows(head)
            action_names.append(name)
            action_slices.append((offset, offset + head.out_features))
            action_rows.extend(rows)
            if bias is None:
                action_bias.extend([0.0] * head.out_features)
            else:
                action_bias.extend(bias)
            offset += head.out_features

        head_h_gate_rows, head_h_gate_bias = _linear_rows(self.head_h_gate)
        head_h_cand_rows, head_h_cand_bias = _linear_rows(self.head_h_cand)
        latent_ffn_noop = all(x == 0.0 for x in self.latent_ffn_down.weight.data)
        if self.latent_ffn_down.bias is not None:
            latent_ffn_noop = latent_ffn_noop and all(x == 0.0 for x in self.latent_ffn_down.bias.data)
        latent_ffn_norm = None
        latent_ffn_up = None
        latent_ffn_down = None
        if not latent_ffn_noop:
            latent_ffn_norm = (list(self.latent_ffn_norm.gamma.data), self.latent_ffn_norm.eps)
            latent_ffn_up = _linear_rows(self.latent_ffn_up)
            latent_ffn_down = _linear_rows(self.latent_ffn_down)

        tile_rows_full, tile_bias = _linear_rows(self.tile_linear)
        tile_q_rows, tile_q_bias = _linear_rows(self.tile_sa_q_proj)
        tile_k_rows, tile_k_bias = _linear_rows(self.tile_sa_k_proj)
        tile_v_rows, tile_v_bias = _linear_rows(self.tile_sa_v_proj)
        if tile_bias is None:
            tile_bias = [0.0] * TILE_EMBED_DIM
        if tile_q_bias is None:
            tile_q_bias = [0.0] * TILE_EMBED_DIM
        if tile_k_bias is None:
            tile_k_bias = [0.0] * TILE_EMBED_DIM
        if tile_v_bias is None:
            tile_v_bias = [0.0] * TILE_EMBED_DIM
        tile_dx_w = [row[0] for row in tile_rows_full]
        tile_dy_w = [row[1] for row in tile_rows_full]
        tile_rows = [row[2:] for row in tile_rows_full]
        base_pos = []
        for dx, dy in self._tile_dxdy:
            base_pos.append([
                tile_bias[j] + dx * tile_dx_w[j] + dy * tile_dy_w[j]
                for j in range(TILE_EMBED_DIM)
            ])
        env_lut = [[0.0] * TILE_EMBED_DIM for _ in range(4)]  # empty, wall, ti, ax
        imp_lut = [[0.0] * TILE_EMBED_DIM for _ in range(2)]
        build_lut = [[0.0] * TILE_EMBED_DIM for _ in range(N_BLD_TYPES)]
        team_lut = [[0.0] * TILE_EMBED_DIM for _ in range(3)]  # -1, 0, +1
        dir_lut = [[0.0] * TILE_EMBED_DIM for _ in range(9)]  # NW..SE
        stored_lut = [[0.0] * TILE_EMBED_DIM for _ in range(3)]  # -1, 0, +1
        unit_lut = [[0.0] * TILE_EMBED_DIM for _ in range(3)]  # none, friendly, enemy
        cooldown_w = [0.0] * TILE_EMBED_DIM
        for j, row in enumerate(tile_rows):
            env_lut[1][j] = row[0]
            env_lut[2][j] = row[1]
            env_lut[3][j] = row[2]
            imp_lut[1][j] = row[3]
            for bidx in range(N_BLD_TYPES):
                build_lut[bidx][j] = row[4 + bidx]
            team_lut[0][j] = -row[14]
            team_lut[2][j] = row[14]
            for dir_y in (-1, 0, 1):
                for dir_x in (-1, 0, 1):
                    dir_lut[(dir_y + 1) * 3 + (dir_x + 1)][j] = (
                        dir_x * row[15] + dir_y * row[16]
                    )
            cooldown_w[j] = row[17]
            stored_lut[0][j] = -row[18]
            stored_lut[2][j] = row[18]
            unit_lut[1][j] = row[19] + row[20]
            unit_lut[2][j] = row[19] - row[20]
        symbolic_tiles = {
            "base": _flat_vecs(base_pos),
            "env": _flat_vecs(env_lut),
            "imp": _flat_vecs(imp_lut),
            "build": _flat_vecs(build_lut),
            "team": _flat_vecs(team_lut),
            "dir": _flat_vecs(dir_lut),
            "stored": _flat_vecs(stored_lut),
            "unit": _flat_vecs(unit_lut),
            "cooldown": array("d", cooldown_w),
        }

        tile_sa_norm_gamma = list(self.tile_sa_norm.gamma.data)
        cache = {
            "symbolic_tiles": symbolic_tiles,
            "context_proj": _linear_rows(self.context_proj),
            "context_k_proj": _linear_rows(self.context_k_proj),
            "context_v_proj": _linear_rows(self.context_v_proj),
            "tile_sa_q_proj": (_scale_input_rows(tile_q_rows, tile_sa_norm_gamma), tile_q_bias),
            "tile_sa_k_proj": (_scale_input_rows(tile_k_rows, tile_sa_norm_gamma), tile_k_bias),
            "tile_sa_v_proj": (_scale_input_rows(tile_v_rows, tile_sa_norm_gamma), tile_v_bias),
            "cls_ca_q_proj": _linear_rows(self.cls_ca_q_proj),
            "cls_ca_k_proj": _linear_rows(self.cls_ca_k_proj),
            "cls_ca_v_proj": _linear_rows(self.cls_ca_v_proj),
            "context_ca_norm": (list(self.context_ca_norm.gamma.data), self.context_ca_norm.eps),
            "tile_sa_norm": (tile_sa_norm_gamma, self.tile_sa_norm.eps),
            "cls_ca_norm": (list(self.cls_ca_norm.gamma.data), self.cls_ca_norm.eps),
            "latent_norm": (list(self.latent_norm.gamma.data), self.latent_norm.eps),
            "context_ca_input_norm": (list(self.context_ca_input_norm.gamma.data), self.context_ca_input_norm.eps),
            "tile_sa_out_norm": (list(self.tile_sa_out_norm.gamma.data), self.tile_sa_out_norm.eps),
            "tile_ffn_out_norm": (list(self.tile_ffn_out_norm.gamma.data), self.tile_ffn_out_norm.eps),
            "cls_ca_input_norm": (list(self.cls_ca_input_norm.gamma.data), self.cls_ca_input_norm.eps),
            "cls_tokens": list(self.cls_tokens.data),
            "context_tau": [_EXP(t) for t in self.context_tau.data],
            "tile_sa_gamma": _EXP(self.tile_sa_gamma.data[0]),
            "cls_ca_tau": [_EXP(t) for t in self.cls_ca_tau.data],
            "latent_ffn_noop": latent_ffn_noop,
            "latent_ffn_norm": latent_ffn_norm,
            "latent_ffn_out_norm": (list(self.latent_ffn_out_norm.gamma.data), self.latent_ffn_out_norm.eps),
            "latent_ffn_up": latent_ffn_up,
            "latent_ffn_down": latent_ffn_down,
            "action_heads": (action_names, action_slices, action_rows, action_bias),
            "head_h_gate": (head_h_gate_rows, head_h_gate_bias),
            "head_h_cand": (head_h_cand_rows, head_h_cand_bias),
            "tile_ffn_norm": (list(self.tile_ffn_norm.gamma.data), self.tile_ffn_norm.eps),
            "tile_ffn_up": _linear_rows(self.tile_ffn_up),
            "tile_ffn_down": _linear_rows(self.tile_ffn_down),
        }
        return cache

    def infer(self, features: list[float]) -> dict[str, list[float]]:
        if self._infer_cache is None:
            self._infer_cache = self._build_infer_cache()
        cache = self._infer_cache

        symbolic_tiles = cache["symbolic_tiles"]
        context_rows, context_bias = cache["context_proj"]
        ctx_k_rows, ctx_k_bias = cache["context_k_proj"]
        ctx_v_rows, ctx_v_bias = cache["context_v_proj"]
        tile_q_rows, tile_q_bias = cache["tile_sa_q_proj"]
        tile_k_rows, tile_k_bias = cache["tile_sa_k_proj"]
        tile_v_rows, tile_v_bias = cache["tile_sa_v_proj"]
        cls_q_rows, cls_q_bias = cache["cls_ca_q_proj"]
        cls_k_rows, cls_k_bias = cache["cls_ca_k_proj"]
        cls_v_rows, cls_v_bias = cache["cls_ca_v_proj"]
        context_norm_gamma, context_norm_eps = cache["context_ca_norm"]
        _tile_norm_gamma, tile_norm_eps = cache["tile_sa_norm"]
        cls_norm_gamma, cls_norm_eps = cache["cls_ca_norm"]
        latent_norm_gamma, latent_norm_eps = cache["latent_norm"]
        context_in_norm_gamma, context_in_norm_eps = cache["context_ca_input_norm"]
        tile_sa_out_norm_gamma, tile_sa_out_norm_eps = cache["tile_sa_out_norm"]
        tile_ffn_out_norm_gamma, tile_ffn_out_norm_eps = cache["tile_ffn_out_norm"]
        cls_in_norm_gamma, cls_in_norm_eps = cache["cls_ca_input_norm"]
        latent_ffn_noop = cache["latent_ffn_noop"]
        latent_ffn_norm = cache["latent_ffn_norm"]
        latent_ffn_out_norm_gamma, latent_ffn_out_norm_eps = cache["latent_ffn_out_norm"]
        cls_tokens_flat = cache["cls_tokens"]
        context_tau = cache["context_tau"]
        tile_sa_gamma = cache["tile_sa_gamma"]
        cls_ca_tau = cache["cls_ca_tau"]
        tile_ffn_norm_gamma, tile_ffn_norm_eps = cache["tile_ffn_norm"]
        tile_ffn_up_rows, tile_ffn_up_bias = cache["tile_ffn_up"]
        tile_ffn_down_rows, tile_ffn_down_bias = cache["tile_ffn_down"]
        attn_scores = self._attn_scores
        attn_update = self._attn_update
        tile_ffn_hidden_buf = self._tile_ffn_hidden
        tile_ffn_normed_buf = self._tile_ffn_normed
        cls_q_buf = self._cls_q_buf
        tmp_token = self._tmp_token

        tile_obs_end = TILE_GRID_N * TILE_FEATURES
        globals_end = tile_obs_end + GLOBAL_FEATURES
        globals_obs = features[tile_obs_end:globals_end]
        h_prev = features[globals_end:globals_end + H_DIM]
        base_flat = symbolic_tiles["base"]
        env_flat = symbolic_tiles["env"]
        imp_flat = symbolic_tiles["imp"]
        build_flat = symbolic_tiles["build"]
        team_flat = symbolic_tiles["team"]
        dir_flat = symbolic_tiles["dir"]
        stored_flat = symbolic_tiles["stored"]
        unit_flat = symbolic_tiles["unit"]
        cooldown_w = symbolic_tiles["cooldown"]
        encoded_flat = self._encoded_flat
        tile_q_flat = self._tile_q_flat
        tile_k_flat = self._tile_k_flat
        tile_v_flat = self._tile_v_flat
        enriched_flat = self._enriched_flat
        encoded = self._encoded_views
        tile_q = self._tile_q_views
        tile_k = self._tile_k_views
        tile_k_norm_sq = self._tile_k_norm_sq
        cls_v = self._cls_v_cache
        tile_v_cols = self._tile_v_cols
        global_ctx = self._global_ctx
        cls_state = self._cls_state
        projected_ctx_k = self._projected_ctx_k
        projected_ctx_k_scales = self._projected_ctx_k_scales
        scaled_ctx_k = self._scaled_ctx_k
        ctx_attn_scores = self._ctx_attn_scores
        ctx_v_cols = self._ctx_v_cols
        tile_sa_g2 = 2.0 * tile_sa_gamma
        # Fused tile encoding + QKV projection + V transpose
        for t in range(TILE_GRID_N):
            base = t * TILE_FEATURES
            off = t * TILE_EMBED_DIM
            wall = features[base]
            ore_ti = features[base + 1]
            ore_ax = features[base + 2]
            env_id = 1 if wall == 1.0 else 2 if ore_ti == 1.0 else 3 if ore_ax == 1.0 else 0
            imp_id = int(features[base + 3])
            build_idx = 0
            for bidx in range(N_BLD_TYPES):
                if features[base + 4 + bidx] == 1.0:
                    build_idx = bidx
                    break
            b_team = features[base + 14]
            team_id = 0 if b_team < 0.0 else 2 if b_team > 0.0 else 1
            dir_x = int(features[base + 15])
            dir_y = int(features[base + 16])
            dir_id = (dir_y + 1) * 3 + (dir_x + 1)
            cooldown = features[base + 17]
            stored = features[base + 18]
            stored_id = 0 if stored < 0.0 else 2 if stored > 0.0 else 1
            has_unit = features[base + 19]
            unit_friendly = features[base + 20]
            unit_id = 0 if has_unit == 0.0 else 1 if unit_friendly > 0.0 else 2

            env_off = env_id * TILE_EMBED_DIM
            imp_off = imp_id * TILE_EMBED_DIM
            build_off = build_idx * TILE_EMBED_DIM
            team_off = team_id * TILE_EMBED_DIM
            dir_off = dir_id * TILE_EMBED_DIM
            stored_off = stored_id * TILE_EMBED_DIM
            unit_off = unit_id * TILE_EMBED_DIM

            # Tile embedding + SiLU → encoded_flat
            token = encoded[t]
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                x = (
                    base_flat[off + j]
                    + env_flat[env_off + j]
                    + imp_flat[imp_off + j]
                    + build_flat[build_off + j]
                    + team_flat[team_off + j]
                    + dir_flat[dir_off + j]
                    + cooldown * cooldown_w[j]
                    + stored_flat[stored_off + j]
                    + unit_flat[unit_off + j]
                )
                if x >= 0.0:
                    x = x / (1.0 + _EXP(-x))
                else:
                    e = _EXP(x)
                    x = x * e / (1.0 + e)
                encoded_flat[off + j] = x
                mean_sq += x * x
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + tile_norm_eps) ** 0.5
            # Pre-norm QKV projection — no QK-norm, no RoPE
            k_nsq = 0.0
            for j in range(TILE_EMBED_DIM):
                q_val = inv_rms * _DOT(token, tile_q_rows[j]) + tile_q_bias[j]
                k_val = inv_rms * _DOT(token, tile_k_rows[j]) + tile_k_bias[j]
                v_val = inv_rms * _DOT(token, tile_v_rows[j]) + tile_v_bias[j]
                tile_q_flat[off + j] = q_val
                tile_k_flat[off + j] = k_val
                tile_v_flat[off + j] = v_val
                k_nsq += k_val * k_val
                tile_v_cols[j][t] = v_val  # V transpose inline
            tile_k_norm_sq[t] = k_nsq
        # Peri-LN RBF attention: input-norm QKV, then output-norm branch before residual add.
        for i in range(TILE_GRID_N):
            off = i * TILE_EMBED_DIM
            qi = tile_q[i]
            best = -1e30
            for k in range(TILE_GRID_N):
                score = tile_sa_g2 * _DOT(qi, tile_k[k]) - tile_sa_gamma * tile_k_norm_sq[k]
                attn_scores[k] = score
                if score > best:
                    best = score
            z = 0.0
            for k in range(TILE_GRID_N):
                w = _EXP(attn_scores[k] - best)
                attn_scores[k] = w
                z += w
            inv_z = 1.0 / z
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                x = _DOT(attn_scores, tile_v_cols[j]) * inv_z
                tmp_token[j] = x
                mean_sq += x * x
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + tile_sa_out_norm_eps) ** 0.5
            for j in range(TILE_EMBED_DIM):
                enriched_flat[off + j] = encoded_flat[off + j] + tmp_token[j] * inv_rms * tile_sa_out_norm_gamma[j]

        # Tile FFN: pre-norm MLP followed by output-norm branch before residual add.
        tile_ffn_hidden = TILE_EMBED_DIM * 2
        for t in range(TILE_GRID_N):
            off = t * TILE_EMBED_DIM
            token = self._enriched_views[t]
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                x = token[j]
                mean_sq += x * x
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + tile_ffn_norm_eps) ** 0.5
            normed = tile_ffn_normed_buf
            for j in range(TILE_EMBED_DIM):
                normed[j] = token[j] * inv_rms * tile_ffn_norm_gamma[j]
            # Up + SiLU
            hidden = tile_ffn_hidden_buf
            for j in range(tile_ffn_hidden):
                x = _DOT(normed, tile_ffn_up_rows[j]) + tile_ffn_up_bias[j]
                if x >= 0.0:
                    hidden[j] = x / (1.0 + _EXP(-x))
                else:
                    e = _EXP(x)
                    hidden[j] = x * e / (1.0 + e)
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                x = _DOT(hidden, tile_ffn_down_rows[j]) + tile_ffn_down_bias[j]
                tmp_token[j] = x
                mean_sq += x * x
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + tile_ffn_out_norm_eps) ** 0.5
            for j in range(TILE_EMBED_DIM):
                enriched_flat[off + j] = token[j] + tmp_token[j] * inv_rms * tile_ffn_out_norm_gamma[j]

        # CLS K/V projection — will pre-scale keys after cls_ca_scale is known

        for j in range(ACTOR_CONTEXT_DIM):
            global_ctx[j] = _DOT(globals_obs, context_rows[j]) + context_bias[j]
        n_ctx_chunks = ACTOR_CONTEXT_CHUNKS * 2
        for chunk_idx in range(n_ctx_chunks):
            if chunk_idx < ACTOR_CONTEXT_CHUNKS:
                token = global_ctx[
                    chunk_idx * TILE_EMBED_DIM:(chunk_idx + 1) * TILE_EMBED_DIM
                ]
            else:
                h_idx = chunk_idx - ACTOR_CONTEXT_CHUNKS
                token = h_prev[
                    h_idx * TILE_EMBED_DIM:(h_idx + 1) * TILE_EMBED_DIM
                ]
            k_tok = projected_ctx_k[chunk_idx]
            k_nsq = 0.0
            for j in range(TILE_EMBED_DIM):
                k_val = _DOT(token, ctx_k_rows[j]) + ctx_k_bias[j]
                v_val = _DOT(token, ctx_v_rows[j]) + ctx_v_bias[j]
                k_tok[j] = k_val
                ctx_v_cols[j][chunk_idx] = v_val
                k_nsq += k_val * k_val
            projected_ctx_k_scales[chunk_idx] = 1.0 / max(math.sqrt(k_nsq), 1e-6)

        for i in range(ACTOR_CLS_TOKENS):
            token = cls_state[i]
            off = i * TILE_EMBED_DIM
            for j in range(TILE_EMBED_DIM):
                token[j] = cls_tokens_flat[off + j]
        for i, token in enumerate(cls_state):
            ctx_scale = context_tau[i]
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                mean_sq += token[j] * token[j]
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + context_in_norm_eps) ** 0.5
            q_nsq = 0.0
            for j in range(TILE_EMBED_DIM):
                x = token[j] * inv_rms * context_in_norm_gamma[j]
                cls_q_buf[j] = x
                q_nsq += x * x
            q_scale = 1.0 / max(math.sqrt(q_nsq), 1e-6)
            for j in range(TILE_EMBED_DIM):
                cls_q_buf[j] *= q_scale
            for ci in range(n_ctx_chunks):
                key_scale = projected_ctx_k_scales[ci] * ctx_scale
                src_key = projected_ctx_k[ci]
                dst_key = scaled_ctx_k[ci]
                for j in range(TILE_EMBED_DIM):
                    dst_key[j] = src_key[j] * key_scale
            update = _attention_prescaled(
                cls_q_buf,
                scaled_ctx_k,
                ctx_v_cols,
                ctx_attn_scores,
                attn_update,
                n_ctx_chunks,
                TILE_EMBED_DIM,
            )
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                x = update[j]
                tmp_token[j] = x
                mean_sq += x * x
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + context_norm_eps) ** 0.5
            for j in range(TILE_EMBED_DIM):
                x = token[j] + tmp_token[j] * inv_rms * context_norm_gamma[j]
                if x >= 0.0:
                    token[j] = x / (1.0 + _EXP(-x))
                else:
                    e = _EXP(x)
                    token[j] = x * e / (1.0 + e)

        # CLS K/V projection + key pre-scaling (normalize only, tau folded into Q)
        scaled_cls_k = self._scaled_cls_k
        cls_v_cols = self._cls_v_cols
        for idx in range(TILE_GRID_N):
            token = self._enriched_views[idx]
            sk = scaled_cls_k[idx]
            v_tok = cls_v[idx]
            k_nsq = 0.0
            for j in range(TILE_EMBED_DIM):
                k_val = _DOT(token, cls_k_rows[j]) + cls_k_bias[j]
                v_val = _DOT(token, cls_v_rows[j]) + cls_v_bias[j]
                sk[j] = k_val
                v_tok[j] = v_val
                cls_v_cols[j][idx] = v_val  # V transpose inline
                k_nsq += k_val * k_val
            # Pre-scale key: normalize only (per-seed tau folded into Q)
            s = 1.0 / max(math.sqrt(k_nsq), 1e-6)
            for j in range(TILE_EMBED_DIM):
                sk[j] *= s
        for i, token in enumerate(cls_state):
            q_tok = cls_q_buf
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                mean_sq += token[j] * token[j]
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + cls_in_norm_eps) ** 0.5
            for j in range(TILE_EMBED_DIM):
                tmp_token[j] = token[j] * inv_rms * cls_in_norm_gamma[j]
            q_nsq = 0.0
            for j in range(TILE_EMBED_DIM):
                q_val = _DOT(tmp_token, cls_q_rows[j]) + cls_q_bias[j]
                q_tok[j] = q_val
                q_nsq += q_val * q_val
            # Fold per-seed tau * q_norm into query
            q_scale = cls_ca_tau[i] / max(math.sqrt(q_nsq), 1e-6)
            for j in range(TILE_EMBED_DIM):
                q_tok[j] *= q_scale
            update = _attention_prescaled(
                q_tok,
                scaled_cls_k,
                cls_v_cols,
                attn_scores,
                attn_update,
                TILE_GRID_N,
                TILE_EMBED_DIM,
            )
            mean_sq = 0.0
            for j in range(TILE_EMBED_DIM):
                x = update[j]
                tmp_token[j] = x
                mean_sq += x * x
            inv_rms = 1.0 / (mean_sq / TILE_EMBED_DIM + cls_norm_eps) ** 0.5
            for j in range(TILE_EMBED_DIM):
                x = token[j] + tmp_token[j] * inv_rms * cls_norm_gamma[j]
                if x >= 0.0:
                    token[j] = x / (1.0 + _EXP(-x))
                else:
                    e = _EXP(x)
                    token[j] = x * e / (1.0 + e)

        latent_input = self._latent_input
        for i in range(ACTOR_CLS_TOKENS):
            off = i * TILE_EMBED_DIM
            token = cls_state[i]
            for j in range(TILE_EMBED_DIM):
                latent_input[off + j] = token[j]
        latent = list(latent_input)
        if not latent_ffn_noop:
            latent_ffn_norm_gamma, latent_ffn_norm_eps = latent_ffn_norm  # type: ignore[misc]
            latent_ffn_up_rows, latent_ffn_up_bias = cache["latent_ffn_up"]
            latent_ffn_down_rows, latent_ffn_down_bias = cache["latent_ffn_down"]
            latent_hidden = _silu_vec(_linear(_rmsnorm(latent, latent_ffn_norm_gamma, latent_ffn_norm_eps), latent_ffn_up_rows, latent_ffn_up_bias))
            latent_delta = _linear(latent_hidden, latent_ffn_down_rows, latent_ffn_down_bias)
            latent_delta = _rmsnorm(latent_delta, latent_ffn_out_norm_gamma, latent_ffn_out_norm_eps)
            latent = [latent[i] + latent_delta[i] for i in range(ACTOR_LATENT_DIM)]
        latent = _rmsnorm(latent, latent_norm_gamma, latent_norm_eps)

        action_names, action_slices, action_rows, action_bias = cache["action_heads"]
        outputs: dict[str, list[float]] = {}
        for name, (start, end) in zip(action_names, action_slices):
            outputs[name] = [_DOT(latent, action_rows[j]) + action_bias[j] for j in range(start, end)]

        gru_in = latent + h_prev
        gate_rows, gate_bias = cache["head_h_gate"]
        cand_rows, cand_bias = cache["head_h_cand"]
        h_gate = [_DOT(gru_in, gate_rows[j]) + gate_bias[j] for j in range(H_DIM)]
        h_cand = [math.tanh(_DOT(gru_in, cand_rows[j]) + cand_bias[j]) for j in range(H_DIM)]
        outputs["h_gate"] = h_gate
        outputs["h_cand"] = h_cand
        return outputs

    def save(self) -> list[list[float]]:
        return [list(p.data) for p in self.parameters()]

    def load(self, state: list[list[float]]) -> None:
        params = self.parameters()
        supported = (
            len(params),
            len(params) - 5,
            len(params) - 10,
            len(params) - 16,
        )
        if len(state) not in supported:
            raise AssertionError(
                f"State has {len(state)} entries, expected one of {supported}"
            )
        if len(state) != len(params):
            default_state = ActorModel(self._output_heads).save()
            state = list(state) + default_state[len(state):]
        for p, s in zip(params, state):
            if len(p.data) != len(s):
                # Handle shape migration (e.g. cls_ca_tau 1→4): broadcast scalar
                if len(s) == 1 and len(p.data) > 1:
                    s = s * len(p.data)
                else:
                    raise AssertionError(f"Param size mismatch: {len(p.data)} vs {len(s)}")
            p.data[:] = s
        self._infer_cache = None
