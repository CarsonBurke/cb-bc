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


def _project_rows_flat(vecs: list[list[float]], rows: list[list[float]]) -> array:
    out = array("d")
    append = out.append
    for vec in vecs:
        for row in rows:
            append(_DOT(vec, row))
    return out


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


def _attention_update_flat_inplace(
    q_view,
    q_scale: float,
    keys,
    key_scales: list[float],
    values,
    scale: float,
    scores: list[float],
    out: array,
    out_off: int,
) -> None:
    n_keys = len(keys)
    dim = len(values[0])
    best = float("-inf")
    for i in range(n_keys):
        score = scale * q_scale * key_scales[i] * _DOT(q_view, keys[i])
        scores[i] = score
        if score > best:
            best = score

    for j in range(dim):
        out[out_off + j] = 0.0
    z = 0.0
    for i in range(n_keys):
        weight = _EXP(scores[i] - best)
        z += weight
        value = values[i]
        for j in range(dim):
            out[out_off + j] += weight * value[j]

    inv_z = 1.0 / z
    for j in range(dim):
        out[out_off + j] *= inv_z


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


def _build_rope_cache() -> tuple[list[list[float]], list[list[float]]]:
    half = TILE_EMBED_DIM // 2
    freqs = [1.0 / (10000.0 ** (float(i) / half)) for i in range(0, half, 2)]
    pos_cos: list[list[float]] = []
    pos_sin: list[list[float]] = []
    for pos in range(GRID_SIZE):
        angles = [pos * freq for freq in freqs]
        pos_cos.append([math.cos(a) for a in angles])
        pos_sin.append([math.sin(a) for a in angles])
    cos_cache: list[list[float]] = []
    sin_cache: list[list[float]] = []
    for idx in range(TILE_GRID_N):
        y = idx // GRID_SIZE
        x = idx % GRID_SIZE
        cos_cache.append(pos_cos[y] + pos_cos[x])
        sin_cache.append(pos_sin[y] + pos_sin[x])
    return cos_cache, sin_cache


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
        self.latent_ffn_norm = RMSNorm(ACTOR_LATENT_DIM)

        self.cls_tokens = Tensor.zeros((ACTOR_CLS_TOKENS, TILE_EMBED_DIM))
        self.context_tau = Tensor.zeros((ACTOR_CLS_TOKENS,))
        self.tile_sa_tau = Tensor.zeros((1,))
        self.cls_ca_tau = Tensor.zeros((1,))

        self.latent_ffn_up = Linear(ACTOR_LATENT_DIM, ACTOR_LATENT_DIM * 2)
        self.latent_ffn_down = Linear(ACTOR_LATENT_DIM * 2, ACTOR_LATENT_DIM)

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
        self._rope_cos, self._rope_sin = _build_rope_cache()
        self._infer_cache: dict | None = None
        self._attn_scores = [0.0] * TILE_GRID_N
        self._attn_update = [0.0] * TILE_EMBED_DIM
        self._attn_update_flat = _flat_zeros(TILE_EMBED_DIM)
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
        self._tile_v_views = [
            memoryview(self._tile_v_flat)[i * TILE_EMBED_DIM:(i + 1) * TILE_EMBED_DIM]
            for i in range(TILE_GRID_N)
        ]
        self._enriched_views = [
            memoryview(self._enriched_flat)[i * TILE_EMBED_DIM:(i + 1) * TILE_EMBED_DIM]
            for i in range(TILE_GRID_N)
        ]
        self._tile_q_scales = [0.0] * TILE_GRID_N
        self._tile_k_scales = [0.0] * TILE_GRID_N
        self._cached_tile_obs = [0.0] * (TILE_GRID_N * TILE_FEATURES)
        self._cached_tile_obs_valid = False
        self._cls_k_cache = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
        self._cls_v_cache = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
        self._cls_k_scales_cache = [0.0] * TILE_GRID_N

        self._init_identity_linear(self.tile_sa_q_proj)
        self._init_identity_linear(self.cls_ca_q_proj)
        self.latent_ffn_down.weight.data[:] = [0.0] * len(self.latent_ffn_down.weight.data)
        self.latent_ffn_down.bias.data[:] = [0.0] * len(self.latent_ffn_down.bias.data)

    @staticmethod
    def _init_identity_linear(layer: Linear) -> None:
        data = layer.weight.data
        for i in range(layer.in_features):
            row_base = i * layer.out_features
            for j in range(layer.out_features):
                data[row_base + j] = 1.0 if i == j else 0.0
        if layer.bias is not None:
            layer.bias.data[:] = [0.0] * len(layer.bias.data)

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
        params.append(self.tile_sa_tau)
        params.append(self.cls_ca_tau)
        params.extend(self.latent_ffn_norm.parameters())
        params.extend(self.latent_ffn_up.parameters())
        params.extend(self.latent_ffn_down.parameters())
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

        def _project_rows(vecs: list[list[float]], rows: list[list[float]]) -> list[list[float]]:
            return [[_DOT(vec, row) for row in rows] for vec in vecs]

        action_heads = (
            ("build_type", self.head_build_type),
            ("build_dir", self.head_build_dir),
            ("conv_dir", self.head_conv_dir),
            ("bridge_target", self.head_bridge_target),
            ("move_dir", self.head_move_dir),
            ("temperature", self.head_temperature),
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
        if tile_bias is None:
            tile_bias = [0.0] * TILE_EMBED_DIM
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

        cache = {
            "symbolic_tiles": symbolic_tiles,
            "context_proj": _linear_rows(self.context_proj),
            "context_k_proj": _linear_rows(self.context_k_proj),
            "context_v_proj": _linear_rows(self.context_v_proj),
            "tile_sa_q_proj": _linear_rows(self.tile_sa_q_proj),
            "tile_sa_k_proj": _linear_rows(self.tile_sa_k_proj),
            "tile_sa_v_proj": _linear_rows(self.tile_sa_v_proj),
            "cls_ca_q_proj": _linear_rows(self.cls_ca_q_proj),
            "cls_ca_k_proj": _linear_rows(self.cls_ca_k_proj),
            "cls_ca_v_proj": _linear_rows(self.cls_ca_v_proj),
            "context_ca_norm": (list(self.context_ca_norm.gamma.data), self.context_ca_norm.eps),
            "tile_sa_norm": (list(self.tile_sa_norm.gamma.data), self.tile_sa_norm.eps),
            "cls_ca_norm": (list(self.cls_ca_norm.gamma.data), self.cls_ca_norm.eps),
            "latent_norm": (list(self.latent_norm.gamma.data), self.latent_norm.eps),
            "cls_tokens": list(self.cls_tokens.data),
            "context_tau": list(self.context_tau.data),
            "tile_sa_tau": list(self.tile_sa_tau.data),
            "cls_ca_tau": list(self.cls_ca_tau.data),
            "latent_ffn_noop": latent_ffn_noop,
            "latent_ffn_norm": latent_ffn_norm,
            "latent_ffn_up": latent_ffn_up,
            "latent_ffn_down": latent_ffn_down,
            "action_heads": (action_names, action_slices, action_rows, action_bias),
            "head_h_gate": (head_h_gate_rows, head_h_gate_bias),
            "head_h_cand": (head_h_cand_rows, head_h_cand_bias),
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
        tile_norm_gamma, tile_norm_eps = cache["tile_sa_norm"]
        cls_norm_gamma, cls_norm_eps = cache["cls_ca_norm"]
        latent_norm_gamma, latent_norm_eps = cache["latent_norm"]
        latent_ffn_noop = cache["latent_ffn_noop"]
        latent_ffn_norm = cache["latent_ffn_norm"]
        cls_tokens_flat = cache["cls_tokens"]
        context_tau = cache["context_tau"]
        tile_sa_tau = cache["tile_sa_tau"][0]
        cls_ca_tau = cache["cls_ca_tau"][0]
        attn_scores = self._attn_scores
        attn_update = self._attn_update

        tile_obs_end = TILE_GRID_N * TILE_FEATURES
        globals_end = tile_obs_end + GLOBAL_FEATURES
        globals_obs = features[tile_obs_end:globals_end]
        h_prev = features[globals_end:globals_end + H_DIM]
        tile_obs_same = self._cached_tile_obs_valid
        if tile_obs_same:
            cached_tile_obs = self._cached_tile_obs
            for idx in range(tile_obs_end):
                if cached_tile_obs[idx] != features[idx]:
                    tile_obs_same = False
                    break
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
        tile_q_scales = self._tile_q_scales
        tile_k_scales = self._tile_k_scales
        cls_k = self._cls_k_cache
        cls_v = self._cls_v_cache
        cls_k_scales = self._cls_k_scales_cache
        if not tile_obs_same:
            cached_tile_obs = self._cached_tile_obs
            for idx in range(tile_obs_end):
                cached_tile_obs[idx] = features[idx]
            self._cached_tile_obs_valid = True
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
                    encoded_flat[off + j] = _silu_scalar(x)
            for t in range(TILE_GRID_N):
                off = t * TILE_EMBED_DIM
                token = self._encoded_views[t]
                for j in range(TILE_EMBED_DIM):
                    q_val = _DOT(token, tile_q_rows[j])
                    k_val = _DOT(token, tile_k_rows[j])
                    v_val = _DOT(token, tile_v_rows[j])
                    if tile_q_bias is not None:
                        q_val += tile_q_bias[j]
                    if tile_k_bias is not None:
                        k_val += tile_k_bias[j]
                    if tile_v_bias is not None:
                        v_val += tile_v_bias[j]
                    tile_q_flat[off + j] = q_val
                    tile_k_flat[off + j] = k_val
                    tile_v_flat[off + j] = v_val
                tile_q_scales[t] = _norm_inverse_flat(tile_q_flat, off, TILE_EMBED_DIM)
                tile_k_scales[t] = _norm_inverse_flat(tile_k_flat, off, TILE_EMBED_DIM)
                _apply_rotary_inplace(tile_q_flat, off, self._rope_cos[t], self._rope_sin[t])
                _apply_rotary_inplace(tile_k_flat, off, self._rope_cos[t], self._rope_sin[t])

            tile_sa_scale = _EXP(tile_sa_tau)
            for i in range(TILE_GRID_N):
                off = i * TILE_EMBED_DIM
                _attention_update_inplace(
                    self._tile_q_views[i],
                    tile_q_scales[i],  # type: ignore[arg-type]
                    self._tile_k_views,
                    tile_k_scales,
                    self._tile_v_views,
                    tile_sa_scale,
                    attn_scores,
                    attn_update,
                )
                _residual_rmsnorm_silu_flat(
                    encoded_flat,
                    off,
                    attn_update,
                    0,
                    tile_norm_gamma,
                    tile_norm_eps,
                    enriched_flat,
                    off,
                    TILE_EMBED_DIM,
                )

            for idx in range(TILE_GRID_N):
                token = self._enriched_views[idx]
                k_tok = cls_k[idx]
                v_tok = cls_v[idx]
                for j in range(TILE_EMBED_DIM):
                    k_val = _DOT(token, cls_k_rows[j])
                    v_val = _DOT(token, cls_v_rows[j])
                    if cls_k_bias is not None:
                        k_val += cls_k_bias[j]
                    if cls_v_bias is not None:
                        v_val += cls_v_bias[j]
                    k_tok[j] = k_val
                    v_tok[j] = v_val
                cls_k_scales[idx] = _norm_inverse(k_tok)

        global_ctx = [0.0] * ACTOR_CONTEXT_DIM
        for j in range(ACTOR_CONTEXT_DIM):
            x = _DOT(globals_obs, context_rows[j])
            if context_bias is not None:
                x += context_bias[j]
            global_ctx[j] = x
        projected_ctx_k: list[list[float]] = []
        projected_ctx_k_scales: list[float] = []
        projected_ctx_v: list[list[float]] = []
        for chunk_idx in range(ACTOR_CONTEXT_CHUNKS * 2):
            if chunk_idx < ACTOR_CONTEXT_CHUNKS:
                token = global_ctx[
                    chunk_idx * TILE_EMBED_DIM:(chunk_idx + 1) * TILE_EMBED_DIM
                ]
            else:
                h_idx = chunk_idx - ACTOR_CONTEXT_CHUNKS
                token = h_prev[
                    h_idx * TILE_EMBED_DIM:(h_idx + 1) * TILE_EMBED_DIM
                ]
            k_tok = [0.0] * TILE_EMBED_DIM
            v_tok = [0.0] * TILE_EMBED_DIM
            for j in range(TILE_EMBED_DIM):
                k_val = _DOT(token, ctx_k_rows[j])
                v_val = _DOT(token, ctx_v_rows[j])
                if ctx_k_bias is not None:
                    k_val += ctx_k_bias[j]
                if ctx_v_bias is not None:
                    v_val += ctx_v_bias[j]
                k_tok[j] = k_val
                v_tok[j] = v_val
            projected_ctx_k_scales.append(_norm_inverse(k_tok))
            projected_ctx_k.append(k_tok)
            projected_ctx_v.append(v_tok)

        cls_state = [
            cls_tokens_flat[i * TILE_EMBED_DIM:(i + 1) * TILE_EMBED_DIM]
            for i in range(ACTOR_CLS_TOKENS)
        ]
        for i, token in enumerate(cls_state):
            q_scale = _norm_inverse(token)
            update = _attention_update_inplace(
                token,
                q_scale,
                projected_ctx_k,
                projected_ctx_k_scales,
                projected_ctx_v,
                _EXP(context_tau[i]),
                attn_scores,
                attn_update,
            )
            cls_state[i] = _residual_rmsnorm_silu(
                token,
                update,
                context_norm_gamma,
                context_norm_eps,
            )

        cls_ca_scale = _EXP(cls_ca_tau)
        for i, token in enumerate(cls_state):
            q_tok = [0.0] * TILE_EMBED_DIM
            for j in range(TILE_EMBED_DIM):
                q_val = _DOT(token, cls_q_rows[j])
                if cls_q_bias is not None:
                    q_val += cls_q_bias[j]
                q_tok[j] = q_val
            q_scale = _norm_inverse(q_tok)
            update = _attention_update_inplace(
                q_tok,
                q_scale,
                cls_k,
                cls_k_scales,
                cls_v,
                cls_ca_scale,
                attn_scores,
                attn_update,
            )
            cls_state[i] = _residual_rmsnorm_silu(
                token,
                update,
                cls_norm_gamma,
                cls_norm_eps,
            )

        latent_input = [0.0] * ACTOR_LATENT_DIM
        for i, token in enumerate(cls_state):
            base = i * TILE_EMBED_DIM
            for j in range(TILE_EMBED_DIM):
                latent_input[base + j] = token[j]
        latent = _rmsnorm(latent_input, latent_norm_gamma, latent_norm_eps)
        if not latent_ffn_noop:
            latent_ffn_norm_gamma, latent_ffn_norm_eps = latent_ffn_norm  # type: ignore[misc]
            latent_ffn_up_rows, latent_ffn_up_bias = cache["latent_ffn_up"]
            latent_ffn_down_rows, latent_ffn_down_bias = cache["latent_ffn_down"]
            latent_hidden = _silu_vec(_linear(_rmsnorm(latent, latent_ffn_norm_gamma, latent_ffn_norm_eps), latent_ffn_up_rows, latent_ffn_up_bias))
            latent_delta = _linear(latent_hidden, latent_ffn_down_rows, latent_ffn_down_bias)
            latent = [latent[i] + latent_delta[i] for i in range(ACTOR_LATENT_DIM)]

        action_names, action_slices, action_rows, action_bias = cache["action_heads"]
        outputs: dict[str, list[float]] = {}
        for name, (start, end) in zip(action_names, action_slices):
            head_out = [0.0] * (end - start)
            for out_idx, row_idx in enumerate(range(start, end)):
                head_out[out_idx] = _DOT(latent, action_rows[row_idx]) + action_bias[row_idx]
            outputs[name] = head_out

        gru_in = [0.0] * ACTOR_GRU_INPUT_DIM
        for i in range(ACTOR_LATENT_DIM):
            gru_in[i] = latent[i]
        for i in range(H_DIM):
            gru_in[ACTOR_LATENT_DIM + i] = h_prev[i]
        gate_rows, gate_bias = cache["head_h_gate"]
        cand_rows, cand_bias = cache["head_h_cand"]
        h_gate = [_DOT(gru_in, gate_rows[j]) + gate_bias[j] for j in range(H_DIM)]
        h_cand = [math.tanh(_DOT(gru_in, cand_rows[j]) + cand_bias[j]) for j in range(H_DIM)]
        outputs["h_gate"] = h_gate
        outputs["h_cand"] = h_cand
        outputs.pop("temperature", None)
        return outputs

    def save(self) -> list[list[float]]:
        return [list(p.data) for p in self.parameters()]

    def load(self, state: list[list[float]]) -> None:
        params = self.parameters()
        legacy_expected = len(params) - 11
        if len(state) not in (len(params), legacy_expected):
            raise AssertionError(
                f"State has {len(state)} entries, expected {len(params)} or legacy {legacy_expected}"
            )
        if len(state) == legacy_expected:
            default_state = ActorModel(self._output_heads).save()
            state = list(state) + default_state[len(state):]
        for p, s in zip(params, state):
            if len(p.data) != len(s):
                raise AssertionError(f"Param size mismatch: {len(p.data)} vs {len(s)}")
            p.data[:] = s
        self._infer_cache = None
