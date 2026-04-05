"""Dedicated stdlib actor used by tensor deployment and opponent bots."""

from __future__ import annotations

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


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [_EXP(x - m) for x in scores]
    z = sum(exps)
    return [x / z for x in exps]


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


def _reshape_tokens(flat: list[float], n_tokens: int, dim: int) -> list[list[float]]:
    return [flat[i * dim:(i + 1) * dim] for i in range(n_tokens)]


def _flatten_tokens(tokens: list[list[float]]) -> list[float]:
    out: list[float] = []
    for token in tokens:
        out.extend(token)
    return out


def _apply_rotary(vec: list[float], cos: list[float], sin: list[float]) -> list[float]:
    half = len(vec) // 2
    out = [0.0] * len(vec)
    for i in range(half):
        x1 = vec[i]
        x2 = vec[half + i]
        out[i] = x1 * cos[i] - x2 * sin[i]
        out[half + i] = x1 * sin[i] + x2 * cos[i]
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
    """Stdlib-only actor matching the PyTorch CLS-attention actor."""

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

        tile_rows_full, tile_bias = _linear_rows(self.tile_linear)
        tile_dx_w = [row[0] for row in tile_rows_full]
        tile_dy_w = [row[1] for row in tile_rows_full]
        tile_rows = [row[2:] for row in tile_rows_full]

        cache = {
            "tile_linear": (tile_rows, tile_bias, tile_dx_w, tile_dy_w),
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
            "latent_ffn_norm": (list(self.latent_ffn_norm.gamma.data), self.latent_ffn_norm.eps),
            "cls_tokens": _reshape_tokens(list(self.cls_tokens.data), ACTOR_CLS_TOKENS, TILE_EMBED_DIM),
            "context_tau": list(self.context_tau.data),
            "tile_sa_tau": list(self.tile_sa_tau.data),
            "cls_ca_tau": list(self.cls_ca_tau.data),
            "latent_ffn_up": _linear_rows(self.latent_ffn_up),
            "latent_ffn_down": _linear_rows(self.latent_ffn_down),
            "action_heads": (action_names, action_slices, action_rows, action_bias),
            "head_h_gate": (head_h_gate_rows, head_h_gate_bias),
            "head_h_cand": (head_h_cand_rows, head_h_cand_bias),
        }
        return cache

    def infer(self, features: list[float]) -> dict[str, list[float]]:
        if self._infer_cache is None:
            self._infer_cache = self._build_infer_cache()
        cache = self._infer_cache

        tile_rows, tile_bias, tile_dx_w, tile_dy_w = cache["tile_linear"]
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
        latent_ffn_norm_gamma, latent_ffn_norm_eps = cache["latent_ffn_norm"]
        cls_tokens = cache["cls_tokens"]
        context_tau = cache["context_tau"]
        tile_sa_tau = cache["tile_sa_tau"][0]
        cls_ca_tau = cache["cls_ca_tau"][0]
        latent_ffn_up_rows, latent_ffn_up_bias = cache["latent_ffn_up"]
        latent_ffn_down_rows, latent_ffn_down_bias = cache["latent_ffn_down"]

        tile_obs_end = TILE_GRID_N * TILE_FEATURES
        globals_end = tile_obs_end + GLOBAL_FEATURES
        globals_obs = features[tile_obs_end:globals_end]
        h_prev = features[globals_end:globals_end + H_DIM]

        encoded_tokens: list[list[float]] = []
        for t in range(TILE_GRID_N):
            base = t * TILE_FEATURES
            tile_raw = features[base:base + TILE_FEATURES]
            dx, dy = self._tile_dxdy[t]
            token = [0.0] * TILE_EMBED_DIM
            for j in range(TILE_EMBED_DIM):
                x = _DOT(tile_raw, tile_rows[j]) + dx * tile_dx_w[j] + dy * tile_dy_w[j]
                if tile_bias is not None:
                    x += tile_bias[j]
                token[j] = _silu_scalar(x)
            encoded_tokens.append(token)

        global_ctx = [0.0] * ACTOR_CONTEXT_DIM
        for j in range(ACTOR_CONTEXT_DIM):
            x = _DOT(globals_obs, context_rows[j])
            if context_bias is not None:
                x += context_bias[j]
            global_ctx[j] = x
        context_tokens = _reshape_tokens(global_ctx, ACTOR_CONTEXT_CHUNKS, TILE_EMBED_DIM)
        context_tokens.extend(_reshape_tokens(h_prev, ACTOR_CONTEXT_CHUNKS, TILE_EMBED_DIM))

        projected_ctx_k: list[list[float]] = []
        projected_ctx_v: list[list[float]] = []
        for token in context_tokens:
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
            projected_ctx_k.append(_l2_normalize(k_tok))
            projected_ctx_v.append(v_tok)

        cls_state = [list(token) for token in cls_tokens]
        for i, token in enumerate(cls_state):
            q = _l2_normalize(token)
            scale = _EXP(context_tau[i])
            scores = [scale * _DOT(q, projected_ctx_k[j]) for j in range(len(projected_ctx_k))]
            weights = _softmax(scores)
            update = [0.0] * TILE_EMBED_DIM
            for w, value in zip(weights, projected_ctx_v):
                for j in range(TILE_EMBED_DIM):
                    update[j] += w * value[j]
            cls_state[i] = [_silu_scalar(x) for x in _rmsnorm(
                [token[j] + update[j] for j in range(TILE_EMBED_DIM)],
                context_norm_gamma,
                context_norm_eps,
            )]

        tile_q: list[list[float]] = []
        tile_k: list[list[float]] = []
        tile_v: list[list[float]] = []
        for idx, token in enumerate(encoded_tokens):
            q_tok = [0.0] * TILE_EMBED_DIM
            k_tok = [0.0] * TILE_EMBED_DIM
            v_tok = [0.0] * TILE_EMBED_DIM
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
                q_tok[j] = q_val
                k_tok[j] = k_val
                v_tok[j] = v_val
            tile_q.append(_l2_normalize(_apply_rotary(q_tok, self._rope_cos[idx], self._rope_sin[idx])))
            tile_k.append(_l2_normalize(_apply_rotary(k_tok, self._rope_cos[idx], self._rope_sin[idx])))
            tile_v.append(v_tok)

        enriched_tiles: list[list[float]] = []
        for i, token in enumerate(encoded_tokens):
            scores = [_EXP(tile_sa_tau) * _DOT(tile_q[i], tile_k[j]) for j in range(TILE_GRID_N)]
            weights = _softmax(scores)
            update = [0.0] * TILE_EMBED_DIM
            for w, value in zip(weights, tile_v):
                for j in range(TILE_EMBED_DIM):
                    update[j] += w * value[j]
            enriched_tiles.append([_silu_scalar(x) for x in _rmsnorm(
                [token[j] + update[j] for j in range(TILE_EMBED_DIM)],
                tile_norm_gamma,
                tile_norm_eps,
            )])

        cls_k: list[list[float]] = []
        cls_k_norm: list[list[float]] = []
        cls_v: list[list[float]] = []
        for token in enriched_tiles:
            k_tok = [0.0] * TILE_EMBED_DIM
            v_tok = [0.0] * TILE_EMBED_DIM
            for j in range(TILE_EMBED_DIM):
                k_val = _DOT(token, cls_k_rows[j])
                v_val = _DOT(token, cls_v_rows[j])
                if cls_k_bias is not None:
                    k_val += cls_k_bias[j]
                if cls_v_bias is not None:
                    v_val += cls_v_bias[j]
                k_tok[j] = k_val
                v_tok[j] = v_val
            cls_k.append(k_tok)
            cls_k_norm.append(_l2_normalize(k_tok))
            cls_v.append(v_tok)

        for i, token in enumerate(cls_state):
            q_tok = [0.0] * TILE_EMBED_DIM
            for j in range(TILE_EMBED_DIM):
                q_val = _DOT(token, cls_q_rows[j])
                if cls_q_bias is not None:
                    q_val += cls_q_bias[j]
                q_tok[j] = q_val
            q_tok = _l2_normalize(q_tok)
            scores = [_EXP(cls_ca_tau) * _DOT(q_tok, cls_k_norm[j]) for j in range(TILE_GRID_N)]
            weights = _softmax(scores)
            update = [0.0] * TILE_EMBED_DIM
            for w, value in zip(weights, cls_v):
                for j in range(TILE_EMBED_DIM):
                    update[j] += w * value[j]
            cls_state[i] = [_silu_scalar(x) for x in _rmsnorm(
                [token[j] + update[j] for j in range(TILE_EMBED_DIM)],
                cls_norm_gamma,
                cls_norm_eps,
            )]

        latent = _rmsnorm(_flatten_tokens(cls_state), latent_norm_gamma, latent_norm_eps)
        latent_hidden = _silu_vec(_linear(_rmsnorm(latent, latent_ffn_norm_gamma, latent_ffn_norm_eps), latent_ffn_up_rows, latent_ffn_up_bias))
        latent_delta = _linear(latent_hidden, latent_ffn_down_rows, latent_ffn_down_bias)
        latent = [latent[i] + latent_delta[i] for i in range(ACTOR_LATENT_DIM)]

        action_names, action_slices, action_rows, action_bias = cache["action_heads"]
        fused_action = [_DOT(latent, action_rows[j]) + action_bias[j] for j in range(len(action_rows))]
        outputs: dict[str, list[float]] = {}
        for name, (start, end) in zip(action_names, action_slices):
            outputs[name] = fused_action[start:end]

        gru_in = latent + h_prev
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
