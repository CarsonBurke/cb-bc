"""Shared model, constants, and PPO training code."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
import os
from enum import Enum
from operator import attrgetter
from pathlib import Path

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

try:
    from pretrain.gpu import (
        PretrainCudaConfig,
        autocast_context,
        configure_cuda_runtime,
        maybe_compile,
        resolve_cuda_config,
    )
    from replay_buffers import (
        SCALAR_BUFFER,
        STATE_BUFFER,
        coerce_writable_replay_buffer,
        load_torch_replay_buffer,
    )
    from trajectory_schema import TrajectoryStep
    from training_profile import trace_scope
except ImportError:
    from bots.tensor.torch.pretrain.gpu import (
        PretrainCudaConfig,
        autocast_context,
        configure_cuda_runtime,
        maybe_compile,
        resolve_cuda_config,
    )
    from bots.tensor.torch.replay_buffers import (
        SCALAR_BUFFER,
        STATE_BUFFER,
        coerce_writable_replay_buffer,
        load_torch_replay_buffer,
    )
    from bots.tensor.trajectory_schema import TrajectoryStep
    from bots.tensor.torch.training_profile import trace_scope

try:
    from normuon import make_normuon_optimizer
except ImportError:
    from bots.tensor.torch.normuon import make_normuon_optimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
_TRAINING_DTYPE = torch.bfloat16 if _USE_BF16 else torch.float32
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---- Architecture constants (imported from tensor_inference/model_constants.py) ----

import sys as _sys
_CONSTANTS_DIR = str(PROJECT_ROOT / "bots" / "tensor_inference")
if _CONSTANTS_DIR not in _sys.path:
    _sys.path.insert(0, _CONSTANTS_DIR)
from model_constants import (
    GRID_SIZE, GRID_HALF, TILE_FEATURES, GLOBAL_FEATURES,
    H_DIM, HIDDEN_SIZES, COMPRESS_DIM, RMSNORM_LAYERS,
    TILE_EMBED_DIM, TILE_INPUT_DIM, TILE_GRID_N,
    INPUT_SIZE, OBS_SIZE, TRUNK_INPUT_SIZE, OBS_PREFIX_SIZE,
    ACTOR_CLS_TOKENS, ACTOR_CONTEXT_CHUNKS, ACTOR_CONTEXT_DIM,
    ACTOR_LATENT_DIM as MODEL_ACTOR_LATENT_DIM, ACTOR_GRU_INPUT_DIM,
)
HIDDEN = HIDDEN_SIZES
TEACHER_HIDDEN = [128, 128, 128, 64]
ACTOR_LATENT_DIM = MODEL_ACTOR_LATENT_DIM
N_BUILD = 13
N_DIR = 9
N_CONV_DIR = 4  # cardinal only: N, W, E, S (indices 1, 3, 5, 7 in DIR_MAP_9)
N_BRIDGE_TARGETS = 28  # offsets (dx,dy) with 0 < dx²+dy² ≤ 9
N_ACTION_TEMPERATURES = 5  # per-head temperatures: bt, bd, cd, brt, md
ACTOR_RMSNORM_LAYERS = tuple(RMSNORM_LAYERS)
assert ACTOR_RMSNORM_LAYERS[0], "First layer RMSNorm must be True (encoder always uses RMSNorm; inference MLP respects this flag)"
HARVESTER_BUILD_TYPE = 1
BRIDGE_BUILD_TYPE = 12
CONVEYOR_BUILD_TYPES = (2, 7, 8)  # conveyor, splitter, armoured_conveyor
TEMP_MIN = 1.0
TEMP_MAX = 3.0
INITIAL_HEAD_TEMPERATURE = 1.0
_FFN_LEAKY_RELU_SLOPE = 0.01


def _temperature_head_init_bias(
    initial_temperature: float = INITIAL_HEAD_TEMPERATURE,
    temp_min: float = TEMP_MIN,
    temp_max: float = TEMP_MAX,
) -> float:
    if not temp_min <= initial_temperature <= temp_max:
        raise ValueError(
            f"initial_temperature must be in [{temp_min}, {temp_max}], got {initial_temperature}"
        )
    if temp_min <= 0.0 or temp_max <= 0.0:
        raise ValueError("temperature range must be positive")
    if temp_min == temp_max:
        return 0.0
    s = (math.log(initial_temperature) - math.log(temp_min)) / (math.log(temp_max) - math.log(temp_min))
    s = min(max(s, 1e-6), 1.0 - 1e-6)
    return math.log(s / (1.0 - s))

INITIAL_BUILD_TYPE_BIAS = [
    -2.0,  # none
    1.1,   # harvester
    1.0,   # conveyor
    0.0,   # road
    -2.0,  # gunner
    -2.0,  # sentinel
    -2.0,  # breach
    0.7,   # splitter
    -2.0,  # armoured_conveyor
    -2.0,  # barrier
    -2.0,  # foundry
    -2.0,  # launcher
    -0.5,  # bridge
]
class RewardMode(Enum):
    OUTCOME = "outcome"   # terminal win/loss only
    INCOME = "income"     # resource delivered to core per round
    FLOW = "flow"         # weighted conveyor hops toward core per round

REWARD_MODE = RewardMode.FLOW
REWARD_INCOME_SCALE = 2.5 # since each "resource" is worth 10
REWARD_FLOW_SCALE = 5.0
TERMINAL_WIN_REWARD = 10.0
TERMINAL_LOSS_REWARD = -10.0
GAMMA = 0.995 # GAE
LAMBDA = 0.95 # GAE

# ---- Value prediction mode ----
class CriticValueMode(Enum):
    GHL = "ghl"   # Gaussian Histogram Loss (distributional)
    MSE = "mse"   # Mean squared error (scalar, symlog space)

CRITIC_VALUE_MODE = CriticValueMode.GHL

# ---- Gaussian Histogram Loss (GHL) value prediction with symlog/symexp ----
NUM_BUCKETS = 255
# symlog range — covers symexp(±9) ≈ ±8103
BUCKET_LOW = -9.0
BUCKET_HIGH = 9.0
# GHL sigma in symlog space, expressed as a multiple of bin width.
_GHL_SIGMA_RATIO = 0.5
_SQRT2 = math.sqrt(2.0)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.expm1(x.abs())


def _make_bucket_support() -> tuple[torch.Tensor, torch.Tensor, float]:
    """Uniform support in symlog space for GHL.

    Returns:
        centers: (NUM_BUCKETS,) bin centers in symlog space.
        support: (NUM_BUCKETS+1,) bin edges in symlog space.
        sigma: Gaussian kernel width.
    """
    support = torch.linspace(BUCKET_LOW, BUCKET_HIGH, NUM_BUCKETS + 1, dtype=torch.float32)
    centers = (support[:-1] + support[1:]) / 2
    bin_width = (BUCKET_HIGH - BUCKET_LOW) / NUM_BUCKETS
    sigma = _GHL_SIGMA_RATIO * bin_width
    return centers, support, sigma


def _make_bucket_centers() -> torch.Tensor:
    """Return bin centers in symlog space (for backward compat with buffer registration)."""
    centers, _, _ = _make_bucket_support()
    return centers


def _ghl_target_probs(
    targets: torch.Tensor,
    support: torch.Tensor,
    sigma: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Gaussian histogram soft targets (Imani et al. 2018).

    Targets are in original value space; symlog is applied internally.
    Support edges are in symlog space.

    Args:
        targets: (...,) scalar targets in original value space.
        support: (NUM_BUCKETS+1,) bin edges in symlog space.
        sigma: Gaussian kernel width in symlog space.

    Returns:
        (..., NUM_BUCKETS) soft probability vectors.
    """
    t = symlog(targets)
    # CDF at each bin edge: erf((edge - target) / (sigma * sqrt(2)))
    cdf = torch.special.erf((support - t.unsqueeze(-1)) / (sigma * _SQRT2))
    # Bin probs = CDF differences, normalized
    bin_probs = cdf[..., 1:] - cdf[..., :-1]
    z = (cdf[..., -1] - cdf[..., 0]).clamp(min=eps)
    return bin_probs / z.unsqueeze(-1)


def win_probability(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Win confidence from critic logits via linear rescale of decoded value.

    Args:
        logits: (..., NUM_BUCKETS) raw critic logits.
        bins: (NUM_BUCKETS,) bin centers in symlog space.

    Returns:
        (...,) win probability in [0, 1].
    """
    values = decode_value(logits, bins)
    return ((values - TERMINAL_LOSS_REWARD) / (TERMINAL_WIN_REWARD - TERMINAL_LOSS_REWARD)).clamp(0, 1)


def decode_value(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode categorical logits to scalar values.

    Bins are in symlog space; weighted average gives symlog-space prediction,
    then symexp maps back to original value space.

    Args:
        logits: (..., NUM_BUCKETS) raw logits.
        bins: (NUM_BUCKETS,) bin centers in symlog space.

    Returns:
        (...,) predicted values in original value space.
    """
    probs = torch.softmax(logits, dim=-1)
    symlog_value = (probs * bins).sum(dim=-1)
    return symexp(symlog_value)


def value_target_probs(
    targets: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Encode scalar values as GHL soft targets.

    Args:
        targets: (...,) in original value space.
        bins: (NUM_BUCKETS,) bin centers in symlog space (unused, kept for API compat;
              support is rebuilt from global constants).
    """
    # Keep target encoding in fp32 even under autocast; bf16 noticeably distorts
    # the Gaussian bin masses around the peak and weakens the critic signal.
    targets_fp32 = targets.to(dtype=torch.float32)
    _, support, sigma = _make_bucket_support()
    support = support.to(device=targets.device, dtype=torch.float32)
    return _ghl_target_probs(targets_fp32, support, sigma)


def soft_cross_entropy_from_logits(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for soft targets against raw logits."""
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def soft_cross_entropy_from_probs(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross-entropy for soft targets against normalized probabilities."""
    return -(target_probs * pred_probs.clamp_min(eps).log()).sum(dim=-1)



def distributional_value_loss(
    logits: torch.Tensor,
    returns: torch.Tensor,
    bins: torch.Tensor,
    *,
    values: torch.Tensor | None = None,
    old_values: torch.Tensor | None = None,
    clip_vloss: bool = False,
    value_clip: float = 0.4,
) -> torch.Tensor:
    """Distributional value loss with optional PPO-style value clipping."""
    return_target = value_target_probs(returns, bins).detach()
    value_loss_1 = soft_cross_entropy_from_logits(logits, return_target)

    if not clip_vloss or old_values is None or values is None:
        return value_loss_1

    clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
    clipped_value_probs = value_target_probs(clipped_values, bins)
    value_loss_2 = soft_cross_entropy_from_probs(clipped_value_probs, return_target)
    return torch.maximum(value_loss_1, value_loss_2)


def _require_finite(name: str, x: torch.Tensor) -> None:
    """Fail fast on non-finite tensors so training errors are not hidden."""
    if torch.isfinite(x).all():
        return
    bad = (~torch.isfinite(x)).nonzero(as_tuple=False)
    first_idx = tuple(int(i) for i in bad[0].tolist())
    first_val = x[first_idx].detach().float().cpu().item()
    raise RuntimeError(
        f"non-finite {name}: count={int(bad.shape[0])} first_index={first_idx} first_value={first_val!r}"
    )


def _require_finite_tree(name: str, value: object) -> None:
    """Recursively validate nested checkpoint payloads before loading them."""
    if torch.is_tensor(value):
        if torch.is_floating_point(value) or torch.is_complex(value):
            _require_finite(name, value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tree(f"{name}.{key}", item)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _require_finite_tree(f"{name}[{idx}]", item)


def _require_module_params_finite(name: str, module: nn.Module) -> None:
    """Validate all floating-point parameters in a module."""
    for param_name, param in module.named_parameters():
        if torch.is_floating_point(param) or torch.is_complex(param):
            _require_finite(f"{name}.{param_name}", param)


def _require_module_grads_finite(name: str, module: nn.Module) -> None:
    """Validate all floating-point parameter gradients in a module."""
    for param_name, param in module.named_parameters():
        grad = param.grad
        if grad is not None and (torch.is_floating_point(grad) or torch.is_complex(grad)):
            _require_finite(f"{name}.{param_name}.grad", grad)

WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor_inference" / "weights.json"
LOG_PATH = PROJECT_ROOT / "bots" / "tensor" / "train_log.jsonl"
REPLAY_DIR = PROJECT_ROOT / "replays" / "training"

# Precomputed (dx/2, dy/2) for each tile in the 5x5 local grid.
# Tile index i -> grid position (i%5 - 2, i//5 - 2), normalized by /2.
_TILE_DXDY = torch.tensor(
    [[float((i % 5) - 2) / 2.0, float((i // 5) - 2) / 2.0] for i in range(TILE_GRID_N)],
    dtype=torch.float32,
)


def _make_tile_susie(n_tokens: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Unrotated sinusoid table used as additive SuSIE position for the tile grid.

    Layout matches the reference: [sin(t*f₀)…sin(t*f_{d/2-1}), cos(t*f₀)…cos(t*f_{d/2-1})].
    A learnable pos_weight of shape (dim/2,) is tied across sin/cos by repeat(2) at
    forward time, so each frequency's sin and cos share a single scale.
    """
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(n_tokens, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (n_tokens, half)
    return torch.cat((freqs.sin(), freqs.cos()), dim=-1).contiguous()  # (n_tokens, dim)

# ---- Action masking ----
# Build costs (ti, ax) indexed by BUILD_ACTIONS:
# none, harvester, conveyor, road, gunner, sentinel, breach, splitter,
# armoured_conveyor, barrier, foundry, launcher
_MASK_COSTS_TI = [0, 80, 3, 1, 10, 15, 30, 6, 10, 3, 120, 20, 10]
_MASK_COSTS_AX = [0, 0, 0, 0, 0, 0, 10, 0, 5, 0, 0, 0, 0]

# Tile indices in 5x5 grid for the 9 build directions (NW,N,NE,W,C,E,SW,S,SE)
_MASK_BD_TILES = [6, 7, 8, 11, 12, 13, 16, 17, 18]
_MASK_BD_DX = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
_MASK_BD_DY = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
_CONV_DIR_MASK = [True, True, True, True]  # all 4 cardinal dirs always valid

# Observation feature indices for resources/scale
_OBS_GLOBAL_START = 5 * 5 * TILE_FEATURES  # 525
_OBS_TI = _OBS_GLOBAL_START + 3
_OBS_AX = _OBS_GLOBAL_START + 4
_OBS_SCALE = _OBS_GLOBAL_START + 5

# Per-device cached index tensors (created once, reused)
_mask_idx_cache: dict = {}


def _mask_indices(device: torch.device):
    """Get or create cached mask index tensors for the given device."""
    if device not in _mask_idx_cache:
        _mask_idx_cache[device] = (
            torch.tensor(_MASK_COSTS_TI, dtype=torch.float32, device=device),
            torch.tensor(_MASK_COSTS_AX, dtype=torch.float32, device=device),
            torch.tensor([t * TILE_FEATURES for t in _MASK_BD_TILES], dtype=torch.long, device=device),      # is_wall
            torch.tensor([t * TILE_FEATURES + 4 for t in _MASK_BD_TILES], dtype=torch.long, device=device),  # bld_none (no building)
            torch.tensor([t * TILE_FEATURES + 19 for t in _MASK_BD_TILES], dtype=torch.long, device=device), # has_unit
            torch.tensor([t * TILE_FEATURES + 14 for t in _MASK_BD_TILES], dtype=torch.long, device=device), # bld_friendly
            torch.tensor([t * TILE_FEATURES + 13 for t in _MASK_BD_TILES], dtype=torch.long, device=device), # bld_core
        )
    return _mask_idx_cache[device]


def compute_action_masks(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute build_type and build_dir boolean masks.

    Build directions allow both empty tiles and tiles with friendly non-core
    buildings (the execution layer auto-destroys before building).

    Args:
        obs: (B, OBS_SIZE) observation tensor
    Returns:
        bt_mask: (B, N_BUILD) boolean — True = action allowed
        bd_mask: (B, N_DIR) boolean — True = direction allowed
    """
    costs_ti, costs_ax, wall_idx, nobld_idx, unit_idx, friendly_idx, core_idx = _mask_indices(obs.device)

    ti = obs[:, _OBS_TI] * 500.0
    ax = obs[:, _OBS_AX] * 500.0
    scale = obs[:, _OBS_SCALE] * 5.0  # scale_pct / 100

    # Build type: floor(scale * base_cost) <= available resources
    actual_ti = (scale.unsqueeze(1) * costs_ti.unsqueeze(0)).floor()
    actual_ax = (scale.unsqueeze(1) * costs_ax.unsqueeze(0)).floor()
    bt_mask = (ti.unsqueeze(1) >= actual_ti) & (ax.unsqueeze(1) >= actual_ax)
    bt_mask[:, 0] = True  # "none" always allowed

    # Build dir: empty tile OR friendly non-core building (auto-destroyed before build).
    # Must not be a wall and must not have a unit.
    not_wall = obs[:, wall_idx] == 0.0
    no_unit = obs[:, unit_idx] == 0.0
    empty = obs[:, nobld_idx] == 1.0
    replaceable = (obs[:, nobld_idx] == 0.0) & (obs[:, friendly_idx] == 1.0) & (obs[:, core_idx] == 0.0)
    bd_mask = not_wall & no_unit & (empty | replaceable)

    # Force none-only when nothing affordable or no valid direction.
    # Apply unconditionally (no .any() branch) to avoid CPU-GPU sync.
    force_none = ~bt_mask[:, 1:].any(dim=1) | ~bd_mask.any(dim=1)
    bt_mask[:, 1:] &= ~force_none.unsqueeze(1)
    bd_mask |= force_none.unsqueeze(1)  # dir is irrelevant, avoid NaN

    return bt_mask, bd_mask


_harv_idx_cache: dict = {}


def _harv_mask_indices(device: torch.device):
    """Get or create cached harvester mask index tensors."""
    if device not in _harv_idx_cache:
        tile_base = torch.tensor([t * TILE_FEATURES for t in _MASK_BD_TILES], dtype=torch.long, device=device)
        _harv_idx_cache[device] = (tile_base + 1, tile_base + 2)
    return _harv_idx_cache[device]


def bd_has_ore(obs: torch.Tensor, bd_act: torch.Tensor) -> torch.Tensor:
    """Check if the chosen build-direction tile has ore.  Returns (B,) bool."""
    ore_ti_idx, ore_ax_idx = _harv_mask_indices(obs.device)
    # ore_ti_idx/ore_ax_idx are (9,) — gather the tile for each sample's bd_act
    chosen_ti = ore_ti_idx[bd_act.long()]   # (B,)
    chosen_ax = ore_ax_idx[bd_act.long()]   # (B,)
    return (obs.gather(1, chosen_ti.unsqueeze(1)).squeeze(1) == 1.0) | \
           (obs.gather(1, chosen_ax.unsqueeze(1)).squeeze(1) == 1.0)


def compute_harvester_build_dir_mask(
    obs: torch.Tensor,
    bd_mask: torch.Tensor,
) -> torch.Tensor:
    """Restrict build directions to ore-bearing tiles for harvester actions."""
    ore_ti_idx, ore_ax_idx = _harv_mask_indices(obs.device)
    ore_present = (obs[:, ore_ti_idx] == 1.0) | (obs[:, ore_ax_idx] == 1.0)
    return bd_mask & ore_present


def compute_build_dir_mask_for_type(
    obs: torch.Tensor,
    bt_act: torch.Tensor,
    bd_mask: torch.Tensor,
    harv_bd_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the build-dir mask conditioned on the chosen build type."""
    del obs
    return torch.where(
        (bt_act == HARVESTER_BUILD_TYPE).unsqueeze(1),
        harv_bd_mask,
        bd_mask,
    )


_conv_dir_mask_cache: dict = {}


def _conveyor_dir_mask_base(device: torch.device) -> torch.Tensor:
    """Get or create the base conveyor direction mask for the given device."""
    if device not in _conv_dir_mask_cache:
        _conv_dir_mask_cache[device] = torch.tensor(_CONV_DIR_MASK, dtype=torch.bool, device=device)
    return _conv_dir_mask_cache[device]


def conveyor_dir_mask(device: torch.device, batch_size: int) -> torch.Tensor:
    return _conveyor_dir_mask_base(device).unsqueeze(0).expand(batch_size, -1)


_cond_bt_cache: dict = {}


def _cond_bt_indices(device: torch.device):
    """Get or create cached conditional build type index tensors."""
    if device not in _cond_bt_cache:
        _cond_bt_cache[device] = (
            torch.tensor(_MASK_BD_TILES, dtype=torch.long, device=device),
            torch.tensor(_MASK_BD_DX, dtype=torch.float32, device=device),
            torch.tensor(_MASK_BD_DY, dtype=torch.float32, device=device),
        )
    return _cond_bt_cache[device]


def compute_conditional_build_type_mask(
    obs: torch.Tensor,
    bd_act: torch.Tensor,
    bt_mask: torch.Tensor,
) -> torch.Tensor:
    """Refine build_type mask using the chosen build_dir.

    Observations collapse CONVEYOR and ARMOURED_CONVEYOR into one channel, so
    the conditional mask can only conservatively suppress regular conveyor
    rebuilds on same-dir friendly conveyor-family tiles. Exact armoured-vs-
    armoured no-op filtering remains enforced in the execution layer.
    """
    bt_mask = bt_mask.clone()
    bd_act = bd_act.to(torch.long)

    tile_lut, dx_lut, dy_lut = _cond_bt_indices(obs.device)

    tile_base = tile_lut[bd_act] * TILE_FEATURES
    conveyor_idx = (tile_base + 6).unsqueeze(1)
    friendly_idx = (tile_base + 12).unsqueeze(1)
    dir_x_idx = (tile_base + 13).unsqueeze(1)
    dir_y_idx = (tile_base + 14).unsqueeze(1)

    has_conveyor_family = obs.gather(1, conveyor_idx).squeeze(1) == 1.0
    is_friendly = obs.gather(1, friendly_idx).squeeze(1) == 1.0
    same_dir = (
        (obs.gather(1, dir_x_idx).squeeze(1) == dx_lut[bd_act])
        & (obs.gather(1, dir_y_idx).squeeze(1) == dy_lut[bd_act])
    )
    same_dir_conv = has_conveyor_family & is_friendly & same_dir
    bt_mask[same_dir_conv, 2] = False

    # Apply unconditionally to avoid CPU-GPU sync
    force_none = ~bt_mask[:, 1:].any(dim=1)
    bt_mask[:, 1:] &= ~force_none.unsqueeze(1)
    bt_mask[:, 0] |= force_none

    return bt_mask


# ---- Model ----

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean centering).

    Uses F.rms_norm (fused CUDA kernel) when available.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            # Keep bf16/fp16 activations, but accumulate the norm in fp32.
            # Recent critic changes add more RMSNorm applications in deep
            # attention stacks, and bf16-native norm math is prone to rare
            # instability on large PPO critic batches.
            if x.dtype in (torch.float16, torch.bfloat16):
                out = F.rms_norm(x.float(), (self.dim,), self.weight.float(), self.eps)
                return out.to(dtype=x.dtype)
            weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
            return F.rms_norm(x, (self.dim,), weight, self.eps)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


# ---- Centralized Critic constants ----
GLOBAL_MAP_MAX_W = 50
GLOBAL_MAP_MAX_H = 50
GLOBAL_MAP_CHANNELS = 23
GLOBAL_SCALAR_DIM = 19  # 12 base + own_working_harv, Δti_collected, Δax_collected, Δcore_hp, Δworking_harv, flow_reward, delivery_reward
UNIT_CONTEXT_DIM = 7  # x_norm, y_norm, core_dx, core_dy, ecore_dx, ecore_dy, unit_age
CRITIC_BATCHES_PER_EPOCH = 8
_NCU_CRITIC_BATCH_INDEX = int(os.environ.get("CB_NCU_CRITIC_BATCH_INDEX", "-1"))
_NCU_CRITIC_EPOCH_INDEX = int(os.environ.get("CB_NCU_CRITIC_EPOCH_INDEX", "0"))
CRITIC_SUMMARY_DIM = 64
CRITIC_STAGE_A_DIM = 64
CRITIC_STAGE_B_DIM = 96
CRITIC_STAGE_B_HEADS = 6




# ---- Transformer Critic Modules ----

class RotaryEmbedding2D(nn.Module):
    """2D axial rotary position embeddings for spatial grids."""

    def __init__(self, head_dim: int, max_size: int = 25, rotary_dim: int | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if self.rotary_dim <= 0 or self.rotary_dim > head_dim or self.rotary_dim % 4 != 0:
            raise ValueError(
                f"rotary_dim must be in (0, {head_dim}] and divisible by 4, got {self.rotary_dim}"
            )
        half = self.rotary_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(0, half, 2, dtype=torch.float32)[: half // 2] / half))
        self.register_buffer("freqs", freqs)  # (half//2,)
        # Precompute cos/sin for grid up to max_size × max_size
        pos = torch.arange(max_size, dtype=torch.float32)
        angles = pos.unsqueeze(-1) * freqs.unsqueeze(0)  # (max_size, half//2)
        self.register_buffer("_cos_cache", angles.cos().to(_TRAINING_DTYPE))
        self.register_buffer("_sin_cache", angles.sin().to(_TRAINING_DTYPE))

    def forward(
        self, h_coords: torch.Tensor, w_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape (N, head_dim//2)."""
        h_cos = self._cos_cache[h_coords]  # (N, half//2)
        h_sin = self._sin_cache[h_coords]
        w_cos = self._cos_cache[w_coords]
        w_sin = self._sin_cache[w_coords]
        return (
            torch.cat([h_cos, w_cos], dim=-1),
            torch.cat([h_sin, w_sin], dim=-1),
        )


def apply_rotary_2d(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Apply 2D rotary embeddings. x: (B, n_heads, N, head_dim), cos/sin: (N, head_dim//2)."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)  # (B|1, 1, N, hd//2)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1] * 2
    x_rope = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1, x2 = x_rope.chunk(2, dim=-1)
    x_rope = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if x_pass.numel() == 0:
        return x_rope
    return torch.cat([x_rope, x_pass], dim=-1)


def ffn_activation(x: torch.Tensor) -> torch.Tensor:
    """Cheap FFN activation used by actor and critic transformer blocks."""
    return F.leaky_relu(x, negative_slope=_FFN_LEAKY_RELU_SLOPE).square()


def residual_mix_input(
    x: torch.Tensor,
    x0: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    """Blend the running residual stream with a fixed stack anchor."""
    mix = mix.to(dtype=x.dtype)
    view_shape = (1,) * (x.dim() - 1) + (x.shape[-1],)
    return mix[0].view(view_shape) * x + mix[1].view(view_shape) * x0


def exclusive_self_attention(
    attn_out: torch.Tensor,
    self_value: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Remove the component of the attention update aligned with the self value."""
    self_value_hat = F.normalize(self_value, dim=-1, eps=eps)
    return attn_out - (attn_out * self_value_hat).sum(dim=-1, keepdim=True) * self_value_hat


class CriticAttentionBlock(nn.Module):
    """Peri-LN transformer block with GQA, QK-norm, and optional RoPE."""

    def __init__(
        self,
        dim: int = 48,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        mlp_ratio: int = 4,
        n_residual_layers: int = 1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_group_size = n_heads // n_kv_heads
        self.n_residual_layers = n_residual_layers
        kv_dim = n_kv_heads * self.head_dim

        self.attn_input_norm = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_output_norm = RMSNorm(dim)

        self.mlp_input_norm = RMSNorm(dim)
        mlp_hidden = dim * mlp_ratio
        self.mlp_fc1 = nn.Linear(dim, mlp_hidden, bias=False)
        self.mlp_fc2 = nn.Linear(mlp_hidden, dim, bias=False)
        self.mlp_output_norm = RMSNorm(dim)
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

        self._init_weights()

    def _init_weights(self):
        _parameter_golf_ortho_init(self.q_proj)
        _parameter_golf_ortho_init(self.k_proj)
        _parameter_golf_ortho_init(self.v_proj)
        _parameter_golf_ortho_init(
            self.out_proj,
            n_residual_layers=self.n_residual_layers,
            scale_proj=True,
        )
        _parameter_golf_ortho_init(self.mlp_fc1)
        _parameter_golf_ortho_init(
            self.mlp_fc2,
            n_residual_layers=self.n_residual_layers,
            scale_proj=True,
        )

    def project_qkv(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project normalized tokens into Q/K/V, applying RoPE when requested."""
        B, N, _ = x.shape
        h = self.attn_input_norm(x)
        q = self.q_proj(h).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).reshape(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_cos is not None:
            q = apply_rotary_2d(q, rope_cos, rope_sin)
            k = apply_rotary_2d(k, rope_cos, rope_sin)
        return q, k, v

    def project_qkv_grouped(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project grouped local blocks into Q/K/V.

        Args:
            x: (B, G, N, dim)
        Returns:
            Q/K/V in (B, G, heads, N, head_dim) form.
        """
        B, G, N, _ = x.shape
        h = self.attn_input_norm(x)
        q = self.q_proj(h).reshape(B, G, N, self.n_heads, self.head_dim).movedim(3, 2)
        k = self.k_proj(h).reshape(B, G, N, self.n_kv_heads, self.head_dim).movedim(3, 2)
        v = self.v_proj(h).reshape(B, G, N, self.n_kv_heads, self.head_dim).movedim(3, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_cos is not None:
            q = apply_rotary_2d(q, rope_cos, rope_sin)
            k = apply_rotary_2d(k, rope_cos, rope_sin)
        return q, k, v

    def attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        self_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run SDPA on projected heads and return token-major outputs."""
        B, _, N_q, _ = q.shape
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            enable_gqa=self.kv_group_size > 1,
        )
        if self_value is not None:
            if self.kv_group_size > 1:
                self_value = self_value.repeat_interleave(self.kv_group_size, dim=1)
            attn_out = exclusive_self_attention(attn_out, self_value)
        return attn_out.transpose(1, 2).reshape(B, N_q, self.n_heads * self.head_dim)

    def attend_grouped(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        self_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run SDPA over grouped local blocks without flattening them into batch.

        The group axis is folded into the head axis so the underlying SDPA call
        stays 4D and remains flash-attention eligible.
        """
        B, n_groups, _, N_q, _ = q.shape
        N_k = k.shape[-2]
        q_folded = q.reshape(B, n_groups * self.n_heads, N_q, self.head_dim)
        k_folded = k.reshape(B, n_groups * self.n_kv_heads, N_k, self.head_dim)
        v_folded = v.reshape(B, n_groups * self.n_kv_heads, N_k, self.head_dim)
        attn_out = F.scaled_dot_product_attention(
            q_folded,
            k_folded,
            v_folded,
            enable_gqa=self.kv_group_size > 1,
        )
        attn_out = attn_out.reshape(B, n_groups, self.n_heads, N_q, self.head_dim)
        if self_value is not None:
            if self.kv_group_size > 1:
                self_value = self_value.repeat_interleave(self.kv_group_size, dim=2)
            attn_out = exclusive_self_attention(attn_out, self_value)
        return attn_out.transpose(2, 3).reshape(B, n_groups, N_q, self.n_heads * self.head_dim)

    def project_attn_residual(self, attn_out: torch.Tensor) -> torch.Tensor:
        """Project attention outputs for the residual update."""
        return self.out_proj(attn_out)

    def mix_residual_input(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return residual_mix_input(x, x0, self.resid_mix)

    def apply_attn_residual(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Peri-LN attention residual update."""
        return x + self.attn_output_norm(self.project_attn_residual(attn_out))

    def apply_mlp_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Peri-LN MLP residual block."""
        h = self.mlp_input_norm(x)
        return x + self.mlp_output_norm(self.mlp_fc2(ffn_activation(self.mlp_fc1(h))))

    def apply_residual_mlp(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
    ) -> torch.Tensor:
        """Apply output projection, residual update, and MLP block."""
        x = self.apply_attn_residual(x, attn_out)
        return self.apply_mlp_residual(x)

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.mix_residual_input(x, x0)
        q, k, v = self.project_qkv(x, rope_cos=rope_cos, rope_sin=rope_sin)
        attn_out = self.attend(q, k, v, attn_mask=attn_mask, self_value=v)
        return self.apply_residual_mlp(x, attn_out)

    def forward_grouped(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention to grouped local blocks while preserving window locality."""
        x = self.mix_residual_input(x, x0)
        q, k, v = self.project_qkv_grouped(x, rope_cos=rope_cos, rope_sin=rope_sin)
        attn_out = self.attend_grouped(q, k, v, self_value=v)
        return self.apply_residual_mlp(x, attn_out)


def critic_window_partition(
    x: torch.Tensor, window_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Partition (B, H, W, C) into grouped non-overlapping windows.

    Returns ``(B, n_windows, window_size * window_size, C)`` plus the padded
    spatial shape.
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    n_win_h = Hp // window_size
    n_win_w = Wp // window_size
    x = x.view(B, n_win_h, window_size, n_win_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(B, n_win_h * n_win_w, window_size * window_size, C)
    return windows, (Hp, Wp)


def critic_window_unpartition(
    windows: torch.Tensor, window_size: int,
    pad_hw: tuple[int, int], hw: tuple[int, int],
) -> torch.Tensor:
    """Reverse grouped window partition. Returns (B, H, W, C)."""
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0]
    C = windows.shape[-1]
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x



# ---- Transformer Critic High-Level Modules ----

from collections import namedtuple

CriticMapState = namedtuple("CriticMapState", ["patches", "pH", "pW", "global_cls"])


@dataclass(slots=True)
class CriticGlobalLayout:
    n_rounds: int
    unit_start: int
    total_tokens: int
    perm: torch.Tensor
    sorted_round_idx: torch.Tensor
    local_pos: torch.Tensor
    valid_unit_mask: torch.Tensor
    unit_attn_bias: torch.Tensor | None  # float additive mask for mem-efficient SDPA
    shared_rope_cos: torch.Tensor
    shared_rope_sin: torch.Tensor
    unit_rope_cos: torch.Tensor | None
    unit_rope_sin: torch.Tensor | None


class CriticTileEmbed(nn.Module):
    """Per-tile two-layer MLP encoder."""

    def __init__(self, in_channels: int = GLOBAL_MAP_CHANNELS, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            RMSNorm(in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, embed_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _parameter_golf_ortho_init(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (R, C, H, W) → (R, H, W, embed_dim).

        Returns NHWC so downstream patchify / mask broadcast avoid a
        round-trip permute back into NCHW.
        """
        R, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(R * H * W, C)
        x = self.net(x)
        return x.reshape(R, H, W, self.embed_dim)


class CriticWindowedSA(nn.Module):
    """Windowed self-attention with local 2D RoPE."""

    def __init__(
        self,
        dim: int = 48,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        window_size: int = 5,
        mlp_ratio: int = 4,
        rope_dim: int | None = None,
        n_residual_layers: int = 1,
    ):
        super().__init__()
        self.window_size = window_size
        self.attn = CriticAttentionBlock(
            dim,
            n_heads,
            n_kv_heads,
            mlp_ratio,
            n_residual_layers,
        )
        self.rope = RotaryEmbedding2D(dim // n_heads, max_size=window_size, rotary_dim=rope_dim)
        ws = window_size
        h_coords = torch.arange(ws).unsqueeze(1).expand(ws, ws).reshape(-1)
        w_coords = torch.arange(ws).unsqueeze(0).expand(ws, ws).reshape(-1)
        self.register_buffer("_local_h", h_coords)
        self.register_buffer("_local_w", w_coords)
        local_rope_cos, local_rope_sin = self.rope(h_coords, w_coords)
        self.register_buffer("_local_rope_cos", local_rope_cos)
        self.register_buffer("_local_rope_sin", local_rope_sin)

    def forward(
        self,
        patches: torch.Tensor,
        pH: int,
        pW: int,
        anchor_patches: torch.Tensor,
    ) -> torch.Tensor:
        """patches: (R, N, dim) where N = pH * pW. Returns same shape."""
        R, N, C = patches.shape
        x = patches.reshape(R, pH, pW, C)
        x0 = anchor_patches.reshape(R, pH, pW, C)
        windows, pad_hw = critic_window_partition(x, self.window_size)
        anchor_windows, _ = critic_window_partition(x0, self.window_size)
        windows = self.attn.forward_grouped(
            windows,
            anchor_windows,
            rope_cos=self._local_rope_cos,
            rope_sin=self._local_rope_sin,
        )
        x = critic_window_unpartition(windows, self.window_size, pad_hw, (pH, pW))
        return x.reshape(R, N, C)


class CriticGlobalSA(nn.Module):
    """Global self-attention over patches + special tokens with masked attention."""

    def __init__(
        self,
        dim: int = 48,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        n_global_cls: int = 8,
        mlp_ratio: int = 4,
        max_grid_size: int = 25,
        rope_dim: int | None = None,
        n_residual_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_global_cls = n_global_cls
        self.attn = CriticAttentionBlock(
            dim,
            n_heads,
            n_kv_heads,
            mlp_ratio,
            n_residual_layers,
        )
        self.rope = RotaryEmbedding2D(dim // n_heads, max_size=max_grid_size, rotary_dim=rope_dim)
        head_dim_half = self.rope.freqs.shape[0] * 2
        self.register_buffer("_special_token_cos", torch.ones(1 + n_global_cls, head_dim_half, dtype=_TRAINING_DTYPE))
        self.register_buffer("_special_token_sin", torch.zeros(1 + n_global_cls, head_dim_half, dtype=_TRAINING_DTYPE))
        self._patch_rope_cache: dict[
            tuple[int, int, int, tuple[str, int | None]],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    @staticmethod
    def _device_key(device: torch.device) -> tuple[str, int | None]:
        return (device.type, device.index)

    def _get_patch_rope(
        self,
        n_patches: int,
        pH: int,
        pW: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached RoPE for patch tokens shared by all rounds."""
        cache_key = (n_patches, pH, pW, self._device_key(device))
        cached = self._patch_rope_cache.get(cache_key)
        if cached is not None:
            return cached

        ph = torch.arange(pH, device=device)
        pw = torch.arange(pW, device=device)
        grid_h = ph.unsqueeze(1).expand(pH, pW).reshape(-1)[:n_patches]
        grid_w = pw.unsqueeze(0).expand(pH, pW).reshape(-1)[:n_patches]
        rope_cos, rope_sin = self.rope(grid_h, grid_w)
        self._patch_rope_cache[cache_key] = (rope_cos, rope_sin)
        return rope_cos, rope_sin

    def prepare_layout(
        self,
        n_patches: int,
        pH: int,
        pW: int,
        unit_cls: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_round_idx: torch.Tensor,
        n_rounds: int,
    ) -> CriticGlobalLayout:
        """Build all static layout tensors once and reuse them across layers."""
        device = unit_cls.device
        dtype = unit_cls.dtype
        unit_start = n_patches + 1 + self.n_global_cls
        total_units = unit_cls.shape[0]

        patch_rope_cos, patch_rope_sin = self._get_patch_rope(n_patches, pH, pW, device)
        rope_dtype = patch_rope_cos.dtype
        special_cos = self._special_token_cos
        special_sin = self._special_token_sin

        if total_units == 0:
            total_tokens = unit_start
            shared_rope_cos = torch.empty((total_tokens, patch_rope_cos.shape[-1]), device=device, dtype=rope_dtype)
            shared_rope_sin = torch.empty_like(shared_rope_cos)
            shared_rope_cos[:n_patches] = patch_rope_cos
            shared_rope_sin[:n_patches] = patch_rope_sin
            shared_rope_cos[n_patches:unit_start] = special_cos
            shared_rope_sin[n_patches:unit_start] = special_sin
            empty = torch.zeros(0, dtype=torch.long, device=device)
            empty_mask = torch.zeros((n_rounds, 0), dtype=torch.bool, device=device)
            return CriticGlobalLayout(
                n_rounds=n_rounds,
                unit_start=unit_start,
                total_tokens=total_tokens,
                perm=empty,
                sorted_round_idx=empty,
                local_pos=empty,
                valid_unit_mask=empty_mask,
                unit_attn_bias=None,
                shared_rope_cos=shared_rope_cos,
                shared_rope_sin=shared_rope_sin,
                unit_rope_cos=None,
                unit_rope_sin=None,
            )

        counts = torch.bincount(unit_round_idx, minlength=n_rounds)
        max_units = int(counts.max().item())
        perm = unit_round_idx.argsort(stable=True)
        sorted_round_idx = unit_round_idx[perm]
        starts = torch.zeros_like(counts)
        starts[1:] = counts.cumsum(0)[:-1]
        local_pos = torch.arange(total_units, device=device) - starts[sorted_round_idx]

        total_tokens = unit_start + max_units
        valid_unit_mask = torch.zeros((n_rounds, max_units), dtype=torch.bool, device=device)
        valid_unit_mask[sorted_round_idx, local_pos] = True
        # Float additive mask: 0 for visible keys, -inf for padded unit keys.
        # Using float (not boolean) makes SDPA dispatch to the memory-efficient
        # backend instead of the math backend.
        unit_attn_bias = torch.zeros(
            (n_rounds, 1, 1, total_tokens), dtype=dtype, device=device,
        )
        unit_attn_bias[:, :, :, unit_start:].masked_fill_(
            ~valid_unit_mask.unsqueeze(1).unsqueeze(2), float('-inf'),
        )

        shared_rope_cos = torch.empty((unit_start, patch_rope_cos.shape[-1]), device=device, dtype=rope_dtype)
        shared_rope_sin = torch.empty_like(shared_rope_cos)
        shared_rope_cos[:n_patches] = patch_rope_cos
        shared_rope_sin[:n_patches] = patch_rope_sin
        shared_rope_cos[n_patches:unit_start] = special_cos
        shared_rope_sin[n_patches:unit_start] = special_sin
        unit_rope_cos = torch.ones((n_rounds, max_units, patch_rope_cos.shape[-1]), device=device, dtype=rope_dtype)
        unit_rope_sin = torch.zeros_like(unit_rope_cos)
        sorted_xy = unit_xy[perm]
        u_h = (sorted_xy[:, 1] // 2).clamp(0, pH - 1)
        u_w = (sorted_xy[:, 0] // 2).clamp(0, pW - 1)
        u_cos, u_sin = self.rope(u_h, u_w)
        unit_rope_cos[sorted_round_idx, local_pos] = u_cos
        unit_rope_sin[sorted_round_idx, local_pos] = u_sin

        return CriticGlobalLayout(
            n_rounds=n_rounds,
            unit_start=unit_start,
            total_tokens=total_tokens,
            perm=perm,
            sorted_round_idx=sorted_round_idx,
            local_pos=local_pos,
            valid_unit_mask=valid_unit_mask,
            unit_attn_bias=unit_attn_bias,
            shared_rope_cos=shared_rope_cos,
            shared_rope_sin=shared_rope_sin,
            unit_rope_cos=unit_rope_cos,
            unit_rope_sin=unit_rope_sin,
        )

    def forward(
        self,
        patches: torch.Tensor,        # (R, N_patches, dim)
        anchor_patches: torch.Tensor,  # (R, N_patches, dim)
        global_cls: torch.Tensor,      # (R, n_global_cls, dim)
        anchor_global_cls: torch.Tensor,  # (R, n_global_cls, dim)
        scalar_tok: torch.Tensor,      # (R, 1, dim)
        anchor_scalar_tok: torch.Tensor,  # (R, 1, dim)
        unit_cls: torch.Tensor,        # (B_total, dim)
        anchor_unit_cls: torch.Tensor,  # (B_total, dim)
        layout: CriticGlobalLayout,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split attention with cross-group visibility.

        Patches query patches + scalar + CLS + units (full key set).
        CLS query patches + scalar + CLS (see each other).
        Units query everything (unchanged).
        Shared MLP residual applied to all tokens.
        """
        R = layout.n_rounds
        N = patches.shape[1]
        all_tokens = patches.new_zeros((R, layout.total_tokens, patches.shape[-1]))
        all_tokens[:, :N] = patches
        all_tokens[:, N:N + 1] = scalar_tok
        all_tokens[:, N + 1:layout.unit_start] = global_cls
        anchor_tokens = patches.new_zeros((R, layout.total_tokens, patches.shape[-1]))
        anchor_tokens[:, :N] = anchor_patches
        anchor_tokens[:, N:N + 1] = anchor_scalar_tok
        anchor_tokens[:, N + 1:layout.unit_start] = anchor_global_cls
        if layout.perm.numel() > 0:
            all_tokens[layout.sorted_round_idx, layout.unit_start + layout.local_pos] = unit_cls[layout.perm]
            anchor_tokens[layout.sorted_round_idx, layout.unit_start + layout.local_pos] = anchor_unit_cls[layout.perm]

        all_tokens = self.attn.mix_residual_input(all_tokens, anchor_tokens)

        q, k, v = self.attn.project_qkv(all_tokens)

        prefix_end = layout.unit_start
        q[:, :, :prefix_end] = apply_rotary_2d(
            q[:, :, :prefix_end], layout.shared_rope_cos, layout.shared_rope_sin,
        )
        k[:, :, :prefix_end] = apply_rotary_2d(
            k[:, :, :prefix_end], layout.shared_rope_cos, layout.shared_rope_sin,
        )
        if layout.unit_rope_cos is not None:
            q[:, :, prefix_end:] = apply_rotary_2d(
                q[:, :, prefix_end:], layout.unit_rope_cos, layout.unit_rope_sin,
            )
            k[:, :, prefix_end:] = apply_rotary_2d(
                k[:, :, prefix_end:], layout.unit_rope_cos, layout.unit_rope_sin,
            )

        patch_end = N
        scalar_end = N + 1
        shared_end = layout.unit_start

        patch_tokens = all_tokens[:, :patch_end]
        scalar_tokens = all_tokens[:, patch_end:scalar_end]
        global_tokens = all_tokens[:, scalar_end:shared_end]
        unit_tokens = all_tokens[:, shared_end:]

        # Patches query patches + scalar + CLS (no mask needed, FA-eligible)
        patch_tokens = self.attn.apply_attn_residual(
            patch_tokens,
            self.attn.attend(
                q[:, :, :patch_end],
                k[:, :, :shared_end],
                v[:, :, :shared_end],
                self_value=v[:, :, :patch_end],
            ),
        )
        # CLS query patches + scalar + CLS (see each other)
        global_tokens = self.attn.apply_attn_residual(
            global_tokens,
            self.attn.attend(
                q[:, :, scalar_end:shared_end],
                k[:, :, :shared_end],
                v[:, :, :shared_end],
                self_value=v[:, :, scalar_end:shared_end],
            ),
        )
        if layout.unit_attn_bias is not None:
            unit_attn_out = self.attn.attend(
                q[:, :, shared_end:], k, v,
                attn_mask=layout.unit_attn_bias,
                self_value=v[:, :, shared_end:],
            )
            unit_attn_out = unit_attn_out * layout.valid_unit_mask.unsqueeze(-1).to(dtype=unit_attn_out.dtype)
            unit_tokens = self.attn.apply_attn_residual(unit_tokens, unit_attn_out)

        all_tokens = torch.cat([patch_tokens, scalar_tokens, global_tokens, unit_tokens], dim=1)
        all_tokens = self.attn.apply_mlp_residual(all_tokens)
        out_patches = all_tokens[:, :N].to(dtype=patches.dtype)
        out_global_cls = all_tokens[:, N + 1:layout.unit_start].to(dtype=global_cls.dtype)
        out_unit_cls = torch.zeros_like(unit_cls)
        if layout.perm.numel() > 0:
            out_unit_cls[layout.perm] = all_tokens[
                layout.sorted_round_idx,
                layout.unit_start + layout.local_pos,
            ].to(dtype=unit_cls.dtype)
        return out_patches, out_global_cls, out_unit_cls


class CriticUnitCLSBuilder(nn.Module):
    """Construct unit CLS tokens: content (actor latent) + position (sine embed)."""

    def __init__(
        self,
        actor_latent_dim: int = ACTOR_LATENT_DIM,
        context_dim: int = UNIT_CONTEXT_DIM,
        out_dim: int = 48,
        n_freqs: int = 16,
    ):
        super().__init__()
        self.content_proj = nn.Sequential(
            nn.Linear(actor_latent_dim, out_dim, bias=False),
            RMSNorm(out_dim),
            nn.SiLU(),
        )
        sine_dim = context_dim * n_freqs * 2
        self.pos_proj = nn.Linear(sine_dim, out_dim, bias=False)
        freq_bands = torch.arange(1, n_freqs + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freq_bands", freq_bands)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _parameter_golf_ortho_init(m)

    def forward(self, actor_latent: torch.Tensor, unit_context: torch.Tensor) -> torch.Tensor:
        """actor_latent: (B, 64), unit_context: (B, 7) → (B, out_dim)."""
        content = self.content_proj(actor_latent)
        angles = unit_context.unsqueeze(-1) * self.freq_bands  # (B, 7, n_freqs)
        sine_embed = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)  # (B, 224)
        position = self.pos_proj(sine_embed)
        return content + position


class CriticValueHead(nn.Module):
    """Per-unit value prediction from CLS tokens + actor latent."""

    def __init__(
        self,
        dim: int = 48,
        actor_latent_dim: int = ACTOR_LATENT_DIM,
        n_global_cls: int = 8,
        n_buckets: int = NUM_BUCKETS,
        value_mode: CriticValueMode = CRITIC_VALUE_MODE,
    ):
        super().__init__()
        self.value_mode = value_mode
        self.cls_weights = nn.Parameter(torch.zeros(n_global_cls))
        total_in = dim + dim + actor_latent_dim
        out_dim = n_buckets if value_mode == CriticValueMode.GHL else 1
        self.mlp = nn.Sequential(
            nn.Linear(total_in, total_in, bias=False),
            nn.SiLU(),
            nn.Linear(total_in, total_in, bias=False),
            RMSNorm(total_in),
            nn.Linear(total_in, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp[:-1]:
            if isinstance(m, nn.Linear):
                _parameter_golf_ortho_init(m)
        if isinstance(self.mlp[-1], nn.Linear):
            if self.mlp[-1].bias is not None:
                nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(
        self,
        unit_cls: torch.Tensor,
        global_cls: torch.Tensor,
        unit_round_idx: torch.Tensor,
        actor_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits (B, n_buckets) for GHL or (B, 1) for MSE."""
        weights = F.softmax(self.cls_weights, dim=0)
        pooled = (global_cls * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (R, dim)
        pooled_per_unit = pooled[unit_round_idx]  # (B, dim)
        x = torch.cat([unit_cls, pooled_per_unit, actor_latent], dim=-1)
        return self.mlp(x)


class CentralizedCritic(nn.Module):
    """Transformer-based centralized per-unit critic over full replay map state.

    Architecture:
      1. Per-tile MLP encodes 23-channel symbolic features → 8-dim embeddings.
      2. 2×2 reshape packs tiles into patches (dim=32).
      3. Alternating windowed SA (local, translation-equivariant) and global SA
         (full map + special tokens) repeated 3 times.
      4. Per-unit value from unit CLS tokens + global CLS pooling + actor latent.
    """

    _TILE_DIM = 8
    _PATCH_DIM = 32  # 4 tiles × 8 dims
    _N_HEADS = 4
    _N_KV_HEADS = 2
    _WINDOW_SIZE = 5
    _MLP_RATIO = 4
    _N_GLOBAL_CLS = 8
    _N_BLOCKS = 3
    _ROPE_DIM = 4  # partial 2D RoPE over the first half of each 8-dim head

    def __init__(self, map_channels: int = GLOBAL_MAP_CHANNELS, value_mode: CriticValueMode = CRITIC_VALUE_MODE):
        super().__init__()
        self.value_mode = value_mode
        dim = self._PATCH_DIM
        n_residual_layers = self._N_BLOCKS * 3

        self.tile_embed = CriticTileEmbed(map_channels, self._TILE_DIM)

        # 3 blocks, each with 2 windowed SA + 1 global SA
        self.windowed_sa = nn.ModuleList([
            CriticWindowedSA(dim, self._N_HEADS, self._N_KV_HEADS,
                             self._WINDOW_SIZE, self._MLP_RATIO, self._ROPE_DIM, n_residual_layers)
            for _ in range(self._N_BLOCKS * 2)
        ])
        self.global_sa = nn.ModuleList([
            CriticGlobalSA(dim, self._N_HEADS, self._N_KV_HEADS,
                           self._N_GLOBAL_CLS, self._MLP_RATIO, rope_dim=self._ROPE_DIM,
                           n_residual_layers=n_residual_layers)
            for _ in range(self._N_BLOCKS)
        ])

        # Scalar token projection
        self.scalar_proj = nn.Sequential(
            nn.Linear(GLOBAL_SCALAR_DIM, dim, bias=False),
            RMSNorm(dim),
            nn.SiLU(),
        )
        _parameter_golf_ortho_init(self.scalar_proj[0])

        # Learned global CLS tokens
        self.global_cls = nn.Parameter(torch.zeros(self._N_GLOBAL_CLS, dim))
        nn.init.xavier_uniform_(self.global_cls)

        # Final norm for pre-norm transformer output
        self.final_norm = RMSNorm(dim)

        # Unit CLS builder and value head
        self.unit_cls_builder = CriticUnitCLSBuilder(
            ACTOR_LATENT_DIM, UNIT_CONTEXT_DIM, dim,
        )
        self.value_head = CriticValueHead(dim, ACTOR_LATENT_DIM, self._N_GLOBAL_CLS,
                                          value_mode=value_mode)

        # GHL bin centers (uniform in symlog space).
        bucket_centers = _make_bucket_centers()
        self.register_buffer("bucket_centers", bucket_centers)

    def _patchify(
        self, tiles: torch.Tensor, valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Convert tile features to patch tokens via 2×2 reshape.

        Args:
            tiles: (R, H, W, tile_dim) — NHWC, already contig out of tile_embed
            valid_mask: (R, 1, H, W) — NCHW mask from the training pipeline
        Returns:
            patches: (R, N_patches, patch_dim), pH, pW
        """
        R, H, W, D = tiles.shape
        # Pad to even dimensions
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            tiles = F.pad(tiles, (0, 0, 0, pad_w, 0, pad_h))
            valid_mask = F.pad(valid_mask, (0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        pH, pW = Hp // 2, Wp // 2

        # Zero out invalid tiles (broadcast NHWC mask against NHWC tiles)
        mask_nhwc = valid_mask.permute(0, 2, 3, 1).to(tiles.dtype)
        tiles = tiles * mask_nhwc

        # (R, Hp, Wp, D) → (R, pH, 2, pW, 2, D) is a view (stride-compatible
        # on NHWC-contig memory), so only the final reshape forces a copy.
        patches = tiles.reshape(R, pH, 2, pW, 2, D).permute(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(R, pH * pW, 4 * D)  # (R, N, patch_dim)
        return patches, pH, pW

    def encode_map(
        self,
        map_tensor: torch.Tensor,    # (R, C, H, W) — R unique rounds
        valid_mask: torch.Tensor,     # (R, 1, H, W)
    ) -> CriticMapState:
        """Run tile embed + patchify + windowed SA layers. Returns cached state."""
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=_USE_BF16):
            tiles = self.tile_embed(map_tensor)  # (R, H, W, tile_dim)
            patches, pH, pW = self._patchify(tiles, valid_mask)
            patch_anchor = patches
            for i in range(self._N_BLOCKS):
                patches = self.windowed_sa[i * 2](patches, pH, pW, patch_anchor)
                patches = self.windowed_sa[i * 2 + 1](patches, pH, pW, patch_anchor)

        R = map_tensor.shape[0]
        global_cls = self.global_cls.unsqueeze(0).expand(R, -1, -1)
        return CriticMapState(patches=patches, pH=pH, pW=pW, global_cls=global_cls)

    def _run_global_sa(
        self,
        state: CriticMapState,
        round_scalars: torch.Tensor,
        round_idx: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_cls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run global SA layers over cached patches + special tokens.

        Returns:
            (global_cls, unit_cls) after refinement through all global SA layers.
        """
        R = state.patches.shape[0]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=_USE_BF16):
            patches = state.patches
            token_dtype = patches.dtype
            scalar_tok = self.scalar_proj(round_scalars).unsqueeze(1).to(dtype=token_dtype)
            global_cls = state.global_cls.to(dtype=token_dtype)
            unit_cls = unit_cls.to(dtype=token_dtype)
            patch_anchor = patches
            global_cls_anchor = global_cls
            scalar_anchor = scalar_tok
            unit_anchor = unit_cls

            # Remap round_idx to local 0..R-1
            ri = round_idx.long()
            layout = self.global_sa[0].prepare_layout(
                patches.shape[1], state.pH, state.pW,
                unit_cls, unit_xy.long(), ri, R,
            )
            for layer in self.global_sa:
                patches, global_cls, unit_cls = layer(
                    patches,
                    patch_anchor,
                    global_cls,
                    global_cls_anchor,
                    scalar_tok,
                    scalar_anchor,
                    unit_cls,
                    unit_anchor,
                    layout,
                )

        return global_cls, unit_cls

    def _decode_head_output(self, head_out: torch.Tensor) -> torch.Tensor:
        """Decode value head output to scalar values (B,)."""
        if self.value_mode == CriticValueMode.MSE:
            return symexp(head_out.squeeze(-1))
        return decode_value(head_out, self.bucket_centers)

    def unit_values(
        self,
        feat: CriticMapState,
        round_scalars: torch.Tensor,  # (R, GLOBAL_SCALAR_DIM)
        round_idx: torch.Tensor,      # (B,)
        unit_xy: torch.Tensor,        # (B, 2)
        unit_context: torch.Tensor,   # (B, UNIT_CONTEXT_DIM)
        actor_latent: torch.Tensor,   # (B, ACTOR_LATENT_DIM)
    ) -> torch.Tensor:
        """Compute per-unit marginal values. Returns (B,)."""
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=_USE_BF16):
            unit_cls = self.unit_cls_builder(actor_latent.float(), unit_context.float())
            global_cls, unit_cls = self._run_global_sa(
                feat, round_scalars, round_idx, unit_xy, unit_cls,
            )
        head_out = self.value_head(self.final_norm(unit_cls.float()),
                                   self.final_norm(global_cls.float()),
                                   round_idx.long(), actor_latent.float())
        return self._decode_head_output(head_out)

    def unit_value_logits(
        self,
        feat: CriticMapState,
        round_scalars: torch.Tensor,
        round_idx: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_context: torch.Tensor,
        actor_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like unit_values but also returns raw head output for training loss."""
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=_USE_BF16):
            unit_cls = self.unit_cls_builder(actor_latent.float(), unit_context.float())
            global_cls, unit_cls = self._run_global_sa(
                feat, round_scalars, round_idx, unit_xy, unit_cls,
            )
        head_out = self.value_head(self.final_norm(unit_cls.float()),
                                   self.final_norm(global_cls.float()),
                                   round_idx.long(), actor_latent.float())
        values = self._decode_head_output(head_out)
        return values, head_out

    def forward(
        self,
        map_tensor: torch.Tensor,
        valid_mask: torch.Tensor,
        round_scalars: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_context: torch.Tensor,
        actor_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass (non-cached path). Returns per-unit value."""
        feat = self.encode_map(map_tensor, valid_mask)
        round_idx = torch.arange(map_tensor.shape[0], device=map_tensor.device)
        return self.unit_values(
            feat, round_scalars, round_idx, unit_xy, unit_context, actor_latent,
        )


class CriticPpoStep(nn.Module):
    """Compiled PPO critic step body over one round chunk."""

    def __init__(self, critic: CentralizedCritic):
        super().__init__()
        self.critic = critic

    def forward(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        scalars: torch.Tensor,
        round_idx: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_context: torch.Tensor,
        actor_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.critic.encode_map(states, masks)
        return self.critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_context, actor_latent,
        )


class CriticPpoLossStep(nn.Module):
    """Compiled PPO critic forward+loss body over one round-bounded batch."""

    def __init__(self, critic: CentralizedCritic):
        super().__init__()
        self.critic = critic

    def forward(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        scalars: torch.Tensor,
        round_idx: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_context: torch.Tensor,
        actor_latent: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.critic.encode_map(states, masks)
        values, head_out = self.critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_context, actor_latent,
        )
        if self.critic.value_mode == CriticValueMode.MSE:
            pred_symlog = head_out.squeeze(-1)
            target_symlog = symlog(returns)
            v_loss_elems = (pred_symlog - target_symlog).square()
        else:
            v_loss_elems = distributional_value_loss(
                head_out,
                returns,
                self.critic.bucket_centers,
                values=values,
                old_values=old_values,
                clip_vloss=False,
            )
        loss_sum = v_loss_elems.sum()
        return loss_sum, loss_sum.detach(), values.detach()


def _materialize_critic_inputs(centralized_data: dict) -> None:
    """Materialize critic inputs into canonical CPU slabs for batched staging.

    Round tensors are packed once into contiguous CPU buffers. GPU copies happen
    lazily per round batch in ``_load_critic_round_batch`` using reusable
    staging buffers, so end-to-end VRAM scales with the current critic batch
    instead of all deduped rounds in the episode.
    """
    global_states = centralized_data["global_states"]
    sizes = centralized_data["global_sizes"]
    R = len(global_states)
    if R == 0:
        return

    max_w = int(sizes[:, 0].max().item())
    max_h = int(sizes[:, 1].max().item())

    cuda_cfg = _get_critic_cuda_config()
    use_pinned = DEVICE.type == "cuda"
    memory_format = torch.channels_last if cuda_cfg.channels_last else torch.contiguous_format

    # Pack once into final-layout CPU slabs so per-batch staging is just slicing.
    obs_cpu = torch.empty(
        (R, GLOBAL_MAP_CHANNELS, max_h, max_w),
        dtype=torch.float32,
        pin_memory=use_pinned,
        memory_format=memory_format,
    )
    mask_cpu = torch.empty(
        (R, 1, max_h, max_w),
        dtype=torch.bool,
        pin_memory=use_pinned,
        memory_format=memory_format,
    )
    obs_cpu.zero_()
    mask_cpu.zero_()

    for i in range(R):
        mw = int(sizes[i, 0].item())
        mh = int(sizes[i, 1].item())
        if mw > 0 and mh > 0:
            buf = global_states[i]
            if isinstance(buf, torch.Tensor):
                obs_cpu[i, :, :mh, :mw].copy_(buf)
            else:
                state_view = torch.frombuffer(
                    buf, dtype=torch.float32,
                ).reshape(GLOBAL_MAP_CHANNELS, mh, mw)
                obs_cpu[i, :, :mh, :mw].copy_(state_view)
            mask_cpu[i, 0, :mh, :mw] = True

    global_scalars = centralized_data["global_scalars"]
    if use_pinned and not global_scalars.is_pinned():
        global_scalars = global_scalars.pin_memory()
        centralized_data["global_scalars"] = global_scalars

    centralized_data["critic_obs_cpu"] = obs_cpu
    centralized_data["critic_mask_cpu"] = mask_cpu
    centralized_data["critic_input_staging"] = None

    # Free CPU-side bytearrays — no longer needed
    centralized_data["global_states"] = None

    cpu_mb = (
        obs_cpu.nbytes
        + mask_cpu.nbytes
        + global_scalars.nbytes
    ) / (1024 * 1024)
    print(
        f"  Critic inputs staged on CPU: {R} rounds, "
        f"{max_h}x{max_w} maps, {cpu_mb:.0f} MB"
    )


def _coerce_replay_buffer(buf):
    """Return a writable buffer for zero-copy torch.frombuffer views."""
    return coerce_writable_replay_buffer(buf)


def _load_critic_round_batch(
    centralized_data: dict,
    round_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch one round batch, staging from CPU to GPU only as needed."""
    ids = round_ids.long()
    if "gpu_obs" in centralized_data:
        return (
            centralized_data["gpu_obs"][ids],
            centralized_data["gpu_mask"][ids],
            centralized_data["gpu_scalars"][ids],
        )

    obs_cpu = centralized_data["critic_obs_cpu"]
    mask_cpu = centralized_data["critic_mask_cpu"]
    scalars_cpu = centralized_data["global_scalars"]

    if DEVICE.type != "cuda":
        ids_cpu = ids.cpu()
        return (
            obs_cpu.index_select(0, ids_cpu),
            mask_cpu.index_select(0, ids_cpu),
            scalars_cpu.index_select(0, ids_cpu),
        )

    ids_cpu = ids.cpu()
    sizes = centralized_data["global_sizes"].index_select(0, ids_cpu)
    batch_w = int(sizes[:, 0].max().item())
    batch_h = int(sizes[:, 1].max().item())
    n_rounds = int(ids_cpu.numel())

    def _contiguous_range(sel: torch.Tensor) -> tuple[int, int] | None:
        if sel.numel() == 0:
            return 0, 0
        start = int(sel[0].item())
        end = start + sel.numel()
        if int(sel[-1].item()) == end - 1 and torch.equal(
            sel, torch.arange(start, end, dtype=sel.dtype),
        ):
            return start, end
        return None

    contiguous = _contiguous_range(ids_cpu)
    if contiguous is not None:
        start, end = contiguous
        obs_src = obs_cpu[start:end, :, :batch_h, :batch_w]
        mask_src = mask_cpu[start:end, :, :batch_h, :batch_w]
        scalars_src = scalars_cpu[start:end]
    else:
        obs_src = obs_cpu[:, :, :batch_h, :batch_w].index_select(0, ids_cpu)
        mask_src = mask_cpu[:, :, :batch_h, :batch_w].index_select(0, ids_cpu)
        scalars_src = scalars_cpu.index_select(0, ids_cpu)

    gpu_dtype = torch.bfloat16 if _USE_BF16 else torch.float32
    staging = centralized_data.get("critic_input_staging")
    if (
        staging is None
        or staging.capacity_rounds < n_rounds
        or staging.capacity_h < batch_h
        or staging.capacity_w < batch_w
    ):
        cuda_cfg = _get_critic_cuda_config()
        memory_format = torch.channels_last if cuda_cfg.channels_last else torch.contiguous_format
        staging = CriticInputStaging(
            obs=torch.empty(
                (n_rounds, GLOBAL_MAP_CHANNELS, batch_h, batch_w),
                dtype=gpu_dtype,
                device=DEVICE,
                memory_format=memory_format,
            ),
            mask=torch.empty(
                (n_rounds, 1, batch_h, batch_w),
                dtype=torch.bool,
                device=DEVICE,
                memory_format=memory_format,
            ),
            scalars=torch.empty(
                (n_rounds, scalars_cpu.shape[1]),
                dtype=scalars_cpu.dtype,
                device=DEVICE,
            ),
            capacity_rounds=n_rounds,
            capacity_h=batch_h,
            capacity_w=batch_w,
        )
        centralized_data["critic_input_staging"] = staging

    obs_dst = staging.obs[:n_rounds, :, :batch_h, :batch_w]
    mask_dst = staging.mask[:n_rounds, :, :batch_h, :batch_w]
    scalars_dst = staging.scalars[:n_rounds]
    obs_dst.copy_(obs_src, non_blocking=True)
    mask_dst.copy_(mask_src, non_blocking=True)
    scalars_dst.copy_(scalars_src, non_blocking=True)
    return (
        obs_dst,
        mask_dst,
        scalars_dst,
    )


def _critic_chunk_plan(sorted_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique ids plus prefix offsets for a sorted id tensor."""
    if sorted_ids.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=sorted_ids.device)
        return empty, torch.zeros(1, dtype=torch.long, device=sorted_ids.device)
    chunk_ids, chunk_counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    offsets = torch.empty(chunk_counts.numel() + 1, dtype=torch.long, device=sorted_ids.device)
    offsets[0] = 0
    offsets[1:] = chunk_counts.cumsum(0)
    return chunk_ids, offsets


def _critic_round_grouped_perm(step_round_idx: torch.Tensor) -> torch.Tensor:
    """Shuffle critic samples by round while keeping each round contiguous."""
    n_steps = step_round_idx.numel()
    if n_steps <= 1:
        return torch.arange(n_steps, device=step_round_idx.device)
    unique_rounds, inverse = torch.unique(step_round_idx, sorted=True, return_inverse=True)
    shuffled_round_order = torch.randperm(unique_rounds.shape[0], device=step_round_idx.device)
    return shuffled_round_order[inverse].argsort(stable=True)


def _critic_round_batch_plan(
    step_round_idx: torch.Tensor,
    *,
    target_steps: int,
    max_rounds: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Plan round-bounded critic batches from per-step round ids."""
    perm = _critic_round_grouped_perm(step_round_idx)
    sorted_round_idx = step_round_idx[perm]
    unique_rounds, round_offsets = _critic_chunk_plan(sorted_round_idx)
    if unique_rounds.numel() == 0:
        return perm, unique_rounds, round_offsets, []

    target_steps = max(1, int(target_steps))
    if max_rounds is not None:
        max_rounds = max(1, int(max_rounds))
    offsets_cpu = round_offsets.tolist()
    n_unique_rounds = unique_rounds.shape[0]
    batch_ranges: list[tuple[int, int]] = []
    round_start = 0
    while round_start < n_unique_rounds:
        round_cap = n_unique_rounds if max_rounds is None else min(round_start + max_rounds, n_unique_rounds)
        step_start = offsets_cpu[round_start]
        round_end = round_start + 1
        while round_end < round_cap:
            next_steps = offsets_cpu[round_end + 1] - step_start
            if next_steps > target_steps:
                break
            round_end += 1
        batch_ranges.append((round_start, round_end))
        round_start = round_end
    return perm, unique_rounds, round_offsets, batch_ranges


@dataclass(slots=True)
class RoundMajorPack:
    perm: torch.Tensor
    round_ids: torch.Tensor
    round_offsets: torch.Tensor
    round_offsets_cpu: list[int]
    tensors: tuple[torch.Tensor, ...]


@dataclass(slots=True)
class CriticEpochPlan:
    round_ids: torch.Tensor
    round_offsets: torch.Tensor
    round_offsets_cpu: list[int]
    batch_ranges: list[tuple[int, int]]
    local_round_idx_by_batch: list[torch.Tensor]
    packed_xy: torch.Tensor
    packed_unit_float: torch.Tensor
    packed_returns_values: torch.Tensor


@dataclass(slots=True)
class CriticInputStaging:
    obs: torch.Tensor
    mask: torch.Tensor
    scalars: torch.Tensor
    capacity_rounds: int
    capacity_h: int
    capacity_w: int


def _positive_round_mask_from_packed(
    packed: RoundMajorPack,
    positive_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Collapse per-step positive rewards into one boolean per packed round."""
    if positive_mask is None:
        return None
    sorted_positive = positive_mask[packed.perm].to(dtype=torch.int32)
    prefix = torch.empty(sorted_positive.numel() + 1, dtype=torch.int32, device=sorted_positive.device)
    prefix[0] = 0
    prefix[1:] = sorted_positive.cumsum(0)
    positive_counts = prefix[packed.round_offsets[1:]] - prefix[packed.round_offsets[:-1]]
    return positive_counts > 0


def _build_critic_round_batches(
    round_counts: torch.Tensor,
    target_steps: int,
    *,
    positive_round_mask: torch.Tensor | None = None,
    max_rounds: int | None = None,
) -> list[torch.Tensor]:
    """Assign packed rounds to critic batches without splitting round segments."""
    n_rounds = int(round_counts.numel())
    if n_rounds <= 0:
        return []

    dev = round_counts.device
    target_steps = max(1, int(target_steps))
    if max_rounds is not None:
        max_rounds = max(1, int(max_rounds))
    total_steps = int(round_counts.sum().item())
    target_batches = max(1, math.ceil(total_steps / target_steps))

    if positive_round_mask is not None:
        positive_rounds = torch.nonzero(positive_round_mask, as_tuple=False).flatten()
        negative_rounds = torch.nonzero(~positive_round_mask, as_tuple=False).flatten()
        if positive_rounds.numel() > 0:
            positive_rounds = positive_rounds[
                torch.randperm(positive_rounds.numel(), device=dev)
            ]
        if negative_rounds.numel() > 0:
            negative_rounds = negative_rounds[
                torch.randperm(negative_rounds.numel(), device=dev)
            ]
    else:
        positive_rounds = torch.empty(0, dtype=torch.long, device=dev)
        negative_rounds = torch.randperm(n_rounds, device=dev)

    seed_count = min(target_batches, int(positive_rounds.numel()))
    round_batches = [
        [int(positive_rounds[i].item())]
        for i in range(seed_count)
    ]
    batch_steps = [
        int(round_counts[positive_rounds[i]].item())
        for i in range(seed_count)
    ]

    if not round_batches:
        round_batches = [[]]
        batch_steps = [0]

    remaining_parts: list[torch.Tensor] = []
    if positive_rounds.numel() > seed_count:
        remaining_parts.append(positive_rounds[seed_count:])
    if negative_rounds.numel() > 0:
        remaining_parts.append(negative_rounds)
    if remaining_parts:
        remaining_rounds = torch.cat(remaining_parts)
        if remaining_rounds.numel() > 1:
            remaining_rounds = remaining_rounds[
                torch.randperm(remaining_rounds.numel(), device=dev)
            ]
    else:
        remaining_rounds = torch.empty(0, dtype=torch.long, device=dev)

    for round_pos in remaining_rounds:
        round_idx = int(round_pos.item())
        count = int(round_counts[round_idx].item())
        best_batch = -1
        best_steps = -1
        for i, steps in enumerate(batch_steps):
            if max_rounds is not None and len(round_batches[i]) >= max_rounds:
                continue
            next_steps = steps + count
            if steps > 0 and next_steps > target_steps:
                continue
            if steps > best_steps:
                best_batch = i
                best_steps = steps
        if best_batch < 0:
            round_batches.append([round_idx])
            batch_steps.append(count)
        else:
            round_batches[best_batch].append(round_idx)
            batch_steps[best_batch] += count

    return [
        torch.tensor(batch_rounds, dtype=torch.long, device=dev)
        for batch_rounds in round_batches
    ]


def _round_major_take_positions(
    round_offsets: torch.Tensor,
    round_order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a permutation of packed round segments into packed step positions."""
    if round_order.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=round_offsets.device)
        return empty, empty

    starts = round_offsets[:-1].index_select(0, round_order)
    counts = round_offsets[1:].index_select(0, round_order) - starts
    rel = torch.arange(int(counts.max().item()), device=round_order.device)
    positions = starts.unsqueeze(1) + rel.unsqueeze(0)
    take_mask = rel.unsqueeze(0) < counts.unsqueeze(1)
    return positions[take_mask], counts


def _build_critic_turn_minibatches(
    step_round_idx: torch.Tensor,
    target_steps: int,
    positive_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Build shuffled critic turn minibatches without splitting turns.

    Critic updates are not recurrent, but the centralized critic is keyed by
    turn-level map state. We therefore shuffle turns, keep all unit-steps from
    a turn together, and spread positive-reward turns across batches when
    available while preserving exact epoch coverage.
    """
    n_steps = int(step_round_idx.numel())
    if n_steps <= 0:
        return []

    packed = _pack_round_major_tensors(step_round_idx)
    round_counts = packed.round_offsets[1:] - packed.round_offsets[:-1]
    round_batches = _build_critic_round_batches(
        round_counts,
        target_steps,
        positive_round_mask=_positive_round_mask_from_packed(packed, positive_mask),
    )

    offsets_cpu = packed.round_offsets_cpu
    step_batches: list[torch.Tensor] = []
    for batch_rounds in round_batches:
        parts = [
            packed.perm[offsets_cpu[int(round_pos.item())]:offsets_cpu[int(round_pos.item()) + 1]]
            for round_pos in batch_rounds
        ]
        step_batches.append(torch.cat(parts) if len(parts) > 1 else parts[0])
    return step_batches


def _pack_round_major_tensors(
    step_round_idx: torch.Tensor,
    *tensors: torch.Tensor,
    perm: torch.Tensor | None = None,
) -> RoundMajorPack:
    """Pack per-step tensors so all samples from a round stay contiguous."""
    if perm is None:
        perm = step_round_idx.argsort(stable=True)
    sorted_round_idx = step_round_idx[perm]
    round_ids, round_offsets = _critic_chunk_plan(sorted_round_idx)
    return RoundMajorPack(
        perm=perm,
        round_ids=round_ids,
        round_offsets=round_offsets,
        round_offsets_cpu=round_offsets.tolist(),
        tensors=tuple(t[perm] for t in tensors),
    )


def _build_critic_epoch_plan(
    step_round_idx: torch.Tensor,
    step_unit_xy: torch.Tensor,
    step_unit_float: torch.Tensor,
    step_rv: torch.Tensor,
    *,
    target_steps: int,
    positive_mask: torch.Tensor | None = None,
    max_rounds: int | None = None,
) -> CriticEpochPlan:
    """Build one critic epoch plan with round-major packed step tensors."""
    if positive_mask is None:
        perm, _, _, batch_ranges = _critic_round_batch_plan(
            step_round_idx,
            target_steps=target_steps,
            max_rounds=max_rounds,
        )
        packed = _pack_round_major_tensors(
            step_round_idx,
            step_unit_xy,
            step_unit_float,
            step_rv,
            perm=perm,
        )
        local_round_idx_by_batch: list[torch.Tensor] = []
        for round_start, round_end in batch_ranges:
            n_rounds = round_end - round_start
            chunk_round_counts = (
                packed.round_offsets[round_start + 1:round_end + 1]
                - packed.round_offsets[round_start:round_end]
            )
            local_round_idx_by_batch.append(torch.repeat_interleave(
                torch.arange(n_rounds, device=step_round_idx.device),
                chunk_round_counts,
            ))
        return CriticEpochPlan(
            round_ids=packed.round_ids,
            round_offsets=packed.round_offsets,
            round_offsets_cpu=packed.round_offsets_cpu,
            batch_ranges=batch_ranges,
            local_round_idx_by_batch=local_round_idx_by_batch,
            packed_xy=packed.tensors[0],
            packed_unit_float=packed.tensors[1],
            packed_returns_values=packed.tensors[2],
        )

    packed = _pack_round_major_tensors(
        step_round_idx,
        step_unit_xy,
        step_unit_float,
        step_rv,
    )
    round_counts = packed.round_offsets[1:] - packed.round_offsets[:-1]
    round_batches = _build_critic_round_batches(
        round_counts,
        target_steps,
        positive_round_mask=_positive_round_mask_from_packed(packed, positive_mask),
        max_rounds=max_rounds,
    )
    round_order = torch.cat(round_batches) if round_batches else torch.zeros(0, dtype=torch.long, device=step_round_idx.device)
    take_positions, ordered_round_counts = _round_major_take_positions(packed.round_offsets, round_order)
    ordered_round_ids = packed.round_ids.index_select(0, round_order)

    round_offsets = torch.empty(ordered_round_counts.numel() + 1, dtype=torch.long, device=step_round_idx.device)
    round_offsets[0] = 0
    if ordered_round_counts.numel() > 0:
        round_offsets[1:] = ordered_round_counts.cumsum(0)

    batch_ranges: list[tuple[int, int]] = []
    local_round_idx_by_batch: list[torch.Tensor] = []
    round_cursor = 0
    for batch_rounds in round_batches:
        batch_n_rounds = int(batch_rounds.numel())
        next_cursor = round_cursor + batch_n_rounds
        batch_ranges.append((round_cursor, next_cursor))
        counts = ordered_round_counts[round_cursor:next_cursor]
        local_round_idx_by_batch.append(torch.repeat_interleave(
            torch.arange(batch_n_rounds, device=step_round_idx.device),
            counts,
        ))
        round_cursor = next_cursor

    return CriticEpochPlan(
        round_ids=ordered_round_ids,
        round_offsets=round_offsets,
        round_offsets_cpu=round_offsets.tolist(),
        batch_ranges=batch_ranges,
        local_round_idx_by_batch=local_round_idx_by_batch,
        packed_xy=packed.tensors[0].index_select(0, take_positions),
        packed_unit_float=packed.tensors[1].index_select(0, take_positions),
        packed_returns_values=packed.tensors[2].index_select(0, take_positions),
    )


def _resolve_critic_minibatch_size(
    n_steps: int,
) -> int:
    """Pick the fixed critic batch size target for PPO updates."""
    return (n_steps + CRITIC_BATCHES_PER_EPOCH - 1) // CRITIC_BATCHES_PER_EPOCH


def _ortho_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init (CleanRL-style). Improves early exploration."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _parameter_golf_ortho_init(
    layer: nn.Linear,
    *,
    n_residual_layers: int | None = None,
    scale_proj: bool = False,
    min_dim: int = 64,
    bias_const: float = 0.0,
) -> nn.Linear:
    """Mimic parameter-golf: ortho only on large matrices, else keep Xavier fallback."""
    use_ortho = min(layer.weight.shape) >= min_dim
    if use_ortho:
        nn.init.orthogonal_(layer.weight, gain=1.0)
    else:
        nn.init.xavier_uniform_(layer.weight, gain=1.0)
    if use_ortho and scale_proj:
        if n_residual_layers is None or n_residual_layers <= 0:
            raise ValueError("n_residual_layers must be positive when scale_proj=True")
        with torch.no_grad():
            layer.weight.mul_(1.0 / math.sqrt(2.0 * n_residual_layers))
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def _xavier_init(layer: nn.Linear, gain: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
    """Xavier uniform init for Peri-LN transformer-style linear layers."""
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def _temperature_scaled_log_probs(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Return masked log-probs after dividing logits by a per-sample temperature."""
    temperature_t = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
    if temperature_t.dim() == 1:
        temperature_t = temperature_t.unsqueeze(-1)
    temperature_t = temperature_t.clamp_min(1e-4)
    scaled_logits = logits / temperature_t

    if mask is not None:
        masked_logits = scaled_logits.masked_fill(~mask, float("-inf"))
        any_valid = mask.any(dim=-1, keepdim=True)
        masked_logits = torch.where(any_valid, masked_logits, torch.zeros_like(masked_logits))
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        return log_probs.masked_fill(~mask, float("-inf"))

    return torch.log_softmax(scaled_logits, dim=-1)


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def _policy_weight_from_advantages(
    advantages: torch.Tensor,
    *,
    use_pmpo: bool = False,
) -> torch.Tensor:
    if use_pmpo:
        return torch.sign(advantages)
    return advantages


def sample_logits_gumbel(
    logits: torch.Tensor,
    temperature: torch.Tensor | float = 1.0,
    *,
    mask: torch.Tensor | None = None,
    return_log_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample from a masked temperature-scaled categorical via Gumbel-max."""
    log_probs = _temperature_scaled_log_probs(logits, mask, temperature)
    gumbel = -torch.empty_like(log_probs).exponential_().log()
    actions = (log_probs + gumbel).argmax(dim=-1)
    if not return_log_prob:
        return actions, None
    chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return actions, chosen


def action_log_prob_and_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: torch.Tensor | float = 1.0,
    entropy_temperature: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return chosen-action log_probs and categorical entropy from logits."""
    log_probs = _temperature_scaled_log_probs(logits, mask, temperature)
    chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    if entropy_temperature is None:
        entropy_temperature = temperature
    # nan_to_num handles 0 * -inf from masked actions.
    entropy_log_probs = _temperature_scaled_log_probs(logits, mask, entropy_temperature)
    probs = entropy_log_probs.exp()
    entropy = -(probs * entropy_log_probs).nan_to_num().sum(dim=-1)
    return chosen, entropy


def _compose_action_log_prob(
    bt_lp: torch.Tensor,
    bd_lp: torch.Tensor,
    cd_lp: torch.Tensor,
    brt_lp: torch.Tensor,
    md_lp: torch.Tensor,
    *,
    cd_causal: torch.Tensor,
    brt_causal: torch.Tensor,
) -> torch.Tensor:
    """Compose per-head log-probs: p(bd) * p(bt|bd) * p(cd|bt) * p(brt|bt) * p(md)."""
    zeros = torch.zeros_like(bt_lp)
    return (
        bd_lp
        + bt_lp
        + torch.where(cd_causal.to(torch.bool), cd_lp, zeros)
        + torch.where(brt_causal.to(torch.bool), brt_lp, zeros)
        + md_lp
    )


def _sample_actions_from_policy_heads(
    obs: torch.Tensor,
    bt_l: torch.Tensor,
    bd_l: torch.Tensor,
    cd_l: torch.Tensor,
    brt_l: torch.Tensor,
    md_l: torch.Tensor,
    head_temperature: torch.Tensor,
    h_next: torch.Tensor,
    *,
    temperature: float,
    return_log_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Apply masks, sample actions, and optionally compose behavior log-probs."""
    B = obs.shape[0]
    bt_m, bd_m = compute_action_masks(obs)
    cd_m = conveyor_dir_mask(obs.device, B)
    # Sample direction first so we can condition build-type on the target tile.
    bd_act, bd_lp = sample_logits_gumbel(
        bd_l,
        mask=bd_m,
        temperature=head_temperature[:, 1] * temperature,
        return_log_prob=return_log_prob,
    )
    # Harvester only legal when chosen direction points at ore.
    bt_m[:, HARVESTER_BUILD_TYPE] &= bd_has_ore(obs, bd_act)
    bt_act, bt_lp = sample_logits_gumbel(
        bt_l,
        mask=bt_m,
        temperature=head_temperature[:, 0] * temperature,
        return_log_prob=return_log_prob,
    )
    cd_act, cd_lp = sample_logits_gumbel(
        cd_l,
        mask=cd_m,
        temperature=head_temperature[:, 2] * temperature,
        return_log_prob=return_log_prob,
    )
    brt_act, brt_lp = sample_logits_gumbel(
        brt_l,
        temperature=head_temperature[:, 3] * temperature,
        return_log_prob=return_log_prob,
    )
    md_act, md_lp = sample_logits_gumbel(
        md_l,
        temperature=head_temperature[:, 4] * temperature,
        return_log_prob=return_log_prob,
    )

    lp = None
    if return_log_prob:
        assert bt_lp is not None and bd_lp is not None and cd_lp is not None and brt_lp is not None and md_lp is not None
        # bd is sampled unconditionally (upstream of bt), so always include it.
        cd_causal = (
            (bt_act == CONVEYOR_BUILD_TYPES[0])
            | (bt_act == CONVEYOR_BUILD_TYPES[1])
            | (bt_act == CONVEYOR_BUILD_TYPES[2])
        )
        brt_causal = bt_act == BRIDGE_BUILD_TYPE
        lp = _compose_action_log_prob(
            bt_lp,
            bd_lp,
            cd_lp,
            brt_lp,
            md_lp,
            cd_causal=cd_causal,
            brt_causal=brt_causal,
        )
    return bt_act, bd_act, cd_act, brt_act, md_act, lp, h_next


def _normalize_actor_latent_dim(latent: torch.Tensor) -> torch.Tensor:
    """Pad/slice actor latents so critic input width is stable across models."""
    cur_dim = latent.shape[-1]
    if cur_dim == ACTOR_LATENT_DIM:
        return latent
    if cur_dim > ACTOR_LATENT_DIM:
        return latent[:, :ACTOR_LATENT_DIM]
    pad = latent.new_zeros(latent.shape[0], ACTOR_LATENT_DIM - cur_dim)
    return torch.cat([latent, pad], dim=-1)


class GraphedInference:
    """CUDA-graphed full inference pipeline: model forward + masks + sampling.

    Captures the entire forward+mask+sample path in a CUDA graph, eliminating
    all CPU kernel launch overhead (~100 launches → 1 graph replay).
    Pads inputs to MAX_B and slices outputs to the actual batch size.
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch: int = 16,
        temperature: float = 1.0,
        *,
        return_log_prob: bool = True,
        graph_sampling: bool = True,
    ):
        self.max_batch = max_batch
        self._model = model
        self._temperature = temperature
        self._return_log_prob = return_log_prob
        self._graph_sampling = graph_sampling
        self._graph: torch.cuda.CUDAGraph | None = None
        # Static buffers (set during capture)
        self._static_input: torch.Tensor | None = None
        # Output references (point into graph memory)
        self._s_bt_act = self._s_bd_act = self._s_cd_act = self._s_md_act = None
        self._s_lp = self._s_h = None
        self._s_bt_l = self._s_bd_l = self._s_cd_l = self._s_brt_l = self._s_md_l = self._s_head_temperature = None

    def capture(self) -> None:
        """Capture the CUDA graph. Call once after model is on device."""
        M = self.max_batch
        # Match model dtype for all buffers
        mdtype = next(self._model.parameters()).dtype
        self._dtype = mdtype
        self._static_input = torch.zeros(M, OBS_SIZE + H_DIM, device=DEVICE, dtype=mdtype)
        if not self._graph_sampling:
            with torch.no_grad():
                self._model.infer_packed(self._static_input)
            torch.cuda.synchronize()

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                with torch.no_grad():
                    self._s_bt_l, self._s_bd_l, self._s_cd_l, self._s_brt_l, self._s_md_l, self._s_head_temperature, self._s_h = self._model.infer_packed(self._static_input)
            return

        costs_ti, costs_ax, wall_idx, nobld_idx, unit_idx, friendly_idx, core_idx = _mask_indices(DEVICE)
        ore_ti_idx, ore_ax_idx = _harv_mask_indices(DEVICE)
        cd_mask = torch.tensor(_CONV_DIR_MASK, dtype=torch.bool, device=DEVICE)
        static_g_bt = torch.empty(M, N_BUILD, device=DEVICE, dtype=mdtype)
        static_g_bd = torch.empty(M, N_DIR, device=DEVICE, dtype=mdtype)
        static_g_cd = torch.empty(M, N_CONV_DIR, device=DEVICE, dtype=mdtype)
        static_g_brt = torch.empty(M, N_BRIDGE_TARGETS, device=DEVICE, dtype=mdtype)
        static_g_md = torch.empty(M, N_DIR, device=DEVICE, dtype=mdtype)

        # Warmup
        with torch.no_grad():
            self._model.infer_packed(self._static_input)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            with torch.no_grad():
                bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, s_h = self._model.infer_packed(self._static_input)
                obs = self._static_input[:, :OBS_SIZE]

                ti = obs[:, _OBS_TI] * 500.0
                ax = obs[:, _OBS_AX] * 500.0
                scale = obs[:, _OBS_SCALE] * 5.0
                actual_ti = (scale.unsqueeze(1) * costs_ti.unsqueeze(0)).floor()
                actual_ax = (scale.unsqueeze(1) * costs_ax.unsqueeze(0)).floor()
                bt_m = (ti.unsqueeze(1) >= actual_ti) & (ax.unsqueeze(1) >= actual_ax)
                bt_m[:, 0] = True
                not_wall = obs[:, wall_idx] == 0.0
                no_unit = obs[:, unit_idx] == 0.0
                empty = obs[:, nobld_idx] == 1.0
                replaceable = (obs[:, nobld_idx] == 0.0) & (obs[:, friendly_idx] == 1.0) & (obs[:, core_idx] == 0.0)
                bd_m = not_wall & no_unit & (empty | replaceable)
                force_none = ~bt_m[:, 1:].any(dim=1) | ~bd_m.any(dim=1)
                bt_m[:, 1:] &= ~force_none.unsqueeze(1)
                bd_m |= force_none.unsqueeze(1)

                # Sample direction first, then condition build-type on ore.
                bd_l.masked_fill_(~bd_m, -1e9)
                bd_l = _temperature_scaled_log_probs(
                    bd_l,
                    bd_m,
                    head_temperature[:, 1] * self._temperature,
                )
                md_l = _temperature_scaled_log_probs(
                    md_l,
                    temperature=head_temperature[:, 4] * self._temperature,
                )

                static_g_bd.exponential_()
                s_bd_act = (bd_l - static_g_bd.log_()).argmax(dim=-1)

                # Harvester only legal when chosen direction has ore.
                bt_m[:, HARVESTER_BUILD_TYPE] &= bd_has_ore(obs, s_bd_act)

                bt_l.masked_fill_(~bt_m, -1e9)
                cd_l.masked_fill_(~cd_mask, -1e9)
                bt_l = _temperature_scaled_log_probs(
                    bt_l,
                    bt_m,
                    head_temperature[:, 0] * self._temperature,
                )
                cd_l = _temperature_scaled_log_probs(
                    cd_l,
                    cd_mask,
                    head_temperature[:, 2] * self._temperature,
                )
                brt_l = _temperature_scaled_log_probs(
                    brt_l,
                    temperature=head_temperature[:, 3] * self._temperature,
                )

                static_g_bt.exponential_()
                s_bt_act = (bt_l - static_g_bt.log_()).argmax(dim=-1)

                static_g_cd.exponential_()
                static_g_brt.exponential_()
                static_g_md.exponential_()
                s_cd_act = (cd_l - static_g_cd.log_()).argmax(dim=-1)
                s_brt_act = (brt_l - static_g_brt.log_()).argmax(dim=-1)
                s_md_act = (md_l - static_g_md.log_()).argmax(dim=-1)

                s_lp = None
                if self._return_log_prob:
                    # bt_l etc. are already masked log-probs — just gather.
                    s_bt_lp = bt_l.gather(1, s_bt_act.unsqueeze(1)).squeeze(1)
                    s_bd_lp = bd_l.gather(1, s_bd_act.unsqueeze(1)).squeeze(1)
                    s_cd_lp = cd_l.gather(1, s_cd_act.unsqueeze(1)).squeeze(1)
                    s_brt_lp = brt_l.gather(1, s_brt_act.unsqueeze(1)).squeeze(1)
                    s_md_lp = md_l.gather(1, s_md_act.unsqueeze(1)).squeeze(1)

                    cd_causal = (
                        (s_bt_act == CONVEYOR_BUILD_TYPES[0])
                        | (s_bt_act == CONVEYOR_BUILD_TYPES[1])
                        | (s_bt_act == CONVEYOR_BUILD_TYPES[2])
                    )
                    brt_causal = s_bt_act == BRIDGE_BUILD_TYPE
                    s_lp = _compose_action_log_prob(
                        s_bt_lp,
                        s_bd_lp,
                        s_cd_lp,
                        s_brt_lp,
                        s_md_lp,
                        cd_causal=cd_causal,
                        brt_causal=brt_causal,
                    )

        self._s_bt_act = s_bt_act
        self._s_bd_act = s_bd_act
        self._s_cd_act = s_cd_act
        self._s_brt_act = s_brt_act
        self._s_md_act = s_md_act
        self._s_lp = s_lp
        self._s_h = s_h

    @torch.no_grad()
    def policy_heads(
        self,
        trunk_in: torch.Tensor,
        *,
        trace=None,
        scope_prefix: str = "inference",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw actor policy heads and recurrent state after the model forward."""
        B = trunk_in.shape[0]
        if self._graph is not None and not self._graph_sampling:
            self._static_input[:B].copy_(trunk_in)
            if B < self.max_batch:
                self._static_input[B:].zero_()
            with trace_scope(trace, f"{scope_prefix}.forward"):
                self._graph.replay()
            return (
                self._s_bt_l[:B].clone(),
                self._s_bd_l[:B].clone(),
                self._s_cd_l[:B].clone(),
                self._s_brt_l[:B].clone(),
                self._s_md_l[:B].clone(),
                self._s_head_temperature[:B].clone(),
                self._s_h[:B].clone(),
            )
        with trace_scope(trace, f"{scope_prefix}.forward"):
            return self._model.infer_packed(trunk_in)

    @torch.no_grad()
    def __call__(
        self,
        trunk_in: torch.Tensor,
        *,
        trace=None,
        scope_prefix: str = "inference",
    ):
        """Run graphed inference. Returns (bt, bd, cd, brt, md, lp, h_next) sliced to B."""
        B = trunk_in.shape[0]
        if self._graph is not None:
            self._static_input[:B].copy_(trunk_in)
            if B < self.max_batch:
                self._static_input[B:].zero_()
            with trace_scope(trace, f"{scope_prefix}.forward"):
                self._graph.replay()
            if not self._graph_sampling:
                bt_l = self._s_bt_l[:B].clone()
                bd_l = self._s_bd_l[:B].clone()
                cd_l = self._s_cd_l[:B].clone()
                brt_l = self._s_brt_l[:B].clone()
                md_l = self._s_md_l[:B].clone()
                head_temperature = self._s_head_temperature[:B].clone()
                h_next = self._s_h[:B].clone()
                with trace_scope(trace, f"{scope_prefix}.sample"):
                    return _sample_actions_from_policy_heads(
                        trunk_in[:, :OBS_SIZE],
                        bt_l,
                        bd_l,
                        cd_l,
                        brt_l,
                        md_l,
                        head_temperature,
                        h_next,
                        temperature=self._temperature,
                        return_log_prob=self._return_log_prob,
                    )
            return (
                self._s_bt_act[:B], self._s_bd_act[:B], self._s_cd_act[:B],
                self._s_brt_act[:B], self._s_md_act[:B],
                self._s_lp[:B] if self._s_lp is not None else None,
                self._s_h[:B],
            )
        obs = trunk_in[:, :OBS_SIZE]
        bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_next = self.policy_heads(
            trunk_in,
            trace=trace,
            scope_prefix=scope_prefix,
        )
        with trace_scope(trace, f"{scope_prefix}.sample"):
            return _sample_actions_from_policy_heads(
                obs,
                bt_l,
                bd_l,
                cd_l,
                brt_l,
                md_l,
                head_temperature,
                h_next,
                temperature=self._temperature,
                return_log_prob=self._return_log_prob,
            )

    @property
    def captured(self) -> bool:
        return self._graph is not None

    def disable_graph(self) -> None:
        self._graph = None


class BuilderMLP(nn.Module):
    """CLS-attention actor with a dedicated recurrent update path.

    The optional ``hidden`` argument is retained for API compatibility with the
    previous trunked actor, but the architecture now has a fixed 32-dim actor
    latent exported directly to the critic.
    """

    def __init__(self, hidden: list[int] | None = None):
        super().__init__()
        self._hidden = hidden or HIDDEN
        if hidden is None or hidden == HIDDEN:
            self._actor_cls_tokens = ACTOR_CLS_TOKENS
            self._actor_latent_dim = ACTOR_LATENT_DIM
        else:
            requested_latent = max(TILE_EMBED_DIM, hidden[-1])
            self._actor_cls_tokens = max(
                ACTOR_CLS_TOKENS,
                math.ceil(requested_latent / TILE_EMBED_DIM),
            )
            self._actor_latent_dim = self._actor_cls_tokens * TILE_EMBED_DIM
        self._recurrent_input_dim = self._actor_latent_dim + H_DIM
        n_actor_residual_layers = 5

        self.tile_linear = _parameter_golf_ortho_init(nn.Linear(TILE_INPUT_DIM, TILE_EMBED_DIM))
        self.context_proj = _parameter_golf_ortho_init(nn.Linear(GLOBAL_FEATURES, ACTOR_CONTEXT_DIM))
        self.context_k_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.context_v_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.tile_sa_q_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.tile_sa_k_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.tile_sa_v_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.cls_ca_q_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.cls_ca_k_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.cls_ca_v_proj = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))

        self.context_ca_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_sa_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_sa_q_norm = RMSNorm(TILE_EMBED_DIM)
        self.cls_ca_norm = RMSNorm(TILE_EMBED_DIM)
        self.latent_norm = RMSNorm(self._actor_latent_dim)
        self.context_ca_input_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_ffn_out_norm = RMSNorm(TILE_EMBED_DIM)
        self.cls_ca_input_norm = RMSNorm(TILE_EMBED_DIM)

        self.actor_cls = nn.Parameter(torch.zeros(self._actor_cls_tokens, TILE_EMBED_DIM))
        self.context_tau = nn.Parameter(torch.zeros(self._actor_cls_tokens))
        self.tile_sa_gamma = nn.Parameter(torch.zeros(1))  # exp(0) = 1.0 init
        # SuSIE additive position for tile self-attention: sinusoidal table
        # scaled by a learnable per-freq weight (tied across sin/cos).
        self.tile_sa_pos_weight = nn.Parameter(torch.full((TILE_EMBED_DIM // 2,), 0.5))
        # Look-away register: small scaled-Gaussian init so the register's
        # softmax score isn't a free zero-centered sink. Zero-init would give
        # ||register_k||² ≈ 0 under softmax RBF, making register score ≈ 0
        # while real tiles score `2γ⟨q,k⟩ - γ||k||² ≈ -D` — the sink then
        # dominates attention mass and starves downstream blocks of spatial
        # signal. randn * D^-0.5 matches the ref's standard-softmax register init.
        self.tile_sa_lookaway = nn.Parameter(
            torch.randn(TILE_EMBED_DIM) * (TILE_EMBED_DIM ** -0.5)
        )
        self.register_buffer(
            "tile_sa_susie", _make_tile_susie(TILE_GRID_N, TILE_EMBED_DIM), persistent=False
        )
        self.cls_ca_tau = nn.Parameter(torch.zeros(self._actor_cls_tokens))  # per-seed temperature
        self.context_resid_mix = nn.Parameter(
            torch.stack((torch.ones(TILE_EMBED_DIM), torch.zeros(TILE_EMBED_DIM))).float()
        )
        self.cls_resid_mix = nn.Parameter(
            torch.stack((torch.ones(TILE_EMBED_DIM), torch.zeros(TILE_EMBED_DIM))).float()
        )

        tile_ffn_hidden = TILE_EMBED_DIM * 2
        self.tile_ffn_norm = RMSNorm(TILE_EMBED_DIM)
        self.tile_ffn_up = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, tile_ffn_hidden))
        self.tile_ffn_down = _parameter_golf_ortho_init(
            nn.Linear(tile_ffn_hidden, TILE_EMBED_DIM),
            n_residual_layers=n_actor_residual_layers,
            scale_proj=True,
        )

        latent_ffn_hidden = self._actor_latent_dim * 2
        self.latent_ffn_norm = RMSNorm(self._actor_latent_dim)
        self.latent_ffn_out_norm = RMSNorm(self._actor_latent_dim)
        self.latent_ffn_up = _parameter_golf_ortho_init(nn.Linear(self._actor_latent_dim, latent_ffn_hidden))
        self.latent_ffn_down = _parameter_golf_ortho_init(
            nn.Linear(latent_ffn_hidden, self._actor_latent_dim),
            n_residual_layers=n_actor_residual_layers,
            scale_proj=True,
        )

        self.head_build_type = _ortho_init(nn.Linear(self._actor_latent_dim, N_BUILD), std=0.01)
        self.head_build_dir = _ortho_init(nn.Linear(self._actor_latent_dim, N_DIR), std=0.01)
        self.head_conv_dir = _ortho_init(nn.Linear(self._actor_latent_dim, N_CONV_DIR), std=0.01)
        self.head_bridge_target = _ortho_init(nn.Linear(self._actor_latent_dim, N_BRIDGE_TARGETS), std=0.01)
        self.head_move_dir = _ortho_init(nn.Linear(self._actor_latent_dim, N_DIR), std=0.01)
        self.head_temperature = _ortho_init(nn.Linear(self._actor_latent_dim, N_ACTION_TEMPERATURES), std=0.01)
        self.head_h_gate = _ortho_init(nn.Linear(self._recurrent_input_dim, H_DIM), std=0.01)
        self.head_h_cand = _ortho_init(nn.Linear(self._recurrent_input_dim, H_DIM), std=0.01)

        self.register_buffer("_tile_dxdy", _TILE_DXDY.clone())

        with torch.no_grad():
            trunc_normal_(self.actor_cls, std=0.02)
            self.head_build_type.bias.copy_(
                torch.tensor(INITIAL_BUILD_TYPE_BIAS, dtype=torch.float32),
            )
            self.head_temperature.bias.fill_(_temperature_head_init_bias())

    def train(self, mode=True):
        if mode:
            for attr in ("_cached_action_head_w", "_cached_action_head_b"):
                if hasattr(self, attr):
                    delattr(self, attr)
        return super().train(mode)

    def _context_cross_attention(
        self,
        cls: torch.Tensor,
        context_tokens: torch.Tensor,
        cls0: torch.Tensor,
    ) -> torch.Tensor:
        cls = residual_mix_input(cls, cls0, self.context_resid_mix)
        # Fold per-token tau into q so SDPA uses scale=1.0 (CUDA-graph safe).
        q = F.normalize(self.context_ca_input_norm(cls), dim=-1, eps=1e-6)
        q = q * self.context_tau.exp().view(1, -1, 1)
        k = F.normalize(self.context_k_proj(context_tokens), dim=-1, eps=1e-6)
        v = self.context_v_proj(context_tokens)
        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), scale=1.0,
        ).squeeze(1)
        return F.silu(cls + self.context_ca_norm(attn_out))

    def _tile_self_attention(self, tiles: torch.Tensor) -> torch.Tensor:
        """Peri-LN softmax RBF SA with SuSIE position and a look-away register.

        Weights are softmax(2γ⟨q,k⟩ - γ‖k‖²) — equivalent to softmax(-γ‖q-k‖²)
        since softmax is translation-invariant and cancels the ‖q‖² constant.
        Position is additive SuSIE sinusoids on q and k. A zero-initialized
        register token is prepended so the softmax has a competing "look-away"
        sink: queries with no nearby real tile concentrate weight on the
        register, whose projected V is near zero at init.
        """
        B, N, D = tiles.shape
        lookaway = self.tile_sa_lookaway.view(1, 1, D).expand(B, 1, D)
        seq = torch.cat([lookaway, tiles], dim=1)  # (B, N+1, D)

        normed = self.tile_sa_norm(seq)
        q = self.tile_sa_q_norm(self.tile_sa_q_proj(normed))
        k = self.tile_sa_k_proj(normed)
        v = self.tile_sa_v_proj(normed)

        pos_weight = self.tile_sa_pos_weight.repeat(2).to(q.dtype)  # (D,)
        pos_real = self.tile_sa_susie.to(q.dtype) * pos_weight  # (N, D)
        pos_emb = F.pad(pos_real, (0, 0, 1, 0))  # (N+1, D), zero row for register
        q = q + pos_emb
        k = k + pos_emb

        gamma = self.tile_sa_gamma.exp()
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)  # (B, 1, N+1)
        qk = torch.matmul(q, k.transpose(-1, -2))  # (B, N+1, N+1)
        scores = 2.0 * gamma * qk - gamma * k_norm_sq
        weights = torch.softmax(scores, dim=-1)

        attn_out = torch.matmul(weights, v)  # (B, N+1, D)
        attn_out = attn_out[:, 1:, :]  # drop register output

        # Softmax already bounds attn_out to the convex hull of V, so no
        # post-norm needed — the residual stream stays well-scaled.
        tiles = tiles + attn_out
        tiles = tiles + self.tile_ffn_out_norm(
            self.tile_ffn_down(ffn_activation(self.tile_ffn_up(self.tile_ffn_norm(tiles)))),
        )
        return tiles

    def _cls_cross_attention(
        self,
        cls: torch.Tensor,
        tiles: torch.Tensor,
        cls0: torch.Tensor,
    ) -> torch.Tensor:
        cls = residual_mix_input(cls, cls0, self.cls_resid_mix)
        q = F.normalize(self.cls_ca_q_proj(self.cls_ca_input_norm(cls)), dim=-1, eps=1e-6)
        q = q * self.cls_ca_tau.exp().view(1, -1, 1)  # per-seed temperature
        k = F.normalize(self.cls_ca_k_proj(tiles), dim=-1, eps=1e-6)
        v = self.cls_ca_v_proj(tiles)
        attn_out = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), scale=1.0,
        ).squeeze(1)
        return F.silu(cls + self.cls_ca_norm(attn_out))

    def _latent_ffn(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = ffn_activation(self.latent_ffn_up(self.latent_ffn_norm(latent)))
        return latent + self.latent_ffn_out_norm(self.latent_ffn_down(hidden))

    def _encode_tiles(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        tile_flat = obs[:, :TILE_GRID_N * TILE_FEATURES]
        tiles = tile_flat.reshape(B, TILE_GRID_N, TILE_FEATURES)
        dxdy = self._tile_dxdy.unsqueeze(0).expand(B, -1, -1)
        tile_in = torch.cat([dxdy, tiles], dim=-1)
        return F.silu(self.tile_linear(tile_in))

    def _obs_encoder_pre(self, obs: torch.Tensor) -> torch.Tensor:
        tiles = self._encode_tiles(obs).reshape(obs.shape[0], TILE_GRID_N * TILE_EMBED_DIM)
        globals_obs = obs[:, TILE_GRID_N * TILE_FEATURES:]
        global_ctx = self.context_proj(globals_obs)
        return torch.cat([tiles, global_ctx], dim=-1)

    def _split_obs_pre(self, obs_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tiles_flat, global_ctx = obs_pre.split([TILE_GRID_N * TILE_EMBED_DIM, ACTOR_CONTEXT_DIM], dim=-1)
        tiles = tiles_flat.reshape(obs_pre.shape[0], TILE_GRID_N, TILE_EMBED_DIM)
        return tiles, global_ctx.reshape(obs_pre.shape[0], ACTOR_CONTEXT_CHUNKS, TILE_EMBED_DIM)

    def _latent_from_obs_pre(self, obs_pre: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        tiles, global_ctx = self._split_obs_pre(obs_pre)
        cls0 = self.actor_cls.unsqueeze(0).expand(obs_pre.shape[0], -1, -1)
        cls = cls0
        recurrent_ctx = h_prev.reshape(obs_pre.shape[0], ACTOR_CONTEXT_CHUNKS, TILE_EMBED_DIM)
        cls = self._context_cross_attention(cls, torch.cat([global_ctx, recurrent_ctx], dim=1), cls0)
        tiles = self._tile_self_attention(tiles)
        cls = self._cls_cross_attention(cls, tiles, cls0)
        latent = self._latent_ffn(cls.reshape(obs_pre.shape[0], self._actor_latent_dim))
        return self.latent_norm(latent)

    def actor_latent(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        obs_pre = self._obs_encoder_pre(x)
        return self._latent_from_obs_pre(obs_pre, h_prev)

    def _project_policy_heads(
        self,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            combined_w = torch.cat([
                self.head_build_type.weight,
                self.head_build_dir.weight,
                self.head_conv_dir.weight,
                self.head_bridge_target.weight,
                self.head_move_dir.weight,
                self.head_temperature.weight,
            ], dim=0)
            combined_b = torch.cat([
                self.head_build_type.bias,
                self.head_build_dir.bias,
                self.head_conv_dir.bias,
                self.head_bridge_target.bias,
                self.head_move_dir.bias,
                self.head_temperature.bias,
            ])
        else:
            if not hasattr(self, "_cached_action_head_w"):
                self._cached_action_head_w = torch.cat([
                    self.head_build_type.weight,
                    self.head_build_dir.weight,
                    self.head_conv_dir.weight,
                    self.head_bridge_target.weight,
                    self.head_move_dir.weight,
                    self.head_temperature.weight,
                ], dim=0)
                self._cached_action_head_b = torch.cat([
                    self.head_build_type.bias,
                    self.head_build_dir.bias,
                    self.head_conv_dir.bias,
                    self.head_bridge_target.bias,
                    self.head_move_dir.bias,
                    self.head_temperature.bias,
                ])
            combined_w = self._cached_action_head_w
            combined_b = self._cached_action_head_b

        combined = F.linear(latent, combined_w, combined_b)
        bt_l, bd_l, cd_l, brt_l, md_l, temp_raw = combined.split(
            [N_BUILD, N_DIR, N_CONV_DIR, N_BRIDGE_TARGETS, N_DIR, N_ACTION_TEMPERATURES],
            dim=-1,
        )
        log_temp = torch.sigmoid(temp_raw) * (math.log(TEMP_MAX) - math.log(TEMP_MIN)) + math.log(TEMP_MIN)
        head_temperature = log_temp.exp()
        return bt_l, bd_l, cd_l, brt_l, md_l, head_temperature

    def _project_recurrent_heads(
        self,
        latent: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_in = torch.cat([latent, h_prev], dim=-1)
        h_gate = self.head_h_gate(recurrent_in)
        h_cand = torch.tanh(self.head_h_cand(recurrent_in))
        return h_gate, h_cand

    def _forward_from_obs_pre(self, obs_pre: torch.Tensor, h_prev: torch.Tensor):
        latent = self._latent_from_obs_pre(obs_pre, h_prev)
        bt_l, bd_l, cd_l, brt_l, md_l, head_temperature = self._project_policy_heads(latent)
        h_gate, h_cand = self._project_recurrent_heads(latent, h_prev)
        return bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_gate, h_cand

    def forward(self, x, h_prev):
        obs_pre = self._obs_encoder_pre(x)
        bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_gate, h_cand = self._forward_from_obs_pre(obs_pre, h_prev)
        z = torch.sigmoid(h_gate)
        h_next = (1.0 - z) * h_prev + z * h_cand
        return bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_next

    @torch.no_grad()
    def mu_center_policy_heads(self) -> None:
        """Post-optimizer mu-centering projection on policy heads.

        Subtracts the mean row from each head's weight matrix and the mean
        from each bias vector.  Because softmax is shift-invariant this does
        not change the model's output distribution — it only constrains the
        weight space to prevent unbounded logit drift under Adam.

        Reference: Stollenwerk et al., "Output Embedding Centering", 2025.
        """
        for head in (
            self.head_build_type,
            self.head_build_dir,
            self.head_conv_dir,
            self.head_bridge_target,
            self.head_move_dir,
        ):
            head.weight.sub_(head.weight.mean(dim=0, keepdim=True))
            head.bias.sub_(head.bias.mean())

    def infer_packed(self, packed: torch.Tensor):
        """Inference-only path for already-packed [obs, h_prev] batches."""
        obs = packed[:, :OBS_SIZE]
        h_prev = packed[:, OBS_SIZE:]
        obs_pre = self._obs_encoder_pre(obs)
        bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_gate, h_cand = self._forward_from_obs_pre(obs_pre, h_prev)
        z = torch.sigmoid(h_gate)
        h_next = (1.0 - z) * h_prev + z * h_cand
        return bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_next

    def _policy_state_entries(self) -> list[tuple[str, object]]:
        return [
            ("linear", self.tile_linear),
            ("linear", self.context_proj),
            ("linear", self.context_k_proj),
            ("linear", self.context_v_proj),
            ("linear", self.tile_sa_k_proj),
            ("linear", self.tile_sa_v_proj),
            ("linear", self.cls_ca_k_proj),
            ("linear", self.cls_ca_v_proj),
            ("param", self.actor_cls),
            ("param", self.context_tau),
            ("norm", self.context_ca_norm),
            ("norm", self.tile_sa_norm),
            ("norm", self.cls_ca_norm),
            ("norm", self.latent_norm),
            ("linear", self.head_build_type),
            ("linear", self.head_build_dir),
            ("linear", self.head_conv_dir),
            ("linear", self.head_bridge_target),
            ("linear", self.head_move_dir),
            ("linear", self.head_temperature),
            ("linear", self.head_h_gate),
            ("linear", self.head_h_cand),
            ("linear", self.tile_sa_q_proj),
            ("linear", self.cls_ca_q_proj),
            ("param", self.tile_sa_gamma),
            ("param", self.cls_ca_tau),
            ("norm", self.latent_ffn_norm),
            ("linear", self.latent_ffn_up),
            ("linear", self.latent_ffn_down),
            ("norm", self.tile_ffn_norm),
            ("linear", self.tile_ffn_up),
            ("linear", self.tile_ffn_down),
            ("norm", self.context_ca_input_norm),
            ("norm", self.tile_ffn_out_norm),
            ("norm", self.cls_ca_input_norm),
            ("norm", self.latent_ffn_out_norm),
            ("param", self.context_resid_mix),
            ("param", self.cls_resid_mix),
            ("param", self.tile_sa_pos_weight),
            ("param", self.tile_sa_lookaway),
            ("norm", self.tile_sa_q_norm),
        ]

    def export_weights(self) -> list[list[float]]:
        """Export weights as JSON list. Transposes Linear to match tensor_lib."""
        state: list[list[float]] = []
        for kind, obj in self._policy_state_entries():
            if kind == "linear":
                layer = obj
                state.append(layer.weight.T.contiguous().view(-1).tolist())
                state.append(layer.bias.view(-1).tolist())
            elif kind == "norm":
                layer = obj
                state.append(layer.weight.view(-1).tolist())
            else:
                param = obj
                state.append(param.view(-1).tolist())
        return state

    def import_weights(self, state: list[list[float]]) -> None:
        """Load weights from tensor_lib JSON format."""
        entries = self._policy_state_entries()
        expected = sum(2 if kind == "linear" else 1 for kind, _ in entries)
        supported = (
            expected,
            expected - 1,
            expected - 3,
            expected - 5,
            expected - 9,
            expected - 14,
            expected - 20,
        )
        if len(state) not in supported:
            raise AssertionError(
                f"State has {len(state)} entries, expected one of {supported}"
            )
        dev = next(self.parameters()).device
        if len(state) != expected:
            state = list(state) + self._legacy_policy_padding(len(state))
        idx = 0
        for kind, obj in entries:
            if kind == "linear":
                layer = obj
                w_flat = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                in_f, out_f = layer.in_features, layer.out_features
                layer.weight.data = w_flat.view(in_f, out_f).T.contiguous()
                idx += 1
                layer.bias.data = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                idx += 1
            elif kind == "norm":
                layer = obj
                layer.weight.data = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                idx += 1
            else:
                param = obj
                s = state[idx]
                if len(s) == 1 and param.numel() > 1:
                    s = s * param.numel()  # broadcast scalar to match new shape
                param.data = torch.tensor(s, dtype=torch.float32, device=dev).view_as(param)
                idx += 1

    def _legacy_policy_padding(self, state_len: int) -> list[list[float]]:
        """Deterministic compatibility padding for old inference schemas."""
        pad: list[list[float]] = []

        if state_len <= 43:
            pad.append([0.0] * self.cls_ca_tau.numel())
            pad.append([1.0] * self.latent_ffn_norm.weight.numel())
            pad.append([0.0] * (self.latent_ffn_up.in_features * self.latent_ffn_up.out_features))
            pad.append([0.0] * self.latent_ffn_up.out_features)
            pad.append([0.0] * (self.latent_ffn_down.in_features * self.latent_ffn_down.out_features))
            pad.append([0.0] * self.latent_ffn_down.out_features)

        if state_len <= 49:
            pad.append([1.0] * self.tile_ffn_norm.weight.numel())
            pad.append([0.0] * (self.tile_ffn_up.in_features * self.tile_ffn_up.out_features))
            pad.append([0.0] * self.tile_ffn_up.out_features)
            pad.append([0.0] * (self.tile_ffn_down.in_features * self.tile_ffn_down.out_features))
            pad.append([0.0] * self.tile_ffn_down.out_features)

        if state_len <= 54:
            for norm in (
                self.context_ca_input_norm,
                self.tile_ffn_out_norm,
                self.cls_ca_input_norm,
                self.latent_ffn_out_norm,
            ):
                pad.append([1.0] * norm.weight.numel())

        if state_len <= 58:
            pad.append([1.0] * TILE_EMBED_DIM + [0.0] * TILE_EMBED_DIM)
            pad.append([1.0] * TILE_EMBED_DIM + [0.0] * TILE_EMBED_DIM)

        if state_len <= 60:
            pad.append([0.5] * (TILE_EMBED_DIM // 2))  # tile_sa_pos_weight
            pad.append([0.0] * TILE_EMBED_DIM)          # tile_sa_lookaway

        if state_len <= 62:
            pad.append([1.0] * self.tile_sa_q_norm.weight.numel())  # tile_sa_q_norm

        return pad

    def _legacy_state_dict_defaults(self) -> dict[str, torch.Tensor]:
        """Compatibility defaults for parameters absent from old actor state_dicts."""
        defaults: dict[str, torch.Tensor] = {
            "cls_ca_tau": torch.zeros_like(self.cls_ca_tau),
            "latent_ffn_norm.weight": torch.ones_like(self.latent_ffn_norm.weight),
            "latent_ffn_up.weight": torch.zeros_like(self.latent_ffn_up.weight),
            "latent_ffn_up.bias": torch.zeros_like(self.latent_ffn_up.bias),
            "latent_ffn_down.weight": torch.zeros_like(self.latent_ffn_down.weight),
            "latent_ffn_down.bias": torch.zeros_like(self.latent_ffn_down.bias),
            "tile_ffn_norm.weight": torch.ones_like(self.tile_ffn_norm.weight),
            "tile_ffn_up.weight": torch.zeros_like(self.tile_ffn_up.weight),
            "tile_ffn_up.bias": torch.zeros_like(self.tile_ffn_up.bias),
            "tile_ffn_down.weight": torch.zeros_like(self.tile_ffn_down.weight),
            "tile_ffn_down.bias": torch.zeros_like(self.tile_ffn_down.bias),
            "context_ca_input_norm.weight": torch.ones_like(self.context_ca_input_norm.weight),
            "tile_ffn_out_norm.weight": torch.ones_like(self.tile_ffn_out_norm.weight),
            "cls_ca_input_norm.weight": torch.ones_like(self.cls_ca_input_norm.weight),
            "latent_ffn_out_norm.weight": torch.ones_like(self.latent_ffn_out_norm.weight),
            "context_resid_mix": torch.stack((
                torch.ones_like(self.context_resid_mix[0]),
                torch.zeros_like(self.context_resid_mix[1]),
            )),
            "cls_resid_mix": torch.stack((
                torch.ones_like(self.cls_resid_mix[0]),
                torch.zeros_like(self.cls_resid_mix[1]),
            )),
            "tile_sa_pos_weight": torch.full_like(self.tile_sa_pos_weight, 0.5),
            "tile_sa_lookaway": torch.zeros_like(self.tile_sa_lookaway),
            "tile_sa_q_norm.weight": torch.ones_like(self.tile_sa_q_norm.weight),
        }
        return defaults

    def _import_legacy_state_dict(self, data: dict[str, torch.Tensor]) -> None:
        """Load legacy actor state_dicts with deterministic defaults for new params."""
        current = self.state_dict()
        compat = {k: v.clone() for k, v in current.items()}
        compat.update({k: v.clone() for k, v in self._legacy_state_dict_defaults().items()})

        for key, value in data.items():
            if key not in compat:
                continue
            target = compat[key]
            if not isinstance(value, torch.Tensor):
                continue
            incoming = value.to(device=target.device, dtype=target.dtype)
            if incoming.shape == target.shape:
                compat[key] = incoming
            elif incoming.numel() == 1 and target.numel() > 1:
                compat[key] = incoming.reshape(1).expand_as(target).clone()

        self.load_state_dict(compat, strict=True)

    def save_weights(self, path: str | Path = WEIGHT_PATH) -> None:
        """Save weights. Uses JSON for .json, pickle list format for .pkl (bot compat)."""
        path = str(path)
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(self.export_weights(), f)
        else:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.export_weights(), f)

    def load_weights(self, path: str | Path = WEIGHT_PATH) -> bool:
        path = str(path)
        if not os.path.exists(path):
            return False
        try:
            if path.endswith(".json"):
                with open(path) as f:
                    self.import_weights(json.load(f))
            else:
                import pickle
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    # torch state_dict format (legacy, may have extra critic keys)
                    self._import_legacy_state_dict(data)
                else:
                    # list-of-lists format (current + old pickle)
                    self.import_weights(data)
            return True
        except Exception as e:
            print(f"Could not load weights from {path}: {e}")
            return False


# ---- PPO ----

BPTT_K = 32  # truncated BPTT unroll length


def _build_bptt_chunks(
    ep_boundaries: list[tuple[int, int]],
    bptt_k: int = BPTT_K,
) -> list[tuple[int, int]]:
    """Split episodes into non-overlapping (start, length) chunks for BPTT."""
    chunks: list[tuple[int, int]] = []
    for ep_start, ep_end in ep_boundaries:
        ep_len = ep_end - ep_start
        for t in range(0, ep_len, bptt_k):
            chunk_len = min(bptt_k, ep_len - t)
            chunks.append((ep_start + t, chunk_len))
    return chunks


def _chunk_metadata_tensors(
    chunks: list[tuple[int, int]],
    device: torch.device,
    *,
    sort_desc: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize chunk start/length metadata on-device."""
    chunk_starts = torch.tensor([start for start, _ in chunks], device=device)
    chunk_lens = torch.tensor([length for _, length in chunks], device=device)
    if sort_desc and chunk_lens.numel() > 1:
        order = chunk_lens.argsort(descending=True)
        chunk_starts = chunk_starts[order]
        chunk_lens = chunk_lens[order]
    return chunk_starts, chunk_lens


def _precompute_policy_masks(
    obs: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute policy masks and causal coefficients for fixed rollout actions."""
    bt_mask, bd_mask = compute_action_masks(obs)
    # Harvester legal only when chosen direction points at ore.
    bt_mask = bt_mask.clone()
    bt_mask[:, HARVESTER_BUILD_TYPE] &= bd_has_ore(obs, bd_act)
    cd_causal = (
        (bt_act == CONVEYOR_BUILD_TYPES[0])
        | (bt_act == CONVEYOR_BUILD_TYPES[1])
        | (bt_act == CONVEYOR_BUILD_TYPES[2])
    ).float()
    return bt_mask, bd_mask, cd_causal


def _split_chunk_ids(
    chunk_lens: torch.Tensor,
    bptt_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split chunk ids into full-length and tail-length groups."""
    all_ids = torch.arange(chunk_lens.shape[0], device=chunk_lens.device)
    full_ids = all_ids[chunk_lens == bptt_k]
    tail_ids = all_ids[chunk_lens != bptt_k]
    if tail_ids.numel() > 1:
        tail_ids = tail_ids[chunk_lens[tail_ids].argsort(descending=True)]
    return full_ids, tail_ids


def _logprob_recurrent_step(
    model: 'BuilderMLP',
    obs_pre_t: torch.Tensor,
    h_t: torch.Tensor,
    bt_act_t: torch.Tensor,
    bd_act_t: torch.Tensor,
    cd_act_t: torch.Tensor,
    brt_act_t: torch.Tensor,
    md_act_t: torch.Tensor,
    bt_mask_t: torch.Tensor,
    bd_mask_t: torch.Tensor,
    cd_mask: torch.Tensor,
    cd_causal_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One recurrent timestep: forward → GRU update → log-prob (no entropy)."""
    bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_gate, h_cand = model._forward_from_obs_pre(obs_pre_t, h_t)
    z = torch.sigmoid(h_gate)
    h_new = (1.0 - z) * h_t + z * h_cand

    bt_l = bt_l.masked_fill(~bt_mask_t, -1e9)
    bd_l = bd_l.masked_fill(~bd_mask_t, -1e9)
    cd_l = cd_l.masked_fill(~cd_mask, -1e9)

    bt_lp = _temperature_scaled_log_probs(bt_l, bt_mask_t, head_temperature[:, 0]).gather(1, bt_act_t.unsqueeze(1)).squeeze(1)
    bd_lp = _temperature_scaled_log_probs(bd_l, bd_mask_t, head_temperature[:, 1]).gather(1, bd_act_t.unsqueeze(1)).squeeze(1)
    cd_lp = _temperature_scaled_log_probs(cd_l, cd_mask, head_temperature[:, 2]).gather(1, cd_act_t.unsqueeze(1)).squeeze(1)
    brt_lp = _temperature_scaled_log_probs(brt_l, temperature=head_temperature[:, 3]).gather(1, brt_act_t.unsqueeze(1)).squeeze(1)
    md_lp = _temperature_scaled_log_probs(md_l, temperature=head_temperature[:, 4]).gather(1, md_act_t.unsqueeze(1)).squeeze(1)
    brt_causal = bt_act_t == BRIDGE_BUILD_TYPE
    lp = _compose_action_log_prob(
        bt_lp,
        bd_lp,
        cd_lp,
        brt_lp,
        md_lp,
        cd_causal=cd_causal_t,
        brt_causal=brt_causal,
    )
    return lp, h_new


_compiled_logprob_step: Callable | None = None


def _get_logprob_step_fn() -> Callable:
    """Get the (possibly compiled) logprob recurrent step function."""
    global _compiled_logprob_step
    if _compiled_logprob_step is None:
        _compiled_logprob_step = maybe_compile(
            _logprob_recurrent_step,
            _get_actor_cuda_config(),
            label="logprob PPO step",
        )
    return _compiled_logprob_step


def _recurrent_logprob_pass(
    model: 'BuilderMLP',
    obs_pre: torch.Tensor,
    h0: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    brt_act: torch.Tensor,
    md_act: torch.Tensor,
    bt_mask: torch.Tensor,
    bd_mask: torch.Tensor,
    cd_causal: torch.Tensor,
) -> torch.Tensor:
    """Sequential recurrent pass that returns chosen-action log-probs."""
    B, K_max, _ = obs_pre.shape
    dev = obs_pre.device
    h_t = h0
    log_probs = torch.empty(B, K_max, device=dev)
    cd_mask = conveyor_dir_mask(dev, B)

    step_fn = _get_logprob_step_fn()
    with autocast_context(_get_actor_cuda_config()):
        for t in range(K_max):
            log_probs[:, t], h_t = step_fn(
                model, obs_pre[:, t], h_t,
                bt_act[:, t], bd_act[:, t], cd_act[:, t], brt_act[:, t], md_act[:, t],
                bt_mask[:, t], bd_mask[:, t], cd_mask, cd_causal[:, t],
            )

    return log_probs


def _actor_recurrent_step(
    model: 'BuilderMLP',
    obs_pre_t: torch.Tensor,
    h_t: torch.Tensor,
    bt_act_t: torch.Tensor,
    bd_act_t: torch.Tensor,
    cd_act_t: torch.Tensor,
    brt_act_t: torch.Tensor,
    md_act_t: torch.Tensor,
    bt_mask_t: torch.Tensor,
    bd_mask_t: torch.Tensor,
    cd_mask: torch.Tensor,
    cd_causal_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One recurrent timestep: forward → GRU update → log-probs + entropy."""
    bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_gate, h_cand = model._forward_from_obs_pre(obs_pre_t, h_t)
    z = torch.sigmoid(h_gate)
    h_new = (1.0 - z) * h_t + z * h_cand

    bt_l = bt_l.masked_fill(~bt_mask_t, -1e9)
    bd_l = bd_l.masked_fill(~bd_mask_t, -1e9)
    cd_l = cd_l.masked_fill(~cd_mask, -1e9)

    bt_lp, bt_ent = action_log_prob_and_entropy(
        bt_l, bt_act_t, bt_mask_t,
        temperature=head_temperature[:, 0],
        entropy_temperature=head_temperature[:, 0].detach(),
    )
    bd_lp, bd_ent = action_log_prob_and_entropy(
        bd_l, bd_act_t, bd_mask_t,
        temperature=head_temperature[:, 1],
        entropy_temperature=head_temperature[:, 1].detach(),
    )
    cd_lp, cd_ent = action_log_prob_and_entropy(
        cd_l, cd_act_t, cd_mask,
        temperature=head_temperature[:, 2],
        entropy_temperature=head_temperature[:, 2].detach(),
    )
    brt_lp, brt_ent = action_log_prob_and_entropy(
        brt_l, brt_act_t,
        temperature=head_temperature[:, 3],
        entropy_temperature=head_temperature[:, 3].detach(),
    )
    md_lp, md_ent = action_log_prob_and_entropy(
        md_l, md_act_t,
        temperature=head_temperature[:, 4],
        entropy_temperature=head_temperature[:, 4].detach(),
    )
    brt_causal = bt_act_t == BRIDGE_BUILD_TYPE

    new_lp = _compose_action_log_prob(
        bt_lp,
        bd_lp,
        cd_lp,
        brt_lp,
        md_lp,
        cd_causal=cd_causal_t,
        brt_causal=brt_causal,
    )

    ent = (
        bt_ent
        + bd_ent
        + cd_ent * cd_causal_t
        + brt_ent * brt_causal
        + md_ent
    )
    return new_lp, ent, head_temperature, h_new


_compiled_actor_step: Callable | None = None
_actor_cuda_cfg: PretrainCudaConfig | None = None


def _get_actor_cuda_config() -> PretrainCudaConfig:
    global _actor_cuda_cfg
    if _actor_cuda_cfg is None:
        _actor_cuda_cfg = resolve_cuda_config(
            DEVICE,
            spatial_inputs=False,
            enable_compile=True,
        )
    return _actor_cuda_cfg


def _get_actor_step_fn() -> Callable:
    """Get the (possibly compiled) actor recurrent step function."""
    global _compiled_actor_step
    if _compiled_actor_step is None:
        _compiled_actor_step = maybe_compile(
            _actor_recurrent_step,
            _get_actor_cuda_config(),
            label="actor PPO step",
        )
    return _compiled_actor_step


def _recurrent_actor_pass(
    model: 'BuilderMLP',
    obs_pre: torch.Tensor,
    h0: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    brt_act: torch.Tensor,
    md_act: torch.Tensor,
    bt_mask: torch.Tensor,
    bd_mask: torch.Tensor,
    cd_causal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential recurrent pass that returns log-probs, entropies, and per-head temperatures."""
    B, K_max, _ = obs_pre.shape
    dev = obs_pre.device
    h_t = h0
    new_lps = torch.empty(B, K_max, device=dev)
    entropies = torch.empty(B, K_max, device=dev)
    all_temperatures = torch.empty(B, K_max, N_ACTION_TEMPERATURES, device=dev)
    cd_mask = conveyor_dir_mask(dev, B)

    step_fn = _get_actor_step_fn()
    with autocast_context(_get_actor_cuda_config()):
        for t in range(K_max):
            new_lps[:, t], entropies[:, t], all_temperatures[:, t], h_t = step_fn(
                model, obs_pre[:, t], h_t,
                bt_act[:, t], bd_act[:, t], cd_act[:, t], brt_act[:, t], md_act[:, t],
                bt_mask[:, t], bd_mask[:, t], cd_mask, cd_causal[:, t],
            )

    return new_lps, entropies, all_temperatures


def _sequential_logprobs(
    model: 'BuilderMLP',
    obs: torch.Tensor,
    h: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    brt_act: torch.Tensor,
    md_act: torch.Tensor,
    chunks: list[tuple[int, int]],
    bptt_k: int = BPTT_K,
    batch_size: int = 8192,
    obs_pre_all: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute log-probs with sequential h propagation (no grad).

    Batches across chunks at each timestep (same pattern as BPTT actor loop)
    for good GPU utilization. Chunks are sorted by length to minimize padding.
    """
    N = obs.shape[0]
    dev = obs.device
    log_probs = torch.zeros(N, device=dev)

    B_chunk = max(1, batch_size // bptt_k)
    all_starts, all_lens = _chunk_metadata_tensors(chunks, dev, sort_desc=True)
    full_chunk_ids, tail_chunk_ids = _split_chunk_ids(all_lens, bptt_k)
    if obs_pre_all is None:
        with autocast_context(_get_actor_cuda_config()):
            obs_pre_all = model._obs_encoder_pre(obs)
    bt_mask_all, bd_mask_all, cd_causal_all = _precompute_policy_masks(obs, bt_act, bd_act)
    t_range_full = torch.arange(bptt_k, device=dev).unsqueeze(0)

    def _run_chunk_group(chunk_ids: torch.Tensor, *, full_length: bool) -> None:
        for cb_start in range(0, chunk_ids.shape[0], B_chunk):
            batch_ids = chunk_ids[cb_start:cb_start + B_chunk]
            B = batch_ids.shape[0]
            chunk_starts = all_starts[batch_ids]
            chunk_lens = all_lens[batch_ids]
            if full_length:
                K_max = bptt_k
                t_range = t_range_full
            else:
                K_max = int(chunk_lens[0].item())
                t_range = t_range_full[:, :K_max]
            idx_grid = (chunk_starts.unsqueeze(1) + t_range).clamp(max=N - 1)
            flat_idx = idx_grid.reshape(-1)

            all_obs_pre = obs_pre_all[flat_idx].reshape(B, K_max, -1)
            all_bt = bt_act[flat_idx].reshape(B, K_max)
            all_bd = bd_act[flat_idx].reshape(B, K_max)
            all_cd = cd_act[flat_idx].reshape(B, K_max)
            all_brt = brt_act[flat_idx].reshape(B, K_max)
            all_md = md_act[flat_idx].reshape(B, K_max)
            all_bt_mask = bt_mask_all[flat_idx].reshape(B, K_max, -1)
            all_bd_mask = bd_mask_all[flat_idx].reshape(B, K_max, -1)
            all_cd_causal = cd_causal_all[flat_idx].reshape(B, K_max)

            step_lps = _recurrent_logprob_pass(
                model,
                all_obs_pre,
                h[chunk_starts],
                all_bt,
                all_bd,
                all_cd,
                all_brt,
                all_md,
                all_bt_mask,
                all_bd_mask,
                all_cd_causal,
            )

            valid_mask = t_range < chunk_lens.unsqueeze(1)
            valid_flat_idx = idx_grid[valid_mask]
            log_probs[valid_flat_idx] = step_lps[valid_mask]

    if full_chunk_ids.numel() > 0:
        _run_chunk_group(full_chunk_ids, full_length=True)
    if tail_chunk_ids.numel() > 0:
        _run_chunk_group(tail_chunk_ids, full_length=False)

    return log_probs


def ppo_update(
    model: BuilderMLP,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    h: torch.Tensor,
    obs_pre_all: torch.Tensor | None,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    brt_act: torch.Tensor,
    md_act: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    centralized_data: dict,
    old_values: torch.Tensor | None = None,
    ep_boundaries: list[tuple[int, int]] | None = None,
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    actor_minibatch_size: int | None = None,
    bptt_k: int = BPTT_K,
    clip_eps: float = 0.2,
    clip_eps_higher: float | None = 0.28,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.0,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    profile: dict[str, float] | None = None,
    trace=None,
    bptt_chunks: list[tuple[int, int]] | None = None,
    freeze_critic: bool = False,
    use_pmpo: bool = False,
) -> dict[str, float]:
    """PPO update with truncated BPTT for recurrent hidden state.

    Processes contiguous K-step windows from episodes. Within each window,
    h is recomputed sequentially so gradients flow through K timesteps.
    """
    N = obs.shape[0]
    dev = obs.device
    # GPU-side scalar accumulators — avoids per-minibatch .item() syncs.
    # Layout: [pi_loss, ent_loss, actor_gn, kl, clip_frac]
    _actor_acc = torch.zeros(5, device=dev)
    _critic_acc = torch.zeros(3, device=dev)  # [vf_loss, critic_gn, critic_delta]
    sum_temperature = torch.zeros(N_ACTION_TEMPERATURES, device=dev)
    sum_temperature_sq = torch.zeros(N_ACTION_TEMPERATURES, device=dev)
    n_temperature = 0
    n_updates = 0

    bucket_centers = _make_bucket_centers().to(device=obs.device)
    critic_cuda_cfg = None

    # Explained variance (computed on GPU; reduced to scalar at final batched sync)
    with torch.no_grad():
        if old_values is not None:
            ev_var_y = returns.var()
            ev_var_res = (returns - old_values).var()
        else:
            ev_var_y = torch.zeros((), device=dev)
            ev_var_res = torch.zeros((), device=dev)

    # Build BPTT chunks from episode boundaries
    if bptt_chunks is None:
        if ep_boundaries is None:
            ep_boundaries = [(0, N)]
        bptt_chunks = _build_bptt_chunks(ep_boundaries, bptt_k)

    # Pre-build chunk metadata as GPU tensors (avoids per-batch CPU-GPU sync)
    all_chunk_starts, all_chunk_lens = _chunk_metadata_tensors(bptt_chunks, dev)

    # PMPO uses sign-weighted policy updates, so advantage normalization
    # should be disabled on that path.
    if not use_pmpo:
        advantages = _normalize_advantages(advantages)
    with torch.no_grad():
        bt_mask_all, bd_mask_all, cd_causal_all = _precompute_policy_masks(obs, bt_act, bd_act)
    # obs is only used for mask precomputation when obs_pre_all is provided.
    # Drop both references so the (N, OBS_SIZE) rollout slab can be reclaimed
    # before actor/critic forward allocates new activations.
    if obs_pre_all is not None:
        centralized_data.pop("obs", None)
        obs = None
    full_chunk_ids, tail_chunk_ids = _split_chunk_ids(all_chunk_lens, bptt_k)

    # 8 minibatches per epoch: scales B_chunk with N so GPU utilization
    # stays high regardless of dataset size while keeping update count fixed.
    n_chunks = len(bptt_chunks)
    if actor_minibatch_size is not None:
        B_chunk = max(1, actor_minibatch_size // bptt_k)
    else:
        B_chunk = max(1, (n_chunks + 7) // 8)

    _hi = clip_eps_higher if clip_eps_higher is not None else clip_eps
    # Precompute max-length arange once; slice per-batch for tail chunks.
    t_range_full = torch.arange(bptt_k, device=dev).unsqueeze(0)

    # ---- Phase 1: Actor updates with BPTT ----
    actor_stopped_early = False
    actor_epochs_ran = 0
    for epoch in range(ppo_epochs):
        actor_epochs_ran += 1
        _epoch_kl_acc = torch.zeros(1, device=dev)
        n_mb = 0

        full_perm = full_chunk_ids
        if full_perm.numel() > 1:
            full_perm = full_perm[torch.randperm(full_perm.numel(), device=dev)]

        tail_perm = tail_chunk_ids
        if tail_perm.numel() > 1:
            tail_perm = tail_perm[torch.randperm(tail_perm.numel(), device=dev)]
            tail_perm = tail_perm[all_chunk_lens[tail_perm].argsort(descending=True, stable=True)]

        def _run_actor_group(chunk_ids: torch.Tensor, *, full_length: bool) -> None:
            nonlocal _epoch_kl_acc, n_mb, n_updates, sum_temperature, sum_temperature_sq, n_temperature
            for cb_start in range(0, chunk_ids.shape[0], B_chunk):
                with trace_scope(trace, "ppo.actor_minibatch"):
                    batch_ids = chunk_ids[cb_start:cb_start + B_chunk]
                    B = batch_ids.shape[0]
                    chunk_starts = all_chunk_starts[batch_ids]
                    chunk_lens = all_chunk_lens[batch_ids]
                    if full_length:
                        K_max = bptt_k
                        t_range = t_range_full
                    else:
                        K_max = int(chunk_lens[0].item())
                        t_range = t_range_full[:, :K_max]
                    idx_grid = (chunk_starts.unsqueeze(1) + t_range).clamp(max=N - 1)
                    flat_idx = idx_grid.reshape(-1)

                    if obs_pre_all is None:
                        all_obs_pre = model._obs_encoder_pre(obs[flat_idx]).reshape(B, K_max, -1)
                    else:
                        all_obs_pre = obs_pre_all[flat_idx].reshape(B, K_max, -1)
                    all_bt = bt_act[flat_idx].reshape(B, K_max)
                    all_bd = bd_act[flat_idx].reshape(B, K_max)
                    all_cd = cd_act[flat_idx].reshape(B, K_max)
                    all_brt = brt_act[flat_idx].reshape(B, K_max)
                    all_md = md_act[flat_idx].reshape(B, K_max)
                    all_old_lp = old_log_probs[flat_idx].reshape(B, K_max)
                    all_adv = advantages[flat_idx].reshape(B, K_max)
                    all_bt_mask = bt_mask_all[flat_idx].reshape(B, K_max, -1)
                    all_bd_mask = bd_mask_all[flat_idx].reshape(B, K_max, -1)
                    all_cd_causal = cd_causal_all[flat_idx].reshape(B, K_max)

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    new_lps, entropies, batch_temperature = _recurrent_actor_pass(
                        model,
                        all_obs_pre,
                        h[chunk_starts].detach(),
                        all_bt,
                        all_bd,
                        all_cd,
                        all_brt,
                        all_md,
                        all_bt_mask,
                        all_bd_mask,
                        all_cd_causal,
                    )
                    if profile is not None:
                        profile["ppo_forward_s"] = profile.get("ppo_forward_s", 0.0) + (time.perf_counter() - t_phase)

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    valid_mask = t_range < chunk_lens.unsqueeze(1)
                    inv_valid = ~valid_mask
                    valid_denom = valid_mask.sum()

                    # Zero padding BEFORE ratio computation to prevent inf*0=NaN.
                    # Padded positions have garbage log_probs that diverge across
                    # PPO epochs; exp(large) → inf, and inf*0.0 = NaN in IEEE 754.
                    new_lps = new_lps.masked_fill(inv_valid, 0.0)
                    all_old_lp = all_old_lp.masked_fill(inv_valid, 0.0)

                    logratio = new_lps - all_old_lp
                    ratio = logratio.exp()

                    mb_adv = all_adv.masked_fill(inv_valid, 0.0)
                    policy_weight = _policy_weight_from_advantages(mb_adv, use_pmpo=use_pmpo)

                    with torch.no_grad():
                        approx_kl = (((ratio - 1) - logratio) * valid_mask).sum() / valid_denom
                        _epoch_kl_acc += approx_kl
                        clip_frac = (((ratio < (1 - clip_eps)) | (ratio > (1 + _hi))) & valid_mask).sum() / valid_denom
                        _actor_acc[3] += approx_kl
                        _actor_acc[4] += clip_frac
                        n_mb += 1
                    pg1 = -policy_weight * ratio
                    pg2 = -policy_weight * torch.clamp(ratio, 1 - clip_eps, 1 + _hi)
                    pi_loss = (torch.max(pg1, pg2) * valid_mask).sum() / valid_denom
                    ent_loss = (entropies * valid_mask).sum() / valid_denom

                    if profile is not None:
                        profile["ppo_policy_loss_s"] = profile.get("ppo_policy_loss_s", 0.0) + (time.perf_counter() - t_phase)

                    optimizer.zero_grad(set_to_none=True)
                    actor_loss = pi_loss - entropy_coeff * ent_loss
                    actor_loss.backward()
                    actor_gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    if hasattr(model, "mu_center_policy_heads"):
                        model.mu_center_policy_heads()

                    with torch.no_grad():
                        _actor_acc[0] += pi_loss
                        _actor_acc[1] += ent_loss
                        _actor_acc[2] += actor_gn
                    n_updates += 1

                    with torch.no_grad():
                        if full_length:
                            flat_temperature = batch_temperature.reshape(-1, N_ACTION_TEMPERATURES)
                        else:
                            flat_temperature = batch_temperature[valid_mask]
                        sum_temperature += flat_temperature.sum(dim=0)
                        sum_temperature_sq += flat_temperature.square().sum(dim=0)
                        n_temperature += flat_temperature.shape[0]

        if full_perm.numel() > 0:
            _run_actor_group(full_perm, full_length=True)
        if tail_perm.numel() > 0:
            _run_actor_group(tail_perm, full_length=False)

        if target_kl is not None and n_mb > 0 and _epoch_kl_acc.item() / n_mb > target_kl:
            actor_stopped_early = True
            break

    # Actor phase done — drop per-step tensors that the critic doesn't need so
    # the critic forward pass starts with a clean VRAM budget.
    del bt_mask_all, bd_mask_all, cd_causal_all
    obs_pre_all = None
    centralized_data.pop("obs_pre", None)

    # ---- Phase 2: Critic updates with random minibatches (full GPU utilization) ----
    # Always run full ppo_epochs for critic regardless of actor early stopping —
    # critic has its own loss landscape and benefits from more updates.
    n_critic_updates = 0
    critic_epochs = 0 if freeze_critic else ppo_epochs
    if critic_epochs > 0:
        critic, critic_opt = get_centralized_critic()
        critic_loss_step = get_centralized_critic_loss_step()
        critic_cuda_cfg = _get_critic_cuda_config()
        critic.train()
        critic_minibatch_size = _resolve_critic_minibatch_size(N)

        step_round_idx = centralized_data["step_round_idx"]

        # Pack per-step float data once — 2 index ops → 1 + slice in the loop
        step_unit_xy = centralized_data["step_unit_xy"]
        step_unit_float = torch.cat([
            centralized_data["step_unit_ctx"],
            centralized_data["step_actor_latent"],
        ], dim=1)
        positive_reward_mask = centralized_data["rewards"] > 0
        _ctx_dim = centralized_data["step_unit_ctx"].shape[1]
        old_values_safe = old_values if old_values is not None else returns
        step_rv = torch.stack([returns, old_values_safe], dim=1)

        for epoch in range(critic_epochs):
            with trace_scope(trace, "ppo.critic_epoch_plan"):
                epoch_plan = _build_critic_epoch_plan(
                    step_round_idx,
                    step_unit_xy,
                    step_unit_float,
                    step_rv,
                    target_steps=critic_minibatch_size,
                    positive_mask=positive_reward_mask,
                )

            for batch_idx, (round_start, round_end) in enumerate(epoch_plan.batch_ranges):
                with trace_scope(trace, "ppo.critic_minibatch"):
                    t_value = time.perf_counter() if profile is not None else 0.0
                    critic_opt.zero_grad(set_to_none=True)
                    capture_this_batch = (
                        DEVICE.type == "cuda"
                        and _NCU_CRITIC_BATCH_INDEX >= 0
                        and epoch == _NCU_CRITIC_EPOCH_INDEX
                        and batch_idx == _NCU_CRITIC_BATCH_INDEX
                    )
                    if capture_this_batch:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.start()

                    step_lo = epoch_plan.round_offsets_cpu[round_start]
                    step_hi = epoch_plan.round_offsets_cpu[round_end]
                    mb_xy = epoch_plan.packed_xy[step_lo:step_hi]
                    mb_uf = epoch_plan.packed_unit_float[step_lo:step_hi]
                    mb_rv = epoch_plan.packed_returns_values[step_lo:step_hi]
                    vf_denom = max(mb_xy.shape[0], 1)
                    c_rounds = epoch_plan.round_ids[round_start:round_end]
                    g_obs, g_mask, g_scalars = _load_critic_round_batch(centralized_data, c_rounds)
                    local_round_idx = epoch_plan.local_round_idx_by_batch[batch_idx]
                    u_ctx = mb_uf[:, :_ctx_dim]
                    u_latent = mb_uf[:, _ctx_dim:]
                    mb_returns = mb_rv[:, 0]
                    mb_old_values = mb_rv[:, 1]
                    with autocast_context(critic_cuda_cfg):
                        loss_sum, vf_loss_sum, new_values = critic_loss_step(
                            g_obs,
                            g_mask,
                            g_scalars,
                            local_round_idx,
                            mb_xy,
                            u_ctx,
                            u_latent,
                            mb_returns,
                            mb_old_values,
                        )
                    batch_ctx = (
                        f"critic minibatch epoch={epoch} batch={batch_idx} "
                        f"n_rounds={int(c_rounds.numel())} n_steps={int(mb_xy.shape[0])} "
                        f"round_lo={int(c_rounds[0].item()) if c_rounds.numel() > 0 else -1} "
                        f"round_hi={int(c_rounds[-1].item()) if c_rounds.numel() > 0 else -1}"
                    )
                    try:
                        _require_finite(f"{batch_ctx}.loss_sum", loss_sum.detach())
                        _require_finite(f"{batch_ctx}.vf_loss_sum", vf_loss_sum)
                        _require_finite(f"{batch_ctx}.new_values", new_values)
                    except RuntimeError as e:
                        raise RuntimeError(str(e)) from e
                    (vf_coeff * loss_sum / vf_denom).backward()

                    critic_gn = torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                    if not torch.isfinite(critic_gn):
                        _require_module_grads_finite(f"{batch_ctx}.critic_grads", critic)
                        raise RuntimeError(f"non-finite {batch_ctx}.critic_grad_norm: value={critic_gn!r}")
                    critic_opt.step()
                    _require_module_params_finite(f"{batch_ctx}.critic_params", critic)
                    if capture_this_batch:
                        torch.cuda.synchronize()
                        torch.cuda.profiler.stop()

                    with torch.no_grad():
                        _critic_acc[0] += vf_loss_sum / vf_denom
                        _critic_acc[1] += critic_gn
                        _critic_acc[2] += new_values.abs().mean()
                    n_critic_updates += 1

                    if profile is not None:
                        profile["ppo_value_loss_s"] = profile.get("ppo_value_loss_s", 0.0) + (time.perf_counter() - t_value)

    if n_temperature > 0:
        mean_temperature = sum_temperature / n_temperature
        std_temperature = (sum_temperature_sq / n_temperature - mean_temperature.square()).clamp_min(0.0).sqrt()
    else:
        mean_temperature = torch.ones(N_ACTION_TEMPERATURES, device=dev)
        std_temperature = torch.zeros(N_ACTION_TEMPERATURES, device=dev)

    # Single GPU→CPU sync for all accumulated scalars
    with torch.no_grad():
        if CRITIC_VALUE_MODE == CriticValueMode.GHL:
            bucket_lo_orig = symexp(bucket_centers[0:1])[0]
            bucket_hi_orig = symexp(bucket_centers[-1:])[-1]
            oob_frac = ((returns <= bucket_lo_orig) | (returns >= bucket_hi_orig)).float().mean()
        else:
            bucket_lo_orig = torch.tensor(0.0, device=dev)
            bucket_hi_orig = torch.tensor(0.0, device=dev)
            oob_frac = torch.tensor(0.0, device=dev)
        _diag = torch.stack([
            returns.min(), returns.max(), returns.mean(),
            oob_frac, bucket_lo_orig.float(), bucket_hi_orig.float(),
            ev_var_y, ev_var_res,
        ])
        # Batch all .item() calls into one contiguous CPU transfer
        _all_scalars = torch.cat([_actor_acc, _critic_acc, _diag, mean_temperature, std_temperature]).cpu()
    # Unpack after single sync
    _a = _all_scalars
    _na = max(n_updates, 1)
    _nc = max(n_critic_updates, 1)
    _var_y = _a[14].item()
    ev = 0.0 if _var_y == 0 else 1.0 - _a[15].item() / _var_y
    return {
        "pi_loss": _a[0].item() / _na,
        "vf_loss": _a[5].item() / _nc,
        "entropy": _a[1].item() / _na,
        "approx_kl": _a[3].item() / _na,
        "clip_frac": _a[4].item() / _na,
        "explained_var": round(ev, 4),
        "actor_grad_norm": _a[2].item() / _na,
        "critic_grad_norm": _a[6].item() / _nc,
        "critic_value_delta": _a[7].item() / _nc,
        "ret_min": round(_a[8].item(), 3),
        "ret_max": round(_a[9].item(), 3),
        "ret_mean": round(_a[10].item(), 3),
        "ret_oob_frac": round(_a[11].item(), 4),
        "n_updates": n_updates,
        "temp_bt_mean": _a[16].item(),
        "temp_bd_mean": _a[17].item(),
        "temp_cd_mean": _a[18].item(),
        "temp_brt_mean": _a[19].item(),
        "temp_md_mean": _a[20].item(),
        "temp_bt_std": _a[21].item(),
        "temp_bd_std": _a[22].item(),
        "temp_cd_std": _a[23].item(),
        "temp_brt_std": _a[24].item(),
        "temp_md_std": _a[25].item(),
    }


_STEP_FIELDS_GETTER = attrgetter("bt", "bd", "cd", "brt", "md", "old_log_prob", "reward")


def _parse_trajectories(
    trajectories: list[list[TrajectoryStep]],
    global_states_list: list[dict | None] | None = None,
    traj_game_map: list[int] | None = None,
    game_winners: list[str | None] | None = None,
) -> dict[str, torch.Tensor | list[tuple[int, int]]] | None:
    """Parse engine-collected trajectories into flat tensors (no GAE yet).

    Returns dict with keys: obs, h, bt, bd, cd, md, old_log_probs, rewards,
    ep_boundaries. Or None if no data.

    If global_states_list is provided, also returns centralized critic tensors:
    round_idx and unit_xy/unit_ctx mappings into replay-derived map states.
    """
    # Pre-count total steps to allocate tensors up front, avoiding
    # 7 large Python lists and the GC pressure they cause.
    N = sum(len(traj) for traj in trajectories)
    if N == 0:
        return None

    # Pre-allocate CPU tensors, fill in-place, transfer to device once.
    obs_t = torch.empty(N, OBS_SIZE, dtype=torch.float32)
    h_t = torch.empty(N, H_DIM, dtype=torch.float32)
    bt_t = torch.empty(N, dtype=torch.long)
    bd_t = torch.empty(N, dtype=torch.long)
    cd_t = torch.empty(N, dtype=torch.long)
    brt_t = torch.empty(N, dtype=torch.long)
    md_t = torch.empty(N, dtype=torch.long)
    lp_t = torch.empty(N, dtype=torch.float32)
    rw_t = torch.empty(N, dtype=torch.float32)
    ep_boundaries = []

    # Disable cyclic GC — the loop creates no reference cycles, and the
    # collector's O(live_objects) scan is expensive with many live tensors.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        idx = 0
        for traj_i, traj in enumerate(trajectories):
            start = idx
            traj_len = len(traj)
            end = idx + traj_len
            if traj_len == 0:
                ep_boundaries.append((start, end))
                continue

            if not all(isinstance(step, TrajectoryStep) for step in traj):
                bad_type = type(next(step for step in traj if not isinstance(step, TrajectoryStep))).__name__
                raise TypeError(f"Expected TrajectoryStep in trajectory, got {bad_type}")

            obs_blob = bytearray().join(step.obs_bytes for step in traj)
            h_blob = bytearray().join(step.h_prev_bytes for step in traj)
            obs_t[start:end] = torch.frombuffer(
                memoryview(obs_blob), dtype=torch.float32,
            ).reshape(traj_len, OBS_SIZE)
            h_t[start:end] = torch.frombuffer(
                memoryview(h_blob), dtype=torch.float32,
            ).reshape(traj_len, H_DIM)
            # Single pass over traj, yielding all scalar fields as column tuples.
            bts, bds, cds, brts, mds, lps, rws = zip(*map(_STEP_FIELDS_GETTER, traj))
            bt_t[start:end] = torch.as_tensor(bts, dtype=torch.long)
            bd_t[start:end] = torch.as_tensor(bds, dtype=torch.long)
            cd_t[start:end] = torch.as_tensor(cds, dtype=torch.long)
            brt_t[start:end] = torch.as_tensor(brts, dtype=torch.long)
            md_t[start:end] = torch.as_tensor(mds, dtype=torch.long)
            lp_t[start:end] = torch.as_tensor(lps, dtype=torch.float32)
            if REWARD_MODE in (RewardMode.INCOME, RewardMode.FLOW):
                rw_t[start:end] = torch.as_tensor(rws, dtype=torch.float32)
            else:
                rw_t[start:end] = 0.0
            idx = end
            ep_boundaries.append((start, end))
    finally:
        if gc_was_enabled:
            gc.enable()

    round_idx_t = _round_index_from_obs(obs_t)

    # Rewards are already per-round team-level values written identically to
    # every unit in augment_trajectories_with_flow_rewards — no averaging needed.

    # Terminal reward: add win/loss bonus to the last step of each trajectory (outcome mode only)
    if REWARD_MODE == RewardMode.OUTCOME and game_winners:
        for traj_i, (start, end) in enumerate(ep_boundaries):
            if start >= end:
                continue
            game_idx = traj_game_map[traj_i] if traj_game_map and traj_i < len(traj_game_map) else 0
            if game_idx < len(game_winners):
                w = game_winners[game_idx]
                if w == "a":
                    rw_t[end - 1] += TERMINAL_WIN_REWARD
                elif w == "b":
                    rw_t[end - 1] += TERMINAL_LOSS_REWARD

    # Bulk H2D: pin all CPU tensors, issue non_blocking transfers, single sync.
    _pin = DEVICE.type == "cuda"
    if _pin:
        obs_t = obs_t.pin_memory()
        h_t = h_t.pin_memory()
        bt_t = bt_t.pin_memory()
        bd_t = bd_t.pin_memory()
        cd_t = cd_t.pin_memory()
        brt_t = brt_t.pin_memory()
        md_t = md_t.pin_memory()
        if lp_t is not None:
            lp_t = lp_t.pin_memory()
        rw_t = rw_t.pin_memory()
    _nb = _pin  # non_blocking only safe with pinned memory
    result = {
        "obs": obs_t.to(DEVICE, non_blocking=_nb),
        "h": h_t.to(DEVICE, non_blocking=_nb),
        "bt": bt_t.to(DEVICE, non_blocking=_nb),
        "bd": bd_t.to(DEVICE, non_blocking=_nb),
        "cd": cd_t.to(DEVICE, non_blocking=_nb),
        "brt": brt_t.to(DEVICE, non_blocking=_nb),
        "md": md_t.to(DEVICE, non_blocking=_nb),
        "old_log_probs": lp_t.to(DEVICE, non_blocking=_nb),
        "rewards": rw_t.to(DEVICE, non_blocking=_nb),
        "ep_boundaries": ep_boundaries,
        "n_steps": N,
    }
    if _pin:
        torch.cuda.synchronize()

    # Build centralized critic data if global states available
    if global_states_list and any(gs is not None for gs in global_states_list):
        result.update(_build_centralized_critic_data(
            obs_t, round_idx_t, global_states_list, ep_boundaries, N,
            traj_game_map=traj_game_map,
        ))

    return result


# Obs feature indices for unit position (x_norm, y_norm at global feature offsets)
_OBS_X_NORM = _OBS_GLOBAL_START + 1  # x / map_w
_OBS_Y_NORM = _OBS_GLOBAL_START + 2  # y / map_h
_OBS_CORE_DX = _OBS_GLOBAL_START + 6
_OBS_CORE_DY = _OBS_GLOBAL_START + 7
_OBS_ECORE_DX = _OBS_GLOBAL_START + 8
_OBS_ECORE_DY = _OBS_GLOBAL_START + 9
_OBS_SPAWN_ROUND = _OBS_GLOBAL_START + 14  # spawn_round / 2000.0


def _build_centralized_critic_data(
    obs_t: torch.Tensor,
    round_idx_t: torch.Tensor,
    global_states_list: list[dict | None],
    ep_boundaries: list[tuple[int, int]],
    N: int,
    traj_game_map: list[int] | None = None,
) -> dict[str, object]:
    """Build per-round global state tensors and per-step index mappings.

    De-duplicates global states by (game_idx, round) to save GPU memory.
    Accepts both contiguous-buffer format (states_buf/scalars_buf) and
    legacy per-turn dict format (states/scalars dicts).

    Args:
        traj_game_map: Maps trajectory index -> game index in global_states_list.
            If None, all trajectories are assigned to game 0.
    """
    max_w = GLOBAL_MAP_MAX_W
    max_h = GLOBAL_MAP_MAX_H
    n_ch = GLOBAL_MAP_CHANNELS
    n_sc = GLOBAL_SCALAR_DIM

    # Reserve index 0 as a zero fallback state for games without global states.
    # Use bytearray so torch.frombuffer works uniformly for all entries.
    _pin_nb = DEVICE.type == "cuda"  # non_blocking safe with pinned memory
    zero_state = bytearray(n_ch * 1 * 1 * 4)
    zero_scalars = [0.0] * n_sc
    unique_states: list[object] = [zero_state]  # bytearray or buffer slice
    unique_valid_masks = [(1, 1)]
    unique_scalars: list[list[float]] = [zero_scalars]  # for legacy path
    _scalar_chunks: list[torch.Tensor] = []  # batch-appended tensor chunks (bulk path)
    _scalar_chunk_offsets: list[tuple[int, int]] = []  # (base_idx, count)
    # Per-step: index into unique states, unit xy, unit context
    step_round_idx = torch.zeros(N, dtype=torch.long)
    step_unit_xy = torch.zeros(N, 2, dtype=torch.long)
    step_unit_ctx = torch.zeros(N, UNIT_CONTEXT_DIM, dtype=torch.float32)

    n_games = len(global_states_list)
    game_round_lookup: dict[int, tuple[int, int, torch.Tensor]] = {}

    for ep_i, (start, end) in enumerate(ep_boundaries):
        if end <= start:
            continue

        game_idx = traj_game_map[ep_i] if traj_game_map and ep_i < len(traj_game_map) else 0
        gs = global_states_list[game_idx] if game_idx < n_games else None
        if gs is None:
            continue

        cached = game_round_lookup.get(game_idx)
        if cached is None:
            gs_map_w = gs.get("map_w", max_w)
            gs_map_h = gs.get("map_h", max_h)
            states_buf = load_torch_replay_buffer(gs, STATE_BUFFER)
            scalars_buf_raw = load_torch_replay_buffer(gs, SCALAR_BUFFER)

            if states_buf is not None:
                import struct

                state_stride_bytes = gs.get("state_stride", n_ch * gs_map_h * gs_map_w) * 4
                scalars_buf = gs.get("scalars_buf", b"")
                scalar_stride_bytes = gs.get("n_scalars", n_sc) * 4
                gs_n_turns = gs.get("n_turns", 0)
                available_rounds_t = torch.arange(gs_n_turns, dtype=torch.long)
                available_unique_idx = torch.empty(gs_n_turns, dtype=torch.long)

                # Single frombuffer + reshape for all rounds (no per-round bytearray copy)
                all_states_t = torch.frombuffer(
                    states_buf, dtype=torch.float32,
                ).reshape(gs_n_turns, n_ch, gs_map_h, gs_map_w)

                # Bulk-parse scalars if present
                if scalar_stride_bytes > 0:
                    n_scalar_floats = scalar_stride_bytes // 4
                    all_scalars_t = torch.frombuffer(
                        scalars_buf, dtype=torch.float32,
                    ).reshape(gs_n_turns, n_scalar_floats)

                base_idx = len(unique_states)
                for rnd in range(gs_n_turns):
                    unique_states.append(all_states_t[rnd])  # tensor view, no copy
                    unique_valid_masks.append((gs_map_w, gs_map_h))
                available_unique_idx[:] = torch.arange(base_idx, base_idx + gs_n_turns, dtype=torch.long)
                # Batch-append scalars as tensor rows (no per-round .tolist() sync)
                if scalar_stride_bytes > 0:
                    _scalar_chunks.append(all_scalars_t)
                else:
                    _scalar_chunks.append(torch.zeros(gs_n_turns, n_sc))
                _scalar_chunk_offsets.append((base_idx, gs_n_turns))
            else:
                gs_states = gs.get("states", {})
                gs_scalars_d = gs.get("scalars", {})
                available_rounds = sorted(gs_states.keys()) if gs_states else []
                available_rounds_t = torch.tensor(available_rounds, dtype=torch.long)
                available_unique_idx = torch.empty(len(available_rounds), dtype=torch.long)

                for pos, rnd in enumerate(available_rounds):
                    state = gs_states[rnd]
                    if not isinstance(state, (bytes, bytearray)):
                        state = bytearray(state.tobytes() if hasattr(state, "tobytes") else bytes(state))
                    scalars = gs_scalars_d.get(rnd)
                    unique_states.append(state)
                    unique_valid_masks.append((gs_map_w, gs_map_h))
                    unique_scalars.append(list(scalars) if scalars is not None else zero_scalars)
                    available_unique_idx[pos] = len(unique_states) - 1

            if available_rounds_t.numel() > 0:
                nearest_pos = _nearest_round_positions(available_rounds_t)
                round_lookup = available_unique_idx[nearest_pos]
            else:
                round_lookup = torch.zeros(2000, dtype=torch.long)

            cached = (gs_map_w, gs_map_h, round_lookup)
            game_round_lookup[game_idx] = cached

        gs_map_w, gs_map_h, round_lookup = cached
        step_round_idx[start:end] = round_lookup[round_idx_t[start:end]]

        x_norm = obs_t[start:end, _OBS_X_NORM]
        y_norm = obs_t[start:end, _OBS_Y_NORM]
        step_unit_xy[start:end, 0] = torch.clamp((x_norm * gs_map_w).round().to(torch.long), 0, max_w - 1)
        step_unit_xy[start:end, 1] = torch.clamp((y_norm * gs_map_h).round().to(torch.long), 0, max_h - 1)

        step_unit_ctx[start:end, 0] = x_norm
        step_unit_ctx[start:end, 1] = y_norm
        step_unit_ctx[start:end, 2] = obs_t[start:end, _OBS_CORE_DX]
        step_unit_ctx[start:end, 3] = obs_t[start:end, _OBS_CORE_DY]
        step_unit_ctx[start:end, 4] = obs_t[start:end, _OBS_ECORE_DX]
        step_unit_ctx[start:end, 5] = obs_t[start:end, _OBS_ECORE_DY]
        step_unit_ctx[start:end, 6] = (
            obs_t[start:end, _OBS_GLOBAL_START] - obs_t[start:end, _OBS_SPAWN_ROUND]
        )

    global_sizes = torch.tensor(unique_valid_masks, dtype=torch.int16)
    # Merge list-based scalars (legacy path) with tensor chunks (bulk path)
    if _scalar_chunks:
        n_total = len(unique_scalars) + sum(c for _, c in _scalar_chunk_offsets)
        global_scalars = torch.zeros(n_total, n_sc, dtype=torch.float32)
        if unique_scalars:
            global_scalars[:len(unique_scalars)] = torch.tensor(unique_scalars, dtype=torch.float32)
        for chunk_t, (base_idx, count) in zip(_scalar_chunks, _scalar_chunk_offsets):
            global_scalars[base_idx:base_idx + count, :chunk_t.shape[1]] = chunk_t
    else:
        global_scalars = torch.tensor(unique_scalars, dtype=torch.float32)

    if _pin_nb:
        global_scalars = global_scalars.pin_memory()
        step_round_idx = step_round_idx.pin_memory()
        step_unit_xy = step_unit_xy.pin_memory()
        step_unit_ctx = step_unit_ctx.pin_memory()
    return {
        "global_states": unique_states,
        "global_sizes": global_sizes,
        "global_scalars": global_scalars,
        "step_round_idx": step_round_idx.to(DEVICE, non_blocking=_pin_nb),
        "step_unit_xy": step_unit_xy.to(DEVICE, non_blocking=_pin_nb),
        "step_unit_ctx": step_unit_ctx.to(DEVICE, non_blocking=_pin_nb),
        "has_centralized": True,
    }


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ep_boundaries: list[tuple[int, int]],
    gamma: float = GAMMA,
    gae_lambda: float = LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GAE on GPU for terminal per-unit trajectories.

    Training episodes are complete unit lifetimes: they end on unit death or
    game termination, not on an arbitrary collector cutoff. That means the
    final step should be treated as terminal and bootstrap with 0.
    """
    dev = rewards.device
    N = rewards.shape[0]
    n_eps = len(ep_boundaries)
    max_len = max(end - start for start, end in ep_boundaries)
    gl = gamma * gae_lambda

    # Vectorized pad: build index tensors once, scatter into (E, T) tensors
    ep_starts = torch.tensor([s for s, _ in ep_boundaries], dtype=torch.long, device=dev)
    ep_lens = torch.tensor([e - s for s, e in ep_boundaries], dtype=torch.long, device=dev)
    t_range = torch.arange(max_len, device=dev).unsqueeze(0)  # (1, T)
    valid = t_range < ep_lens.unsqueeze(1)  # (E, T)
    flat_idx = (ep_starts.unsqueeze(1) + t_range).clamp(max=N - 1)  # (E, T)
    valid_idx = flat_idx[valid]

    r_pad = torch.zeros(n_eps, max_len, device=dev)
    v_pad = torch.zeros(n_eps, max_len, device=dev)
    r_pad[valid] = rewards[valid_idx]
    v_pad[valid] = values[valid_idx]

    # Nonterminal mask: only interior steps bootstrap from the next value.
    interior = t_range < (ep_lens - 1).clamp(min=0).unsqueeze(1)
    nonterminal = interior.float()

    # Build next_values by shifting within each episode; terminal tail stays 0.
    next_v = torch.zeros_like(v_pad)
    if max_len > 1:
        next_v[:, :-1] = v_pad[:, 1:]

    # TD residuals: delta = r + gamma * next_v * nonterminal - v
    delta = r_pad + gamma * next_v * nonterminal - v_pad

    # Reverse scan for GAE (sequential over T, vectorized over E)
    gae = torch.zeros(n_eps, device=dev)
    adv_pad = torch.zeros_like(r_pad)
    for t in range(max_len - 1, -1, -1):
        gae = delta[:, t] + gl * nonterminal[:, t] * gae
        adv_pad[:, t] = gae

    # Vectorized unpad back to flat (N,) tensor
    advantages = torch.zeros(N, device=dev)
    advantages[valid_idx] = adv_pad[valid]

    returns = advantages + values
    return advantages, returns


def _round_index_from_obs(obs_t: torch.Tensor) -> torch.Tensor:
    """Decode normalized round feature into integer round indices on CPU/GPU."""
    return torch.clamp((obs_t[:, _OBS_GLOBAL_START] * 2000.0).round().to(torch.long), 0, 1999)


def _nearest_round_positions(
    available_rounds: torch.Tensor,
    max_rounds: int = 2000,
) -> torch.Tensor:
    """Map each target round in [0, max_rounds) to the nearest available-round position."""
    if available_rounds.numel() == 0:
        return torch.zeros(max_rounds, dtype=torch.long)

    targets = torch.arange(max_rounds, dtype=torch.long)
    insert_pos = torch.bucketize(targets, available_rounds)
    left_pos = (insert_pos - 1).clamp(min=0)
    right_pos = insert_pos.clamp(max=available_rounds.numel() - 1)

    left_rounds = available_rounds[left_pos]
    right_rounds = available_rounds[right_pos]

    choose_left = (insert_pos == available_rounds.numel()) | (
        (insert_pos > 0) & ((targets - left_rounds) <= (right_rounds - targets))
    )
    return torch.where(choose_left, left_pos, right_pos)


# ---- Reward normalizer (running variance of discounted returns) ----
# Inspired by gym.wrappers.NormalizeReward. Tracks Var(discounted_return) via
# Welford's online algorithm over full episode returns, divides each reward
# by sqrt(var + eps), clips to [-clip, clip].

_centralized_critic: CentralizedCritic | None = None
_centralized_optimizer: torch.optim.Optimizer | None = None
_centralized_critic_step: nn.Module | None = None
_centralized_critic_loss_step: nn.Module | None = None
_critic_cuda_cfg: PretrainCudaConfig | None = None
_critic_compile_enabled = True
_critic_cuda_runtime_configured = False


def configure_centralized_critic_runtime(*, enable_compile: bool = True) -> None:
    """Configure PPO critic CUDA execution before the critic is instantiated."""
    global _critic_cuda_cfg, _critic_compile_enabled, _centralized_critic_step, _centralized_critic_loss_step
    _critic_compile_enabled = enable_compile
    _critic_cuda_cfg = resolve_cuda_config(
        DEVICE,
        spatial_inputs=True,
        enable_compile=enable_compile,
    )
    _centralized_critic_step = None
    _centralized_critic_loss_step = None


def _get_critic_cuda_config() -> PretrainCudaConfig:
    global _critic_cuda_cfg
    if _critic_cuda_cfg is None:
        configure_centralized_critic_runtime(enable_compile=_critic_compile_enabled)
    return _critic_cuda_cfg


def _ensure_critic_cuda_runtime() -> PretrainCudaConfig:
    global _critic_cuda_runtime_configured
    cuda_cfg = _get_critic_cuda_config()
    if cuda_cfg.enabled and not _critic_cuda_runtime_configured:
        configure_cuda_runtime(DEVICE)
        _critic_cuda_runtime_configured = True
    return cuda_cfg


def critic_checkpoint_path(weight_path: str | Path = WEIGHT_PATH) -> Path:
    """Sidecar path used to persist centralized critic state for training resumes."""
    path = Path(weight_path)
    return path.with_name(f"{path.stem}.critic.pt")


def reset_centralized_critic() -> None:
    """Drop the singleton critic/optimizer so a fresh instance is created on demand."""
    global _centralized_critic, _centralized_optimizer, _centralized_critic_step, _centralized_critic_loss_step
    global _compiled_actor_step, _compiled_logprob_step, _actor_cuda_cfg
    _centralized_critic = None
    _centralized_optimizer = None
    _centralized_critic_step = None
    _centralized_critic_loss_step = None
    _compiled_actor_step = None
    _compiled_logprob_step = None
    _actor_cuda_cfg = None


def get_centralized_critic(lr: float = 3e-4) -> tuple[CentralizedCritic, torch.optim.Optimizer]:
    """Get or create the singleton centralized critic and its optimizer."""
    global _centralized_critic, _centralized_optimizer
    cuda_cfg = _ensure_critic_cuda_runtime()
    if _centralized_critic is None:
        _centralized_critic = CentralizedCritic().to(DEVICE)
        if cuda_cfg.compile_model:
            _centralized_critic.encode_map = torch.compile(
                _centralized_critic.encode_map,
            )
        _centralized_optimizer = make_normuon_optimizer(
            _centralized_critic,
            lr=lr,
            weight_decay=0.0,
        )
        n_params = sum(p.numel() for p in _centralized_critic.parameters())
        print(
            f"  CentralizedCritic: {n_params:,} params on {DEVICE} "
            f"(compile={'on' if cuda_cfg.compile_model else 'off'})"
        )
    return _centralized_critic, _centralized_optimizer


def get_centralized_critic_step() -> nn.Module:
    """Get or create the compiled PPO critic step wrapper."""
    global _centralized_critic_step
    critic, _ = get_centralized_critic()
    if _centralized_critic_step is None:
        # encode_map is compiled separately; global SA has dynamic per-round
        # loops that prevent whole-module compile.
        _centralized_critic_step = CriticPpoStep(critic)
    _centralized_critic_step.train()
    return _centralized_critic_step


def get_centralized_critic_loss_step() -> nn.Module:
    """Get or create the compiled PPO critic loss wrapper."""
    global _centralized_critic_loss_step
    critic, _ = get_centralized_critic()
    if _centralized_critic_loss_step is None:
        # encode_map is compiled separately; global SA has dynamic per-round
        # loops that prevent whole-module compile.
        _centralized_critic_loss_step = CriticPpoLossStep(critic)
    _centralized_critic_loss_step.train()
    return _centralized_critic_loss_step


def load_centralized_critic_checkpoint(
    weight_path: str | Path = WEIGHT_PATH,
    *,
    lr: float = 3e-4,
) -> bool:
    """Load centralized critic state from the session sidecar, if present."""
    ckpt_path = critic_checkpoint_path(weight_path)
    if not ckpt_path.exists():
        return False

    try:
        reset_centralized_critic()
        critic, critic_opt = get_centralized_critic(lr=lr)
        payload = torch.load(ckpt_path, map_location=DEVICE)
        critic_state = payload.get("critic") if isinstance(payload, dict) else None
        opt_state = payload.get("optimizer") if isinstance(payload, dict) else None
        if critic_state is None:
            raise ValueError("checkpoint missing critic state")
        _require_finite_tree("critic checkpoint", critic_state)
        if opt_state is not None:
            _require_finite_tree("critic optimizer checkpoint", opt_state)
        critic.load_state_dict(critic_state)
        if opt_state is not None:
            critic_opt.load_state_dict(opt_state)
        return True
    except Exception as e:
        print(f"Could not load centralized critic from {ckpt_path}: {e}")
        reset_centralized_critic()
        return False


def save_centralized_critic_checkpoint(weight_path: str | Path = WEIGHT_PATH) -> Path | None:
    """Persist centralized critic state alongside the actor session weights."""
    if _centralized_critic is None or _centralized_optimizer is None:
        return None

    ckpt_path = critic_checkpoint_path(weight_path)
    critic_state = _centralized_critic.state_dict()
    opt_state = _centralized_optimizer.state_dict()
    _require_finite_tree("critic checkpoint", critic_state)
    _require_finite_tree("critic optimizer checkpoint", opt_state)
    payload = {
        "critic": critic_state,
        "optimizer": opt_state,
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def _compute_all_values(
    critic: CentralizedCritic,
    data: dict,
    cnn_chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute per-unit critic values for all steps using chunked CNN.

    Processes rounds in chunks to bound memory. Feature maps are not retained
    across chunks — only the per-step values are accumulated.
    """
    round_idx = data["step_round_idx"]    # (N,) long
    unit_xy = data["step_unit_xy"]        # (N, 2) long
    unit_ctx = data["step_unit_ctx"]      # (N, UNIT_CONTEXT_DIM)
    actor_latent = data["step_actor_latent"]  # (N, ACTOR_LATENT_DIM)
    N = data["n_steps"]

    values = torch.zeros(N, device=DEVICE)
    packed = _pack_round_major_tensors(
        round_idx,
        unit_xy,
        unit_ctx,
        actor_latent,
    )
    packed_xy, packed_ctx, packed_latent = packed.tensors
    n_unique_rounds = packed.round_ids.shape[0]

    round_chunk = n_unique_rounds if cnn_chunk_size is None else max(1, int(cnn_chunk_size))
    for round_start in range(0, n_unique_rounds, round_chunk):
        round_end = min(round_start + round_chunk, n_unique_rounds)
        step_lo = packed.round_offsets_cpu[round_start]
        step_hi = packed.round_offsets_cpu[round_end]
        if step_lo == step_hi:
            continue
        c_rounds = packed.round_ids[round_start:round_end]
        g_obs, g_mask, g_scalars = _load_critic_round_batch(data, c_rounds)
        feat = critic.encode_map(g_obs, g_mask)

        # Compute per-unit values for these steps (all at once — unit_head is small)
        step_indices = packed.perm[step_lo:step_hi]
        chunk_round_counts = (
            packed.round_offsets[round_start + 1:round_end + 1]
            - packed.round_offsets[round_start:round_end]
        )
        local_ri = torch.repeat_interleave(
            torch.arange(round_end - round_start, device=step_indices.device),
            chunk_round_counts,
        )
        batch_values = critic.unit_values(
            feat,
            g_scalars,
            local_ri,
            packed_xy[step_lo:step_hi],
            packed_ctx[step_lo:step_hi],
            packed_latent[step_lo:step_hi],
        )
        if not torch.isfinite(batch_values).all():
            bad = (~torch.isfinite(batch_values)).nonzero(as_tuple=False)
            b = int(bad[0, 0].item())
            local_round = int(local_ri[b].item())
            round_id = int(c_rounds[local_round].item())
            step_index = int(step_indices[b].item())
            raise RuntimeError(
                "non-finite critic batch value: "
                f"step_index={step_index} "
                f"round_id={round_id} "
                f"local_round={local_round} "
                f"unit_xy={packed_xy[step_lo + b].detach().cpu().tolist()} "
                f"unit_ctx_finite={bool(torch.isfinite(packed_ctx[step_lo:step_hi]).all().item())} "
                f"actor_latent_finite={bool(torch.isfinite(packed_latent[step_lo:step_hi]).all().item())} "
                f"round_scalars_finite={bool(torch.isfinite(g_scalars).all().item())} "
                f"patches_finite={bool(torch.isfinite(feat.patches).all().item())} "
                f"global_template_finite={bool(torch.isfinite(feat.global_cls).all().item())}"
            )
        values[step_indices] = batch_values

    return values



def train_on_trajectories(
    model: BuilderMLP,
    optimizer: torch.optim.Optimizer,
    trajectories: list[list],
    gamma: float = GAMMA,
    gae_lambda: float = LAMBDA,
    profile: dict[str, float] | None = None,
    trace=None,
    global_states_list: list[dict | None] | None = None,
    traj_game_map: list[int] | None = None,
    game_winners: list[str | None] | None = None,
    freeze_critic: bool = False,
    **ppo_kwargs,
) -> dict[str, float]:
    """End-to-end: trajectories -> model values -> GAE -> PPO update."""
    t_phase = time.perf_counter() if profile is not None else 0.0
    with trace_scope(trace, "train.parse_trajectories"):
        data = _parse_trajectories(
            trajectories,
            global_states_list=global_states_list,
            traj_game_map=traj_game_map,
            game_winners=game_winners,
        )
    if profile is not None:
        profile["parse_s"] = time.perf_counter() - t_phase
    if data is None:
        return {"pi_loss": 0, "vf_loss": 0, "entropy": 0, "n_updates": 0, "n_steps": 0}

    obs, h = data["obs"], data["h"]
    N = data["n_steps"]

    if not data.get("has_centralized", False):
        return {"pi_loss": 0, "vf_loss": 0, "entropy": 0, "n_updates": 0, "n_steps": N,
                "error": "no global states available for centralized critic"}

    # Pre-materialize all round tensors on GPU once (eliminates per-chunk CPU→GPU copies)
    t_phase = time.perf_counter() if profile is not None else 0.0
    with trace_scope(trace, "train.materialize_critic_inputs"):
        _materialize_critic_inputs(data)
    if profile is not None:
        profile["materialize_s"] = time.perf_counter() - t_phase

    # Compute obs_pre once — shared by actor_latent (critic conditioning)
    # and value baseline.
    t_phase = time.perf_counter() if profile is not None else 0.0
    model.eval()
    with trace_scope(trace, "train.actor_latent"), torch.no_grad(), autocast_context(_get_actor_cuda_config()):
        obs_pre = model._obs_encoder_pre(obs)
        data["step_actor_latent"] = _normalize_actor_latent_dim(
            model._latent_from_obs_pre(obs_pre, h).detach(),
        )
    # obs_pre is fed to _latent_from_obs_pre under bf16 autocast during the
    # actor PPO update anyway, so storing the persistent (N, OBS_PREFIX_SIZE)
    # buffer in bf16 halves its VRAM footprint with no downstream precision
    # change. Keep fp32 on CPU to preserve behavioural parity there.
    if _USE_BF16 and obs_pre.is_cuda and obs_pre.dtype == torch.float32:
        obs_pre = obs_pre.to(torch.bfloat16)
    if profile is not None:
        profile["critic_latent_s"] = time.perf_counter() - t_phase

    # Compute value estimates via centralized critic (chunked, memory-bounded)
    t_phase = time.perf_counter() if profile is not None else 0.0
    critic, critic_opt = get_centralized_critic()
    critic.eval()
    with trace_scope(trace, "train.compute_values"), torch.no_grad():
        values = _compute_all_values(critic, data)
    _require_finite("critic values", values)
    if profile is not None:
        profile["value_forward_s"] = time.perf_counter() - t_phase

    t_phase = time.perf_counter() if profile is not None else 0.0
    with trace_scope(trace, "train.gae"):
        advantages, returns = _compute_gae(
            data["rewards"], values, data["ep_boundaries"], gamma, gae_lambda,
        )
    _require_finite("advantages", advantages)
    _require_finite("returns", returns)
    if profile is not None:
        profile["gae_s"] = time.perf_counter() - t_phase

    ppo_profile = profile if profile is not None else None
    model.train()

    bptt_k = int(ppo_kwargs.get("bptt_k", BPTT_K))
    bptt_chunks = _build_bptt_chunks(data["ep_boundaries"], bptt_k)
    old_log_probs = data["old_log_probs"]

    # Hand obs / obs_pre ownership to the ppo_update call via the shared data
    # dict and drop every outer reference so ppo_update can free them mid-run.
    data["obs"] = obs
    data["obs_pre"] = obs_pre
    reward_std = data["rewards"].std()
    n_steps = data["n_steps"]
    obs = None
    obs_pre = None

    with trace_scope(trace, "train.ppo_update"):
        stats = ppo_update(
            model, optimizer,
            data["obs"], h, data["obs_pre"], data["bt"], data["bd"], data["cd"], data["brt"], data["md"],
            old_log_probs, advantages, returns,
            old_values=values,
            centralized_data=data,
            ep_boundaries=data["ep_boundaries"],
            bptt_chunks=bptt_chunks,
            profile=ppo_profile,
            trace=trace,
            freeze_critic=freeze_critic,
            **ppo_kwargs,
        )
    # Release large critic staging buffers — these are pinned CPU slabs (and a
    # GPU scratch area) that otherwise live until the next PPO update GCs the
    # dict.
    data.pop("critic_obs_cpu", None)
    data.pop("critic_mask_cpu", None)
    data.pop("critic_input_staging", None)
    data.pop("step_actor_latent", None)
    stats["n_steps"] = n_steps
    stats["reward_raw_std"] = reward_std.item()
    return stats


def distill(
    teacher: BuilderMLP,
    student: BuilderMLP,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    h: torch.Tensor,
    epochs: int = 4,
    minibatch_size: int = 2048,
    temperature: float = 2.0,
) -> dict[str, float]:
    """Distill teacher policy into student via KL divergence on logits.

    Both models must share the same input/output dimensions (same obs space
    and action heads), but can have different hidden sizes.
    """
    N = obs.shape[0]
    teacher.eval()
    student.train()

    # Get teacher logits (no grad needed)
    t_bt, t_bd, t_cd, t_md = [], [], [], []
    with torch.no_grad():
        for i in range(0, N, 4096):
            packed = torch.cat([obs[i:i + 4096], h[i:i + 4096]], dim=-1)
            bt_l, bd_l, cd_l, _brt_l, md_l, _head_temperature, _ = teacher.infer_packed(packed)
            t_bt.append(bt_l)
            t_bd.append(bd_l)
            t_cd.append(cd_l)
            t_md.append(md_l)
    t_bt = torch.cat(t_bt)
    t_bd = torch.cat(t_bd)
    t_cd = torch.cat(t_cd)
    t_md = torch.cat(t_md)

    sum_loss = 0.0
    n_updates = 0
    T = temperature

    for _ in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, minibatch_size):
            idx = perm[start:start + minibatch_size]
            s_bt, s_bd, s_cd, _s_brt, s_md, _head_temperature, _ = student(obs[idx], h[idx])

            # KL(teacher || student) on softened logits
            loss = (
                F.kl_div(F.log_softmax(s_bt / T, dim=-1), F.softmax(t_bt[idx] / T, dim=-1), reduction='batchmean')
                + F.kl_div(F.log_softmax(s_bd / T, dim=-1), F.softmax(t_bd[idx] / T, dim=-1), reduction='batchmean')
                + F.kl_div(F.log_softmax(s_cd / T, dim=-1), F.softmax(t_cd[idx] / T, dim=-1), reduction='batchmean')
                + F.kl_div(F.log_softmax(s_md / T, dim=-1), F.softmax(t_md[idx] / T, dim=-1), reduction='batchmean')
            ) * (T * T)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            optimizer.step()

            sum_loss += loss.item()
            n_updates += 1

    return {"distill_loss": sum_loss / max(n_updates, 1), "n_updates": n_updates}
