"""Shared model, constants, and PPO training code."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---- Architecture constants (must match main.py) ----

TILE_FEATURES = 19
GLOBAL_FEATURES = 15
POS_EMBED_DIM = 8
OBS_SIZE = 5 * 5 * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN = [64, 48]
TEACHER_HIDDEN = [128, 128, 128, 64]
H_DIM = 24
INPUT_SIZE = OBS_SIZE + H_DIM
TILE_EMBED_DIM = 8
TILE_INPUT_DIM = TILE_FEATURES + 2  # 21: raw features + (dx/2, dy/2)
TILE_GRID_N = 5 * 5  # 25 tiles in the 5x5 local grid
COMPRESS_DIM = 64  # tile compress output dim (spatial pooling)
TRUNK_INPUT_SIZE = COMPRESS_DIM + GLOBAL_FEATURES + POS_EMBED_DIM + H_DIM  # 103
OBS_PREFIX_SIZE = TRUNK_INPUT_SIZE - H_DIM
ACTOR_LATENT_DIM = max(HIDDEN[-1], TEACHER_HIDDEN[-1])
N_BUILD = 12
N_DIR = 9
N_CONV_DIR = 4  # cardinal only: N, W, E, S (indices 1, 3, 5, 7 in DIR_MAP_9)
ACTOR_RMSNORM_LAYERS = (True, True)
assert ACTOR_RMSNORM_LAYERS[0], "First layer RMSNorm must be True (encoder always uses RMSNorm; inference MLP respects this flag)"
HARVESTER_BUILD_TYPE = 1
CONVEYOR_BUILD_TYPES = (2, 7, 8)  # conveyor, splitter, armoured_conveyor
INITIAL_BUILD_TYPE_BIAS = [
    0.0,   # none
    0.8,   # harvester
    1.0,   # conveyor
    0.0,   # road
    -0.4,  # gunner
    -0.4,  # sentinel
    -0.6,  # breach
    0.7,   # splitter
    0.0,   # armoured_conveyor
    -0.3,  # barrier
    -0.7,  # foundry
    -0.5,  # launcher
]
REWARD_FLOW_THROUGHPUT = 50.0
REWARD_ENEMY_CORE_DAMAGE = 20.0
REWARD_OWN_CORE_DAMAGE = -40.0  # heavy punishment for damage taken to our core

# ---- DreamerV3-style symlog twohot value prediction ----
NUM_BUCKETS = 255
BUCKET_LOW = -20.0
BUCKET_HIGH = 20.0
BUCKET_STEP = (BUCKET_HIGH - BUCKET_LOW) / (NUM_BUCKETS - 1)
VALUE_TARGET_KERNEL_SIGMA = BUCKET_STEP * 0.5


def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * torch.expm1(x.abs())


def _make_bucket_centers() -> torch.Tensor:
    """DreamerV3-style symexp-spaced bins in original value space.

    Builds 255 bins via symexp(linspace(-20, 0, 128)) mirrored around 0.
    Dense near zero, sparse at extremes — matches DreamerV3 heads.py:132.
    """
    n_half = (NUM_BUCKETS - 1) // 2 + 1  # 128
    half = torch.linspace(BUCKET_LOW, 0.0, n_half, dtype=torch.float32)
    half = symexp(half)
    return torch.cat([half, -half[:-1].flip(0)])


def twohot_encode(targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets as twohot distributions over bins.

    Bins are symexp-spaced in original value space. Targets are raw
    returns (NOT symlog'd) — same space as bins.

    Args:
        targets: (...,) scalar targets in original value space.
        bins: (NUM_BUCKETS,) bin centers in original value space.

    Returns:
        (..., NUM_BUCKETS) twohot probability vectors.
    """
    targets = targets.clamp(bins[0], bins[-1])
    below = torch.bucketize(targets, bins) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    dist_to_below = (targets - bins[below]).abs()
    dist_to_above = (bins[above] - targets).abs()
    total = (dist_to_below + dist_to_above).clamp(min=1e-8)
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    result = torch.zeros(*targets.shape, len(bins), device=targets.device, dtype=targets.dtype)
    result.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    result.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return result


def decode_value(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode categorical logits to scalar values via symmetric summation.

    Bins are already in original value space (symexp-spaced), so the
    weighted average is the predicted value directly — no symexp needed.
    Symmetric sum ensures uniform logits decode to exactly 0.

    Args:
        logits: (..., NUM_BUCKETS) raw logits.
        bins: (NUM_BUCKETS,) bin centers in original value space.

    Returns:
        (...,) predicted values in original value space.
    """
    n = bins.shape[0]
    m = (n - 1) // 2  # 127 — center index
    probs = torch.softmax(logits, dim=-1)
    # Symmetric sum: pair bins equidistant from center to cancel float error.
    # p_left * bins_left (reversed) + p_right * bins_right cancel perfectly
    # when probs are uniform over symmetric bins.
    left = probs[..., :m] * bins[:m]          # (*, 127) — negative side
    center = probs[..., m:m+1] * bins[m]      # (*, 1) — center bin (=0)
    right = probs[..., m+1:] * bins[m+1:]     # (*, 127) — positive side
    wavg = center.sum(dim=-1) + (left.flip(-1) + right).sum(dim=-1)
    return wavg


def value_target_logits(
    targets: torch.Tensor,
    bin_symlog: torch.Tensor,
    sigma: float = VALUE_TARGET_KERNEL_SIGMA,
) -> torch.Tensor:
    """Project scalar values to dense logits over bins in symlog space.

    Use the same smooth target family for both unclipped and clipped branches
    so the PPO max compares like with like while still avoiding sparse-support
    dead zones.
    """
    target_symlog = symlog(targets).unsqueeze(-1)
    if bin_symlog.device != targets.device or bin_symlog.dtype != targets.dtype:
        bin_symlog = bin_symlog.to(device=targets.device, dtype=targets.dtype)
    diff = (target_symlog - bin_symlog) / sigma
    return -0.5 * diff.square()

WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor_inference" / "weights.json"
LOG_PATH = PROJECT_ROOT / "bots" / "tensor" / "train_log.jsonl"
REPLAY_DIR = PROJECT_ROOT / "replays" / "training"

# Precomputed (dx/2, dy/2) for each tile in the 5x5 local grid.
# Tile index i -> grid position (i%5 - 2, i//5 - 2), normalized by /2.
_TILE_DXDY = torch.tensor(
    [[float((i % 5) - 2) / 2.0, float((i // 5) - 2) / 2.0] for i in range(TILE_GRID_N)],
    dtype=torch.float32,
)

# ---- Action masking ----
# Build costs (ti, ax) indexed by BUILD_ACTIONS:
# none, harvester, conveyor, road, gunner, sentinel, breach, splitter,
# armoured_conveyor, barrier, foundry, launcher
_MASK_COSTS_TI = [0, 80, 3, 1, 10, 15, 30, 6, 10, 3, 120, 20]
_MASK_COSTS_AX = [0, 0, 0, 0, 0, 0, 10, 0, 5, 0, 0, 0]

# Tile indices in 5x5 grid for the 9 build directions (NW,N,NE,W,C,E,SW,S,SE)
_MASK_BD_TILES = [6, 7, 8, 11, 12, 13, 16, 17, 18]
_MASK_BD_DX = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
_MASK_BD_DY = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
_CONV_DIR_MASK = [True, True, True, True]  # all 4 cardinal dirs always valid

# Observation feature indices for resources/scale
_OBS_GLOBAL_START = 5 * 5 * TILE_FEATURES  # 475
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
            torch.tensor([t * TILE_FEATURES + 17 for t in _MASK_BD_TILES], dtype=torch.long, device=device), # has_unit
            torch.tensor([t * TILE_FEATURES + 12 for t in _MASK_BD_TILES], dtype=torch.long, device=device), # bld_friendly
            torch.tensor([t * TILE_FEATURES + 11 for t in _MASK_BD_TILES], dtype=torch.long, device=device), # bld_core
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


def compute_harvester_build_dir_mask(obs: torch.Tensor, bd_mask: torch.Tensor) -> torch.Tensor:
    """Harvester-specific direction mask.

    Harvester placement is only legal on empty ore tiles.
    """
    _, _, _, nobld_idx, _, _, _ = _mask_indices(obs.device)
    ore_ti_idx, ore_ax_idx = _harv_mask_indices(obs.device)
    ore_here = (obs[:, ore_ti_idx] == 1.0) | (obs[:, ore_ax_idx] == 1.0)
    empty = obs[:, nobld_idx] == 1.0
    return bd_mask & ore_here & empty


_conv_dir_mask_cache: dict = {}


def _conveyor_dir_mask_base(device: torch.device) -> torch.Tensor:
    """Get or create the base conveyor direction mask for the given device."""
    if device not in _conv_dir_mask_cache:
        _conv_dir_mask_cache[device] = torch.tensor(_CONV_DIR_MASK, dtype=torch.bool, device=device)
    return _conv_dir_mask_cache[device]


def compute_build_dir_mask_for_type(
    obs: torch.Tensor,
    bt_act: torch.Tensor,
    bd_mask: torch.Tensor,
    harv_bd_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Refine build_dir mask using the chosen build type."""
    if harv_bd_mask is None:
        harv_bd_mask = compute_harvester_build_dir_mask(obs, bd_mask)
    # Use where() instead of conditional indexing to avoid CPU-GPU sync
    harv_rows = bt_act.to(torch.long) == HARVESTER_BUILD_TYPE
    return torch.where(harv_rows.unsqueeze(1), harv_bd_mask, bd_mask)


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
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


# ---- Centralized Critic constants ----
GLOBAL_MAP_MAX_W = 50
GLOBAL_MAP_MAX_H = 50
GLOBAL_MAP_CHANNELS = 19
GLOBAL_SCALAR_DIM = 17  # 12 base + own_working_harv, Δti, Δax, Δcore_hp, Δworking_harv
UNIT_CONTEXT_DIM = 7  # x_norm, y_norm, core_dx, core_dy, ecore_dx, ecore_dy, unit_age
CENTRALIZED_CNN_CHUNK = 2048


class RMSNorm2d(nn.Module):
    """RMSNorm for 2D spatial feature maps (B, C, H, W).

    Uses F.rms_norm (fused CUDA kernel) via NHWC permutation for faster
    forward+backward.  Falls back to manual implementation on CPU.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — normalize over C dimension per spatial position
        if x.is_cuda:
            # F.rms_norm needs C as the last dim.  When x is channels_last the
            # permute is a free stride relabel; for contiguous NCHW it copies.
            xp = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            return F.rms_norm(xp, (self.num_channels,), self.weight, self.eps).permute(0, 3, 1, 2)
        return x * torch.rsqrt(x.square().mean(1, keepdim=True) + self.eps) * self.weight.view(1, -1, 1, 1)


class CentralizedCritic(nn.Module):
    """Centralized per-unit critic over full replay-derived map state.

    Takes full map state (omniscient, from replay) and produces per-unit
    value estimates. Only used during training.

    Architecture:
      1. Per-tile 1×1 conv encodes sparse 19-channel indicators into dense
         embeddings (preserves directional info for supply line analysis).
      2. Full-resolution 3×3 convs with dilations for spatial context.
         No striding — preserves per-tile directional precision needed
         for supply line connectivity analysis.
      3. Per-unit indexing + MLP produces value estimates.
    """

    def __init__(self, map_channels: int = GLOBAL_MAP_CHANNELS, conv_dim: int = 48):
        super().__init__()
        self.conv_dim = conv_dim

        self.conv = nn.Sequential(
            # Per-tile encode: sparse binary channels → dense embedding.
            # 1×1 kernel = shared MLP per tile, no spatial mixing yet.
            nn.Conv2d(map_channels, conv_dim, 1),
            RMSNorm2d(conv_dim),
            nn.SiLU(),
            # Full-resolution spatial layers — no stride so per-tile
            # conveyor directions are preserved for supply line analysis.
            nn.Conv2d(conv_dim, conv_dim, 3, padding=1),
            RMSNorm2d(conv_dim),
            nn.SiLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, padding=2, dilation=2),
            RMSNorm2d(conv_dim),
            nn.SiLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, padding=4, dilation=4),
            RMSNorm2d(conv_dim),
            nn.SiLU(),
        )

        # Single per-unit value head from tile features + replay/global context.
        # Outputs NUM_BUCKETS logits for DreamerV3-style symlog twohot prediction.
        unit_in = conv_dim + UNIT_CONTEXT_DIM + GLOBAL_SCALAR_DIM + ACTOR_LATENT_DIM
        self.unit_head = nn.Sequential(
            nn.Linear(unit_in, 128),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            RMSNorm(64),
            nn.SiLU(),
            nn.Linear(64, NUM_BUCKETS),
        )

        # DreamerV3-style symexp-spaced bins in original value space.
        bucket_centers = _make_bucket_centers()
        self.register_buffer("bucket_centers", bucket_centers)
        self.register_buffer("bucket_symlog", symlog(bucket_centers))

        self._init_weights()

        # Cast conv stack to bf16 for native bf16 execution (no autocast overhead).
        # unit_head stays fp32 for value precision.
        if _USE_BF16:
            self.conv = self.conv.to(dtype=torch.bfloat16)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Near-uniform init for final categorical logits (DreamerV3 convention)
        nn.init.zeros_(self.unit_head[-1].weight)
        nn.init.zeros_(self.unit_head[-1].bias)

    def encode_map(
        self,
        map_tensor: torch.Tensor,    # (R, C, H, W) — R unique rounds
        valid_mask: torch.Tensor,     # (R, 1, H, W)
    ) -> torch.Tensor:
        """Run CNN once per unique round and return spatial feature maps.

        Conv stack runs in bf16 natively (weights are bf16).
        valid_mask applied so padded area contributes no signal.
        """
        x = map_tensor
        if _USE_BF16 and x.dtype != torch.bfloat16:
            x = x.bfloat16()
        feat = self.conv(x)
        if feat.dtype != valid_mask.dtype:
            valid_mask = valid_mask.to(feat.dtype)
        return feat * valid_mask

    def unit_values(
        self,
        feat: torch.Tensor,          # (R, conv_dim, H', W') — cached from encode_map
        round_scalars: torch.Tensor,  # (R, GLOBAL_SCALAR_DIM)
        round_idx: torch.Tensor,      # (B,) — maps each unit-step to its round index
        unit_xy: torch.Tensor,        # (B, 2) — x, y positions in ORIGINAL map coords
        unit_context: torch.Tensor,   # (B, UNIT_CONTEXT_DIM)
        actor_latent: torch.Tensor,   # (B, ACTOR_LATENT_DIM) — detached rollout latent
    ) -> torch.Tensor:
        """Compute per-unit marginal values by indexing cached feature maps.

        Returns:
            v_unit: (B,) — per-unit marginal value
        """
        fH, fW = feat.shape[2], feat.shape[3]
        if unit_xy.dtype == torch.long:
            ux = unit_xy[:, 0]
            uy = unit_xy[:, 1]
        else:
            ux = unit_xy[:, 0].round().to(torch.long)
            uy = unit_xy[:, 1].round().to(torch.long)
        if round_idx.dtype == torch.long:
            ri = round_idx
        else:
            ri = round_idx.to(torch.long)
        ux = ux.clamp(0, fW - 1)
        uy = uy.clamp(0, fH - 1)

        # Index into cached feature maps: (B, conv_dim)
        # feat may be bf16 from conv stack; cast to fp32 for the value MLP
        unit_tile_feat = feat[ri, :, uy, ux].float()
        unit_round_scalars = round_scalars[ri]

        # Per-unit categorical value prediction (DreamerV3-style)
        unit_in = torch.cat(
            [unit_tile_feat, unit_round_scalars, unit_context, actor_latent],
            dim=-1,
        )
        logits = self.unit_head(unit_in)  # (B, NUM_BUCKETS)
        return decode_value(logits, self.bucket_centers)  # (B,)

    def unit_value_logits(
        self,
        feat: torch.Tensor,
        round_scalars: torch.Tensor,
        round_idx: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_context: torch.Tensor,
        actor_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like unit_values but also returns raw logits for training loss.

        Returns:
            (decoded_values (B,), logits (B, NUM_BUCKETS))
        """
        fH, fW = feat.shape[2], feat.shape[3]
        if unit_xy.dtype == torch.long:
            ux = unit_xy[:, 0]
            uy = unit_xy[:, 1]
        else:
            ux = unit_xy[:, 0].round().to(torch.long)
            uy = unit_xy[:, 1].round().to(torch.long)
        if round_idx.dtype == torch.long:
            ri = round_idx
        else:
            ri = round_idx.to(torch.long)
        ux = ux.clamp(0, fW - 1)
        uy = uy.clamp(0, fH - 1)

        unit_tile_feat = feat[ri, :, uy, ux].float()
        unit_round_scalars = round_scalars[ri]

        unit_in = torch.cat(
            [unit_tile_feat, unit_round_scalars, unit_context, actor_latent],
            dim=-1,
        )
        logits = self.unit_head(unit_in)  # (B, NUM_BUCKETS)
        values = decode_value(logits, self.bucket_centers)  # (B,)
        return values, logits

    def forward(
        self,
        map_tensor: torch.Tensor,    # (B, C, H, W)
        valid_mask: torch.Tensor,     # (B, 1, H, W)
        round_scalars: torch.Tensor,  # (B, GLOBAL_SCALAR_DIM)
        unit_xy: torch.Tensor,        # (B, 2) — x, y positions (long)
        unit_context: torch.Tensor,   # (B, UNIT_CONTEXT_DIM)
        actor_latent: torch.Tensor,   # (B, ACTOR_LATENT_DIM)
    ) -> torch.Tensor:
        """Full forward pass (non-cached path). Returns per-unit value."""
        feat = self.encode_map(map_tensor, valid_mask)
        round_idx = torch.arange(map_tensor.shape[0], device=map_tensor.device)
        return self.unit_values(
            feat, round_scalars, round_idx, unit_xy, unit_context, actor_latent,
        )


def _materialize_critic_inputs(centralized_data: dict) -> None:
    """Pre-materialize all round tensors on GPU as bf16.

    Replaces the per-chunk CPU→GPU copy in _load_critic_round_batch with a
    single bulk transfer. Stores results in centralized_data so subsequent
    calls just index into the pre-built tensors.
    """
    global_states = centralized_data["global_states"]
    sizes = centralized_data["global_sizes"]
    R = len(global_states)
    if R == 0:
        return

    max_w = int(sizes[:, 0].max().item())
    max_h = int(sizes[:, 1].max().item())
    gpu_dtype = torch.bfloat16 if _USE_BF16 else torch.float32

    # Build on CPU then transfer once
    obs_cpu = torch.zeros(R, GLOBAL_MAP_CHANNELS, max_h, max_w, dtype=torch.float32)
    mask_cpu = torch.zeros(R, 1, max_h, max_w, dtype=torch.float32)

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
            mask_cpu[i, 0, :mh, :mw] = 1.0

    centralized_data["gpu_obs"] = obs_cpu.to(device=DEVICE, dtype=gpu_dtype)
    centralized_data["gpu_mask"] = mask_cpu.to(device=DEVICE, dtype=gpu_dtype)
    centralized_data["gpu_scalars"] = centralized_data["global_scalars"].to(device=DEVICE)

    # Free CPU-side bytearrays — no longer needed
    centralized_data["global_states"] = None

    vram_mb = (centralized_data["gpu_obs"].nbytes
               + centralized_data["gpu_mask"].nbytes
               + centralized_data["gpu_scalars"].nbytes) / (1024 * 1024)
    print(f"  Critic inputs materialized on GPU: {R} rounds, "
          f"{max_h}x{max_w} maps, {gpu_dtype}, {vram_mb:.0f} MB")


def _load_critic_round_batch(
    centralized_data: dict,
    round_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch a round batch from pre-materialized GPU tensors."""
    ids = round_ids.long()
    return (
        centralized_data["gpu_obs"][ids],
        centralized_data["gpu_mask"][ids],
        centralized_data["gpu_scalars"][ids],
    )


def _chunk_offsets_from_ids(
    chunk_ids: torch.Tensor,
    n_chunks: int,
) -> list[int]:
    """Return CPU-side prefix offsets for contiguous sorted chunk ids."""
    if n_chunks <= 0:
        return [0]
    if chunk_ids.numel() == 0:
        return [0] * (n_chunks + 1)
    counts = torch.bincount(chunk_ids, minlength=n_chunks)
    offsets = torch.empty(n_chunks + 1, dtype=torch.long, device=counts.device)
    offsets[0] = 0
    offsets[1:] = counts.cumsum(0)
    return offsets.cpu().tolist()


def _ortho_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init (CleanRL-style). Improves early exploration."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def sample_logits_gumbel(
    logits: torch.Tensor,
    temperature: float = 1.0,
    *,
    return_log_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample exactly from softmax(logits / temperature) via Gumbel-max."""
    scaled = logits if temperature == 1.0 else logits / temperature
    gumbel = -torch.empty_like(scaled).exponential_().log()
    actions = (scaled + gumbel).argmax(dim=-1)
    if not return_log_prob:
        return actions, None
    chosen_logits = scaled.gather(1, actions.unsqueeze(1)).squeeze(1)
    chosen = chosen_logits - torch.logsumexp(scaled, dim=-1)
    return actions, chosen


def action_log_prob_and_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return chosen-action log_probs and categorical entropy from logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return chosen, entropy


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
    ):
        self.max_batch = max_batch
        self._model = model
        self._temperature = temperature
        self._return_log_prob = return_log_prob
        self._graph: torch.cuda.CUDAGraph | None = None
        # Static buffers (set during capture)
        self._static_input: torch.Tensor | None = None
        # Output references (point into graph memory)
        self._s_bt_act = self._s_bd_act = self._s_cd_act = self._s_md_act = None
        self._s_lp = self._s_h = None

    def capture(self) -> None:
        """Capture the CUDA graph. Call once after model is on device."""
        M = self.max_batch
        costs_ti, costs_ax, wall_idx, nobld_idx, unit_idx, friendly_idx, core_idx = _mask_indices(DEVICE)
        ore_ti_idx, ore_ax_idx = _harv_mask_indices(DEVICE)
        cd_mask = torch.tensor(_CONV_DIR_MASK, dtype=torch.bool, device=DEVICE)

        # Match model dtype for all buffers
        mdtype = next(self._model.parameters()).dtype
        self._dtype = mdtype
        self._static_input = torch.zeros(M, OBS_SIZE + H_DIM, device=DEVICE, dtype=mdtype)
        static_g_bt = torch.empty(M, N_BUILD, device=DEVICE, dtype=mdtype)
        static_g_bd = torch.empty(M, N_DIR, device=DEVICE, dtype=mdtype)
        static_g_cd = torch.empty(M, N_CONV_DIR, device=DEVICE, dtype=mdtype)
        static_g_md = torch.empty(M, N_DIR, device=DEVICE, dtype=mdtype)

        # Warmup
        with torch.no_grad():
            self._model.infer_packed(self._static_input)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            with torch.no_grad():
                bt_l, bd_l, cd_l, md_l, s_h = self._model.infer_packed(self._static_input)
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

                ore_here = (obs[:, ore_ti_idx] == 1.0) | (obs[:, ore_ax_idx] == 1.0)
                empty_dir = obs[:, nobld_idx] == 1.0
                harv_bd_m = bd_m & ore_here & empty_dir
                bt_m[:, HARVESTER_BUILD_TYPE] &= harv_bd_m.any(dim=1)

                bt_l.masked_fill_(~bt_m, -1e9)
                cd_l.masked_fill_(~cd_mask, -1e9)

                # Apply temperature scaling for Gumbel-max sampling
                _T = self._temperature
                if _T != 1.0:
                    bt_l = bt_l / _T
                    cd_l = cd_l / _T

                static_g_bt.exponential_()
                s_bt_act = (bt_l - static_g_bt.log_()).argmax(dim=-1)

                harv_rows = s_bt_act == HARVESTER_BUILD_TYPE
                bd_m_final = torch.where(harv_rows.unsqueeze(1), harv_bd_m, bd_m)
                bd_l.masked_fill_(~bd_m_final, -1e9)
                if _T != 1.0:
                    bd_l = bd_l / _T
                    md_l = md_l / _T

                static_g_bd.exponential_()
                static_g_cd.exponential_()
                static_g_md.exponential_()
                s_bd_act = (bd_l - static_g_bd.log_()).argmax(dim=-1)
                s_cd_act = (cd_l - static_g_cd.log_()).argmax(dim=-1)
                s_md_act = (md_l - static_g_md.log_()).argmax(dim=-1)

                s_lp = None
                if self._return_log_prob:
                    s_bt_lp = bt_l.gather(1, s_bt_act.unsqueeze(1)).squeeze(1) - bt_l.logsumexp(dim=-1)
                    s_bd_lp = bd_l.gather(1, s_bd_act.unsqueeze(1)).squeeze(1) - bd_l.logsumexp(dim=-1)
                    s_cd_lp = cd_l.gather(1, s_cd_act.unsqueeze(1)).squeeze(1) - cd_l.logsumexp(dim=-1)
                    s_md_lp = md_l.gather(1, s_md_act.unsqueeze(1)).squeeze(1) - md_l.logsumexp(dim=-1)

                    bd_causal = (s_bt_act != 0).float()
                    cd_causal = (
                        (s_bt_act == CONVEYOR_BUILD_TYPES[0])
                        | (s_bt_act == CONVEYOR_BUILD_TYPES[1])
                        | (s_bt_act == CONVEYOR_BUILD_TYPES[2])
                    ).float()
                    s_lp = s_bt_lp + s_bd_lp * bd_causal + s_cd_lp * cd_causal + s_md_lp

        self._s_bt_act = s_bt_act
        self._s_bd_act = s_bd_act
        self._s_cd_act = s_cd_act
        self._s_md_act = s_md_act
        self._s_lp = s_lp
        self._s_h = s_h

    @torch.no_grad()
    def __call__(self, trunk_in: torch.Tensor):
        """Run graphed inference. Returns (bt, bd, cd, md, lp, h_next) sliced to B."""
        B = trunk_in.shape[0]
        self._static_input[:B].copy_(trunk_in)
        if B < self.max_batch:
            self._static_input[B:].zero_()
        self._graph.replay()
        return (
            self._s_bt_act[:B], self._s_bd_act[:B], self._s_cd_act[:B],
            self._s_md_act[:B],
            self._s_lp[:B] if self._s_lp is not None else None,
            self._s_h[:B],
        )

    @property
    def captured(self) -> bool:
        return self._graph is not None


class BuilderMLP(nn.Module):
    """MLP actor with shared encoder and separate policy heads.

    Value estimation is handled by CentralizedCritic (not part of this model).

    Args:
        hidden: Hidden layer sizes. Defaults to HIDDEN (production).
                Use TEACHER_HIDDEN for larger teacher model.
    """

    def __init__(self, hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or HIDDEN
        self._hidden = hidden

        # Per-tile linear encoder
        self.tile_linear = _ortho_init(nn.Linear(TILE_INPUT_DIM, TILE_EMBED_DIM))
        # Depthwise 3x3 conv for spatial mixing between adjacent tiles
        self.tile_conv = nn.Conv2d(
            TILE_EMBED_DIM, TILE_EMBED_DIM, 3, padding=1,
            groups=TILE_EMBED_DIM, bias=True,
        )
        # Pointwise 1x1 conv mixes channels after depthwise spatial aggregation.
        self.tile_pointwise = nn.Conv2d(
            TILE_EMBED_DIM, TILE_EMBED_DIM, 1, bias=True,
        )
        nn.init.orthogonal_(self.tile_conv.weight, gain=0.1)
        nn.init.zeros_(self.tile_conv.bias)
        nn.init.orthogonal_(self.tile_pointwise.weight, gain=1.0)
        nn.init.zeros_(self.tile_pointwise.bias)
        self.tile_rms = RMSNorm(TILE_EMBED_DIM)
        self.tile_act = nn.SiLU()
        # Compress 25×8=200 spatially-mixed tile features → 64-dim summary
        self.tile_compress = nn.Sequential(
            _ortho_init(nn.Linear(TILE_GRID_N * TILE_EMBED_DIM, COMPRESS_DIM)),
            RMSNorm(COMPRESS_DIM),
            nn.SiLU(),
        )
        self.register_buffer('_tile_dxdy', _TILE_DXDY.clone())

        # Shared encoder: first hidden layer only
        enc_dim = hidden[0]
        self.encoder = nn.Sequential(
            _ortho_init(nn.Linear(TRUNK_INPUT_SIZE, enc_dim)),
            RMSNorm(enc_dim),
            nn.SiLU(),
        )

        # Actor trunk: remaining hidden layers
        actor_layers: list[nn.Module] = []
        prev = enc_dim
        for i, h in enumerate(hidden[1:], start=1):
            actor_layers.append(_ortho_init(nn.Linear(prev, h)))
            if ACTOR_RMSNORM_LAYERS[i]:
                actor_layers.append(RMSNorm(h))
            actor_layers.append(nn.SiLU())
            prev = h
        self.actor_trunk = nn.Sequential(*actor_layers)

        # Policy heads: small std for near-uniform initial policy
        self.head_build_type = _ortho_init(nn.Linear(prev, N_BUILD), std=0.01)
        self.head_build_dir = _ortho_init(nn.Linear(prev, N_DIR), std=0.01)
        self.head_conv_dir = _ortho_init(nn.Linear(prev, N_CONV_DIR), std=0.01)
        self.head_move_dir = _ortho_init(nn.Linear(prev, N_DIR), std=0.01)
        self.head_h_gate = _ortho_init(nn.Linear(prev, H_DIM), std=0.01)  # Wz: update gate
        self.head_h_cand = _ortho_init(nn.Linear(prev, H_DIM), std=0.01)  # Wh: candidate

        # Give new policies a structured exploration prior toward economy actions.
        with torch.no_grad():
            self.head_build_type.bias.copy_(
                torch.tensor(INITIAL_BUILD_TYPE_BIAS, dtype=torch.float32),
            )

    def train(self, mode=True):
        # Invalidate fused head cache when entering train mode
        if mode and hasattr(self, '_cached_head_w'):
            del self._cached_head_w
            del self._cached_head_b
        return super().train(mode)

    def _encode_tiles(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract tiles, encode per-tile, spatial conv, compress to summary."""
        B = obs.shape[0]
        tile_flat = obs[:, :TILE_GRID_N * TILE_FEATURES]
        tiles = tile_flat.reshape(B, TILE_GRID_N, TILE_FEATURES)
        dxdy = self._tile_dxdy.unsqueeze(0).expand(B, -1, -1)
        tile_in = torch.cat([dxdy, tiles], dim=-1)          # (B, 25, 21)
        h = F.silu(self.tile_linear(tile_in))               # (B, 25, 8)
        # Depthwise 3x3 + pointwise 1x1 + residual + SiLU
        h_grid = h.reshape(B, 5, 5, TILE_EMBED_DIM).permute(0, 3, 1, 2)
        conv_out = self.tile_pointwise(self.tile_conv(h_grid))  # (B, 8, 5, 5)
        conv_out = conv_out.permute(0, 2, 3, 1).reshape(B, TILE_GRID_N, TILE_EMBED_DIM)
        h = self.tile_rms(h + conv_out)
        h = self.tile_act(h)                                # (B, 25, 8)
        # Compress to fixed-size tile summary
        h_flat = h.reshape(B, TILE_GRID_N * TILE_EMBED_DIM) # (B, 200)
        return self.tile_compress(h_flat)                    # (B, 64)

    def _build_trunk_input(self, obs: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Build trunk input: compressed tiles + globals + pos_embed + h_prev."""
        obs_prefix = self._build_obs_prefix(obs)
        return torch.cat([obs_prefix, h_prev], dim=-1)  # (B, 103)

    def _build_obs_prefix(self, obs: torch.Tensor) -> torch.Tensor:
        """Build the observation-only trunk prefix."""
        tile_summary = self._encode_tiles(obs)  # (B, 64)
        globals_pos = obs[:, TILE_GRID_N * TILE_FEATURES:]  # (B, 23) globals+pos_embed
        return torch.cat([tile_summary, globals_pos], dim=-1)  # (B, 87)

    def _obs_encoder_pre(self, obs: torch.Tensor) -> torch.Tensor:
        """Precompute the obs-only contribution to the first encoder linear."""
        obs_prefix = self._build_obs_prefix(obs)
        enc_linear = self.encoder[0]
        return F.linear(obs_prefix, enc_linear.weight[:, :OBS_PREFIX_SIZE], enc_linear.bias)

    def _latent_from_obs_pre(self, obs_pre: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Run the recurrent actor from a precomputed obs-side encoder term."""
        enc_linear = self.encoder[0]
        h_pre = F.linear(h_prev, enc_linear.weight[:, OBS_PREFIX_SIZE:], None)
        enc = self.encoder[1](obs_pre + h_pre)
        enc = self.encoder[2](enc)
        return self.actor_trunk(enc)

    def _forward_from_obs_pre(self, obs_pre: torch.Tensor, h_prev: torch.Tensor):
        """Forward pass from precomputed obs-side encoder activations."""
        h = self._latent_from_obs_pre(obs_pre, h_prev)
        return self._forward_from_hidden(h)

    def actor_latent(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Return the pre-head actor latent used for policy logits."""
        obs_pre = self._obs_encoder_pre(x)
        return self._latent_from_obs_pre(obs_pre, h_prev)

    def _forward_from_hidden(self, h: torch.Tensor):
        """Project actor latent to policy heads."""
        # Fuse 4 head matmuls into one kernel launch
        # Cache fused weights in eval mode (inference hot path)
        if self.training:
            combined_w = torch.cat([
                self.head_build_type.weight, self.head_build_dir.weight,
                self.head_conv_dir.weight,
                self.head_move_dir.weight, self.head_h_gate.weight, self.head_h_cand.weight,
            ], dim=0)
            combined_b = torch.cat([
                self.head_build_type.bias, self.head_build_dir.bias,
                self.head_conv_dir.bias,
                self.head_move_dir.bias, self.head_h_gate.bias, self.head_h_cand.bias,
            ])
        else:
            if not hasattr(self, '_cached_head_w'):
                self._cached_head_w = torch.cat([
                    self.head_build_type.weight, self.head_build_dir.weight,
                    self.head_conv_dir.weight,
                    self.head_move_dir.weight, self.head_h_gate.weight, self.head_h_cand.weight,
                ], dim=0)
                self._cached_head_b = torch.cat([
                    self.head_build_type.bias, self.head_build_dir.bias,
                    self.head_conv_dir.bias,
                    self.head_move_dir.bias, self.head_h_gate.bias, self.head_h_cand.bias,
                ])
            combined_w = self._cached_head_w
            combined_b = self._cached_head_b

        combined = F.linear(h, combined_w, combined_b)
        bt_l, bd_l, cd_l, md_l, h_gate, h_cand = combined.split([N_BUILD, N_DIR, N_CONV_DIR, N_DIR, H_DIM, H_DIM], dim=-1)
        return bt_l, bd_l, cd_l, md_l, h_gate, h_cand

    def _forward_from_trunk_in(self, trunk_in: torch.Tensor):
        enc = self.encoder(trunk_in)
        h = self.actor_trunk(enc)
        return self._forward_from_hidden(h)

    def forward(self, x, h_prev):
        obs_pre = self._obs_encoder_pre(x)
        bt_l, bd_l, cd_l, md_l, h_gate, h_cand = self._forward_from_obs_pre(obs_pre, h_prev)
        z = torch.sigmoid(h_gate)
        h_next = (1.0 - z) * h_prev + z * h_cand
        return bt_l, bd_l, cd_l, md_l, h_next

    def infer_packed(self, packed: torch.Tensor):
        """Inference-only path for already-packed [obs, h_prev] batches."""
        obs = packed[:, :OBS_SIZE]
        h_prev = packed[:, OBS_SIZE:]
        obs_pre = self._obs_encoder_pre(obs)
        bt_l, bd_l, cd_l, md_l, h_gate, h_cand = self._forward_from_obs_pre(obs_pre, h_prev)
        z = torch.sigmoid(h_gate)
        h_next = (1.0 - z) * h_prev + z * h_cand
        return bt_l, bd_l, cd_l, md_l, h_next

    def _policy_state_modules(self) -> list[nn.Module]:
        """Policy modules only in tensor_lib serialization order (critic excluded)."""
        tile_modules = [self.tile_linear, self.tile_conv, self.tile_pointwise, self.tile_rms]
        compress_modules = [m for m in self.tile_compress if isinstance(m, (nn.Linear, RMSNorm))]
        enc_modules = [m for m in self.encoder if isinstance(m, (nn.Linear, RMSNorm))]
        actor_modules = [m for m in self.actor_trunk if isinstance(m, (nn.Linear, RMSNorm))]
        return tile_modules + compress_modules + enc_modules + actor_modules + [
            self.head_build_type, self.head_build_dir, self.head_conv_dir, self.head_move_dir,
            self.head_h_gate, self.head_h_cand,
        ]

    def export_weights(self) -> list[list[float]]:
        """Export weights as JSON list. Transposes Linear to match tensor_lib (in, out) format."""
        state: list[list[float]] = []
        for layer in self._policy_state_modules():
            if isinstance(layer, nn.Linear):
                state.append(layer.weight.T.contiguous().view(-1).tolist())
                state.append(layer.bias.view(-1).tolist())
            elif isinstance(layer, nn.Conv2d):
                # (C_out, C_in/groups, kH, kW) → flatten as-is
                state.append(layer.weight.view(-1).tolist())
                if layer.bias is not None:
                    state.append(layer.bias.view(-1).tolist())
            else:
                state.append(layer.weight.view(-1).tolist())
        return state

    def import_weights(self, state: list[list[float]]) -> None:
        """Load weights from tensor_lib JSON format."""
        modules = self._policy_state_modules()
        full_entries = sum(
            (2 if isinstance(m, (nn.Linear, nn.Conv2d)) else 1)
            for m in modules
        )
        # Also support loading without conv_dir head (2 fewer entries)
        full_no_conv_dir = full_entries - 2
        # Legacy exports before the pointwise 1x1 conv existed.
        full_no_pointwise = full_entries - 3
        full_no_pointwise_no_conv_dir = full_no_conv_dir - 3
        # Legacy exports before the tile RMSNorm existed.
        full_no_tile_rms = full_entries - 1
        full_no_tile_rms_no_conv_dir = full_no_conv_dir - 1
        pointwise_weight_size = self.tile_pointwise.weight.numel()
        tile_rms_size = self.tile_rms.weight.numel()
        load_pointwise = True
        load_tile_rms = True

        if len(state) == full_entries:
            load_conv_dir = True
        elif len(state) == full_no_conv_dir and len(state[4]) == pointwise_weight_size:
            load_conv_dir = False
        elif (
            len(state) == full_no_tile_rms
            and len(state[4]) == pointwise_weight_size
            and len(state[6]) != tile_rms_size
        ):
            load_conv_dir = True
            load_tile_rms = False
        elif (
            len(state) == full_no_tile_rms_no_conv_dir
            and len(state[4]) == pointwise_weight_size
            and len(state[6]) != tile_rms_size
        ):
            load_conv_dir = False
            load_tile_rms = False
        elif len(state) == full_no_pointwise:
            load_conv_dir = True
            load_pointwise = False
            load_tile_rms = False
        elif len(state) == full_no_pointwise_no_conv_dir:
            load_conv_dir = False
            load_pointwise = False
            load_tile_rms = False
        elif len(state) == full_no_conv_dir:
            load_conv_dir = False
        else:
            raise AssertionError(
                f"State has {len(state)} entries, expected {full_entries} "
                f"(or {full_no_conv_dir} without conv_dir head)"
            )

        idx = 0
        dev = next(self.parameters()).device
        for layer in modules:
            if layer is self.head_conv_dir and not load_conv_dir:
                continue
            if layer is self.tile_pointwise and not load_pointwise:
                continue
            if layer is self.tile_rms and not load_tile_rms:
                continue
            if isinstance(layer, nn.Linear):
                w_flat = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                in_f, out_f = layer.in_features, layer.out_features
                layer.weight.data = w_flat.view(in_f, out_f).T.contiguous()
                idx += 1
                layer.bias.data = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                idx += 1
            elif isinstance(layer, nn.Conv2d):
                w_flat = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                layer.weight.data = w_flat.view(layer.weight.shape)
                idx += 1
                if layer.bias is not None:
                    layer.bias.data = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                    idx += 1
            else:
                # RMSNorm
                layer.weight.data = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                idx += 1

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
                    self.load_state_dict(data, strict=False)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute policy masks and causal coefficients for fixed rollout actions."""
    bt_mask, bd_mask = compute_action_masks(obs)
    harv_bd_mask = compute_harvester_build_dir_mask(obs, bd_mask)
    bt_mask = bt_mask.clone()
    bt_mask[:, HARVESTER_BUILD_TYPE] &= harv_bd_mask.any(dim=1)
    bd_mask = compute_build_dir_mask_for_type(obs, bt_act, bd_mask, harv_bd_mask)
    bd_causal = (bt_act != 0).float()
    cd_causal = (
        (bt_act == CONVEYOR_BUILD_TYPES[0])
        | (bt_act == CONVEYOR_BUILD_TYPES[1])
        | (bt_act == CONVEYOR_BUILD_TYPES[2])
    ).float()
    return bt_mask, bd_mask, bd_causal, cd_causal


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


def _recurrent_logprob_pass(
    model: 'BuilderMLP',
    obs_pre: torch.Tensor,
    h0: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    md_act: torch.Tensor,
    bt_mask: torch.Tensor,
    bd_mask: torch.Tensor,
    bd_causal: torch.Tensor,
    cd_causal: torch.Tensor,
) -> torch.Tensor:
    """Sequential recurrent pass that returns chosen-action log-probs."""
    B, K_max, _ = obs_pre.shape
    dev = obs_pre.device
    h_t = h0
    log_probs = torch.empty(B, K_max, device=dev)
    cd_mask = conveyor_dir_mask(dev, B)

    for t in range(K_max):
        bt_l, bd_l, cd_l, md_l, h_gate, h_cand = model._forward_from_obs_pre(obs_pre[:, t], h_t)
        z = torch.sigmoid(h_gate)
        h_t = (1.0 - z) * h_t + z * h_cand

        bt_l = bt_l.masked_fill(~bt_mask[:, t], -1e9)
        bd_l = bd_l.masked_fill(~bd_mask[:, t], -1e9)
        cd_l = cd_l.masked_fill(~cd_mask, -1e9)

        bt_lp = F.log_softmax(bt_l, dim=-1).gather(1, bt_act[:, t].unsqueeze(1)).squeeze(1)
        bd_lp = F.log_softmax(bd_l, dim=-1).gather(1, bd_act[:, t].unsqueeze(1)).squeeze(1)
        cd_lp = F.log_softmax(cd_l, dim=-1).gather(1, cd_act[:, t].unsqueeze(1)).squeeze(1)
        md_lp = F.log_softmax(md_l, dim=-1).gather(1, md_act[:, t].unsqueeze(1)).squeeze(1)
        log_probs[:, t] = bt_lp + bd_lp * bd_causal[:, t] + cd_lp * cd_causal[:, t] + md_lp

    return log_probs


def _recurrent_actor_pass(
    model: 'BuilderMLP',
    obs_pre: torch.Tensor,
    h0: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    md_act: torch.Tensor,
    bt_mask: torch.Tensor,
    bd_mask: torch.Tensor,
    bd_causal: torch.Tensor,
    cd_causal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential recurrent pass that returns log-probs and entropies."""
    B, K_max, _ = obs_pre.shape
    dev = obs_pre.device
    h_t = h0
    new_lps = torch.empty(B, K_max, device=dev)
    entropies = torch.empty(B, K_max, device=dev)
    cd_mask = conveyor_dir_mask(dev, B)

    for t in range(K_max):
        bt_l, bd_l, cd_l, md_l, h_gate, h_cand = model._forward_from_obs_pre(obs_pre[:, t], h_t)
        z = torch.sigmoid(h_gate)
        h_t = (1.0 - z) * h_t + z * h_cand

        bt_l = bt_l.masked_fill(~bt_mask[:, t], -1e9)
        bd_l = bd_l.masked_fill(~bd_mask[:, t], -1e9)
        cd_l = cd_l.masked_fill(~cd_mask, -1e9)

        bt_lp, bt_ent = action_log_prob_and_entropy(bt_l, bt_act[:, t])
        bd_lp, bd_ent = action_log_prob_and_entropy(bd_l, bd_act[:, t])
        cd_lp, cd_ent = action_log_prob_and_entropy(cd_l, cd_act[:, t])
        md_lp, md_ent = action_log_prob_and_entropy(md_l, md_act[:, t])

        new_lps[:, t] = bt_lp + bd_lp * bd_causal[:, t] + cd_lp * cd_causal[:, t] + md_lp
        entropies[:, t] = bt_ent + bd_ent * bd_causal[:, t] + cd_ent * cd_causal[:, t] + md_ent

    return new_lps, entropies


def _sequential_logprobs(
    model: 'BuilderMLP',
    obs: torch.Tensor,
    h: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    md_act: torch.Tensor,
    chunks: list[tuple[int, int]],
    bptt_k: int = BPTT_K,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Recompute log-probs with sequential h propagation (no grad).

    Batches across chunks at each timestep (same pattern as BPTT actor loop)
    for good GPU utilization. Chunks are sorted by length to minimize padding.
    """
    N = obs.shape[0]
    dev = obs.device
    log_probs = torch.empty(N, device=dev)

    B_chunk = max(1, batch_size // bptt_k)
    all_starts, all_lens = _chunk_metadata_tensors(chunks, dev, sort_desc=True)
    full_chunk_ids, tail_chunk_ids = _split_chunk_ids(all_lens, bptt_k)
    obs_pre_all = model._obs_encoder_pre(obs)
    bt_mask_all, bd_mask_all, bd_causal_all, cd_causal_all = _precompute_policy_masks(obs, bt_act)

    def _run_chunk_group(chunk_ids: torch.Tensor, *, full_length: bool) -> None:
        for cb_start in range(0, chunk_ids.shape[0], B_chunk):
            batch_ids = chunk_ids[cb_start:cb_start + B_chunk]
            B = batch_ids.shape[0]
            chunk_starts = all_starts[batch_ids]
            chunk_lens = all_lens[batch_ids]
            K_max = bptt_k if full_length else int(chunk_lens[0].item())

            t_range = torch.arange(K_max, device=dev).unsqueeze(0)
            idx_grid = (chunk_starts.unsqueeze(1) + t_range).clamp(max=N - 1)
            flat_idx = idx_grid.reshape(-1)

            all_obs_pre = obs_pre_all[flat_idx].reshape(B, K_max, -1)
            all_bt = bt_act[flat_idx].reshape(B, K_max)
            all_bd = bd_act[flat_idx].reshape(B, K_max)
            all_cd = cd_act[flat_idx].reshape(B, K_max)
            all_md = md_act[flat_idx].reshape(B, K_max)
            all_bt_mask = bt_mask_all[flat_idx].reshape(B, K_max, -1)
            all_bd_mask = bd_mask_all[flat_idx].reshape(B, K_max, -1)
            all_bd_causal = bd_causal_all[flat_idx].reshape(B, K_max)
            all_cd_causal = cd_causal_all[flat_idx].reshape(B, K_max)

            step_lps = _recurrent_logprob_pass(
                model,
                all_obs_pre,
                h[chunk_starts],
                all_bt,
                all_bd,
                all_cd,
                all_md,
                all_bt_mask,
                all_bd_mask,
                all_bd_causal,
                all_cd_causal,
            )

            if full_length:
                log_probs[flat_idx] = step_lps.reshape(-1)
                continue

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
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    cd_act: torch.Tensor,
    md_act: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    centralized_data: dict,
    old_values: torch.Tensor | None = None,
    ep_boundaries: list[tuple[int, int]] | None = None,
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    bptt_k: int = BPTT_K,
    clip_eps: float = 0.2,
    clip_eps_higher: float | None = 0.28,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    clip_vloss: bool = True,
    value_clip: float = 0.4,
    profile: dict[str, float] | None = None,
    bptt_chunks: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    """PPO update with truncated BPTT for recurrent hidden state.

    Processes contiguous K-step windows from episodes. Within each window,
    h is recomputed sequentially so gradients flow through K timesteps.
    """
    N = obs.shape[0]
    sum_pi, sum_vf, sum_ent = 0.0, 0.0, 0.0
    sum_kl, sum_clip_frac = 0.0, 0.0
    n_updates = 0

    critic, critic_opt = get_centralized_critic()
    critic.train()

    # Explained variance
    if old_values is not None:
        with torch.no_grad():
            var_y = returns.var().item()
            ev = 0.0 if var_y == 0 else 1.0 - (returns - old_values).var().item() / var_y
    else:
        ev = 0.0

    # Build BPTT chunks from episode boundaries
    if bptt_chunks is None:
        if ep_boundaries is None:
            ep_boundaries = [(0, N)]
        bptt_chunks = _build_bptt_chunks(ep_boundaries, bptt_k)
    n_chunks = len(bptt_chunks)
    B_chunk = max(1, minibatch_size // bptt_k)
    dev = obs.device

    # Pre-build chunk metadata as GPU tensors (avoids per-batch CPU-GPU sync)
    all_chunk_starts, all_chunk_lens = _chunk_metadata_tensors(bptt_chunks, dev)

    # Normalize advantages globally (not per-minibatch) to avoid noisy
    # normalization on small/correlated chunk batches from length sorting
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    with torch.no_grad():
        bt_mask_all, bd_mask_all, bd_causal_all, cd_causal_all = _precompute_policy_masks(obs, bt_act)
    full_chunk_ids, tail_chunk_ids = _split_chunk_ids(all_chunk_lens, bptt_k)

    # ---- Phase 1: Actor updates with BPTT ----
    actor_stopped_early = False
    actor_epochs_ran = 0
    for epoch in range(ppo_epochs):
        actor_epochs_ran += 1
        approx_kl_acc, n_mb = 0.0, 0

        full_perm = full_chunk_ids
        if full_perm.numel() > 1:
            full_perm = full_perm[torch.randperm(full_perm.numel(), device=dev)]

        tail_perm = tail_chunk_ids
        if tail_perm.numel() > 1:
            tail_perm = tail_perm[torch.randperm(tail_perm.numel(), device=dev)]
            tail_perm = tail_perm[all_chunk_lens[tail_perm].argsort(descending=True, stable=True)]

        def _run_actor_group(chunk_ids: torch.Tensor, *, full_length: bool) -> None:
            nonlocal approx_kl_acc, n_mb, sum_kl, sum_clip_frac, sum_pi, sum_ent, n_updates
            for cb_start in range(0, chunk_ids.shape[0], B_chunk):
                batch_ids = chunk_ids[cb_start:cb_start + B_chunk]
                B = batch_ids.shape[0]
                chunk_starts = all_chunk_starts[batch_ids]
                chunk_lens = all_chunk_lens[batch_ids]
                K_max = bptt_k if full_length else int(chunk_lens[0].item())
                t_range = torch.arange(K_max, device=dev).unsqueeze(0)
                idx_grid = (chunk_starts.unsqueeze(1) + t_range).clamp(max=N - 1)
                flat_idx = idx_grid.reshape(-1)

                all_obs_pre = model._obs_encoder_pre(obs[flat_idx]).reshape(B, K_max, -1)
                all_bt = bt_act[flat_idx].reshape(B, K_max)
                all_bd = bd_act[flat_idx].reshape(B, K_max)
                all_cd = cd_act[flat_idx].reshape(B, K_max)
                all_md = md_act[flat_idx].reshape(B, K_max)
                all_old_lp = old_log_probs[flat_idx].reshape(B, K_max)
                all_adv = advantages[flat_idx].reshape(B, K_max)
                all_bt_mask = bt_mask_all[flat_idx].reshape(B, K_max, -1)
                all_bd_mask = bd_mask_all[flat_idx].reshape(B, K_max, -1)
                all_bd_causal = bd_causal_all[flat_idx].reshape(B, K_max)
                all_cd_causal = cd_causal_all[flat_idx].reshape(B, K_max)

                t_phase = time.perf_counter() if profile is not None else 0.0
                new_lps, entropies = _recurrent_actor_pass(
                    model,
                    all_obs_pre,
                    h[chunk_starts].detach(),
                    all_bt,
                    all_bd,
                    all_cd,
                    all_md,
                    all_bt_mask,
                    all_bd_mask,
                    all_bd_causal,
                    all_cd_causal,
                )
                if profile is not None:
                    profile["ppo_forward_s"] = profile.get("ppo_forward_s", 0.0) + (time.perf_counter() - t_phase)

                t_phase = time.perf_counter() if profile is not None else 0.0
                logratio = new_lps - all_old_lp
                ratio = logratio.exp()

                if full_length:
                    valid_denom = max(B * K_max, 1)
                    mb_adv = all_adv
                    with torch.no_grad():
                        approx_kl = (((ratio - 1) - logratio).sum() / valid_denom)
                        approx_kl_acc += approx_kl.item()
                        _hi = clip_eps_higher if clip_eps_higher is not None else clip_eps
                        clip_frac = (((ratio < (1 - clip_eps)) | (ratio > (1 + _hi))).float().sum() / valid_denom)
                        sum_kl += approx_kl.item()
                        sum_clip_frac += clip_frac.item()
                        n_mb += 1
                    pg1 = -mb_adv * ratio
                    _hi = clip_eps_higher if clip_eps_higher is not None else clip_eps
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + _hi)
                    pi_loss = torch.max(pg1, pg2).mean()
                    ent_loss = entropies.mean()
                else:
                    valid_mask = t_range < chunk_lens.unsqueeze(1)
                    valid_denom = valid_mask.sum()
                    mb_adv = all_adv.clone()
                    mb_adv[~valid_mask] = 0.0
                    with torch.no_grad():
                        approx_kl = (((ratio - 1) - logratio) * valid_mask).sum() / valid_denom
                        approx_kl_acc += approx_kl.item()
                        _hi = clip_eps_higher if clip_eps_higher is not None else clip_eps
                        clip_frac = (((ratio < (1 - clip_eps)) | (ratio > (1 + _hi))).float() * valid_mask).sum() / valid_denom
                        sum_kl += approx_kl.item()
                        sum_clip_frac += clip_frac.item()
                        n_mb += 1
                    pg1 = -mb_adv * ratio
                    _hi = clip_eps_higher if clip_eps_higher is not None else clip_eps
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + _hi)
                    pi_loss = (torch.max(pg1, pg2) * valid_mask).sum() / valid_denom
                    ent_loss = (entropies * valid_mask).sum() / valid_denom

                if profile is not None:
                    profile["ppo_policy_loss_s"] = profile.get("ppo_policy_loss_s", 0.0) + (time.perf_counter() - t_phase)

                optimizer.zero_grad(set_to_none=True)
                actor_loss = pi_loss - entropy_coeff * ent_loss
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                sum_pi += pi_loss.item()
                sum_ent += ent_loss.item()
                n_updates += 1

        if full_perm.numel() > 0:
            _run_actor_group(full_perm, full_length=True)
        if tail_perm.numel() > 0:
            _run_actor_group(tail_perm, full_length=False)

        if target_kl is not None and n_mb > 0 and approx_kl_acc / n_mb > target_kl:
            actor_stopped_early = True
            break

    # ---- Phase 2: Critic updates with random minibatches (full GPU utilization) ----
    n_critic_updates = 0
    critic_epochs = actor_epochs_ran if actor_stopped_early else ppo_epochs
    for epoch in range(critic_epochs):
        perm = torch.randperm(N, device=obs.device)
        for start in range(0, N, minibatch_size):
            idx = perm[start:start + minibatch_size]

            t_value = time.perf_counter() if profile is not None else 0.0
            critic_opt.zero_grad(set_to_none=True)

            vf_loss_sum = 0.0
            vf_denom = max(idx.shape[0], 1)
            ri = centralized_data["step_round_idx"][idx]
            u_xy = centralized_data["step_unit_xy"][idx]
            u_ctx = centralized_data["step_unit_ctx"][idx]
            u_latent = centralized_data["step_actor_latent"][idx]
            mb_returns = returns[idx]
            mb_old_values = old_values[idx] if old_values is not None else None
            unique_rounds, inverse = ri.unique(return_inverse=True)
            _CNN_CHUNK = CENTRALIZED_CNN_CHUNK
            _UNIT_CHUNK = 4096
            round_order = inverse.argsort(stable=True)
            sorted_inverse = inverse[round_order]
            n_round_chunks = (unique_rounds.shape[0] + _CNN_CHUNK - 1) // _CNN_CHUNK
            chunk_offsets = _chunk_offsets_from_ids(
                torch.div(sorted_inverse, _CNN_CHUNK, rounding_mode='floor'),
                n_round_chunks,
            )
            for chunk_idx in range(n_round_chunks):
                lo = chunk_offsets[chunk_idx]
                hi = chunk_offsets[chunk_idx + 1]
                if lo == hi:
                    continue
                c_start = chunk_idx * _CNN_CHUNK
                c_end = min(c_start + _CNN_CHUNK, unique_rounds.shape[0])
                c_rounds = unique_rounds[c_start:c_end]
                g_obs, g_mask, g_scalars = _load_critic_round_batch(centralized_data, c_rounds)
                feat = critic.encode_map(g_obs, g_mask)
                step_indices = round_order[lo:hi]
                local_round_idx = sorted_inverse[lo:hi] - c_start
                for u_start in range(0, step_indices.shape[0], _UNIT_CHUNK):
                    u_end = min(u_start + _UNIT_CHUNK, step_indices.shape[0])
                    si = step_indices[u_start:u_end]
                    lr = local_round_idx[u_start:u_end]
                    newvalue, logits = critic.unit_value_logits(
                        feat, g_scalars, lr, u_xy[si], u_ctx[si], u_latent[si],
                    )
                    # Dreamer4-style value clipping: decode, clip in scalar
                    # space, re-project, then compare both branches in the same
                    # smooth value-target family.
                    return_target = F.softmax(
                        value_target_logits(
                            mb_returns[si], critic.bucket_symlog,
                        ),
                        dim=-1,
                    )
                    v_loss_unclipped = F.cross_entropy(
                        logits, return_target, reduction='none',
                    )

                    if clip_vloss and mb_old_values is not None:
                        clipped_val = mb_old_values[si] + (newvalue - mb_old_values[si]).clamp(
                            -value_clip, value_clip,
                        )
                        clipped_logits = value_target_logits(
                            clipped_val, critic.bucket_symlog,
                        )
                        v_loss_clipped = F.cross_entropy(
                            clipped_logits, return_target, reduction='none',
                        )
                        v_loss_elems = torch.max(v_loss_unclipped, v_loss_clipped)
                    else:
                        v_loss_elems = v_loss_unclipped

                    vf_loss_sum += v_loss_elems.detach().sum().item()
                    (vf_coeff * v_loss_elems.sum() / vf_denom).backward()

            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()

            sum_vf += vf_loss_sum / vf_denom
            n_critic_updates += 1

            if profile is not None:
                profile["ppo_value_loss_s"] = profile.get("ppo_value_loss_s", 0.0) + (time.perf_counter() - t_value)

    return {
        "pi_loss": sum_pi / max(n_updates, 1),
        "vf_loss": sum_vf / max(n_critic_updates, 1),
        "entropy": sum_ent / max(n_updates, 1),
        "approx_kl": sum_kl / max(n_updates, 1),
        "clip_frac": sum_clip_frac / max(n_updates, 1),
        "explained_var": round(ev, 4),
        "n_updates": n_updates,
    }


def _parse_trajectories(
    trajectories: list[list[dict]],
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
    md_t = torch.empty(N, dtype=torch.long)
    lp_t = torch.empty(N, dtype=torch.float32)
    rw_t = torch.empty(N, dtype=torch.float32)
    step_game_t = torch.empty(N, dtype=torch.int32)
    ep_boundaries = []

    # Disable cyclic GC — the loop creates no reference cycles, and the
    # collector's O(live_objects) scan is expensive with many live tensors.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        idx = 0
        for traj_i, traj in enumerate(trajectories):
            start = idx
            game_idx = traj_game_map[traj_i] if traj_game_map and traj_i < len(traj_game_map) else 0
            traj_len = len(traj)
            end = idx + traj_len
            if traj_len == 0:
                ep_boundaries.append((start, end))
                continue

            step0 = traj[0]
            if isinstance(step0, (list, tuple)) and all(isinstance(step, (list, tuple)) for step in traj):
                obs_t[start:end] = torch.as_tensor([step[0] for step in traj], dtype=torch.float32)
                h_t[start:end] = torch.as_tensor([step[1] for step in traj], dtype=torch.float32)
                lengths = [len(step) for step in traj]
                bt_t[start:end] = torch.as_tensor([step[2] for step in traj], dtype=torch.long)
                bd_t[start:end] = torch.as_tensor([step[3] for step in traj], dtype=torch.long)

                if all(n > 12 for n in lengths):
                    cd_t[start:end] = torch.as_tensor([step[4] for step in traj], dtype=torch.long)
                    md_t[start:end] = torch.as_tensor([step[5] for step in traj], dtype=torch.long)
                    n_working_t = torch.as_tensor([step[7] if len(step) > 7 else 0 for step in traj], dtype=torch.float32)
                    dmg_t = torch.as_tensor([step[11] if len(step) > 11 else 0.0 for step in traj], dtype=torch.float32)
                    lp_t[start:end] = torch.as_tensor([step[12] if len(step) > 12 else 0.0 for step in traj], dtype=torch.float32)
                    own_dmg_t = torch.as_tensor([step[21] if len(step) > 21 else 0.0 for step in traj], dtype=torch.float32)
                else:
                    cd_t[start:end] = torch.as_tensor([step[3] for step in traj], dtype=torch.long)
                    md_t[start:end] = torch.as_tensor([step[4] for step in traj], dtype=torch.long)
                    n_working_t = torch.as_tensor([step[6] if len(step) > 6 else 0 for step in traj], dtype=torch.float32)
                    dmg_t = torch.as_tensor([step[10] if len(step) > 10 else 0.0 for step in traj], dtype=torch.float32)
                    lp_t[start:end] = torch.as_tensor([step[11] if len(step) > 11 else 0.0 for step in traj], dtype=torch.float32)
                    own_dmg_t = torch.zeros(traj_len, dtype=torch.float32)
            elif isinstance(step0, dict) and all(isinstance(step, dict) for step in traj):
                obs_t[start:end] = torch.as_tensor([step["obs"] for step in traj], dtype=torch.float32)
                h_t[start:end] = torch.as_tensor([step["h_prev"] for step in traj], dtype=torch.float32)
                bt_t[start:end] = torch.as_tensor([step["bt"] for step in traj], dtype=torch.long)
                bd_t[start:end] = torch.as_tensor([step["bd"] for step in traj], dtype=torch.long)
                cd_t[start:end] = torch.as_tensor([step.get("cd", step["bd"]) for step in traj], dtype=torch.long)
                md_t[start:end] = torch.as_tensor([step["md"] for step in traj], dtype=torch.long)
                n_working_t = torch.as_tensor(
                    [step.get("n_working", 0) for step in traj],
                    dtype=torch.float32,
                )
                dmg_t = torch.as_tensor([step.get("dmg", 0.0) for step in traj], dtype=torch.float32)
                own_dmg_t = torch.as_tensor([step.get("own_dmg", 0.0) for step in traj], dtype=torch.float32)
                lp_t[start:end] = torch.as_tensor([step.get("old_lp", 0.0) for step in traj], dtype=torch.float32)
            else:
                for offset, step in enumerate(traj):
                    i = start + offset
                    if isinstance(step, (list, tuple)):
                        n = len(step)
                        obs_t[i] = torch.as_tensor(step[0], dtype=torch.float32)
                        h_t[i] = torch.as_tensor(step[1], dtype=torch.float32)
                        bt_t[i] = step[2]
                        bd_t[i] = step[3]
                        if n > 12:
                            cd_t[i] = step[4]
                            md_t[i] = step[5]
                            n_working = step[7] if n > 7 else 0
                            dmg = step[11] if n > 11 else 0.0
                            lp_t[i] = step[12] if n > 12 else 0.0
                            own_dmg = step[21] if n > 21 else 0.0
                        else:
                            cd_t[i] = step[3]
                            md_t[i] = step[4]
                            n_working = step[6] if n > 6 else 0
                            dmg = step[10] if n > 10 else 0.0
                            lp_t[i] = step[11] if n > 11 else 0.0
                            own_dmg = 0.0
                    else:
                        obs_t[i] = torch.as_tensor(step["obs"], dtype=torch.float32)
                        h_t[i] = torch.as_tensor(step["h_prev"], dtype=torch.float32)
                        bt_t[i] = step["bt"]
                        bd_t[i] = step["bd"]
                        cd_t[i] = step.get("cd", step["bd"])
                        md_t[i] = step["md"]
                        n_working = step.get("n_working", 0)
                        dmg = step.get("dmg", 0)
                        own_dmg = step.get("own_dmg", 0)
                        lp_t[i] = step.get("old_lp", 0)
                    rw_t[i] = (
                        n_working * REWARD_FLOW_THROUGHPUT
                        + dmg * REWARD_ENEMY_CORE_DAMAGE
                        + own_dmg * REWARD_OWN_CORE_DAMAGE
                    )
                step_game_t[start:end] = game_idx
                idx = end
                ep_boundaries.append((start, end))
                continue

            rw_t[start:end] = (
                n_working_t * REWARD_FLOW_THROUGHPUT
                + dmg_t * REWARD_ENEMY_CORE_DAMAGE
                + own_dmg_t * REWARD_OWN_CORE_DAMAGE
            )
            step_game_t[start:end] = game_idx
            idx = end
            ep_boundaries.append((start, end))
    finally:
        if gc_was_enabled:
            gc.enable()

    round_idx_t = _round_index_from_obs(obs_t)
    reward_keys = step_game_t.to(torch.long) * 2000 + round_idx_t
    _, reward_inverse = reward_keys.unique(sorted=False, return_inverse=True)
    round_reward = torch.zeros(int(reward_inverse.max().item()) + 1, dtype=rw_t.dtype)
    round_reward.index_add_(0, reward_inverse, rw_t)
    rw_t = round_reward[reward_inverse]

    result = {
        "obs": obs_t.to(DEVICE),
        "h": h_t.to(DEVICE),
        "bt": bt_t.to(DEVICE),
        "bd": bd_t.to(DEVICE),
        "cd": cd_t.to(DEVICE),
        "md": md_t.to(DEVICE),
        "old_log_probs": lp_t.to(DEVICE),
        "rewards": rw_t.to(DEVICE),
        "ep_boundaries": ep_boundaries,
        "n_steps": N,
    }

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
    zero_state = bytearray(n_ch * 1 * 1 * 4)
    zero_scalars = [0.0] * n_sc
    unique_states: list[object] = [zero_state]  # bytearray or buffer slice
    unique_valid_masks = [(1, 1)]
    unique_scalars: list[list[float]] = [zero_scalars]
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
            states_buf = gs.get("states_buf")

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
                    bytearray(states_buf), dtype=torch.float32,
                ).reshape(gs_n_turns, n_ch, gs_map_h, gs_map_w)

                # Bulk-parse scalars if present
                if scalar_stride_bytes > 0:
                    n_scalar_floats = scalar_stride_bytes // 4
                    all_scalars_t = torch.frombuffer(
                        bytearray(scalars_buf), dtype=torch.float32,
                    ).reshape(gs_n_turns, n_scalar_floats)

                base_idx = len(unique_states)
                for rnd in range(gs_n_turns):
                    unique_states.append(all_states_t[rnd])  # tensor view, no copy
                    unique_valid_masks.append((gs_map_w, gs_map_h))
                    if scalar_stride_bytes > 0:
                        unique_scalars.append(all_scalars_t[rnd].tolist())
                    else:
                        unique_scalars.append(zero_scalars)
                    available_unique_idx[rnd] = base_idx + rnd
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
    global_scalars = torch.tensor(unique_scalars, dtype=torch.float32)

    return {
        "global_states": unique_states,
        "global_sizes": global_sizes,
        "global_scalars": global_scalars,
        "step_round_idx": step_round_idx.to(DEVICE),
        "step_unit_xy": step_unit_xy.to(DEVICE),
        "step_unit_ctx": step_unit_ctx.to(DEVICE),
        "has_centralized": True,
    }


# Max steps a single unit can take in a full game (~2000 rounds).
# Episodes reaching this are truncated (tiebreaker), not terminated.
_TRUNCATION_THRESHOLD = 1900


def _compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    ep_boundaries: list[tuple[int, int]],
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GAE on GPU with proper truncation handling.

    Pads episodes to (E, max_len), computes TD residuals in one shot,
    then reverse-scans over T (sequential) but vectorized across all E episodes.
    Episodes >= _TRUNCATION_THRESHOLD steps bootstrap with V(s_final);
    shorter episodes (core destroyed / unit killed) bootstrap with 0.
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

    # Nonterminal mask: 1.0 for interior steps + last step of truncated episodes
    truncated = ep_lens >= _TRUNCATION_THRESHOLD
    interior = t_range < (ep_lens - 1).clamp(min=0).unsqueeze(1)
    last_step = t_range == (ep_lens - 1).clamp(min=0).unsqueeze(1)
    nonterminal = (interior | (last_step & truncated.unsqueeze(1) & (ep_lens > 0).unsqueeze(1))).float()

    # Build next_values: shift left, bootstrap truncated with V(s_T)
    next_v = torch.zeros_like(v_pad)
    if max_len > 1:
        next_v[:, :-1] = v_pad[:, 1:]
    trunc_mask = truncated & (ep_lens > 0)
    if trunc_mask.any():
        ep_idx = torch.arange(n_eps, device=dev)
        trunc_i = ep_idx[trunc_mask]
        trunc_last = (ep_lens[trunc_mask] - 1).clamp(min=0)
        next_v[trunc_i, trunc_last] = v_pad[trunc_i, trunc_last]

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


def critic_checkpoint_path(weight_path: str | Path = WEIGHT_PATH) -> Path:
    """Sidecar path used to persist centralized critic state for training resumes."""
    path = Path(weight_path)
    return path.with_name(f"{path.stem}.critic.pt")


def reset_centralized_critic() -> None:
    """Drop the singleton critic/optimizer so a fresh instance is created on demand."""
    global _centralized_critic, _centralized_optimizer
    _centralized_critic = None
    _centralized_optimizer = None


def get_centralized_critic(lr: float = 3e-4) -> tuple[CentralizedCritic, torch.optim.Optimizer]:
    """Get or create the singleton centralized critic and its optimizer."""
    global _centralized_critic, _centralized_optimizer
    if _centralized_critic is None:
        _centralized_critic = CentralizedCritic().to(DEVICE)
        _centralized_optimizer = torch.optim.Adam(
            _centralized_critic.parameters(), lr=lr, eps=1e-5,
        )
        n_params = sum(p.numel() for p in _centralized_critic.parameters())
        print(f"  CentralizedCritic: {n_params:,} params on {DEVICE}")
    return _centralized_critic, _centralized_optimizer


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
    payload = {
        "critic": _centralized_critic.state_dict(),
        "optimizer": _centralized_optimizer.state_dict(),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def _compute_all_values(
    critic: CentralizedCritic,
    data: dict,
    cnn_chunk_size: int = CENTRALIZED_CNN_CHUNK,
    unit_chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute per-unit critic values for all steps using chunked CNN.

    Processes rounds in chunks to bound memory. Feature maps are not retained
    across chunks — only the per-step values are accumulated.
    """
    round_idx = data["step_round_idx"]    # (N,) long
    unit_xy = data["step_unit_xy"]        # (N, 2) long
    unit_ctx = data["step_unit_ctx"]      # (N, UNIT_CONTEXT_DIM)
    actor_latent = data["step_actor_latent"]  # (N, ACTOR_LATENT_DIM)
    R = data["gpu_obs"].shape[0] if "gpu_obs" in data else len(data["global_states"])
    N = data["n_steps"]

    values = torch.zeros(N, device=DEVICE)

    # Process rounds in chunks; for each chunk, compute CNN features,
    # then scatter V_team + V_unit to the steps that reference those rounds.
    step_chunk_ids = torch.div(round_idx, cnn_chunk_size, rounding_mode='floor')
    step_order = step_chunk_ids.argsort(stable=True)
    sorted_chunk_ids = step_chunk_ids[step_order]
    n_round_chunks = (R + cnn_chunk_size - 1) // cnn_chunk_size
    chunk_offsets = _chunk_offsets_from_ids(sorted_chunk_ids, n_round_chunks)

    for chunk_idx in range(n_round_chunks):
        lo = chunk_offsets[chunk_idx]
        hi = chunk_offsets[chunk_idx + 1]
        if lo == hi:
            continue
        r_start = chunk_idx * cnn_chunk_size
        r_end = min(r_start + cnn_chunk_size, R)
        g_obs = data["gpu_obs"][r_start:r_end]
        g_mask = data["gpu_mask"][r_start:r_end]
        g_scalars = data["gpu_scalars"][r_start:r_end]
        feat = critic.encode_map(g_obs, g_mask)

        # Compute per-unit values for these steps
        step_indices = step_order[lo:hi]
        local_ri = round_idx[step_indices] - r_start  # remap to chunk-local index
        for u_start in range(0, step_indices.shape[0], unit_chunk_size):
            u_end = min(u_start + unit_chunk_size, step_indices.shape[0])
            si = step_indices[u_start:u_end]
            lr = local_ri[u_start:u_end]
            values[si] = critic.unit_values(
                feat, g_scalars, lr, unit_xy[si], unit_ctx[si], actor_latent[si],
            )

    return values



def train_on_trajectories(
    model: BuilderMLP,
    optimizer: torch.optim.Optimizer,
    trajectories: list[list],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    profile: dict[str, float] | None = None,
    global_states_list: list[dict | None] | None = None,
    traj_game_map: list[int] | None = None,
    game_winners: list[str | None] | None = None,
    **ppo_kwargs,
) -> dict[str, float]:
    """End-to-end: trajectories -> model values -> GAE -> PPO update."""
    t_phase = time.perf_counter() if profile is not None else 0.0
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
    _materialize_critic_inputs(data)
    if profile is not None:
        profile["materialize_s"] = time.perf_counter() - t_phase

    # Freeze actor-side unit summaries once so the critic sees a stable
    # per-step latent throughout PPO updates.
    t_phase = time.perf_counter() if profile is not None else 0.0
    model.eval()
    with torch.no_grad():
        data["step_actor_latent"] = _normalize_actor_latent_dim(
            model.actor_latent(obs, h).detach(),
        )
    if profile is not None:
        profile["critic_latent_s"] = time.perf_counter() - t_phase

    # Compute value estimates via centralized critic (chunked, memory-bounded)
    t_phase = time.perf_counter() if profile is not None else 0.0
    critic, critic_opt = get_centralized_critic()
    critic.eval()
    with torch.no_grad():
        values = _compute_all_values(critic, data)
    if profile is not None:
        profile["value_forward_s"] = time.perf_counter() - t_phase

    t_phase = time.perf_counter() if profile is not None else 0.0
    advantages, returns = _compute_gae(
        data["rewards"], values, data["ep_boundaries"], gamma, gae_lambda,
    )
    if profile is not None:
        profile["gae_s"] = time.perf_counter() - t_phase

    ppo_profile = profile if profile is not None else None
    model.train()

    # Recompute old_log_probs with sequential h propagation through BPTT chunks.
    # This ensures old_log_probs match what the BPTT training loop produces,
    # preventing ratio drift from stored-vs-recomputed h mismatch.
    t_phase = time.perf_counter() if profile is not None else 0.0
    bptt_k = int(ppo_kwargs.get("bptt_k", BPTT_K))
    bptt_chunks = _build_bptt_chunks(data["ep_boundaries"], bptt_k)
    with torch.no_grad():
        data["old_log_probs"] = _sequential_logprobs(
            model, obs, h,
            data["bt"], data["bd"], data["cd"], data["md"],
            bptt_chunks,
            bptt_k=bptt_k,
        )
    if profile is not None:
        profile["recompute_lp_s"] = time.perf_counter() - t_phase

    stats = ppo_update(
        model, optimizer,
        obs, h, data["bt"], data["bd"], data["cd"], data["md"],
        data["old_log_probs"], advantages, returns,
        old_values=values,
        centralized_data=data,
        ep_boundaries=data["ep_boundaries"],
        bptt_chunks=bptt_chunks,
        profile=ppo_profile,
        **ppo_kwargs,
    )
    stats["n_steps"] = data["n_steps"]
    stats["reward_raw_std"] = data["rewards"].std().item()
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
        for i in range(0, N, 8192):
            packed = torch.cat([obs[i:i + 8192], h[i:i + 8192]], dim=-1)
            bt_l, bd_l, cd_l, md_l, _ = teacher.infer_packed(packed)
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
            s_bt, s_bd, s_cd, s_md, _ = student(obs[idx], h[idx])

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
