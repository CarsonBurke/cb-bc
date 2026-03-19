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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---- Architecture constants (must match main.py) ----

TILE_FEATURES = 17
GLOBAL_FEATURES = 15
POS_EMBED_DIM = 8
OBS_SIZE = 5 * 5 * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN = [64, 64, 48, 48]
TEACHER_HIDDEN = [128, 128, 128, 64]
H_DIM = 16
INPUT_SIZE = OBS_SIZE + H_DIM
ACTOR_LATENT_DIM = max(HIDDEN[-1], TEACHER_HIDDEN[-1])
N_BUILD = 12
N_DIR = 9
ACTOR_RMSNORM_LAYERS = (True, True, True, True)
REWARD_WORKING_HARVESTER = 12.5
REWARD_ENEMY_CORE_DAMAGE = 20.0
TERMINAL_WIN_REWARD = 10.0
TERMINAL_LOSS_REWARD = -10.0

WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"
LOG_PATH = PROJECT_ROOT / "bots" / "tensor" / "train_log.jsonl"
REPLAY_DIR = PROJECT_ROOT / "replays" / "training"

# ---- Action masking ----
# Build costs (ti, ax) indexed by BUILD_ACTIONS:
# none, harvester, conveyor, road, gunner, sentinel, breach, splitter,
# armoured_conveyor, barrier, foundry, launcher
_MASK_COSTS_TI = [0, 80, 3, 1, 10, 15, 30, 6, 10, 3, 120, 20]
_MASK_COSTS_AX = [0, 0, 0, 0, 0, 0, 10, 0, 5, 0, 0, 0]

# Tile indices in 5x5 grid for the 9 build directions (NW,N,NE,W,C,E,SW,S,SE)
_MASK_BD_TILES = [6, 7, 8, 11, 12, 13, 16, 17, 18]

# Observation feature indices for resources/scale
_OBS_GLOBAL_START = 5 * 5 * TILE_FEATURES  # 425
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
            torch.tensor([t * TILE_FEATURES + 15 for t in _MASK_BD_TILES], dtype=torch.long, device=device), # has_unit
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

    # Force none-only when nothing affordable or no valid direction
    force_none = ~bt_mask[:, 1:].any(dim=1) | ~bd_mask.any(dim=1)
    if force_none.any():
        bt_mask[force_none, 1:] = False
        bd_mask[force_none] = True  # dir is irrelevant, avoid NaN

    return bt_mask, bd_mask


# Sinusoidal position embedding frequencies
_POS_QUARTER = POS_EMBED_DIM // 4
_POS_FREQS = torch.tensor(
    [1.0 / (10000.0 ** (2.0 * i / POS_EMBED_DIM)) for i in range(_POS_QUARTER)],
    device=DEVICE,
)


# ---- Model ----

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---- Centralized Critic constants ----
GLOBAL_MAP_MAX_W = 50
GLOBAL_MAP_MAX_H = 50
GLOBAL_MAP_CHANNELS = 19
GLOBAL_SCALAR_DIM = 12
UNIT_CONTEXT_DIM = 7  # x_norm, y_norm, core_dx, core_dy, ecore_dx, ecore_dy, unit_age
CENTRALIZED_CNN_CHUNK = 248


class RMSNorm2d(nn.Module):
    """RMSNorm for 2D spatial feature maps (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — normalize over C dimension per spatial position
        rms = torch.sqrt(x.pow(2).mean(1, keepdim=True) + self.eps)
        return x / rms * self.weight.view(1, -1, 1, 1)


class CentralizedCritic(nn.Module):
    """Centralized per-unit critic over full replay-derived map state.

    Takes full map state (omniscient, from replay) and produces per-unit
    value estimates. Only used during training.

    The CNN encodes the map once per round. Per-unit computation is just
    an index into the feature map + a small MLP (cheap, O(1) per unit).
    """

    def __init__(self, map_channels: int = GLOBAL_MAP_CHANNELS, conv_dim: int = 64):
        super().__init__()
        self.conv_dim = conv_dim

        # Dilated conv stack: full spatial resolution, 31-tile receptive field
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 32, 3, padding=1, dilation=1),
            RMSNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, conv_dim, 3, padding=2, dilation=2),
            RMSNorm2d(conv_dim),
            nn.SiLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, padding=4, dilation=4),
            RMSNorm2d(conv_dim),
            nn.SiLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, padding=8, dilation=8),
            RMSNorm2d(conv_dim),
            nn.SiLU(),
        )

        # Single per-unit value head from tile features + replay/global context.
        unit_in = conv_dim + UNIT_CONTEXT_DIM + GLOBAL_SCALAR_DIM + ACTOR_LATENT_DIM
        self.unit_head = nn.Sequential(
            nn.Linear(unit_in, 128),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            RMSNorm(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

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
        # Small std for final value outputs
        nn.init.orthogonal_(self.unit_head[-1].weight, gain=1.0)

    def encode_map(
        self,
        map_tensor: torch.Tensor,    # (R, C, H, W) — R unique rounds
        valid_mask: torch.Tensor,     # (R, 1, H, W)
    ) -> torch.Tensor:
        """Run CNN once per unique round and return full spatial feature maps.

        valid_mask is applied so padded map area contributes no signal.
        """
        return self.conv(map_tensor) * valid_mask

    def unit_values(
        self,
        feat: torch.Tensor,          # (R, conv_dim, H, W) — cached from encode_map
        round_scalars: torch.Tensor,  # (R, GLOBAL_SCALAR_DIM)
        round_idx: torch.Tensor,      # (B,) — maps each unit-step to its round index
        unit_xy: torch.Tensor,        # (B, 2) — x, y positions (long)
        unit_context: torch.Tensor,   # (B, UNIT_CONTEXT_DIM)
        actor_latent: torch.Tensor,   # (B, ACTOR_LATENT_DIM) — detached rollout latent
    ) -> torch.Tensor:
        """Compute per-unit marginal values by indexing cached feature maps.

        Returns:
            v_unit: (B,) — per-unit marginal value
        """
        B = round_idx.shape[0]
        H, W = feat.shape[2], feat.shape[3]
        ux = unit_xy[:, 0].clamp(0, W - 1).long()
        uy = unit_xy[:, 1].clamp(0, H - 1).long()
        ri = round_idx.long()

        # Index into cached feature maps: (B, conv_dim)
        unit_tile_feat = feat[ri, :, uy, ux]
        unit_round_scalars = round_scalars[ri]

        # Per-unit value
        unit_in = torch.cat(
            [unit_tile_feat, unit_round_scalars, unit_context, actor_latent],
            dim=-1,
        )
        return self.unit_head(unit_in).squeeze(-1)  # (B,)

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


_critic_stage_obs: torch.Tensor | None = None
_critic_stage_mask: torch.Tensor | None = None
_critic_stage_shape = (0, 0, 0)  # n, h, w capacity


def _get_critic_staging_buffers(
    batch_n: int,
    batch_h: int,
    batch_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse pinned CPU staging buffers for critic chunk assembly."""
    global _critic_stage_obs, _critic_stage_mask, _critic_stage_shape

    cap_n, cap_h, cap_w = _critic_stage_shape
    if (
        _critic_stage_obs is None
        or batch_n > cap_n
        or batch_h > cap_h
        or batch_w > cap_w
    ):
        alloc_n = max(batch_n, cap_n)
        alloc_h = max(batch_h, cap_h)
        alloc_w = max(batch_w, cap_w)
        _critic_stage_obs = torch.empty(
            alloc_n, GLOBAL_MAP_CHANNELS, alloc_h, alloc_w, dtype=torch.float32,
        )
        _critic_stage_mask = torch.empty(
            alloc_n, 1, alloc_h, alloc_w, dtype=torch.float32,
        )
        if DEVICE.type == "cuda":
            _critic_stage_obs = _critic_stage_obs.pin_memory()
            _critic_stage_mask = _critic_stage_mask.pin_memory()
        _critic_stage_shape = (alloc_n, alloc_h, alloc_w)

    obs = _critic_stage_obs[:batch_n, :, :batch_h, :batch_w]
    mask = _critic_stage_mask[:batch_n, :, :batch_h, :batch_w]
    obs.zero_()
    mask.zero_()
    return obs, mask


def _load_critic_round_batch(
    centralized_data: dict,
    round_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch a round batch for the centralized critic and move it to DEVICE."""
    round_ids_cpu = round_ids.detach().to("cpu")
    round_list = round_ids_cpu.tolist()
    sizes = centralized_data["global_sizes"][round_ids_cpu]
    widths = sizes[:, 0]
    heights = sizes[:, 1]
    batch_w = int(widths.max().item()) if widths.numel() else 1
    batch_h = int(heights.max().item()) if heights.numel() else 1
    batch_n = len(round_list)

    g_obs_cpu, g_mask_cpu = _get_critic_staging_buffers(
        batch_n, batch_h, batch_w,
    )

    global_states = centralized_data["global_states"]
    for batch_i, round_idx in enumerate(round_list):
        mw = int(widths[batch_i].item())
        mh = int(heights[batch_i].item())
        if mw > 0 and mh > 0:
            buf = global_states[round_idx]
            state_view = torch.frombuffer(
                buf, dtype=torch.float32,
            ).reshape(GLOBAL_MAP_CHANNELS, mh, mw)
            g_obs_cpu[batch_i, :, :mh, :mw].copy_(state_view)
            g_mask_cpu[batch_i, 0, :mh, :mw] = 1.0

    return (
        g_obs_cpu.to(device=DEVICE, non_blocking=True),
        g_mask_cpu.to(device=DEVICE, non_blocking=True),
        centralized_data["global_scalars"][round_ids_cpu].to(device=DEVICE, non_blocking=True),
    )


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

        # Shared encoder: first hidden layer only
        enc_dim = hidden[0]
        self.encoder = nn.Sequential(
            _ortho_init(nn.Linear(INPUT_SIZE, enc_dim)),
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
        self.head_move_dir = _ortho_init(nn.Linear(prev, N_DIR), std=0.01)
        self.head_h_next = _ortho_init(nn.Linear(prev, H_DIM), std=0.01)

    def train(self, mode=True):
        # Invalidate fused head cache when entering train mode
        if mode and hasattr(self, '_cached_head_w'):
            del self._cached_head_w
            del self._cached_head_b
        return super().train(mode)

    def actor_latent(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Return the pre-head actor latent used for policy logits."""
        trunk_in = torch.cat([x, h_prev], dim=-1)
        enc = self.encoder(trunk_in)
        return self.actor_trunk(enc)

    def _forward_from_hidden(self, h: torch.Tensor):
        """Project actor latent to policy heads."""
        # Fuse 4 head matmuls into one kernel launch
        # Cache fused weights in eval mode (inference hot path)
        if self.training:
            combined_w = torch.cat([
                self.head_build_type.weight, self.head_build_dir.weight,
                self.head_move_dir.weight, self.head_h_next.weight,
            ], dim=0)
            combined_b = torch.cat([
                self.head_build_type.bias, self.head_build_dir.bias,
                self.head_move_dir.bias, self.head_h_next.bias,
            ])
        else:
            if not hasattr(self, '_cached_head_w'):
                self._cached_head_w = torch.cat([
                    self.head_build_type.weight, self.head_build_dir.weight,
                    self.head_move_dir.weight, self.head_h_next.weight,
                ], dim=0)
                self._cached_head_b = torch.cat([
                    self.head_build_type.bias, self.head_build_dir.bias,
                    self.head_move_dir.bias, self.head_h_next.bias,
                ])
            combined_w = self._cached_head_w
            combined_b = self._cached_head_b

        combined = F.linear(h, combined_w, combined_b)
        bt_l, bd_l, md_l, h_raw = combined.split([N_BUILD, N_DIR, N_DIR, H_DIM], dim=-1)
        return bt_l, bd_l, md_l, torch.tanh(h_raw)

    def _forward_from_trunk_in(self, trunk_in: torch.Tensor):
        enc = self.encoder(trunk_in)
        h = self.actor_trunk(enc)
        return self._forward_from_hidden(h)

    def forward(self, x, h_prev):
        trunk_in = torch.cat([x, h_prev], dim=-1)
        return self._forward_from_trunk_in(trunk_in)

    def infer_packed(self, trunk_in: torch.Tensor):
        """Inference-only path for already-packed [obs, h_prev] batches."""
        return self._forward_from_trunk_in(trunk_in)

    def _policy_state_modules(self) -> list[nn.Module]:
        """Policy modules only in tensor_lib serialization order (critic excluded)."""
        enc_modules = [m for m in self.encoder if isinstance(m, (nn.Linear, RMSNorm))]
        actor_modules = [m for m in self.actor_trunk if isinstance(m, (nn.Linear, RMSNorm))]
        return enc_modules + actor_modules + [
            self.head_build_type, self.head_build_dir, self.head_move_dir,
            self.head_h_next,
        ]

    def export_weights(self) -> list[list[float]]:
        """Export weights as JSON list. Transposes to match tensor_lib (in, out) format."""
        state: list[list[float]] = []
        for layer in self._policy_state_modules():
            if isinstance(layer, nn.Linear):
                state.append(layer.weight.T.contiguous().view(-1).tolist())
                state.append(layer.bias.view(-1).tolist())
            else:
                state.append(layer.weight.view(-1).tolist())
        return state

    def import_weights(self, state: list[list[float]]) -> None:
        """Load weights from tensor_lib JSON format, including legacy no-RMSNorm states."""
        modules = self._policy_state_modules()
        full_entries = sum(2 if isinstance(layer, nn.Linear) else 1 for layer in modules)
        legacy_entries = sum(2 for layer in modules if isinstance(layer, nn.Linear))
        if len(state) == full_entries:
            load_rmsnorm = True
        elif len(state) == legacy_entries:
            load_rmsnorm = False
        else:
            raise AssertionError(
                f"State has {len(state)} entries, expected {full_entries} "
                f"({legacy_entries} legacy without RMSNorm)"
            )

        idx = 0
        dev = next(self.parameters()).device
        for layer in modules:
            if isinstance(layer, nn.Linear):
                w_flat = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                in_f, out_f = layer.in_features, layer.out_features
                layer.weight.data = w_flat.view(in_f, out_f).T.contiguous()
                idx += 1
                layer.bias.data = torch.tensor(state[idx], dtype=torch.float32, device=dev)
                idx += 1
            elif load_rmsnorm:
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

def ppo_update(
    model: BuilderMLP,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    h: torch.Tensor,
    bt_act: torch.Tensor,
    bd_act: torch.Tensor,
    md_act: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    centralized_data: dict,
    old_values: torch.Tensor | None = None,
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    clip_eps: float = 0.2,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    clip_vloss: bool = True,
    profile: dict[str, float] | None = None,
) -> dict[str, float]:
    """PPO update on flattened rollout data. Returns stats dict."""
    N = obs.shape[0]
    sum_pi, sum_vf, sum_ent = 0.0, 0.0, 0.0
    sum_kl, sum_clip_frac = 0.0, 0.0
    n_updates = 0

    critic, critic_opt = get_centralized_critic()
    critic.train()

    # Explained variance: pre-update values vs returns (CleanRL-style, no extra fwd pass)
    if old_values is not None:
        with torch.no_grad():
            var_y = returns.var().item()
            ev = float('nan') if var_y == 0 else 1.0 - (returns - old_values).var().item() / var_y
    else:
        ev = 0.0

    for epoch in range(ppo_epochs):
        approx_kl_acc, n_mb = 0.0, 0

        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, minibatch_size):
            idx = perm[start:start + minibatch_size]

            t_phase = time.perf_counter() if profile is not None else 0.0
            bt_l, bd_l, md_l, _ = model(obs[idx], h[idx])
            if profile is not None:
                profile["ppo_forward_s"] = profile.get("ppo_forward_s", 0.0) + (time.perf_counter() - t_phase)

            t_phase = time.perf_counter() if profile is not None else 0.0
            # Apply action masks to logits (affordability + spatial validity)
            bt_m, bd_m = compute_action_masks(obs[idx])
            bt_l = bt_l.masked_fill(~bt_m, -1e9)
            bd_l = bd_l.masked_fill(~bd_m, -1e9)

            # Mask build_dir when build_type is "none" (idx 0) — causally irrelevant
            bd_causal = (bt_act[idx] != 0).float()
            bt_lp, bt_ent = action_log_prob_and_entropy(bt_l, bt_act[idx])
            bd_lp, bd_ent = action_log_prob_and_entropy(bd_l, bd_act[idx])
            md_lp, md_ent = action_log_prob_and_entropy(md_l, md_act[idx])
            new_lp = bt_lp + bd_lp * bd_causal + md_lp
            entropy = bt_ent + bd_ent * bd_causal + md_ent

            logratio = new_lp - old_log_probs[idx]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                approx_kl_acc += approx_kl.item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
                sum_kl += approx_kl.item()
                sum_clip_frac += clip_frac.item()
                n_mb += 1
            if profile is not None:
                profile["ppo_dist_s"] = profile.get("ppo_dist_s", 0.0) + (time.perf_counter() - t_phase)

            t_phase = time.perf_counter() if profile is not None else 0.0
            mb_adv = advantages[idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            pi_loss = torch.max(pg1, pg2).mean()
            if profile is not None:
                profile["ppo_policy_loss_s"] = profile.get("ppo_policy_loss_s", 0.0) + (time.perf_counter() - t_phase)

            t_phase = time.perf_counter() if profile is not None else 0.0
            optimizer.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)

            actor_loss = pi_loss - entropy_coeff * entropy.mean()
            actor_loss.backward()

            vf_loss_sum = 0.0
            vf_denom = max(idx.shape[0], 1)
            # Value loss with optional clipping (CleanRL-style), but backpropagated
            # chunk-by-chunk so the centralized critic never retains a full
            # minibatch worth of spatial feature maps.
            t_value = time.perf_counter() if profile is not None else 0.0
            ri = centralized_data["step_round_idx"][idx]
            u_xy = centralized_data["step_unit_xy"][idx]
            u_ctx = centralized_data["step_unit_ctx"][idx]
            u_latent = centralized_data["step_actor_latent"][idx]
            mb_returns = returns[idx]
            mb_old_values = old_values[idx] if old_values is not None else None
            unique_rounds, inverse = ri.unique(return_inverse=True)
            _CNN_CHUNK = CENTRALIZED_CNN_CHUNK
            _UNIT_CHUNK = 4096
            for c_start in range(0, unique_rounds.shape[0], _CNN_CHUNK):
                c_end = min(c_start + _CNN_CHUNK, unique_rounds.shape[0])
                c_rounds = unique_rounds[c_start:c_end]
                g_obs, g_mask, g_scalars = _load_critic_round_batch(centralized_data, c_rounds)
                feat = critic.encode_map(g_obs, g_mask)
                step_mask = (inverse >= c_start) & (inverse < c_end)
                step_indices = step_mask.nonzero(as_tuple=True)[0]
                if step_indices.numel() == 0:
                    continue

                local_round_idx = inverse[step_indices] - c_start
                for u_start in range(0, step_indices.shape[0], _UNIT_CHUNK):
                    u_end = min(u_start + _UNIT_CHUNK, step_indices.shape[0])
                    si = step_indices[u_start:u_end]
                    lr = local_round_idx[u_start:u_end]
                    newvalue = critic.unit_values(
                        feat, g_scalars, lr, u_xy[si], u_ctx[si], u_latent[si],
                    )
                    if clip_vloss and mb_old_values is not None:
                        v_loss_unclipped = (newvalue - mb_returns[si]) ** 2
                        v_clipped = mb_old_values[si] + torch.clamp(
                            newvalue - mb_old_values[si], -clip_eps, clip_eps,
                        )
                        v_loss_elems = torch.max(v_loss_unclipped, (v_clipped - mb_returns[si]) ** 2)
                    else:
                        v_loss_elems = (newvalue - mb_returns[si]) ** 2

                    vf_loss_sum += 0.5 * v_loss_elems.detach().sum().item()
                    (vf_coeff * 0.5 * v_loss_elems.sum() / vf_denom).backward()

            vf_loss = vf_loss_sum / vf_denom
            if profile is not None:
                profile["ppo_value_loss_s"] = profile.get("ppo_value_loss_s", 0.0) + (time.perf_counter() - t_value)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer.step()
            critic_opt.step()
            if profile is not None:
                profile["ppo_backward_step_s"] = profile.get("ppo_backward_step_s", 0.0) + (time.perf_counter() - t_phase)

            sum_pi += pi_loss.item()
            sum_vf += vf_loss
            sum_ent += entropy.mean().item()
            n_updates += 1

        if target_kl is not None and n_mb > 0 and approx_kl_acc / n_mb > target_kl:
            break

    return {
        "pi_loss": sum_pi / max(n_updates, 1),
        "vf_loss": sum_vf / max(n_updates, 1),
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

    Returns dict with keys: obs, h, bt, bd, md, old_log_probs, rewards,
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
    md_t = torch.empty(N, dtype=torch.long)
    lp_t = torch.empty(N, dtype=torch.float32)
    rw_t = torch.empty(N, dtype=torch.float32)
    step_game_t = torch.empty(N, dtype=torch.int32)
    ep_boundaries = []
    ep_game_idx = []

    # Disable cyclic GC — the loop creates no reference cycles, and the
    # collector's O(live_objects) scan is expensive with many live tensors.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        idx = 0
        for traj_i, traj in enumerate(trajectories):
            start = idx
            game_idx = traj_game_map[traj_i] if traj_game_map and traj_i < len(traj_game_map) else 0
            for step in traj:
                if isinstance(step, (list, tuple)):
                    n = len(step)
                    obs_t[idx] = torch.tensor(step[0], dtype=torch.float32)
                    h_t[idx] = torch.tensor(step[1], dtype=torch.float32)
                    bt_t[idx] = step[2]
                    bd_t[idx] = step[3]
                    md_t[idx] = step[4]
                    n_working = step[6] if n > 6 else 0
                    conv_r = step[7] if n > 7 else 0.0
                    harv_r = step[9] if n > 9 else 0.0
                    dmg = step[10] if n > 10 else 0.0
                    lp_t[idx] = step[11] if n > 11 else 0.0
                else:
                    obs_t[idx] = torch.tensor(step["obs"], dtype=torch.float32)
                    h_t[idx] = torch.tensor(step["h_prev"], dtype=torch.float32)
                    bt_t[idx] = step["bt"]
                    bd_t[idx] = step["bd"]
                    md_t[idx] = step["md"]
                    n_working = step.get("n_working", 0)
                    conv_r = step.get("conv_r", 0)
                    harv_r = step.get("harv_r", 0)
                    dmg = step.get("dmg", 0)
                    lp_t[idx] = step.get("old_lp", 0)
                rw_t[idx] = (
                    harv_r
                    + conv_r
                    + n_working * REWARD_WORKING_HARVESTER
                    + dmg * REWARD_ENEMY_CORE_DAMAGE
                )
                step_game_t[idx] = game_idx
                idx += 1
            ep_boundaries.append((start, idx))
            ep_game_idx.append(game_idx)
    finally:
        if gc_was_enabled:
            gc.enable()

    # Share each round's total reward across all unit-steps from that game/round.
    round_reward: dict[tuple[int, int], float] = {}
    for i in range(N):
        rnd = int(round(obs_t[i, _OBS_GLOBAL_START].item() * 2000.0))
        rnd = max(0, min(rnd, 1999))
        key = (int(step_game_t[i].item()), rnd)
        round_reward[key] = round_reward.get(key, 0.0) + float(rw_t[i].item())
    for i in range(N):
        rnd = int(round(obs_t[i, _OBS_GLOBAL_START].item() * 2000.0))
        rnd = max(0, min(rnd, 1999))
        rw_t[i] = round_reward[(int(step_game_t[i].item()), rnd)]

    # Add a small terminal outcome reward to the final step of every unit
    # trajectory so dead units can still learn from game outcome.
    if game_winners:
        for ep_i, (start, end) in enumerate(ep_boundaries):
            if end <= start:
                continue
            game_idx = ep_game_idx[ep_i]
            winner = game_winners[game_idx] if game_idx < len(game_winners) else None
            if winner == "a":
                rw_t[end - 1] += TERMINAL_WIN_REWARD
            elif winner == "b":
                rw_t[end - 1] += TERMINAL_LOSS_REWARD

    result = {
        "obs": obs_t.to(DEVICE),
        "h": h_t.to(DEVICE),
        "bt": bt_t.to(DEVICE),
        "bd": bd_t.to(DEVICE),
        "md": md_t.to(DEVICE),
        "old_log_probs": lp_t.to(DEVICE),
        "rewards": rw_t.to(DEVICE),
        "ep_boundaries": ep_boundaries,
        "n_steps": N,
    }

    # Build centralized critic data if global states available
    if global_states_list and any(gs is not None for gs in global_states_list):
        result.update(_build_centralized_critic_data(
            obs_t, trajectories, global_states_list, ep_boundaries, N,
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
    trajectories: list[list],
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
    unique_keys = {}  # (game_idx, round) → flat index (starting from 1)

    # Per-step: index into unique states, unit xy, unit context
    step_round_idx = torch.zeros(N, dtype=torch.long)
    step_unit_xy = torch.zeros(N, 2, dtype=torch.long)
    step_unit_ctx = torch.zeros(N, UNIT_CONTEXT_DIM, dtype=torch.float32)

    n_games = len(global_states_list)

    for ep_i, (start, end) in enumerate(ep_boundaries):
        if end <= start:
            continue

        game_idx = traj_game_map[ep_i] if traj_game_map and ep_i < len(traj_game_map) else 0
        gs = global_states_list[game_idx] if game_idx < n_games else None
        if gs is None:
            continue

        gs_map_w = gs.get("map_w", max_w)
        gs_map_h = gs.get("map_h", max_h)

        # Detect format: contiguous buffer or legacy per-turn dicts
        states_buf = gs.get("states_buf")
        if states_buf is not None:
            # Contiguous buffer format — index by byte offset
            state_stride_bytes = gs.get("state_stride", n_ch * gs_map_h * gs_map_w) * 4
            scalars_buf = gs.get("scalars_buf", b"")
            scalar_stride_bytes = gs.get("n_scalars", n_sc) * 4
            gs_n_turns = gs.get("n_turns", 0)

            def _get_state(rnd):
                if 0 <= rnd < gs_n_turns:
                    off = rnd * state_stride_bytes
                    return bytearray(states_buf[off:off + state_stride_bytes])
                return zero_state

            def _get_scalars(rnd):
                if 0 <= rnd < gs_n_turns:
                    import struct
                    off = rnd * scalar_stride_bytes
                    n = scalar_stride_bytes // 4
                    return list(struct.unpack_from(f"<{n}f", scalars_buf, off))
                return zero_scalars

            available_rounds = list(range(gs_n_turns)) if gs_n_turns > 0 else []
        else:
            # Legacy per-turn dict format
            gs_states = gs.get("states", {})
            gs_scalars_d = gs.get("scalars", {})
            available_rounds = sorted(gs_states.keys()) if gs_states else []

            def _get_state(rnd):
                s = gs_states.get(rnd)
                if s is not None:
                    if isinstance(s, (bytes, bytearray)):
                        return s
                    # array.array — convert to bytearray for torch.frombuffer
                    return bytearray(s.tobytes() if hasattr(s, "tobytes") else bytes(s))
                return zero_state

            def _get_scalars(rnd):
                s = gs_scalars_d.get(rnd)
                if s is not None:
                    return list(s)
                return zero_scalars

        for i in range(start, end):
            rnd = int(round(obs_t[i, _OBS_GLOBAL_START].item() * 2000.0))
            rnd = max(0, min(rnd, 1999))

            key = (game_idx, rnd)
            if key not in unique_keys:
                state = _get_state(rnd)
                scalars = _get_scalars(rnd)
                if state is zero_state and available_rounds:
                    # Round not in replay — find nearest
                    nearest = min(available_rounds, key=lambda r: abs(r - rnd))
                    state = _get_state(nearest)
                    scalars = _get_scalars(nearest)
                unique_keys[key] = len(unique_states)
                unique_states.append(state)
                unique_valid_masks.append((gs_map_w, gs_map_h))
                unique_scalars.append(scalars)

            step_round_idx[i] = unique_keys[key]

            # Extract unit position from obs (denormalize)
            x_norm = obs_t[i, _OBS_X_NORM].item()
            y_norm = obs_t[i, _OBS_Y_NORM].item()
            ux = int(round(x_norm * gs_map_w))
            uy = int(round(y_norm * gs_map_h))
            step_unit_xy[i, 0] = max(0, min(ux, max_w - 1))
            step_unit_xy[i, 1] = max(0, min(uy, max_h - 1))

            step_unit_ctx[i, 0] = obs_t[i, _OBS_X_NORM]
            step_unit_ctx[i, 1] = obs_t[i, _OBS_Y_NORM]
            step_unit_ctx[i, 2] = obs_t[i, _OBS_CORE_DX]
            step_unit_ctx[i, 3] = obs_t[i, _OBS_CORE_DY]
            step_unit_ctx[i, 4] = obs_t[i, _OBS_ECORE_DX]
            step_unit_ctx[i, 5] = obs_t[i, _OBS_ECORE_DY]
            rnd_norm = obs_t[i, _OBS_GLOBAL_START].item()
            spawn_norm = obs_t[i, _OBS_SPAWN_ROUND].item()
            step_unit_ctx[i, 6] = rnd_norm - spawn_norm

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
    gamma: float = 0.99,
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

    # Pad episodes into (E, T) tensors
    r_pad = torch.zeros(n_eps, max_len, device=dev)
    v_pad = torch.zeros(n_eps, max_len, device=dev)
    # nonterminal: 1.0 for all valid steps; 0.0 at last step of terminal episodes
    # and for all padding positions (prevents GAE leaking across episode boundaries)
    nonterminal = torch.zeros(n_eps, max_len, device=dev)

    for i, (start, end) in enumerate(ep_boundaries):
        ep_len = end - start
        r_pad[i, :ep_len] = rewards[start:end]
        v_pad[i, :ep_len] = values[start:end]
        truncated = ep_len >= _TRUNCATION_THRESHOLD
        # Interior steps are nonterminal; last step only if truncated
        if ep_len > 1:
            nonterminal[i, :ep_len - 1] = 1.0
        if truncated:
            nonterminal[i, ep_len - 1] = 1.0

    # Build next_values: shift values left, last position gets bootstrap
    next_v = torch.zeros_like(v_pad)
    if max_len > 1:
        next_v[:, :-1] = v_pad[:, 1:]
    for i, (start, end) in enumerate(ep_boundaries):
        ep_len = end - start
        # Truncated: bootstrap with V(s_T) as approx for V(s_{T+1})
        if ep_len >= _TRUNCATION_THRESHOLD:
            next_v[i, ep_len - 1] = v_pad[i, ep_len - 1]
        # Terminal: next_v stays 0 (already zero-initialized)

    # TD residuals: delta = r + gamma * next_v * nonterminal - v
    delta = r_pad + gamma * next_v * nonterminal - v_pad

    # Reverse scan for GAE (sequential over T, vectorized over E)
    gae = torch.zeros(n_eps, device=dev)
    adv_pad = torch.zeros_like(r_pad)
    for t in range(max_len - 1, -1, -1):
        gae = delta[:, t] + gl * nonterminal[:, t] * gae
        adv_pad[:, t] = gae

    # Unpad back to flat (N,) tensor
    advantages = torch.zeros(N, device=dev)
    for i, (start, end) in enumerate(ep_boundaries):
        advantages[start:end] = adv_pad[i, :end - start]

    returns = advantages + values
    return advantages, returns



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
    R = len(data["global_states"])
    N = data["n_steps"]

    values = torch.zeros(N, device=DEVICE)

    # Process rounds in chunks; for each chunk, compute CNN features,
    # then scatter V_team + V_unit to the steps that reference those rounds.
    for r_start in range(0, R, cnn_chunk_size):
        r_end = min(r_start + cnn_chunk_size, R)
        round_ids = torch.arange(r_start, r_end, dtype=torch.long)
        g_obs, g_mask, g_scalars = _load_critic_round_batch(data, round_ids)
        feat = critic.encode_map(g_obs, g_mask)

        # Find all steps that reference rounds in this chunk
        step_mask = (round_idx >= r_start) & (round_idx < r_end)
        step_indices = step_mask.nonzero(as_tuple=True)[0]
        if step_indices.shape[0] == 0:
            continue

        # Compute per-unit values for these steps
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
    stats = ppo_update(
        model, optimizer,
        obs, h, data["bt"], data["bd"], data["md"],
        data["old_log_probs"], advantages, returns,
        old_values=values,
        centralized_data=data,
        profile=ppo_profile,
        **ppo_kwargs,
    )
    stats["n_steps"] = data["n_steps"]
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
    t_bt, t_bd, t_md = [], [], []
    with torch.no_grad():
        for i in range(0, N, 8192):
            packed = torch.cat([obs[i:i + 8192], h[i:i + 8192]], dim=-1)
            bt_l, bd_l, md_l, _ = teacher.infer_packed(packed)
            t_bt.append(bt_l)
            t_bd.append(bd_l)
            t_md.append(md_l)
    t_bt = torch.cat(t_bt)
    t_bd = torch.cat(t_bd)
    t_md = torch.cat(t_md)

    sum_loss = 0.0
    n_updates = 0
    T = temperature

    for _ in range(epochs):
        perm = torch.randperm(N, device=DEVICE)
        for start in range(0, N, minibatch_size):
            idx = perm[start:start + minibatch_size]
            s_bt, s_bd, s_md, _ = student(obs[idx], h[idx])

            # KL(teacher || student) on softened logits
            loss = (
                F.kl_div(F.log_softmax(s_bt / T, dim=-1), F.softmax(t_bt[idx] / T, dim=-1), reduction='batchmean')
                + F.kl_div(F.log_softmax(s_bd / T, dim=-1), F.softmax(t_bd[idx] / T, dim=-1), reduction='batchmean')
                + F.kl_div(F.log_softmax(s_md / T, dim=-1), F.softmax(t_md[idx] / T, dim=-1), reduction='batchmean')
            ) * (T * T)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            optimizer.step()

            sum_loss += loss.item()
            n_updates += 1

    return {"distill_loss": sum_loss / max(n_updates, 1), "n_updates": n_updates}
