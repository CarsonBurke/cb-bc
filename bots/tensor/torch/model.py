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
GLOBAL_FEATURES = 14
POS_EMBED_DIM = 8
OBS_SIZE = 5 * 5 * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN = [64, 64, 48, 48]
TEACHER_HIDDEN = [128, 128, 128, 64]
H_DIM = 16
INPUT_SIZE = OBS_SIZE + H_DIM
N_BUILD = 12
N_DIR = 9
ACTOR_RMSNORM_LAYERS = (True, True, True, True)

WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"
LOG_PATH = PROJECT_ROOT / "bots" / "tensor" / "train_log.json"
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


CRITIC_DIM = 128


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


class BuilderMLP(nn.Module):
    """MLP with shared encoder, separate actor trunk and deep critic.

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

        # Separate deep critic (training-only, not exported)
        critic_dim = max(CRITIC_DIM, enc_dim * 2)
        self.critic = nn.Sequential(
            _ortho_init(nn.Linear(enc_dim, critic_dim)),
            RMSNorm(critic_dim),
            nn.SiLU(),
            _ortho_init(nn.Linear(critic_dim, critic_dim)),
            RMSNorm(critic_dim),
            nn.SiLU(),
            _ortho_init(nn.Linear(critic_dim, 1), std=1.0),
        )

    def train(self, mode=True):
        # Invalidate fused head cache when entering train mode
        if mode and hasattr(self, '_cached_head_w'):
            del self._cached_head_w
            del self._cached_head_b
        return super().train(mode)

    def _forward_from_trunk_in(
        self,
        trunk_in: torch.Tensor,
        *,
        include_value: bool,
    ):
        enc = self.encoder(trunk_in)
        h = self.actor_trunk(enc)

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

        if include_value:
            value = self.critic(enc).squeeze(-1)
            return bt_l, bd_l, md_l, value, torch.tanh(h_raw)
        return bt_l, bd_l, md_l, torch.tanh(h_raw)

    def forward(self, x, h_prev):
        trunk_in = torch.cat([x, h_prev], dim=-1)
        return self._forward_from_trunk_in(trunk_in, include_value=True)

    def infer_packed(self, trunk_in: torch.Tensor):
        """Inference-only path for already-packed [obs, h_prev] batches."""
        return self._forward_from_trunk_in(trunk_in, include_value=False)

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
                    # torch state_dict format (legacy)
                    self.load_state_dict(data)
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
            bt_l, bd_l, md_l, value, _ = model(obs[idx], h[idx])
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

            # Value loss with optional clipping (CleanRL-style)
            t_phase = time.perf_counter() if profile is not None else 0.0
            newvalue = value.view(-1)
            if clip_vloss and old_values is not None:
                v_loss_unclipped = (newvalue - returns[idx]) ** 2
                v_clipped = old_values[idx] + torch.clamp(
                    newvalue - old_values[idx], -clip_eps, clip_eps,
                )
                v_loss_clipped = (v_clipped - returns[idx]) ** 2
                vf_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                vf_loss = 0.5 * ((newvalue - returns[idx]) ** 2).mean()
            if profile is not None:
                profile["ppo_value_loss_s"] = profile.get("ppo_value_loss_s", 0.0) + (time.perf_counter() - t_phase)

            loss = pi_loss + vf_coeff * vf_loss - entropy_coeff * entropy.mean()

            t_phase = time.perf_counter() if profile is not None else 0.0
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if profile is not None:
                profile["ppo_backward_step_s"] = profile.get("ppo_backward_step_s", 0.0) + (time.perf_counter() - t_phase)

            sum_pi += pi_loss.item()
            sum_vf += vf_loss.item()
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
) -> dict[str, torch.Tensor | list[tuple[int, int]]] | None:
    """Parse engine-collected trajectories into flat tensors (no GAE yet).

    Returns dict with keys: obs, h, bt, bd, md, old_log_probs, rewards,
    flow_bonuses, ep_boundaries. Or None if no data.

    flow_bonuses is a (E,) tensor of per-episode terminal bonuses extracted
    from trajectory index 12. These are kept separate from step rewards so
    _compute_gae can inject them undiscounted at the terminal step.
    """
    # Pre-count total steps to allocate tensors up front, avoiding
    # 7 large Python lists and the GC pressure they cause.
    N = sum(len(traj) for traj in trajectories)
    if N == 0:
        return None

    n_eps = len(trajectories)

    # Pre-allocate CPU tensors, fill in-place, transfer to device once.
    obs_t = torch.empty(N, OBS_SIZE, dtype=torch.float32)
    h_t = torch.empty(N, H_DIM, dtype=torch.float32)
    bt_t = torch.empty(N, dtype=torch.long)
    bd_t = torch.empty(N, dtype=torch.long)
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
        for ep_i, traj in enumerate(trajectories):
            start = idx
            ep_harv = ep_work = ep_dmg = ep_flow = 0.0
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
                    # index 12 is flow_bonus — only present on terminal step,
                    # read once per episode from the last step below.
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
                flow_r = step[12] if n > 12 else 0.0
                rw_t[idx] = harv_r + n_working * 2.5 + dmg * 20.0 + flow_r
                ep_harv += harv_r
                ep_work += n_working
                ep_dmg += dmg
                ep_flow += flow_r
                idx += 1

            ep_boundaries.append((start, idx))

            # flow bonus is pre-baked per step in rw_t (injected by game_worker)

            print(
                f"  [traj {ep_i:3d}] steps={idx - start:4d} "
                f"harv={ep_harv:.2f} "
                f"work_sum={ep_work:.0f} dmg={ep_dmg:.2f} "
                f"flow={ep_flow:.3f}"
            )

    finally:
        if gc_was_enabled:
            gc.enable()

    return {
        "obs": obs_t.to(DEVICE),
        "h": h_t.to(DEVICE),
        "bt": bt_t.to(DEVICE),
        "bd": bd_t.to(DEVICE),
        "md": md_t.to(DEVICE),
        "old_log_probs": lp_t.to(DEVICE),
        "rewards": rw_t.to(DEVICE),
        "flow_bonuses": torch.zeros(n_eps, dtype=torch.float32).to(DEVICE),
        "ep_boundaries": ep_boundaries,
        "n_steps": N,
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
    flow_bonuses: torch.Tensor | None = None,
    bt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized GAE on GPU with proper truncation handling.

    Pads episodes to (E, max_len), computes TD residuals in one shot,
    then reverse-scans over T (sequential) but vectorized across all E episodes.
    Episodes >= _TRUNCATION_THRESHOLD steps bootstrap with V(s_final);
    shorter episodes (core destroyed / unit killed) bootstrap with 0.

    flow_bonuses: optional (E,) tensor of per-episode terminal bonuses.
    bt: optional (N,) tensor of build-type actions (indices into BUILD_ACTIONS).

    Credit assignment for flow_bonuses:
    Rather than injecting the bonus only at the terminal step (which only
    trains the last action), the bonus is distributed equally across every
    conveyor-placing step in the episode (bt == 2 for conveyor, bt == 8 for
    armoured_conveyor). This directly reinforces the actions that built the
    chain being scored, regardless of when in the episode they occurred.
    Added to advantages only (not returns) so the critic's value target is
    unaffected — the critic learns normal episode value without the flow
    signal mixed in, keeping it stable.
    If bt is not provided, falls back to terminal-step injection.
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

    # Distribute flow bonuses across conveyor-placing steps in each episode.
    # bt indices: 2 = conveyor, 8 = armoured_conveyor (matches BUILD_ACTIONS).
    # If no bt provided, fall back to terminal-step injection.
    if flow_bonuses is not None:
        for i, (start, end) in enumerate(ep_boundaries):
            bonus = flow_bonuses[i]
            if bonus == 0.0:
                continue
            if bt is not None:
                ep_bt = bt[start:end]
                conv_mask = (ep_bt == 2) | (ep_bt == 8)
                n_conv = int(conv_mask.sum().item())
                if n_conv > 0:
                    # Equal share to each conveyor-placing step
                    per_step = bonus / n_conv
                    # Scatter add via boolean index
                    conv_indices = conv_mask.nonzero(as_tuple=True)[0] + start
                    advantages[conv_indices] += per_step
                else:
                    # No conveyors placed — fall back to terminal step
                    advantages[end - 1] += bonus
            else:
                advantages[end - 1] += bonus

    returns = advantages + values
    return advantages, returns


def trajectories_to_tensors(
    trajectories: list[list[dict]],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> dict[str, torch.Tensor] | None:
    """Convert engine-collected trajectories to flat tensors with GAE advantages.

    DEPRECATED: Prefer train_on_trajectories() which computes values via model
    forward pass instead of using the zero-placeholder values from the bot.
    """
    data = _parse_trajectories(trajectories)
    if data is None:
        return None

    # Legacy path: use zero values (same as before for backward compat)
    N = data["n_steps"]
    values = torch.zeros(N, device=DEVICE)
    advantages, returns = _compute_gae(
        data["rewards"], values, data["ep_boundaries"], gamma, gae_lambda,
        flow_bonuses=data["flow_bonuses"],
        bt=data["bt"],
    )
    return {
        "obs": data["obs"],
        "h": data["h"],
        "bt": data["bt"],
        "bd": data["bd"],
        "md": data["md"],
        "old_log_probs": data["old_log_probs"],
        "advantages": advantages,
        "returns": returns,
        "n_steps": N,
    }


def train_on_trajectories(
    model: BuilderMLP,
    optimizer: torch.optim.Optimizer,
    trajectories: list[list],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    profile: dict[str, float] | None = None,
    **ppo_kwargs,
) -> dict[str, float]:
    """End-to-end: trajectories -> model values -> GAE -> PPO update."""
    t_phase = time.perf_counter() if profile is not None else 0.0
    data = _parse_trajectories(trajectories)
    if profile is not None:
        profile["parse_s"] = time.perf_counter() - t_phase
    if data is None:
        return {"pi_loss": 0, "vf_loss": 0, "entropy": 0, "n_updates": 0, "n_steps": 0}

    obs, h = data["obs"], data["h"]

    # Compute value estimates with the current model (replaces zero placeholders)
    # Chunked to avoid OOM on large trajectory batches
    N = data["n_steps"]
    values = torch.empty(N, device=DEVICE)
    t_phase = time.perf_counter() if profile is not None else 0.0
    with torch.no_grad():
        for i in range(0, N, 8192):
            _, _, _, v, _ = model(obs[i:i + 8192], h[i:i + 8192])
            values[i:i + 8192] = v.view(-1)
    if profile is not None:
        profile["value_forward_s"] = time.perf_counter() - t_phase

    t_phase = time.perf_counter() if profile is not None else 0.0
    advantages, returns = _compute_gae(
        data["rewards"], values, data["ep_boundaries"], gamma, gae_lambda,
        flow_bonuses=data["flow_bonuses"],
        bt=data["bt"],
    )
    if profile is not None:
        profile["gae_s"] = time.perf_counter() - t_phase

    ppo_profile = profile if profile is not None else None
    stats = ppo_update(
        model, optimizer,
        obs, h, data["bt"], data["bd"], data["md"],
        data["old_log_probs"], advantages, returns,
        old_values=values,
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
            s_bt, s_bd, s_md, _, _ = student(obs[idx], h[idx])

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