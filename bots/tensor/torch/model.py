"""Shared model, constants, and PPO training code.

Used by train_gpu.py, train_engine.py, and train_selfplay.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ---- Architecture constants (must match main.py) ----

TILE_FEATURES = 16
GLOBAL_FEATURES = 12
POS_EMBED_DIM = 8
OBS_SIZE = 5 * 5 * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN = [32, 32, 32, 16]
H_DIM = 16
INPUT_SIZE = OBS_SIZE + H_DIM
N_BUILD = 12
N_DIR = 9

WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"
LOG_PATH = PROJECT_ROOT / "bots" / "tensor" / "train_log.json"
REPLAY_DIR = PROJECT_ROOT / "replays" / "training"

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


class BuilderMLP(nn.Module):
    """MLP with shared encoder, separate actor trunk and deep critic.

    Architecture:
        [obs, h_prev] -> encoder (Linear(335,32) + SiLU, shared)
                          ├─ actor_trunk ([32→32, 32→32, 32→16] + SiLU) -> {policy heads, h_next}
                          └─ critic ([128→128→1] with SiLU + RMSNorm, training-only)
    """

    def __init__(self):
        super().__init__()
        # Shared encoder: first hidden layer only
        enc_dim = HIDDEN[0]
        self.encoder = nn.Sequential(
            _ortho_init(nn.Linear(INPUT_SIZE, enc_dim)),
            nn.SiLU(),
        )

        # Actor trunk: remaining hidden layers
        actor_layers: list[nn.Module] = []
        prev = enc_dim
        for h in HIDDEN[1:]:
            actor_layers.append(_ortho_init(nn.Linear(prev, h)))
            actor_layers.append(nn.SiLU())
            prev = h
        self.actor_trunk = nn.Sequential(*actor_layers)

        # Policy heads: small std for near-uniform initial policy
        self.head_build_type = _ortho_init(nn.Linear(prev, N_BUILD), std=0.01)
        self.head_build_dir = _ortho_init(nn.Linear(prev, N_DIR), std=0.01)
        self.head_move_dir = _ortho_init(nn.Linear(prev, N_DIR), std=0.01)
        self.head_h_next = _ortho_init(nn.Linear(prev, H_DIM), std=0.01)

        # Separate deep critic (training-only, not exported)
        self.critic = nn.Sequential(
            _ortho_init(nn.Linear(enc_dim, CRITIC_DIM)),
            RMSNorm(CRITIC_DIM),
            nn.SiLU(),
            _ortho_init(nn.Linear(CRITIC_DIM, CRITIC_DIM)),
            RMSNorm(CRITIC_DIM),
            nn.SiLU(),
            _ortho_init(nn.Linear(CRITIC_DIM, 1), std=1.0),
        )

    def forward(self, x, h_prev):
        trunk_in = torch.cat([x, h_prev], dim=-1)
        enc = self.encoder(trunk_in)
        h = self.actor_trunk(enc)
        return (
            self.head_build_type(h),
            self.head_build_dir(h),
            self.head_move_dir(h),
            self.critic(enc).squeeze(-1),
            torch.tanh(self.head_h_next(h)),
        )

    def _linear_layers(self) -> list[nn.Linear]:
        """Policy layers only — used for export/import (critic excluded)."""
        enc_linears = [m for m in self.encoder if isinstance(m, nn.Linear)]
        actor_linears = [m for m in self.actor_trunk if isinstance(m, nn.Linear)]
        return enc_linears + actor_linears + [
            self.head_build_type, self.head_build_dir, self.head_move_dir,
            self.head_h_next,
        ]

    def export_weights(self) -> list[list[float]]:
        """Export weights as JSON list. Transposes to match tensor_lib (in, out) format."""
        state: list[list[float]] = []
        for layer in self._linear_layers():
            state.append(layer.weight.T.contiguous().view(-1).tolist())
            state.append(layer.bias.view(-1).tolist())
        return state

    def import_weights(self, state: list[list[float]]) -> None:
        """Load weights from tensor_lib JSON format."""
        idx = 0
        dev = next(self.parameters()).device
        for layer in self._linear_layers():
            w_flat = torch.tensor(state[idx], dtype=torch.float32, device=dev)
            in_f, out_f = layer.in_features, layer.out_features
            layer.weight.data = w_flat.view(in_f, out_f).T.contiguous()
            idx += 1
            layer.bias.data = torch.tensor(state[idx], dtype=torch.float32, device=dev)
            idx += 1

    def save_weights(self, path: str | Path = WEIGHT_PATH) -> None:
        with open(str(path), "w") as f:
            json.dump(self.export_weights(), f)

    def load_weights(self, path: str | Path = WEIGHT_PATH) -> bool:
        path = str(path)
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                self.import_weights(json.load(f))
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

            bt_l, bd_l, md_l, value, _ = model(obs[idx], h[idx])
            bt_d = Categorical(logits=bt_l)
            bd_d = Categorical(logits=bd_l)
            md_d = Categorical(logits=md_l)

            new_lp = bt_d.log_prob(bt_act[idx]) + bd_d.log_prob(bd_act[idx]) + md_d.log_prob(md_act[idx])
            entropy = bt_d.entropy() + bd_d.entropy() + md_d.entropy()

            logratio = new_lp - old_log_probs[idx]
            ratio = logratio.exp()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                approx_kl_acc += approx_kl.item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
                sum_kl += approx_kl.item()
                sum_clip_frac += clip_frac.item()
                n_mb += 1

            mb_adv = advantages[idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            pg1 = -mb_adv * ratio
            pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            pi_loss = torch.max(pg1, pg2).mean()

            # Value loss with optional clipping (CleanRL-style)
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

            loss = pi_loss + vf_coeff * vf_loss - entropy_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

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
    ep_boundaries. Or None if no data.
    """
    all_obs, all_h = [], []
    all_bt, all_bd, all_md = [], [], []
    all_rewards, all_old_lp = [], []
    ep_boundaries = []

    idx = 0
    for traj in trajectories:
        start = idx
        for step in traj:
            if isinstance(step, (list, tuple)):
                n = len(step)
                obs, h_prev, bt, bd, md = step[:5]
                dti = step[6] if n > 6 else 0.0
                harv = step[9] if n > 9 else 0.0
                dmg = step[10] if n > 10 else 0.0
                old_lp = step[11] if n > 11 else 0.0
                conv = step[12] if n > 12 else 0.0
            else:
                obs = step["obs"]
                h_prev = step["h_prev"]
                bt, bd, md = step["bt"], step["bd"], step["md"]
                dti = step.get("dti", 0)
                harv = step.get("harv", 0)
                dmg = step.get("dmg", 0)
                old_lp = step.get("old_lp", 0)
                conv = step.get("conv", 0)
            all_obs.append(obs)
            all_h.append(h_prev)
            all_bt.append(bt)
            all_bd.append(bd)
            all_md.append(md)
            all_old_lp.append(old_lp)

            # Reward: 4 signals
            reward = harv * 5.0 + conv * 2.0 + max(dti, 0) * 0.2 + dmg * 1.0
            all_rewards.append(reward)
            idx += 1
        ep_boundaries.append((start, idx))

    N = len(all_obs)
    if N == 0:
        return None

    return {
        "obs": torch.tensor(all_obs, dtype=torch.float32, device=DEVICE),
        "h": torch.tensor(all_h, dtype=torch.float32, device=DEVICE),
        "bt": torch.tensor(all_bt, dtype=torch.long, device=DEVICE),
        "bd": torch.tensor(all_bd, dtype=torch.long, device=DEVICE),
        "md": torch.tensor(all_md, dtype=torch.long, device=DEVICE),
        "old_log_probs": torch.tensor(all_old_lp, dtype=torch.float32, device=DEVICE),
        "rewards": torch.tensor(all_rewards, dtype=torch.float32, device=DEVICE),
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with proper truncation handling.

    Episodes >= _TRUNCATION_THRESHOLD steps bootstrap with V(s_final);
    shorter episodes (core destroyed / unit killed) bootstrap with 0.
    """
    N = rewards.shape[0]
    advantages = torch.zeros(N, device=rewards.device)
    for start, end in ep_boundaries:
        ep_len = end - start
        # Truncated episodes bootstrap with V(s_T) as approximation for V(s_{T+1}).
        # Trajectory data doesn't include the post-action next state, so this is the
        # best available estimate. Bias is ~(gamma-1)*V(s_T) ≈ -0.01*V(s_T), small.
        bootstrap_val = values[end - 1].item() if ep_len >= _TRUNCATION_THRESHOLD else 0.0
        gae = 0.0
        for t in range(end - 1, start - 1, -1):
            if t < end - 1:
                next_val = values[t + 1]
                next_nonterminal = 1.0
            else:
                next_val = bootstrap_val
                next_nonterminal = 1.0 if ep_len >= _TRUNCATION_THRESHOLD else 0.0
            delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
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
    **ppo_kwargs,
) -> dict[str, float]:
    """End-to-end: trajectories -> model values -> GAE -> PPO update."""
    data = _parse_trajectories(trajectories)
    if data is None:
        return {"pi_loss": 0, "vf_loss": 0, "entropy": 0, "n_updates": 0, "n_steps": 0}

    obs, h = data["obs"], data["h"]

    # Compute value estimates with the current model (replaces zero placeholders)
    with torch.no_grad():
        _, _, _, values, _ = model(obs, h)
        values = values.view(-1)

    advantages, returns = _compute_gae(
        data["rewards"], values, data["ep_boundaries"], gamma, gae_lambda,
    )

    stats = ppo_update(
        model, optimizer,
        obs, h, data["bt"], data["bd"], data["md"],
        data["old_log_probs"], advantages, returns,
        old_values=values,
        **ppo_kwargs,
    )
    stats["n_steps"] = data["n_steps"]
    return stats
