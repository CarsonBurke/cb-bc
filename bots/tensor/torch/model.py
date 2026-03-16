"""Shared model, constants, and PPO training code.

Used by train_gpu.py, train_engine.py, and train_selfplay.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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

class BuilderMLP(nn.Module):
    """MLP with recurrent hidden state and value head for PPO.

    Architecture: [obs, h_prev] -> trunk -> {action heads, value, h_next}
    """

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        prev = INPUT_SIZE
        for h in HIDDEN:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head_build_type = nn.Linear(prev, N_BUILD)
        self.head_build_dir = nn.Linear(prev, N_DIR)
        self.head_move_dir = nn.Linear(prev, N_DIR)
        self.head_value = nn.Linear(prev, 1)
        self.head_h_next = nn.Linear(prev, H_DIM)

    def forward(self, x, h_prev):
        trunk_in = torch.cat([x, h_prev], dim=-1)
        h = self.trunk(trunk_in)
        return (
            self.head_build_type(h),
            self.head_build_dir(h),
            self.head_move_dir(h),
            self.head_value(h).squeeze(-1),
            torch.tanh(self.head_h_next(h)),
        )

    def _linear_layers(self) -> list[nn.Linear]:
        trunk_linears = [m for m in self.trunk if isinstance(m, nn.Linear)]
        return trunk_linears + [
            self.head_build_type, self.head_build_dir, self.head_move_dir,
            self.head_value, self.head_h_next,
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
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    clip_eps: float = 0.2,
    vf_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
) -> dict[str, float]:
    """PPO update on flattened rollout data. Returns stats dict."""
    N = obs.shape[0]
    sum_pi, sum_vf, sum_ent = 0.0, 0.0, 0.0
    sum_kl, sum_clip_frac = 0.0, 0.0
    n_updates = 0

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

            v = value.view(-1)
            vf_loss = 0.5 * ((v - returns[idx]) ** 2).mean()

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

    # Explained variance: how well the value function predicts returns
    with torch.no_grad():
        _, _, _, all_values, _ = model(obs, h)
        v_pred = all_values.view(-1)
        var_returns = returns.var()
        ev = 1.0 - (returns - v_pred).var() / (var_returns + 1e-8) if var_returns > 1e-8 else 0.0
        if isinstance(ev, torch.Tensor):
            ev = ev.item()

    return {
        "pi_loss": sum_pi / max(n_updates, 1),
        "vf_loss": sum_vf / max(n_updates, 1),
        "entropy": sum_ent / max(n_updates, 1),
        "approx_kl": sum_kl / max(n_updates, 1),
        "clip_frac": sum_clip_frac / max(n_updates, 1),
        "explained_var": round(ev, 4),
        "n_updates": n_updates,
    }


def trajectories_to_tensors(
    trajectories: list[list[dict]],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> dict[str, torch.Tensor] | None:
    """Convert engine-collected trajectories to flat tensors with GAE advantages.

    Returns dict with keys: obs, h, bt, bd, md, old_log_probs, advantages, returns.
    Or None if no data.
    """
    all_obs, all_h = [], []
    all_bt, all_bd, all_md = [], [], []
    all_rewards, all_values, all_old_lp = [], [], []
    ep_boundaries = []

    idx = 0
    for traj in trajectories:
        start = idx
        for step in traj:
            if isinstance(step, (list, tuple)):
                n = len(step)
                obs, h_prev, bt, bd, md, value = step[:6]
                dti = step[6] if n > 6 else 0.0
                harv = step[9] if n > 9 else 0.0
                dmg = step[10] if n > 10 else 0.0
                old_lp = step[11] if n > 11 else 0.0
                conv = step[12] if n > 12 else 0.0
            else:
                obs = step["obs"]
                h_prev = step["h_prev"]
                bt, bd, md = step["bt"], step["bd"], step["md"]
                value = step["value"]
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
            all_values.append(value)
            all_old_lp.append(old_lp)

            # Reward: 4 signals
            # 1. Harvester proximity to ore (0-1, scaled by 5)
            # 2. Conveyor chain connectivity (0/0.5/1, scaled by 2)
            # 3. Resources delivered to core (+0.2 per ti)
            # 4. Damage dealt to enemy (+1 per HP)
            reward = harv * 5.0 + conv * 2.0 + max(dti, 0) * 0.2 + dmg * 1.0
            all_rewards.append(reward)
            idx += 1
        ep_boundaries.append((start, idx))

    N = len(all_obs)
    if N == 0:
        return None

    obs_t = torch.tensor(all_obs, dtype=torch.float32, device=DEVICE)
    h_t = torch.tensor(all_h, dtype=torch.float32, device=DEVICE)
    bt_t = torch.tensor(all_bt, dtype=torch.long, device=DEVICE)
    bd_t = torch.tensor(all_bd, dtype=torch.long, device=DEVICE)
    md_t = torch.tensor(all_md, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(all_rewards, dtype=torch.float32, device=DEVICE)
    values = torch.tensor(all_values, dtype=torch.float32, device=DEVICE)
    old_lp_t = torch.tensor(all_old_lp, dtype=torch.float32, device=DEVICE)

    # GAE advantage estimation
    advantages = torch.zeros(N, device=DEVICE)
    for start, end in ep_boundaries:
        gae = 0.0
        for t in range(end - 1, start - 1, -1):
            next_val = values[t + 1] if t < end - 1 else 0.0
            delta = rewards[t] + gamma * next_val - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
    returns = advantages + values

    return {
        "obs": obs_t,
        "h": h_t,
        "bt": bt_t,
        "bd": bd_t,
        "md": md_t,
        "old_log_probs": old_lp_t,
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
    """End-to-end: trajectories -> tensors -> PPO update. Returns stats."""
    data = trajectories_to_tensors(trajectories, gamma, gae_lambda)
    if data is None:
        return {"pi_loss": 0, "vf_loss": 0, "entropy": 0, "n_updates": 0, "n_steps": 0}

    stats = ppo_update(
        model, optimizer,
        data["obs"], data["h"], data["bt"], data["bd"], data["md"],
        data["old_log_probs"], data["advantages"], data["returns"],
        **ppo_kwargs,
    )
    stats["n_steps"] = data["n_steps"]
    return stats
