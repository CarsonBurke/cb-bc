#!/usr/bin/env python3
"""GPU-accelerated training with vectorized persistent environments.

Each environment simulates a full round: persistent grid, buildings stay,
bot moves, and resources flow through conveyor chains to core. The primary
reward signal is resources delivered to core per step.

Usage:
    python bots/tensor/torch/train_gpu.py                         # Defaults
    python bots/tensor/torch/train_gpu.py --envs 4096 --rounds 500
    python bots/tensor/torch/train_gpu.py --ep-len 200 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Architecture constants (must match main.py) ----

OBS_GRID = 5  # 5x5 observation window
TILE_FEATURES = 12
GLOBAL_FEATURES = 11
POS_EMBED_DIM = 8
N_BLD_TYPES = 4
INPUT_SIZE = OBS_GRID * OBS_GRID * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN = [32, 32, 32, 16]
H_DIM = 16  # Recurrent hidden state size
N_BUILD = 12
N_DIR = 9

# Simulation grid
SIM_GRID = 20  # 20x20 map

# Building types (internal simulation IDs)
BLD_NONE = 0
BLD_HARV = 1
BLD_CONV = 2
BLD_ROAD = 3
BLD_TURRET = 4
BLD_BARRIER = 5

# Mapping from sim building type to observation one-hot index (N_BLD_TYPES=4)
# 0=none, 1=economy(harv+conv), 2=road, 3=military(turret+barrier)
BLD_TO_OBS = {BLD_NONE: 0, BLD_HARV: 1, BLD_CONV: 1, BLD_ROAD: 2, BLD_TURRET: 3, BLD_BARRIER: 3}

# Environment types
ENV_EMPTY = 0
ENV_WALL = 1
ENV_ORE_TI = 2
ENV_ORE_AX = 3

# Direction offsets: index 0-8 matching DIR_MAP_9 in main.py
# NW, N, NE, W, C, E, SW, S, SE
DIR_DX = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1], device=DEVICE)
DIR_DY = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1], device=DEVICE)

# Max conveyor chain length to propagate
MAX_CHAIN = 15

# Sinusoidal position embedding frequencies
_POS_QUARTER = POS_EMBED_DIM // 4
_POS_FREQS = torch.tensor(
    [1.0 / (10000.0 ** (2.0 * i / POS_EMBED_DIM)) for i in range(_POS_QUARTER)],
    device=DEVICE,
)

WEIGHT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights.json")


# ---- Model ----

class BuilderMLP(nn.Module):
    """MLP with recurrent hidden state and value head for PPO.

    Architecture: [obs, h_prev] -> trunk -> {action heads, value, h_next}
    Built dynamically from HIDDEN so architecture changes propagate.
    """

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        prev = INPUT_SIZE + H_DIM  # obs + recurrent hidden
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

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (bt_logits, bd_logits, md_logits, value, h_next)."""
        trunk_in = torch.cat([x, h_prev], dim=-1)
        h = self.trunk(trunk_in)
        return (
            self.head_build_type(h),
            self.head_build_dir(h),
            self.head_move_dir(h),
            self.head_value(h).squeeze(-1),
            torch.tanh(self.head_h_next(h)),  # bounded hidden state
        )

    def _linear_layers(self) -> list[nn.Linear]:
        """All Linear layers in parameter order (trunk, action heads, value, h_next)."""
        trunk_linears = [m for m in self.trunk if isinstance(m, nn.Linear)]
        return trunk_linears + [
            self.head_build_type,
            self.head_build_dir,
            self.head_move_dir,
            self.head_value,
            self.head_h_next,
        ]

    def export_weights(self) -> list[list[float]]:
        """Export all weights as JSON-compatible nested lists.

        tensor_lib Linear stores weight as (in, out) row-major.
        PyTorch Linear stores weight as (out, in).
        We transpose before flattening.
        """
        state: list[list[float]] = []
        for layer in self._linear_layers():
            state.append(layer.weight.T.contiguous().view(-1).tolist())
            state.append(layer.bias.view(-1).tolist())
        return state

    def import_weights(self, state: list[list[float]]) -> None:
        """Load weights from tensor_lib JSON format."""
        idx = 0
        for layer in self._linear_layers():
            w_flat = torch.tensor(state[idx], dtype=torch.float32)
            in_f, out_f = layer.in_features, layer.out_features
            layer.weight.data = w_flat.view(in_f, out_f).T.contiguous()
            idx += 1
            layer.bias.data = torch.tensor(state[idx], dtype=torch.float32)
            idx += 1


# ---- Grid shift helper ----

def shift_2d(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shift a (N, H, W) tensor by (dy, dx), zero-padding the edges."""
    if dy == 0 and dx == 0:
        return t
    N, H, W = t.shape
    result = torch.zeros_like(t)
    # Compute source/dest slices
    sy = slice(max(0, -dy), H - max(0, dy))
    dy_s = slice(max(0, dy), H - max(0, -dy))
    sx = slice(max(0, -dx), W - max(0, dx))
    dx_s = slice(max(0, dx), W - max(0, -dx))
    result[:, dy_s, dx_s] = t[:, sy, sx]
    return result


# ---- Vectorized Environment ----

class VecEnv:
    """N parallel game environments on GPU.

    Maintains persistent grid state: environment tiles, buildings,
    conveyor directions, bot position, core position, and resource counts.
    """

    def __init__(self, n: int, gs: int = SIM_GRID):
        self.n = n
        self.gs = gs
        self.reset()

    def reset(self):
        N, G = self.n, self.gs

        # Environment grid: 0=empty, 1=wall, 2=ore_ti, 3=ore_ax
        self.env = torch.zeros(N, G, G, dtype=torch.long, device=DEVICE)

        # Border walls
        self.env[:, 0, :] = ENV_WALL
        self.env[:, -1, :] = ENV_WALL
        self.env[:, :, 0] = ENV_WALL
        self.env[:, :, -1] = ENV_WALL

        # Random interior walls (~8% of interior)
        interior_mask = torch.zeros(N, G, G, dtype=torch.bool, device=DEVICE)
        interior_mask[:, 2:-2, 2:-2] = True
        wall_rand = (torch.rand(N, G, G, device=DEVICE) < 0.08) & interior_mask
        self.env[wall_rand] = ENV_WALL

        # Ore clusters: place 4-7 clusters of 2-4 tiles each
        for _ in range(6):
            # Random cluster centers
            cy = torch.randint(3, G - 3, (N,), device=DEVICE)
            cx = torch.randint(3, G - 3, (N,), device=DEVICE)
            ore_type = torch.where(
                torch.rand(N, device=DEVICE) > 0.4,
                torch.full((N,), ENV_ORE_TI, device=DEVICE),
                torch.full((N,), ENV_ORE_AX, device=DEVICE),
            ).long()
            # Place cluster (center + random neighbors)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if abs(dy) + abs(dx) > 1 and torch.rand(1).item() > 0.4:
                        continue
                    ty = (cy + dy).clamp(1, G - 2)
                    tx = (cx + dx).clamp(1, G - 2)
                    # Only place on empty tiles
                    idx = torch.arange(N, device=DEVICE)
                    is_empty = self.env[idx, ty, tx] == ENV_EMPTY
                    self.env[idx[is_empty], ty[is_empty], tx[is_empty]] = ore_type[is_empty]

        # Buildings grid: 0=none
        self.buildings = torch.zeros(N, G, G, dtype=torch.long, device=DEVICE)

        # Conveyor directions: (dx, dy) per tile
        self.conv_dx = torch.zeros(N, G, G, dtype=torch.long, device=DEVICE)
        self.conv_dy = torch.zeros(N, G, G, dtype=torch.long, device=DEVICE)

        # Core position: center of left edge area
        self.core_x = torch.full((N,), 2, dtype=torch.long, device=DEVICE)
        self.core_y = torch.full((N,), G // 2, dtype=torch.long, device=DEVICE)

        # Clear area around core (no walls/ore on core 3x3)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cy = (self.core_y + dy).clamp(0, G - 1)
                cx = (self.core_x + dx).clamp(0, G - 1)
                idx = torch.arange(N, device=DEVICE)
                self.env[idx, cy, cx] = ENV_EMPTY

        # Bot starts adjacent to core (one tile to the right)
        self.bot_x = self.core_x + 2
        self.bot_y = self.core_y.clone()

        # Resource tracking
        self.total_resources = torch.zeros(N, device=DEVICE)
        self.step_count = torch.zeros(N, dtype=torch.long, device=DEVICE)

    def observe(self) -> torch.Tensor:
        """Extract (N, INPUT_SIZE) observation tensor matching bot's _encode_state.

        Per tile (19): env_oh(4) + bld_type_oh(8) + bld_friendly(1) + bld_dir(2)
                       + bld_hp(1) + has_unit(1) + unit_friendly(1) + is_passable(1)
        Global (11): round, x, y, ti, ax, scale, core_dx, core_dy,
                     ecore_dx, ecore_dy, n_builders
        Pos embed (8): sinusoidal
        """
        N, G = self.n, self.gs

        # Pad grids for edge handling
        pad = OBS_GRID // 2
        padded_env = F.pad(self.env.float(), (pad, pad, pad, pad), value=ENV_WALL).long()
        padded_bld = F.pad(self.buildings.float(), (pad, pad, pad, pad), value=0).long()
        padded_cdx = F.pad(self.conv_dx.float(), (pad, pad, pad, pad), value=0)
        padded_cdy = F.pad(self.conv_dy.float(), (pad, pad, pad, pad), value=0)

        # Extract 5x5 patches
        by = self.bot_y + pad
        bx = self.bot_x + pad
        offsets = torch.arange(-pad, pad + 1, device=DEVICE)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        oy = oy.reshape(1, 25)
        ox = ox.reshape(1, 25)
        patch_y = by.unsqueeze(1) + oy  # (N, 25)
        patch_x = bx.unsqueeze(1) + ox

        PW = G + 2 * pad
        flat_idx = patch_y * PW + patch_x  # (N, 25)

        local_env = padded_env.reshape(N, -1).gather(1, flat_idx)  # (N, 25)
        local_bld = padded_bld.reshape(N, -1).gather(1, flat_idx)
        local_cdx = padded_cdx.reshape(N, -1).gather(1, flat_idx)  # (N, 25)
        local_cdy = padded_cdy.reshape(N, -1).gather(1, flat_idx)

        # 1. Environment (3): is_wall, ore_ti, ore_ax
        is_wall = (local_env == ENV_WALL).float().unsqueeze(-1)
        ore_ti = (local_env == ENV_ORE_TI).float().unsqueeze(-1)
        ore_ax = (local_env == ENV_ORE_AX).float().unsqueeze(-1)

        # 2. Building type one-hot (4): none/econ/road/military
        bld_obs_idx = torch.zeros_like(local_bld)
        bld_obs_idx = torch.where(local_bld == BLD_HARV, 1, bld_obs_idx)
        bld_obs_idx = torch.where(local_bld == BLD_CONV, 1, bld_obs_idx)
        bld_obs_idx = torch.where(local_bld == BLD_ROAD, 2, bld_obs_idx)
        bld_obs_idx = torch.where(local_bld == BLD_TURRET, 3, bld_obs_idx)
        bld_obs_idx = torch.where(local_bld == BLD_BARRIER, 3, bld_obs_idx)
        bld_type_oh = F.one_hot(bld_obs_idx, num_classes=N_BLD_TYPES).float()  # (N, 25, 4)

        # 3. Building friendly (1): all own buildings = +1
        has_bld = (local_bld > 0).float()
        bld_friendly = has_bld.unsqueeze(-1)

        # 4. Building direction (2): only conveyors have direction
        is_conv = (local_bld == BLD_CONV).float()
        bld_dir = torch.stack([
            local_cdx * is_conv,
            local_cdy * is_conv,
        ], dim=-1)  # (N, 25, 2)

        # 5. Has unit (1) + unit friendly (1)
        has_unit = torch.zeros(N, 25, 1, device=DEVICE)
        has_unit[:, 12] = 1.0
        unit_friendly = has_unit.clone()

        # Concatenate: 3 + 4 + 1 + 2 + 1 + 1 = 12
        tile_feats = torch.cat([
            is_wall, ore_ti, ore_ax,
            bld_type_oh, bld_friendly, bld_dir,
            has_unit, unit_friendly,
        ], dim=-1)  # (N, 25, 12)
        tile_flat = tile_feats.reshape(N, 25 * TILE_FEATURES)  # (N, 300)

        # Global features (11)
        step_norm = self.step_count.float() / 2000.0
        x_norm = self.bot_x.float() / G
        y_norm = self.bot_y.float() / G
        ti_delivered = self.total_resources / 500.0
        ax_delivered = torch.zeros(N, device=DEVICE)
        bld_count = (self.buildings > 0).float().sum(dim=(1, 2))
        scale = bld_count / 50.0

        # Direction to own core
        cdx = (self.core_x - self.bot_x).float()
        cdy = (self.core_y - self.bot_y).float()
        cd_max = cdx.abs().clamp(min=1).max(cdy.abs().clamp(min=1))
        core_dx = cdx / cd_max
        core_dy = cdy / cd_max

        # Direction to enemy core (opposite side of map)
        ecore_x = G - 1 - self.core_x
        ecore_y = G - 1 - self.core_y
        edx = (ecore_x - self.bot_x).float()
        edy = (ecore_y - self.bot_y).float()
        ed_max = edx.abs().clamp(min=1).max(edy.abs().clamp(min=1))
        ecore_dx = edx / ed_max
        ecore_dy = edy / ed_max

        n_builders = torch.ones(N, device=DEVICE) * 0.1  # sim has 1 builder

        global_feats = torch.stack([
            step_norm, x_norm, y_norm, ti_delivered, ax_delivered, scale,
            core_dx, core_dy, ecore_dx, ecore_dy, n_builders,
        ], dim=-1)  # (N, 11)

        # Sinusoidal position embedding (8)
        pos_feats = torch.zeros(N, POS_EMBED_DIM, device=DEVICE)
        px = self.bot_x.float()
        py = self.bot_y.float()
        half = _POS_QUARTER * 2
        for i in range(_POS_QUARTER):
            ax = px * _POS_FREQS[i]
            pos_feats[:, 2 * i] = torch.sin(ax)
            pos_feats[:, 2 * i + 1] = torch.cos(ax)
            ay = py * _POS_FREQS[i]
            pos_feats[:, half + 2 * i] = torch.sin(ay)
            pos_feats[:, half + 2 * i + 1] = torch.cos(ay)

        return torch.cat([tile_flat, global_feats, pos_feats], dim=-1)  # (N, INPUT_SIZE)

    def step(
        self,
        build_type: torch.Tensor,
        build_dir: torch.Tensor,
        move_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Apply actions, simulate resource flow. Returns per-env reward (N,)."""
        N, G = self.n, self.gs
        idx = torch.arange(N, device=DEVICE)

        reward = torch.zeros(N, device=DEVICE)

        # ---- Build action ----
        bdx = DIR_DX[build_dir]  # (N,)
        bdy = DIR_DY[build_dir]
        tx = (self.bot_x + bdx).clamp(0, G - 1)
        ty = (self.bot_y + bdy).clamp(0, G - 1)

        target_env = self.env[idx, ty, tx]
        target_bld = self.buildings[idx, ty, tx]
        not_wall = target_env != ENV_WALL
        not_built = target_bld == BLD_NONE
        is_adj = build_dir != 4  # not CENTRE

        # Harvester: must be on ore, no existing building
        is_harv = build_type == 1
        harv_valid = is_harv & is_adj & not_wall & not_built & (
            (target_env == ENV_ORE_TI) | (target_env == ENV_ORE_AX)
        )
        self.buildings[idx[harv_valid], ty[harv_valid], tx[harv_valid]] = BLD_HARV

        # Shaped reward: harvester on ore
        reward += harv_valid.float() * 5.0
        reward += (harv_valid & (target_env == ENV_ORE_TI)).float() * 1.0
        reward += (is_harv & ~harv_valid & is_adj).float() * -1.0  # failed harvester

        # Conveyor: point toward core
        is_conv = build_type == 2
        conv_valid = is_conv & is_adj & not_wall & not_built
        self.buildings[idx[conv_valid], ty[conv_valid], tx[conv_valid]] = BLD_CONV
        # Conveyor direction: point from target toward core
        cvdx = torch.sign(self.core_x - tx).long()
        cvdy = torch.sign(self.core_y - ty).long()
        # If both zero (at core), use build direction
        at_core = (cvdx == 0) & (cvdy == 0)
        cvdx = torch.where(at_core, bdx, cvdx)
        cvdy = torch.where(at_core, bdy, cvdy)
        self.conv_dx[idx[conv_valid], ty[conv_valid], tx[conv_valid]] = cvdx[conv_valid]
        self.conv_dy[idx[conv_valid], ty[conv_valid], tx[conv_valid]] = cvdy[conv_valid]

        # Shaped reward: conveyor placement
        reward += conv_valid.float() * 1.0

        # Road
        is_road = build_type == 3
        road_valid = is_road & is_adj & not_wall & not_built
        self.buildings[idx[road_valid], ty[road_valid], tx[road_valid]] = BLD_ROAD
        reward += road_valid.float() * 0.3

        # ---- Move action ----
        mdx = DIR_DX[move_dir]
        mdy = DIR_DY[move_dir]
        mx = (self.bot_x + mdx).clamp(0, G - 1)
        my = (self.bot_y + mdy).clamp(0, G - 1)

        move_env = self.env[idx, my, mx]
        move_bld = self.buildings[idx, my, mx]
        can_move = (move_dir != 4) & (move_env != ENV_WALL) & (
            (move_bld == BLD_ROAD) | (move_bld == BLD_CONV) | (move_bld == BLD_NONE)
        )
        # Can also move onto ore tiles (passable in the real game? only if road/conv)
        # Actually in the game you need a road/conveyor to move. Let's be strict:
        # Can move if target has road or conveyor, OR target is empty (no building, not wall)
        can_move = (move_dir != 4) & (move_env != ENV_WALL) & (
            (move_bld == BLD_ROAD) | (move_bld == BLD_CONV)
        )

        self.bot_x = torch.where(can_move, mx, self.bot_x)
        self.bot_y = torch.where(can_move, my, self.bot_y)

        # ---- Resource flow simulation ----
        resources_this_step = self._compute_resource_flow()
        self.total_resources += resources_this_step

        # Primary reward: resources delivered to core (strongest signal)
        reward += resources_this_step * 20.0

        self.step_count += 1

        return reward

    def _compute_resource_flow(self) -> torch.Tensor:
        """Compute resources delivered this step via conveyor chains.

        Propagates connectivity from core through conveyors. Harvesters on
        ore tiles adjacent to connected conveyors deliver resources.
        Returns (N,) count of delivering harvesters.
        """
        N, G = self.n, self.gs

        # Initialize: tiles within distance 1 of core center are connected
        connected = torch.zeros(N, G, G, dtype=torch.bool, device=DEVICE)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cy = (self.core_y + dy).clamp(0, G - 1)
                cx = (self.core_x + dx).clamp(0, G - 1)
                connected[torch.arange(N, device=DEVICE), cy, cx] = True

        # Propagate: a conveyor is connected if the tile it points to is connected
        is_conv = self.buildings == BLD_CONV

        for _ in range(MAX_CHAIN):
            new_conn = connected.clone()
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    # Conveyors pointing in direction (dx, dy)
                    pointing = is_conv & (self.conv_dx == dx) & (self.conv_dy == dy)
                    if not pointing.any():
                        continue
                    # Their target tile (y+dy, x+dx) — check if connected
                    target_conn = shift_2d(connected, dy, dx)
                    # If target is connected, this conveyor tile is connected
                    new_conn |= (pointing & target_conn)

            if torch.equal(new_conn, connected):
                break  # converged early
            connected = new_conn

        # Harvesters on ore deliver if adjacent to a connected tile
        is_harv_ore = (self.buildings == BLD_HARV) & (
            (self.env == ENV_ORE_TI) | (self.env == ENV_ORE_AX)
        )

        # Check adjacency to connected tiles
        adj_connected = torch.zeros(N, G, G, dtype=torch.bool, device=DEVICE)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                adj_connected |= shift_2d(connected, -dy, -dx)

        delivering = is_harv_ore & adj_connected
        return delivering.float().sum(dim=(1, 2))  # (N,)


# ---- PPO Training ----

def collect_rollout(
    model: BuilderMLP,
    env: VecEnv,
    ep_len: int,
    gamma: float,
    gae_lambda: float,
) -> dict[str, torch.Tensor]:
    """Collect a full rollout and compute GAE advantages.

    Returns dict with keys: obs, h_prev, bt_act, bd_act, md_act,
    log_probs, values, rewards, advantages, returns — each (ep_len, N).
    """
    env.reset()
    N = env.n

    all_obs: list[torch.Tensor] = []
    all_h_prev: list[torch.Tensor] = []
    all_bt: list[torch.Tensor] = []
    all_bd: list[torch.Tensor] = []
    all_md: list[torch.Tensor] = []
    all_log_probs: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []

    h = torch.zeros(N, H_DIM, device=DEVICE)

    with torch.no_grad():
        for _ in range(ep_len):
            obs = env.observe()
            bt_logits, bd_logits, md_logits, value, h_next = model(obs, h)

            bt_dist = Categorical(logits=bt_logits)
            bd_dist = Categorical(logits=bd_logits)
            md_dist = Categorical(logits=md_logits)

            bt_act = bt_dist.sample()
            bd_act = bd_dist.sample()
            md_act = md_dist.sample()

            log_prob = (
                bt_dist.log_prob(bt_act)
                + bd_dist.log_prob(bd_act)
                + md_dist.log_prob(md_act)
            )

            reward = env.step(bt_act, bd_act, md_act)

            all_obs.append(obs)
            all_h_prev.append(h)
            all_bt.append(bt_act)
            all_bd.append(bd_act)
            all_md.append(md_act)
            all_log_probs.append(log_prob)
            all_values.append(value)
            all_rewards.append(reward)

            h = h_next

        # Bootstrap value for last state
        last_obs = env.observe()
        _, _, _, last_value, _ = model(last_obs, h)

    # Stack: (ep_len, N)
    values = torch.stack(all_values)
    rewards = torch.stack(all_rewards)

    # GAE advantage estimation
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=DEVICE)
    for t in range(ep_len - 1, -1, -1):
        next_val = last_value if t == ep_len - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values

    return {
        "obs": torch.stack(all_obs),
        "h_prev": torch.stack(all_h_prev),
        "bt_act": torch.stack(all_bt),
        "bd_act": torch.stack(all_bd),
        "md_act": torch.stack(all_md),
        "old_log_probs": torch.stack(all_log_probs),
        "values": values,
        "rewards": rewards,
        "advantages": advantages,
        "returns": returns,
    }


def ppo_update(
    model: BuilderMLP,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    vf_coeff: float,
    entropy_coeff: float,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
) -> tuple[float, float, float]:
    """PPO update epochs on collected rollout (matches CleanRL ppo.py).

    Returns (policy_loss, value_loss, entropy).
    """
    T, N = rollout["rewards"].shape
    total = T * N

    # Flatten (T, N, ...) -> (T*N, ...)
    obs_flat = rollout["obs"].reshape(total, -1)
    h_flat = rollout["h_prev"].reshape(total, H_DIM)
    bt_flat = rollout["bt_act"].reshape(total)
    bd_flat = rollout["bd_act"].reshape(total)
    md_flat = rollout["md_act"].reshape(total)
    old_lp_flat = rollout["old_log_probs"].reshape(total)
    adv_flat = rollout["advantages"].reshape(total)
    ret_flat = rollout["returns"].reshape(total)
    old_val_flat = rollout["values"].reshape(total)

    sum_pi_loss = 0.0
    sum_vf_loss = 0.0
    sum_entropy = 0.0
    n_updates = 0

    for epoch in range(ppo_epochs):
        approx_kl_acc = 0.0
        n_mb = 0

        perm = torch.randperm(total, device=DEVICE)
        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = perm[start:end]

            mb_obs = obs_flat[idx]
            mb_h = h_flat[idx]
            mb_bt = bt_flat[idx]
            mb_bd = bd_flat[idx]
            mb_md = md_flat[idx]
            mb_old_lp = old_lp_flat[idx]
            mb_adv = adv_flat[idx]
            mb_ret = ret_flat[idx]
            mb_old_val = old_val_flat[idx]

            bt_logits, bd_logits, md_logits, value, _ = model(mb_obs, mb_h)

            bt_dist = Categorical(logits=bt_logits)
            bd_dist = Categorical(logits=bd_logits)
            md_dist = Categorical(logits=md_logits)

            new_lp = (
                bt_dist.log_prob(mb_bt)
                + bd_dist.log_prob(mb_bd)
                + md_dist.log_prob(mb_md)
            )
            entropy = bt_dist.entropy() + bd_dist.entropy() + md_dist.entropy()

            logratio = new_lp - mb_old_lp
            ratio = logratio.exp()

            # KL divergence approximation (Schulman blog)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                approx_kl_acc += approx_kl.item()
                n_mb += 1

            # Per-minibatch advantage normalization (CleanRL style)
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            # Clipped surrogate policy loss
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pi_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Clipped value loss (CleanRL style)
            value = value.view(-1)
            v_loss_unclipped = (value - mb_ret) ** 2
            v_clipped = mb_old_val + torch.clamp(
                value - mb_old_val, -clip_eps, clip_eps,
            )
            v_loss_clipped = (v_clipped - mb_ret) ** 2
            vf_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            # Combined loss
            loss = pi_loss + vf_coeff * vf_loss - entropy_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            sum_pi_loss += pi_loss.item()
            sum_vf_loss += vf_loss.item()
            sum_entropy += entropy.mean().item()
            n_updates += 1

        # Early stopping on KL divergence (CleanRL style)
        if target_kl is not None and n_mb > 0:
            if approx_kl_acc / n_mb > target_kl:
                break

    return (
        sum_pi_loss / max(n_updates, 1),
        sum_vf_loss / max(n_updates, 1),
        sum_entropy / max(n_updates, 1),
    )


def main():
    parser = argparse.ArgumentParser(description="PPO training with vectorized environments")
    parser.add_argument("--envs", type=int, default=2048, help="Parallel environments")
    parser.add_argument("--ep-len", type=int, default=200, help="Game rounds per episode")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--grid", type=int, default=SIM_GRID, help="Simulation grid size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--minibatch", type=int, default=4096, help="Minibatch size for PPO updates")
    parser.add_argument("--vf-coeff", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--entropy", type=float, default=0.01, help="Entropy bonus")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--target-kl", type=float, default=None, help="Target KL for early stopping")
    parser.add_argument("--save", default=WEIGHT_PATH, help="Weight save path")
    parser.add_argument("--load", default=None, help="Load existing weights")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    total = args.envs * args.ep_len * args.episodes
    print(f"PPO Training: {args.episodes} episodes x {args.envs} envs x {args.ep_len} rounds/ep = {total:,} samples")
    print(f"Grid: {args.grid}x{args.grid} | lr={args.lr} gamma={args.gamma} "
          f"gae_lambda={args.gae_lambda} clip={args.clip_eps} epochs={args.ppo_epochs}")

    model = BuilderMLP().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params | hidden={HIDDEN} h_dim={H_DIM}")

    load_path = args.load or (args.save if os.path.exists(args.save) else None)
    if load_path and os.path.exists(load_path):
        try:
            with open(load_path) as f:
                model.import_weights(json.load(f))
            model = model.to(DEVICE)
            print(f"Loaded weights from {load_path}")
        except Exception as e:
            print(f"Could not load weights: {e}, starting fresh")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    env = VecEnv(args.envs, args.grid)

    best_resources = float("-inf")
    t_start = time.time()

    train_log: list[dict] = []
    log_path = os.path.join(os.path.dirname(os.path.abspath(args.save)), "train_log.json")

    for ep in range(args.episodes):
        t0 = time.time()

        rollout = collect_rollout(model, env, args.ep_len, args.gamma, args.gae_lambda)

        pi_loss, vf_loss, entropy = ppo_update(
            model, optimizer, rollout,
            args.ppo_epochs, args.minibatch, args.clip_eps,
            args.vf_coeff, args.entropy,
            args.max_grad_norm, args.target_kl,
        )
        scheduler.step()

        dt = time.time() - t0
        sps = args.envs * args.ep_len / dt
        mean_reward = rollout["rewards"].mean().item()
        mean_resources = env.total_resources.mean().item()
        lr_now = scheduler.get_last_lr()[0]

        if mean_resources > best_resources:
            best_resources = mean_resources
            with open(args.save, "w") as f:
                json.dump(model.export_weights(), f)

        # Log every episode
        train_log.append({
            "episode": ep + 1,
            "reward": round(mean_reward, 4),
            "resources": round(mean_resources, 2),
"best_resources": round(best_resources, 2),
            "pi_loss": round(pi_loss, 5),
            "vf_loss": round(vf_loss, 5),
            "entropy": round(entropy, 4),
            "lr": round(lr_now, 8),
            "sps": int(sps),
            "elapsed": round(time.time() - t_start, 1),
        })

        # Write log incrementally so dashboard can read live
        if (ep + 1) % 5 == 0 or ep == 0 or ep == args.episodes - 1:
            with open(log_path, "w") as f:
                json.dump({"episodes": train_log, "config": {
                    "envs": args.envs, "ep_len": args.ep_len,
                    "grid": args.grid, "lr": args.lr,
                    "gamma": args.gamma, "total_episodes": args.episodes,
                }}, f)

        if (ep + 1) % 10 == 0 or ep == 0:
            elapsed = time.time() - t_start
            print(
                f"  Ep {ep + 1:4d}/{args.episodes}: "
                f"reward={mean_reward:+7.2f}  res={mean_resources:5.1f}  "
                f"pi={pi_loss:+.4f} vf={vf_loss:.4f} ent={entropy:.3f}  "
                f"best={best_resources:5.1f}  "
                f"lr={lr_now:.2e}  {sps:,.0f} s/s  [{elapsed:.0f}s]"
            )

    with open(args.save, "w") as f:
        json.dump(model.export_weights(), f)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s. Best resources: {best_resources:.1f}")
    print(f"Throughput: {total / total_time:,.0f} samples/sec")
    print(f"Weights saved to {args.save}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
