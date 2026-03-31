"""Helpers for replay-derived flow rewards."""

from __future__ import annotations

import struct
from pathlib import Path

# Obs feature indices
_OBS_GLOBAL_START = 5 * 5 * 19  # GRID_SIZE^2 * TILE_FEATURES = 475
_OBS_ROUND_BYTE_OFFSET = _OBS_GLOBAL_START * 4  # byte offset for float32 obs


def compute_dense_flow_rewards(
    global_states: dict,
    team: int = 0,
) -> dict[int, float]:
    """Compute per-round reward from titanium income (delivered to core).

    Primary signal: titanium_delivered — per-turn delta of titanium_collected
    per team (i32 pairs [team_a, team_b]).

    Fallback: transfer_events — weighted conveyor hop count toward core
    (f32 pairs [team_a, team_b]).

    Returns {round: reward} for the requested team.
    """
    # Try titanium delivered first (cleaner signal)
    ti_buf = global_states.get("ti_delivered_buf", b"")
    if not ti_buf:
        ti_path = global_states.get("ti_delivered_path")
        if ti_path:
            try:
                ti_buf = Path(ti_path).read_bytes()
            except OSError:
                ti_buf = b""

    n_turns = global_states.get("n_turns", 0)

    if ti_buf and n_turns > 0:
        rewards: dict[int, float] = {}
        for rnd in range(n_turns):
            a_val, b_val = struct.unpack_from("<ii", ti_buf, rnd * 8)
            delivered = a_val if team == 0 else b_val
            if delivered > 0:
                rewards[rnd] = float(delivered)
        return rewards

    # Fallback: transfer events (weighted conveyor hops toward core)
    buf = global_states.get("transfer_events_buf", b"")
    if not buf:
        transfer_path = global_states.get("transfer_events_path")
        if transfer_path:
            try:
                buf = Path(transfer_path).read_bytes()
            except OSError:
                buf = b""

    if not buf or n_turns == 0:
        return {}

    rewards = {}
    for rnd in range(n_turns):
        a_val, b_val = struct.unpack_from("<ff", buf, rnd * 8)
        reward = a_val if team == 0 else b_val
        if reward != 0.0:
            rewards[rnd] = float(reward)
    return rewards


def augment_trajectories_with_flow_rewards(
    trajectories: list[list],
    rewards: dict[int, float],
) -> list[list]:
    """Write replay-derived round reward into the n_working slot.

    Each trajectory step's round is extracted from obs, and the shared team
    reward for that round (if any) is written into the n_working slot.

    Args:
        trajectories: list of trajectory lists (each step is a tuple/list)
        rewards: {round: reward} from compute_dense_flow_rewards

    Returns:
        Modified trajectories (in-place mutation + return)
    """
    for traj in trajectories:
        for i, step in enumerate(traj):
            if not isinstance(step, (list, tuple)) or len(step) < 7:
                continue
            if isinstance(step, tuple):
                step = list(step)
                traj[i] = step
            obs = step[0]
            if isinstance(obs, (bytes, bytearray, memoryview)):
                rnd_val = struct.unpack_from('<f', obs, _OBS_ROUND_BYTE_OFFSET)[0]
            else:
                rnd_val = obs[_OBS_GLOBAL_START]
            rnd = int(round(rnd_val * 2000.0))
            rnd = max(0, min(rnd, 1999))
            slot = 7 if len(step) > 12 else 6  # n_working slot
            step[slot] = rewards.get(rnd, 0.0)

    return trajectories
