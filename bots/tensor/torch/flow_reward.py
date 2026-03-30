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
    """Decode per-turn dense flow reward from replay parser output.

    The replay parser emits contiguous f32 pairs [team_a, team_b] for each
    round in ``transfer_events_buf``. We expose that as ``{round: reward}``
    for the requested team.
    """
    buf = global_states.get("transfer_events_buf", b"")
    if not buf:
        transfer_path = global_states.get("transfer_events_path")
        if transfer_path:
            buf = Path(transfer_path).read_bytes()
    n_turns = global_states.get("n_turns", 0)
    if not buf or n_turns == 0:
        return {}

    rewards: dict[int, float] = {}
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
