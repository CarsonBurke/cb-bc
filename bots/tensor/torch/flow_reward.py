"""Helpers for replay-derived flow rewards."""

from __future__ import annotations

import struct
from pathlib import Path

# Obs feature indices — must stay in sync with model.py TILE_FEATURES
_TILE_FEATURES = 21
_OBS_GLOBAL_START = 5 * 5 * _TILE_FEATURES  # 525
_OBS_ROUND_BYTE_OFFSET = _OBS_GLOBAL_START * 4  # byte offset for float32 obs

AX_CONVERSION_RATE = 4  # GameConstants.AXIONITE_CONVERSION_TITANIUM_RATE


def _load_buf(global_states: dict, buf_key: str, path_key: str) -> bytes:
    buf = global_states.get(buf_key, b"")
    if not buf:
        path = global_states.get(path_key)
        if path:
            buf = Path(path).read_bytes()
    return buf


def compute_dense_delivery_rewards(
    global_states: dict,
    team: int = 0,
) -> dict[int, float]:
    """Per-round reward from resources delivered to core (ti + ax * 4).

    Requires ti_delivered and ax_delivered buffers. Raises if missing.
    """
    n_turns = global_states.get("n_turns", 0)
    if n_turns == 0:
        return {}

    ti_buf = _load_buf(global_states, "ti_delivered_buf", "ti_delivered_path")
    ax_buf = _load_buf(global_states, "ax_delivered_buf", "ax_delivered_path")
    if not ti_buf:
        raise ValueError("ti_delivered data missing from global_states")
    if not ax_buf:
        raise ValueError("ax_delivered data missing from global_states")

    rewards: dict[int, float] = {}
    for rnd in range(n_turns):
        ti_a, ti_b = struct.unpack_from("<ii", ti_buf, rnd * 8)
        ax_a, ax_b = struct.unpack_from("<ii", ax_buf, rnd * 8)
        delivered = (ti_a if team == 0 else ti_b) + (ax_a if team == 0 else ax_b) * AX_CONVERSION_RATE
        if delivered > 0:
            rewards[rnd] = float(delivered)
    return rewards


def compute_dense_flow_rewards(
    global_states: dict,
    team: int = 0,
) -> dict[int, float]:
    """Per-round reward from weighted conveyor hops toward core.

    Requires transfer_events buffer. Raises if missing.
    """
    n_turns = global_states.get("n_turns", 0)
    if n_turns == 0:
        return {}

    buf = _load_buf(global_states, "transfer_events_buf", "transfer_events_path")
    if not buf:
        raise ValueError("transfer_events data missing from global_states")

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
        rewards: {round: reward} from compute_dense_*_rewards

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
