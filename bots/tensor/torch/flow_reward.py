"""Helpers for replay-derived flow rewards."""

from __future__ import annotations

import struct
from pathlib import Path

<<<<<<< Updated upstream
from model import REWARD_INCOME_SCALE, REWARD_FLOW_SCALE
try:
    from trajectory_schema import TrajectoryStep
except ImportError:
    from bots.tensor.trajectory_schema import TrajectoryStep
try:
    from replay_buffers import (
        SCALAR_BUFFER,
        TRANSFER_EVENTS_BUFFER,
        load_replay_buffer,
    )
except ImportError:
    from bots.tensor.torch.replay_buffers import (
        SCALAR_BUFFER,
        TRANSFER_EVENTS_BUFFER,
        load_replay_buffer,
    )

# Obs feature indices — must stay in sync with model.py TILE_FEATURES
_TILE_FEATURES = 21
_OBS_GLOBAL_START = 5 * 5 * _TILE_FEATURES  # 525
_OBS_ROUND_BYTE_OFFSET = _OBS_GLOBAL_START * 4  # byte offset for float32 obs

# Scalar buffer layout (replay_parser replay.rs): 19 f32s per turn.
# Index 9 = own resources_collected / 500.0
_N_SCALARS = 19
_RESOURCES_COLLECTED_IDX = 9
_RESOURCES_COLLECTED_NORM = 500.0


def _load_buf(global_states: dict, spec) -> bytes:
    return load_replay_buffer(global_states, spec) or b""


def _resources_collected_deltas(global_states: dict) -> list[float]:
    """Per-round resources_collected deltas from the scalars buffer.

    The scalars buffer is team-relative (index 9 = own resources_collected),
    so no team parameter is needed.  Returns unnormalized integer-scale deltas.
    """
    n_turns = global_states.get("n_turns", 0)
    if n_turns == 0:
        return []

    sc_buf = _load_buf(global_states, SCALAR_BUFFER)
    if not sc_buf:
        raise ValueError("scalars data missing from global_states")

    stride = _N_SCALARS * 4  # bytes per turn
    deltas: list[float] = []
    prev = 0.0
    for rnd in range(n_turns):
        (val_norm,) = struct.unpack_from("<f", sc_buf, rnd * stride + _RESOURCES_COLLECTED_IDX * 4)
        val = val_norm * _RESOURCES_COLLECTED_NORM
        deltas.append(val - prev)
        prev = val
    return deltas


def compute_dense_delivery_rewards(
    global_states: dict,
    team: int = 0,
) -> dict[int, float]:
    """Per-round reward from resources delivered to core.

    Uses resources_collected deltas from the scalars buffer, which includes
    both titanium deliveries and axionite refinement output.
    """
    deltas = _resources_collected_deltas(global_states)
    rewards: dict[int, float] = {}
    for rnd, d in enumerate(deltas):
        if d > 0.5:  # threshold at 0.5 to absorb float rounding
            rewards[rnd] = round(d)
    return rewards
=======
# Obs feature indices
_OBS_GLOBAL_START = 5 * 5 * 19  # GRID_SIZE^2 * TILE_FEATURES = 475
_OBS_ROUND_BYTE_OFFSET = _OBS_GLOBAL_START * 4  # byte offset for float32 obs
>>>>>>> Stashed changes


def compute_dense_flow_rewards(
    global_states: dict,
    team: int = 0,
) -> dict[int, float]:
    """Per-round reward from weighted conveyor hops toward core + delivery.

    Conveyor hops use improvement/sqrt(src_dist) weighting with flow-field
    distance. Core deliveries are added from resources_collected deltas
    so transfer hops exclude the final hop into core.

    Requires transfer_events buffer. Raises if missing.
    """
<<<<<<< Updated upstream
=======
    buf = global_states.get("transfer_events_buf", b"")
    if not buf:
        transfer_path = global_states.get("transfer_events_path")
        if transfer_path:
            buf = Path(transfer_path).read_bytes()
>>>>>>> Stashed changes
    n_turns = global_states.get("n_turns", 0)
    if n_turns == 0:
        return {}

    buf = _load_buf(global_states, TRANSFER_EVENTS_BUFFER)
    if not buf:
        raise ValueError("transfer_events data missing from global_states")

    delivery_deltas = _resources_collected_deltas(global_states)

    rewards: dict[int, float] = {}
    for rnd in range(n_turns):
        a_val, b_val = struct.unpack_from("<ff", buf, rnd * 8)
        reward = (a_val if team == 0 else b_val) * REWARD_FLOW_SCALE
        if rnd < len(delivery_deltas) and delivery_deltas[rnd] > 0.5:
            reward += round(delivery_deltas[rnd]) * REWARD_INCOME_SCALE
        if reward != 0.0:
            rewards[rnd] = float(reward)
    return rewards


def augment_trajectories_with_flow_rewards(
    trajectories: list[list[TrajectoryStep]],
    rewards: dict[int, float],
) -> list[list[TrajectoryStep]]:
    """Write replay-derived round reward into each strict trajectory step.

    Each trajectory step already carries its round index, so reward injection
    is a pure schema-level update with no obs decoding or tuple mutation.

    Args:
        trajectories: list of trajectory lists
        rewards: {round: reward} from compute_dense_*_rewards

    Returns:
        Modified trajectories (in-place list replacement + return)
    """
    for traj in trajectories:
        for i, step in enumerate(traj):
<<<<<<< Updated upstream
            if not isinstance(step, TrajectoryStep):
                raise TypeError(f"Expected TrajectoryStep, got {type(step).__name__}")
            traj[i] = step.with_reward(rewards.get(step.round_idx, 0.0))
=======
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
>>>>>>> Stashed changes

    return trajectories
