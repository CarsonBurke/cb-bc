"""Helpers for replay-derived flow rewards."""

from __future__ import annotations

import struct

from model import REWARD_INCOME_SCALE, REWARD_FLOW_SCALE
try:
    from trajectory_schema import TrajectoryStep
except ImportError:
    from bots.tensor.trajectory_schema import TrajectoryStep
try:
    from replay_buffers import (
        AX_DELIVERED_BUFFER,
        TI_DELIVERED_BUFFER,
        TRANSFER_EVENTS_BUFFER,
        load_replay_buffer,
    )
except ImportError:
    from bots.tensor.torch.replay_buffers import (
        AX_DELIVERED_BUFFER,
        TI_DELIVERED_BUFFER,
        TRANSFER_EVENTS_BUFFER,
        load_replay_buffer,
    )

# Obs feature indices — must stay in sync with model.py TILE_FEATURES
_TILE_FEATURES = 21
_OBS_GLOBAL_START = 5 * 5 * _TILE_FEATURES  # 525
_OBS_ROUND_BYTE_OFFSET = _OBS_GLOBAL_START * 4  # byte offset for float32 obs

AX_CONVERSION_RATE = 4  # GameConstants.AXIONITE_CONVERSION_TITANIUM_RATE


def _load_buf(global_states: dict, spec) -> bytes:
    return load_replay_buffer(global_states, spec) or b""


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

    ti_buf = _load_buf(global_states, TI_DELIVERED_BUFFER)
    ax_buf = _load_buf(global_states, AX_DELIVERED_BUFFER)
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
    """Per-round reward from weighted conveyor hops toward core + delivery.

    Conveyor hops use improvement/sqrt(src_dist) weighting with flow-field
    distance. Axionite transfers are weighted 4x (AX_CONVERSION_RATE).
    Core deliveries are added separately from the delivery buffers
    (ti + ax * 4) so transfer hops exclude the final hop into core.

    Requires transfer_events buffer. Raises if missing.
    """
    n_turns = global_states.get("n_turns", 0)
    if n_turns == 0:
        return {}

    buf = _load_buf(global_states, TRANSFER_EVENTS_BUFFER)
    if not buf:
        raise ValueError("transfer_events data missing from global_states")

    ti_buf = _load_buf(global_states, TI_DELIVERED_BUFFER)
    ax_buf = _load_buf(global_states, AX_DELIVERED_BUFFER)
    has_delivery = bool(ti_buf and ax_buf)

    rewards: dict[int, float] = {}
    for rnd in range(n_turns):
        a_val, b_val = struct.unpack_from("<ff", buf, rnd * 8)
        reward = (a_val if team == 0 else b_val) * REWARD_FLOW_SCALE
        if has_delivery:
            ti_a, ti_b = struct.unpack_from("<ii", ti_buf, rnd * 8)
            ax_a, ax_b = struct.unpack_from("<ii", ax_buf, rnd * 8)
            delivered = (ti_a if team == 0 else ti_b) + (ax_a if team == 0 else ax_b) * AX_CONVERSION_RATE
            reward += float(delivered) * REWARD_INCOME_SCALE
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
            if not isinstance(step, TrajectoryStep):
                raise TypeError(f"Expected TrajectoryStep, got {type(step).__name__}")
            traj[i] = step.with_reward(rewards.get(step.round_idx, 0.0))

    return trajectories
