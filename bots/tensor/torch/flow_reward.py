"""
flow_reward.py — Retroactive conveyor build credit assignment.

Computes per-round shared team reward by attributing each conveyor's lifetime
productive transfers back to the round it was built.

Data comes from the replay parser's per-tile buffers:
  - conveyor_births: flat i32 triples [turn, x, y, ...] for friendly placements
  - tile_transfers:  flat i32 triples [turn, x, y, ...] for hop-filtered transfers

A "productive transfer" is one where the destination tile is strictly closer
(Manhattan distance) to the friendly core than the source tile. This filtering
is done in the Rust replay parser.

Entry point:
  compute_retroactive_build_rewards(global_states) -> dict[int, float]
  augment_trajectories_with_build_rewards(trajectories, rewards) -> trajectories
"""

from __future__ import annotations

import struct
from collections import defaultdict

# Obs feature indices
_OBS_GLOBAL_START = 5 * 5 * 19  # GRID_SIZE^2 * TILE_FEATURES = 475


def compute_retroactive_build_rewards(
    global_states: dict,
) -> dict[int, float]:
    """Compute per-round shared team reward from retroactive conveyor credit.

    For each conveyor placement, finds how many productive transfers occurred
    at that tile during the conveyor's lifetime, and attributes that count
    to the round the conveyor was built.

    Args:
        global_states: dict with conveyor_births_buf, tile_transfers_buf, n_turns

    Returns:
        {round: total_lifetime_transfers} for rounds where conveyors were built
    """
    cb_buf = global_states.get("conveyor_births_buf", b"")
    tt_buf = global_states.get("tile_transfers_buf", b"")
    n_turns = global_states.get("n_turns", 0)

    if not cb_buf or n_turns == 0:
        return {}

    # Parse conveyor births: list of (turn, x, y)
    n_births = len(cb_buf) // 12  # 3 × i32
    births: list[tuple[int, int, int]] = []
    for i in range(n_births):
        t, x, y = struct.unpack_from("<iii", cb_buf, i * 12)
        births.append((t, x, y))

    if not births:
        return {}

    # Parse tile transfers into per-tile sorted turn lists
    # tile_at[(x, y)] = sorted list of transfer turns
    tile_at: dict[tuple[int, int], list[int]] = defaultdict(list)
    n_transfers = len(tt_buf) // 12
    for i in range(n_transfers):
        t, x, y = struct.unpack_from("<iii", tt_buf, i * 12)
        tile_at[(x, y)].append(t)

    # Sort each tile's transfer turns for bisect
    for v in tile_at.values():
        v.sort()

    # Build per-tile birth intervals: for each (x, y), sorted list of birth turns.
    # A conveyor's lifetime at (x, y) is [birth_i, birth_{i+1}) or [birth_i, n_turns).
    tile_births: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t, x, y in births:
        tile_births[(x, y)].append(t)
    for v in tile_births.values():
        v.sort()

    # Attribute transfers to birth rounds
    from bisect import bisect_left, bisect_right

    round_rewards: dict[int, float] = {}

    for (x, y), b_turns in tile_births.items():
        transfers = tile_at.get((x, y))
        if not transfers:
            continue

        for idx, birth_t in enumerate(b_turns):
            # Lifetime: [birth_t, next_birth_t) or [birth_t, n_turns)
            death_t = b_turns[idx + 1] if idx + 1 < len(b_turns) else n_turns

            # Count transfers in [birth_t, death_t) using bisect
            lo = bisect_left(transfers, birth_t)
            hi = bisect_left(transfers, death_t)
            count = hi - lo

            if count > 0:
                round_rewards[birth_t] = round_rewards.get(birth_t, 0.0) + count

    return round_rewards


def augment_trajectories_with_build_rewards(
    trajectories: list[list],
    rewards: dict[int, float],
) -> list[list]:
    """Write retroactive build rewards into the n_working slot.

    Each trajectory step's round is extracted from obs, and the shared team
    reward for that round (if any) is written into the n_working slot.

    Args:
        trajectories: list of trajectory lists (each step is a tuple/list)
        rewards: {round: reward} from compute_retroactive_build_rewards

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
            rnd = int(round(obs[_OBS_GLOBAL_START] * 2000.0))
            rnd = max(0, min(rnd, 1999))
            slot = 7 if len(step) > 12 else 6  # n_working slot
            step[slot] = rewards.get(rnd, 0.0)

    return trajectories
