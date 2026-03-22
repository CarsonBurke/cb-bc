"""
flow_reward.py — Post-game flow field conveyor scorer.

Builds a cached Dijkstra flow field from core outward (biased toward ore),
then scores each conveyor by alignment with the field and actual throughput.

Flow field: Dijkstra from core, lower edge cost near ore tiles, producing
a unit vector at each tile pointing toward core. Cached per map geometry.

Throughput: From replay state channels — how often each conveyor tile had
stored > 0 (was actively carrying a resource), recency-weighted.

Score = (throughput × flow_alignment + chain_bonus) × scale

State channels used:
  0-3: environment (empty/wall/ore_ti/ore_ax)
  6: conveyor/foundry/bridge, 7: splitter, 10: armoured conveyor
  11: core, 12: team (+1 friendly), 13-14: direction dx/dy
  17: stored resource

Entry points:
  compute_flow_scores(global_states) -> dict[(x,y), float]
  augment_trajectories_with_flow_scores(trajectories, flow_scores, map_w, map_h)
"""

from __future__ import annotations

import heapq
import math
from pathlib import Path

# BUILD_ACTIONS indices for conveyor-like buildings
_BT_CONVEYOR = 2
_BT_SPLITTER = 7
_BT_ARMOURED_CONVEYOR = 8
_CONV_BT_SET = {_BT_CONVEYOR, _BT_SPLITTER, _BT_ARMOURED_CONVEYOR}

# DIR_MAP_9 index -> (dx, dy), matching shared.py DIR_MAP_9 order
_DIR_DXY_MAP = [
    (-1, -1),  # 0: NORTHWEST
    (0, -1),   # 1: NORTH
    (1, -1),   # 2: NORTHEAST
    (-1, 0),   # 3: WEST
    (0, 0),    # 4: CENTRE
    (1, 0),    # 5: EAST
    (-1, 1),   # 6: SOUTHWEST
    (0, 1),    # 7: SOUTH
    (1, 1),    # 8: SOUTHEAST
]

# Obs feature indices
_OBS_GLOBAL_START = 5 * 5 * 17  # GRID_SIZE^2 * TILE_FEATURES = 425
_OBS_X_NORM = _OBS_GLOBAL_START + 1  # x / map_w
_OBS_Y_NORM = _OBS_GLOBAL_START + 2  # y / map_h

# State channel indices (from Rust replay parser)
_CH_WALL = 1          # wall tile
_CH_ORE_TI = 2        # titanium ore
_CH_ORE_AX = 3        # axionite ore
_CH_CONVEYOR = 6      # conveyor / foundry / bridge
_CH_SPLITTER = 7      # splitter
_CH_ARMOURED = 10     # armoured conveyor
_CH_CORE = 11         # core
_CH_TEAM = 12         # +1 friendly, -1 enemy
_CH_DIR_DX = 13       # direction x component
_CH_DIR_DY = 14       # direction y component
_CH_STORED = 17       # 1.0 if entity is storing a resource
_N_CHANNELS = 19

# Dijkstra flow field tuning
_CARDINALS = ((0, -1), (0, 1), (-1, 0), (1, 0))
ORE_PROXIMITY_RADIUS = 4
ORE_WEIGHT_BASE = 2.0
ORE_WEIGHT_FALLOFF = 0.4

# Scoring tuning
THROUGHPUT_CAP = 100.0      # normalize throughput count to [0, 1]
FLOW_REWARD_SCALE = 3.0     # final score multiplier
CHAIN_BONUS = 0.3           # bonus for downstream tile also having throughput

# Distance field cache: (map_w, map_h, walls, core_xy) -> {(x,y): dist}
_flow_field_cache: dict[tuple, dict[tuple[int, int], float]] = {}


def _build_ore_weights(
    ore_positions: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Ore proximity weight for tiles within radius of any ore."""
    ore_weight: dict[tuple[int, int], float] = {}
    for ox, oy in ore_positions:
        for dy in range(-ORE_PROXIMITY_RADIUS, ORE_PROXIMITY_RADIUS + 1):
            for dx in range(-ORE_PROXIMITY_RADIUS, ORE_PROXIMITY_RADIUS + 1):
                d = abs(dx) + abs(dy)
                if d > ORE_PROXIMITY_RADIUS:
                    continue
                key = (ox + dx, oy + dy)
                w_val = ORE_WEIGHT_BASE + ORE_WEIGHT_FALLOFF * (ORE_PROXIMITY_RADIUS - d)
                if ore_weight.get(key, 0.0) < w_val:
                    ore_weight[key] = w_val
    return ore_weight


def _build_flow_field(
    wall_set: frozenset[tuple[int, int]],
    ore_positions: list[tuple[int, int]],
    core_xy: tuple[int, int],
    map_w: int,
    map_h: int,
) -> dict[tuple[int, int], float]:
    """Dijkstra from core outward, biased toward ore.

    Returns dist: (x,y) -> distance to core (ore-biased).
    Used for gradient-based alignment: a conveyor is good if its output
    tile is closer to core than its input tile, regardless of exact angle.
    """
    cache_key = (map_w, map_h, wall_set, core_xy)
    cached = _flow_field_cache.get(cache_key)
    if cached is not None:
        return cached

    ore_weight = _build_ore_weights(ore_positions)

    INF = float("inf")
    dist: dict[tuple[int, int], float] = {}

    # Multi-source: seed all 3x3 core tiles
    cx, cy = core_xy
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            seed = (cx + dx, cy + dy)
            if 0 <= seed[0] < map_w and 0 <= seed[1] < map_h and seed not in wall_set:
                dist[seed] = 0.0

    heap: list[tuple[float, tuple[int, int]]] = [(0.0, k) for k in dist]
    heapq.heapify(heap)

    while heap:
        d, (px, py) = heapq.heappop(heap)
        if d > dist.get((px, py), INF):
            continue
        for ddx, ddy in _CARDINALS:
            nx, ny = px + ddx, py + ddy
            if nx < 0 or nx >= map_w or ny < 0 or ny >= map_h:
                continue
            if (nx, ny) in wall_set:
                continue
            w_val = ore_weight.get((nx, ny), 1.0)
            edge_cost = 1.0 / max(w_val, 0.1)
            nd = d + edge_cost
            if nd < dist.get((nx, ny), INF):
                dist[(nx, ny)] = nd
                heapq.heappush(heap, (nd, (nx, ny)))

    _flow_field_cache[cache_key] = dist
    return dist


def compute_flow_scores(
    global_states: dict,
) -> dict[tuple[int, int], float]:
    """Compute per-conveyor flow scores from global state channels.

    1. Extract map geometry (walls, ore, core) from first turn state channels
    2. Build/cache Dijkstra flow field from core biased toward ore
    3. Scan all turns for conveyor throughput (stored > 0)
    4. Score each final-turn conveyor: throughput × flow_alignment + chain_bonus

    Args:
        global_states: dict with keys states_buf, map_w, map_h, n_turns
                       (from game_worker's ReplayData)

    Returns:
        {(x,y): score} for each friendly conveyor position at game end
    """
    import torch

    map_w = global_states["map_w"]
    map_h = global_states["map_h"]
    n_turns = global_states["n_turns"]

    if map_w == 0 or map_h == 0 or n_turns == 0:
        return {}

    # Zero-copy view: [n_turns, N_CHANNELS, H, W]
    st = torch.frombuffer(global_states["states_buf"], dtype=torch.float32).clone().reshape(
        n_turns, _N_CHANNELS, map_h, map_w,
    )

    # --- Extract static map geometry from first turn ---
    t0 = st[0]
    wall_coords = (t0[_CH_WALL] > 0.5).nonzero(as_tuple=False)  # [K, 2] (y, x)
    wall_set = frozenset((int(c[1]), int(c[0])) for c in wall_coords)

    ore_coords = ((t0[_CH_ORE_TI] > 0.5) | (t0[_CH_ORE_AX] > 0.5)).nonzero(as_tuple=False)
    ore_positions = [(int(c[1]), int(c[0])) for c in ore_coords]

    # Find friendly core position
    core_mask = (t0[_CH_CORE] > 0.5) & (t0[_CH_TEAM] > 0.5)
    core_coords = core_mask.nonzero(as_tuple=False)
    if core_coords.shape[0] == 0:
        return {}
    # Core center = average of all core tiles (3x3)
    core_y = float(core_coords[:, 0].float().mean())
    core_x = float(core_coords[:, 1].float().mean())
    core_xy = (int(round(core_x)), int(round(core_y)))

    # --- Build/cache distance field ---
    dist_field = _build_flow_field(wall_set, ore_positions, core_xy, map_w, map_h)

    # --- Compute throughput from state channels ---
    is_conv = (st[:, _CH_CONVEYOR] > 0.5) | (st[:, _CH_SPLITTER] > 0.5) | (st[:, _CH_ARMOURED] > 0.5)
    is_friendly = st[:, _CH_TEAM] > 0.5
    is_stored = st[:, _CH_STORED] > 0.5
    active = is_conv & is_friendly & is_stored  # [T, H, W]

    # Recency-weighted throughput
    throughput: dict[tuple[int, int], float] = {}
    if active.any():
        weights = torch.arange(1, n_turns + 1, dtype=torch.float32).div_(n_turns).unsqueeze(-1).unsqueeze(-1)
        tp_map = (active.float() * weights).sum(dim=0)  # [H, W]
        tp_nonzero = tp_map.nonzero(as_tuple=False)
        for i in range(tp_nonzero.shape[0]):
            yi, xi = int(tp_nonzero[i, 0]), int(tp_nonzero[i, 1])
            throughput[(xi, yi)] = float(tp_map[yi, xi])

    # --- Extract final-turn friendly conveyors ---
    final_conv_mask = is_conv[-1] & is_friendly[-1]
    conv_indices = final_conv_mask.nonzero(as_tuple=False)
    conv_data: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(conv_indices.shape[0]):
        yi, xi = int(conv_indices[i, 0]), int(conv_indices[i, 1])
        dx = float(st[-1, _CH_DIR_DX, yi, xi])
        dy = float(st[-1, _CH_DIR_DY, yi, xi])
        mag = math.sqrt(dx * dx + dy * dy)
        if mag > 0:
            conv_data[(xi, yi)] = (dx / mag, dy / mag)

    if not conv_data:
        return {}

    # --- Score each conveyor using distance gradient ---
    # A conveyor is well-aligned if its output tile is closer to core than
    # its current tile. This naturally accepts diagonal, zigzag, or any
    # path that makes progress toward core — no dot-product penalty.
    INF = float("inf")
    scores: dict[tuple[int, int], float] = {}
    for (x, y), (cdx, cdy) in conv_data.items():
        dist_here = dist_field.get((x, y))
        if dist_here is None:
            continue  # tile unreachable from core

        downstream = (x + int(round(cdx)), y + int(round(cdy)))
        dist_down = dist_field.get(downstream, INF)

        # Gradient: positive means moving toward core
        gradient = dist_here - dist_down
        if gradient <= 0:
            # Conveyor points away from core or sideways
            # Still allow small reward if it has throughput (functional chain)
            tile_tp = throughput.get((x, y), 0.0)
            if tile_tp > 0:
                tp_score = min(tile_tp / THROUGHPUT_CAP, 1.0)
                scores[(x, y)] = tp_score * 0.3 * FLOW_REWARD_SCALE
            continue

        # Normalize gradient: how much of the optimal step did this take?
        # Max possible single-step gradient ≈ edge_cost of the optimal neighbor
        # Use gradient directly, capped at 1.0
        align = min(gradient / max(dist_here * 0.1, 0.3), 1.0)

        # Throughput
        tile_tp = throughput.get((x, y), 0.0)
        tp_score = min(tile_tp / THROUGHPUT_CAP, 1.0)

        # Chain bonus: downstream tile also has throughput
        chain = CHAIN_BONUS if throughput.get(downstream, 0.0) > 0 else 0.0

        if tp_score > 0:
            # Active conveyor: throughput + alignment bonus + chain
            raw_score = tp_score * (1.0 + align * 0.5) + chain
        else:
            # Inactive but well-oriented: small bootstrapping reward
            raw_score = align * 0.1

        scores[(x, y)] = raw_score * FLOW_REWARD_SCALE

    return scores


def augment_trajectories_with_flow_scores(
    trajectories: list[list],
    flow_scores: dict[tuple[int, int], float],
    map_w: int,
    map_h: int,
) -> list[list]:
    """Inject flow-based conveyor scores into trajectory steps.

    For each step where a conveyor/splitter/armoured_conveyor was built,
    computes the build position from obs + build_dir and writes the
    flow score into the conv_r slot (index [8] long format, [7] short).

    Args:
        trajectories: list of trajectory lists (each step is a tuple/list)
        flow_scores: {(x,y): score} from compute_flow_scores
        map_w, map_h: map dimensions for position recovery

    Returns:
        Modified trajectories (in-place mutation + return)
    """
    if not flow_scores:
        return trajectories

    # Pass 1: count builds per tile so we can split reward evenly (zero-sum per tile)
    tile_build_count: dict[tuple[int, int], int] = {}
    build_info: list[tuple[int, int, int, tuple[int, int]]] = []  # (traj_i, step_i, conv_r_idx, (bx,by))

    for traj_i, traj in enumerate(trajectories):
        for step_i, step in enumerate(traj):
            if not isinstance(step, (list, tuple)):
                continue
            n = len(step)
            if n < 5:
                continue

            if n > 12:
                bt_idx, bd_idx, conv_r_idx = 2, 3, 8
            else:
                bt_idx, bd_idx, conv_r_idx = 2, 3, 7

            bt = step[bt_idx]
            if bt not in _CONV_BT_SET:
                continue

            obs = step[0]
            ux = int(round(obs[_OBS_X_NORM] * map_w))
            uy = int(round(obs[_OBS_Y_NORM] * map_h))

            bd = step[bd_idx]
            if 0 <= bd < len(_DIR_DXY_MAP):
                ddx, ddy = _DIR_DXY_MAP[bd]
            else:
                ddx, ddy = 0, 0
            bpos = (ux + ddx, uy + ddy)

            if conv_r_idx < n:
                tile_build_count[bpos] = tile_build_count.get(bpos, 0) + 1
                build_info.append((traj_i, step_i, conv_r_idx, bpos))

    # Pass 2: write score / count into each step
    for traj_i, step_i, conv_r_idx, bpos in build_info:
        step = trajectories[traj_i][step_i]
        if isinstance(step, tuple):
            step = list(step)
            trajectories[traj_i][step_i] = step
        score = flow_scores.get(bpos, 0.0)
        step[conv_r_idx] = score / tile_build_count[bpos]

    return trajectories


# ── Per-round resource throughput ─────────────────────────────────────────


def compute_per_round_throughput(
    global_states: dict,
) -> dict[int, float]:
    """Compute per-round count of friendly conveyor tiles actively transporting toward core.

    Only counts conveyors whose output tile is strictly closer to core than
    the conveyor itself (per Dijkstra flow field). This excludes loops and
    dead-end conveyors that hold resources without delivering them.

    Returns {round: throughput_count}.
    """
    import torch

    map_w = global_states["map_w"]
    map_h = global_states["map_h"]
    n_turns = global_states.get("n_turns", 0)

    if map_w == 0 or map_h == 0 or n_turns == 0:
        return {}

    st = torch.frombuffer(global_states["states_buf"], dtype=torch.float32).clone().reshape(
        n_turns, _N_CHANNELS, map_h, map_w,
    )

    # --- Build flow field from first turn (static map geometry) ---
    t0 = st[0]
    wall_coords = (t0[_CH_WALL] > 0.5).nonzero(as_tuple=False)
    wall_set = frozenset((int(c[1]), int(c[0])) for c in wall_coords)

    ore_coords = ((t0[_CH_ORE_TI] > 0.5) | (t0[_CH_ORE_AX] > 0.5)).nonzero(as_tuple=False)
    ore_positions = [(int(c[1]), int(c[0])) for c in ore_coords]

    core_mask = (t0[_CH_CORE] > 0.5) & (t0[_CH_TEAM] > 0.5)
    core_coords = core_mask.nonzero(as_tuple=False)
    if core_coords.shape[0] == 0:
        return {}
    core_xy = (int(round(float(core_coords[:, 1].float().mean()))),
               int(round(float(core_coords[:, 0].float().mean()))))

    dist_field = _build_flow_field(wall_set, ore_positions, core_xy, map_w, map_h)

    # --- Build mask of tiles with positive flow gradient (output closer to core) ---
    # Check each tile: if it's a friendly conveyor, does its direction point toward core?
    # Use last turn to get final conveyor layout, but the mask applies to all turns.
    # Recompute per-turn since conveyors can be built/destroyed.
    INF = float("inf")

    # For each turn, compute the set of valid (flow-aligned) conveyor tiles
    is_conv = (st[:, _CH_CONVEYOR] > 0.5) | (st[:, _CH_SPLITTER] > 0.5) | (st[:, _CH_ARMOURED] > 0.5)
    is_friendly = st[:, _CH_TEAM] > 0.5
    is_stored = st[:, _CH_STORED] > 0.5
    dir_dx = st[:, _CH_DIR_DX]  # [T, H, W]
    dir_dy = st[:, _CH_DIR_DY]  # [T, H, W]

    # Build dist tensor for vectorized gradient check
    dist_tensor = torch.full((map_h, map_w), INF, dtype=torch.float32)
    for (x, y), d in dist_field.items():
        if 0 <= y < map_h and 0 <= x < map_w:
            dist_tensor[y, x] = d

    # For each conveyor, output tile = (x + round(dx), y + round(dy))
    # Check dist[output] < dist[here]
    # Vectorize: shift dist_tensor by each cardinal direction, check gradient
    # Precompute flow-aligned mask once from the dist field.
    # For each tile, check all 4 cardinal directions: if any neighbor is closer
    # to core, a conveyor pointing there would be valid. But we need to check
    # the actual conveyor direction per-turn. Instead, build shifted dist tensors.
    #
    # output_y = y + round(dy), output_x = x + round(dx)
    # We need dist[output] < dist[here] per conveyor per turn.
    # Vectorize: compute output coords, gather dist, compare.

    active_base = is_conv & is_friendly & is_stored  # [T, H, W]
    if not active_base.any():
        return {}

    # Round directions to integer offsets
    ox_map = dir_dx.round().long()  # [T, H, W]
    oy_map = dir_dy.round().long()  # [T, H, W]

    # Create coordinate grids
    yy = torch.arange(map_h).unsqueeze(1).expand(map_h, map_w)  # [H, W]
    xx = torch.arange(map_w).unsqueeze(0).expand(map_h, map_w)  # [H, W]

    # Output coordinates (clamp to valid range)
    out_y = (yy.unsqueeze(0) + oy_map).clamp(0, map_h - 1)  # [T, H, W]
    out_x = (xx.unsqueeze(0) + ox_map).clamp(0, map_w - 1)  # [T, H, W]

    # Gather distances: dist_here and dist_output
    dist_here = dist_tensor.unsqueeze(0).expand(n_turns, -1, -1)  # [T, H, W]
    # Flatten for gather
    out_flat = out_y * map_w + out_x  # [T, H, W]
    dist_flat = dist_tensor.reshape(-1)  # [H*W]
    dist_output = dist_flat[out_flat]  # [T, H, W]

    # Flow-aligned: output strictly closer to core, and tile is reachable
    flow_aligned = (dist_output < dist_here) & (dist_here < INF)

    # Weight each tile by 1/dist_to_core — closer tiles worth more.
    # A resource on a tile 2 steps from core contributes 0.5, one 10 steps away
    # contributes 0.1. This makes short and long chains yield similar total
    # reward for the same delivery rate, while still giving partial credit
    # for incomplete chains (dense learning signal).
    valid = active_base & flow_aligned  # [T, H, W]
    if not valid.any():
        return {}

    inv_dist = torch.where(
        dist_tensor < INF,
        1.0 / dist_tensor.clamp(min=0.5),
        torch.zeros_like(dist_tensor),
    )  # [H, W]
    weighted = valid.float() * inv_dist.unsqueeze(0)  # [T, H, W]
    scores = weighted.sum(dim=(1, 2))  # [T]

    result = {}
    for t in range(n_turns):
        s = float(scores[t].item())
        if s > 0:
            result[t] = s
    return result


def augment_trajectories_with_throughput(
    trajectories: list[list],
    throughput: dict[int, float],
) -> list[list]:
    """Write per-round resource throughput into the n_working slot of trajectory steps.

    Replaces the working-harvester count with actual flow throughput — how many
    friendly conveyor tiles are actively transporting resources each round.
    """
    if not throughput:
        return trajectories

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
            n = len(step)
            slot = 7 if n > 12 else 6  # n_working slot
            step[slot] = throughput.get(rnd, 0)

    return trajectories
