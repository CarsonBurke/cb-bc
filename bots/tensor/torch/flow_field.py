"""
flow_field.py — Post-mortem conveyor alignment scorer for tensor bot.

Two entry points:
  score_conveyor_flow(ct, team, core_pos, ore_positions)
      Vision-limited fallback using the bot's Controller API.
      Used at end of game if no replay is available.

  score_conveyor_flow_from_data(
      wall_set, ore_positions, conv_data, core_xy, map_w, map_h
  )
      Pure-data scorer — no Controller dependency. Called by
      replay_parser.score_conveyor_flow_from_replay() with full
      global map knowledge extracted from the .replay26 file.

Design:
  1. Multi-source Dijkstra from core outward over all non-wall tiles.
     Each tile gets a flow vector pointing back toward core, biased to
     prefer paths near ore (lower edge cost near ore).
  2. Per friendly conveyor: alignment = dot(facing, flow_vec_at_tile).
  3. Connectivity discount: conveyors not reachable from core via the
     conveyor graph are penalised by 1 / (1 + hops_to_nearest_connected).
  4. Final score = log1p(sum of ore_weight * alignment * connectivity)
     clipped per conveyor at MAX_SCORE_PER_CONV.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cambc import Controller, Direction, EntityType, Position

try:
    from cambc import Controller, Direction, EntityType, Environment, Position
    _CAMBC_AVAILABLE = True
except ImportError:
    _CAMBC_AVAILABLE = False

# Tuning knobs (shared by both paths)
ORE_PROXIMITY_RADIUS = 4    # manhattan radius that counts as "near ore"
ORE_WEIGHT_BASE      = 2.0  # weight for a tile directly on ore
ORE_WEIGHT_FALLOFF   = 0.4  # additional weight per tile closer to ore
DISCONNECT_SCALE     = 1.0  # hop penalty steepness
MAX_SCORE_PER_CONV   = 3.0  # clip per-conveyor contribution
WALL_PENALTY         = 1.0  # penalty for conveyor facing into a wall
FEED_BONUS           = 0.5  # bonus per upstream feeder (capped at flow rate limit)
FLOW_RATE_LIMIT      = 2    # max feeders that contribute bonus (game flow rate cap)

# Cardinal (dx, dy) offsets — shared by both paths
_CARDINALS = ((0, -1), (0, 1), (-1, 0), (1, 0))

# replay_parser Direction enum int -> (dx, dy)
_REPLAY_DIR_DXY = {
    0: ( 0,  0),  # DIR_CENTRE
    1: ( 0, -1),  # DIR_NORTH
    2: ( 1, -1),  # DIR_NORTHEAST
    3: ( 1,  0),  # DIR_EAST
    4: ( 1,  1),  # DIR_SOUTHEAST
    5: ( 0,  1),  # DIR_SOUTH
    6: (-1,  1),  # DIR_SOUTHWEST
    7: (-1,  0),  # DIR_WEST
    8: (-1, -1),  # DIR_NORTHWEST
}

# cambc Direction -> (dx, dy) — only populated when cambc is available
if _CAMBC_AVAILABLE:
    _DIR_DXY: dict = {
        Direction.NORTHWEST: (-1, -1),
        Direction.NORTH:     ( 0, -1),
        Direction.NORTHEAST: ( 1, -1),
        Direction.WEST:      (-1,  0),
        Direction.CENTRE:    ( 0,  0),
        Direction.EAST:      ( 1,  0),
        Direction.SOUTHWEST: (-1,  1),
        Direction.SOUTH:     ( 0,  1),
        Direction.SOUTHEAST: ( 1,  1),
    }
    _CONV_TYPES = (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.SPLITTER)
else:
    _DIR_DXY = {}
    _CONV_TYPES = ()


# ---------------------------------------------------------------------------
# Pure-data core algorithms (no Controller dependency)
# ---------------------------------------------------------------------------

def _build_ore_weights(
    ore_positions: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Precompute ore proximity weight for every tile within radius of any ore."""
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


def _build_flow_field_pure(
    wall_set: set[tuple[int, int]],
    ore_positions: list[tuple[int, int]],
    core_xy: tuple[int, int],
    map_w: int,
    map_h: int,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Dijkstra from core outward over all non-wall tiles.

    Returns flow: (x,y) -> (fx, fy) unit vector pointing TOWARD core.
    Edge cost is lower near ore so the field bends toward ore-rich paths.
    """
    ore_weight = _build_ore_weights(ore_positions)

    INF = float("inf")
    dist: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}

    dist[core_xy] = 0.0
    parent[core_xy] = None
    heap: list[tuple[float, tuple[int, int]]] = [(0.0, core_xy)]

    while heap:
        d, (cx, cy) = heapq.heappop(heap)
        if d > dist.get((cx, cy), INF):
            continue
        for ddx, ddy in _CARDINALS:
            nx, ny = cx + ddx, cy + ddy
            if nx < 0 or nx >= map_w or ny < 0 or ny >= map_h:
                continue
            if (nx, ny) in wall_set:
                continue
            w_val = ore_weight.get((nx, ny), 1.0)
            edge_cost = 1.0 / max(w_val, 0.1)
            nd = d + edge_cost
            if nd < dist.get((nx, ny), INF):
                dist[(nx, ny)] = nd
                parent[(nx, ny)] = (cx, cy)
                heapq.heappush(heap, (nd, (nx, ny)))

    # Build flow vectors: each tile points toward its Dijkstra parent (toward core)
    flow: dict[tuple[int, int], tuple[float, float]] = {}
    for (x, y), par in parent.items():
        if par is None:
            flow[(x, y)] = (0.0, 0.0)
            continue
        px, py = par
        fx, fy = px - x, py - y
        mag = math.sqrt(fx * fx + fy * fy)
        flow[(x, y)] = (fx / mag, fy / mag) if mag > 0 else (0.0, 0.0)

    return flow


def _disconnect_distances(
    connected: set[tuple[int, int]],
    adj: dict[tuple[int, int], list[tuple[int, int]]],
    all_conv: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """BFS from the connected chain outward.
    Returns hop distance to nearest connected conveyor for each conveyor."""
    dist: dict[tuple[int, int], int] = {k: 0 for k in connected}
    queue = list(connected)
    head = 0
    while head < len(queue):
        cur = queue[head]; head += 1
        for nb in adj.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    max_dist = len(all_conv) + 1
    for k in all_conv:
        if k not in dist:
            dist[k] = max_dist
    return dist


def _score_from_data(
    flow: dict[tuple[int, int], tuple[float, float]],
    ore_positions: list[tuple[int, int]],
    conv_data: dict[tuple[int, int], tuple[float, float]],
    core_xy: tuple[int, int],
    wall_set: set[tuple[int, int]] | None = None,
) -> float:
    """Score friendly conveyors against the precomputed flow field.

    conv_data: {(x,y): (facing_dx, facing_dy)} — facing vector should be
               unit-length (or zero for unknown direction).
    wall_set:  optional set of wall tile coords. Conveyors facing into a wall
               receive a negative WALL_PENALTY instead of an alignment score.
    Returns log1p-compressed total alignment score.
    """
    if not conv_data:
        return 0.0

    all_conv = set(conv_data.keys())

    # Build adjacency over the conveyor graph
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {k: [] for k in all_conv}
    for (x, y) in all_conv:
        for ddx, ddy in _CARDINALS:
            nk = (x + ddx, y + ddy)
            if nk in all_conv:
                adj[(x, y)].append(nk)

    # BFS connectivity from core's cardinal neighbours
    connected: set[tuple[int, int]] = set()
    queue: list[tuple[int, int]] = []
    cx, cy = core_xy
    for ddx, ddy in _CARDINALS:
        nk = (cx + ddx, cy + ddy)
        if nk in all_conv and nk not in connected:
            connected.add(nk)
            queue.append(nk)
    head = 0
    while head < len(queue):
        cur = queue[head]; head += 1
        for nb in adj.get(cur, []):
            if nb not in connected:
                connected.add(nb)
                queue.append(nb)

    disconnect_dist = _disconnect_distances(connected, adj, all_conv)
    ore_weight_map = _build_ore_weights(ore_positions)

    # Precompute feed counts: for each conveyor, count upstream neighbours
    # pointing toward it (capped at FLOW_RATE_LIMIT).
    # Conveyor at (nx, ny) feeds into (x, y) if its facing vector (ndx, ndy)
    # points from (nx, ny) toward (x, y), i.e. (ndx, ndy) == (x-nx, y-ny).
    feed_counts: dict[tuple[int, int], int] = {k: 0 for k in all_conv}
    for (nx, ny), (ndx, ndy) in conv_data.items():
        # The tile this conveyor is pointing at
        target = (nx + int(round(ndx)), ny + int(round(ndy)))
        if target in feed_counts:
            feed_counts[target] = min(feed_counts[target] + 1, FLOW_RATE_LIMIT)

    total = 0.0
    for key, (cdx, cdy) in conv_data.items():
        x, y = key
        ore_w = ore_weight_map.get(key, 1.0)

        # Penalise conveyors whose output faces directly into a wall
        facing_tile = (x + int(round(cdx)), y + int(round(cdy)))
        if wall_set is not None and facing_tile in wall_set:
            total -= ore_w * WALL_PENALTY
            continue

        fvec = flow.get(key)
        if fvec is None:
            continue  # tile unreachable from core — no reward
        fx, fy = fvec
        alignment = cdx * fx + cdy * fy
        hops = disconnect_dist.get(key, len(all_conv) + 1)
        connectivity = 1.0 / (1.0 + DISCONNECT_SCALE * hops)
        contribution = min(ore_w * alignment * connectivity, MAX_SCORE_PER_CONV)

        # Feed bonus: reward conveyors that receive input from upstream neighbours.
        # Purely additive — no penalty for conveyors with no feeders.
        # Scales with ore proximity so downstream conveyors near ore matter more.
        feeders = feed_counts.get(key, 0)
        contribution += ore_w * FEED_BONUS * feeders

        total += contribution

    return math.log1p(max(total, 0.0))


# ---------------------------------------------------------------------------
# Public entry point: pure-data (replay path)
# ---------------------------------------------------------------------------

def score_conveyor_flow_from_data(
    wall_set: set[tuple[int, int]],
    ore_positions: list[tuple[int, int]],
    conv_data: dict[tuple[int, int], tuple[float, float]],
    core_xy: tuple[int, int],
    map_w: int,
    map_h: int,
) -> float:
    """Score conveyor alignment using globally-accurate data.

    Called by replay_parser.score_conveyor_flow_from_replay() which
    extracts all inputs from the .replay26 file without vision limits.

    Args:
        wall_set     : set of (x,y) wall tile coordinates (complete)
        ore_positions: list of (x,y) ore tile coordinates (complete)
        conv_data    : {(x,y): (facing_dx, facing_dy)} for all friendly
                       conveyors — facing vector must be unit-length or zero
        core_xy      : (x,y) of the friendly core
        map_w, map_h : full map dimensions

    Returns:
        float alignment score (log1p-compressed, unscaled)
    """
    if not ore_positions or not conv_data:
        return 0.0
    flow = _build_flow_field_pure(wall_set, ore_positions, core_xy, map_w, map_h)
    return _score_from_data(flow, ore_positions, conv_data, core_xy, wall_set=wall_set)


# ---------------------------------------------------------------------------
# Public entry point: vision-limited fallback (Controller path)
# ---------------------------------------------------------------------------

def score_conveyor_flow(
    ct,
    team,
    core_pos,
    ore_positions,
) -> float:
    """Vision-limited fallback scorer using the bot's Controller API.

    Prefer score_conveyor_flow_from_replay() in replay_parser.py when a
    replay file is available — it has complete global map knowledge and
    correct wall routing.
    """
    if not _CAMBC_AVAILABLE or not ore_positions:
        return 0.0

    w, h = ct.get_map_width(), ct.get_map_height()
    ore_xy: list[tuple[int, int]] = [(p.x, p.y) for p in ore_positions]

    # Collect walls from visible tiles — non-visible assumed passable
    wall_set: set[tuple[int, int]] = set()
    for y in range(h):
        for x in range(w):
            pos = Position(x, y)
            try:
                if ct.is_in_vision(pos) and ct.get_tile_env(pos) == Environment.WALL:
                    wall_set.add((x, y))
            except Exception:
                pass

    flow = _build_flow_field_pure(wall_set, ore_xy, (core_pos.x, core_pos.y), w, h)

    # Collect friendly conveyors from vision only
    conv_data: dict[tuple[int, int], tuple[float, float]] = {}
    for y in range(h):
        for x in range(w):
            pos = Position(x, y)
            try:
                bid = ct.get_tile_building_id(pos)
                if bid is None:
                    continue
                if ct.get_entity_type(bid) not in _CONV_TYPES:
                    continue
                if ct.get_team(bid) != team:
                    continue
                cdir = ct.get_direction(bid)
                ddx, ddy = _DIR_DXY.get(cdir, (0, 0))
                mag = math.sqrt(ddx * ddx + ddy * ddy)
                conv_data[(x, y)] = (ddx / mag, ddy / mag) if mag > 0 else (0.0, 0.0)
            except Exception:
                continue

    return _score_from_data(flow, ore_xy, conv_data, (core_pos.x, core_pos.y))


# Expose for import by replay_parser
REPLAY_DIR_DXY = _REPLAY_DIR_DXY