"""
flow_field.py — Post-mortem conveyor alignment scorer for tensor bot.

Call `score_conveyor_flow(ct, team)` at end of game (round >= 1999) from
_flush_log or wherever you assemble the terminal reward. Returns a float
bonus to add to the final trajectory step.

Design:
  1. Multi-source Dijkstra from core outward over all passable tiles.
     Each tile gets (distance_to_core, flow_vector_pointing_back_to_core).
  2. Flow vectors are ore-weighted: tiles near ore get a stronger pull,
     boosting alignment rewards for conveyors placed near ore.
  3. Per friendly conveyor: alignment = dot(facing, flow_vec_at_tile).
  4. Connectivity discount: conveyors not in the chain reachable from core
     are discounted by 1/(1 + bfs_hops_to_nearest_connected_conveyor).
  5. Final score = sum of (ore_weight * alignment * connectivity_discount)
     over all friendly conveyors, normalised by conveyor count.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cambc import Controller, Direction, EntityType, Position

# Re-import at runtime (cambc is available in the bot environment)
from cambc import Controller, Direction, EntityType, Environment, Position

# Mirror of DIR_DXY from main.py
_DIR_DXY: dict[Direction, tuple[int, int]] = {
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
_CARDINAL = (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)

# Tuning knobs
ORE_PROXIMITY_RADIUS = 4      # tiles within this manhattan distance count as "near ore"
ORE_WEIGHT_BASE      = 2.0    # multiplier for a tile directly on ore
ORE_WEIGHT_FALLOFF   = 0.4    # weight added per tile closer to ore (linear)
DISCONNECT_SCALE     = 1.0    # disconnect penalty steepness (higher = harsher)
MAX_SCORE_PER_CONV   = 3.0    # clip individual conveyor contribution


# ---------------------------------------------------------------------------
# 1. Dijkstra flow field from core
# ---------------------------------------------------------------------------

def _build_flow_field(
    ct: Controller,
    core_pos: Position,
    ore_positions: list[Position],
) -> tuple[dict[tuple[int,int], float], dict[tuple[int,int], tuple[float,float]]]:
    """
    Multi-source Dijkstra from core outward.

    Returns:
        dist   : (x,y) -> float distance from core (0 at core)
        flow   : (x,y) -> (fx, fy) unit vector pointing TOWARD core
                 (i.e. the direction a conveyor here should face)
    """
    w, h = ct.get_map_width(), ct.get_map_height()

    # Precompute ore weight map (manhattan proximity, capped at radius)
    ore_weight: dict[tuple[int,int], float] = {}
    for ore in ore_positions:
        for dy in range(-ORE_PROXIMITY_RADIUS, ORE_PROXIMITY_RADIUS + 1):
            for dx in range(-ORE_PROXIMITY_RADIUS, ORE_PROXIMITY_RADIUS + 1):
                d = abs(dx) + abs(dy)
                if d > ORE_PROXIMITY_RADIUS:
                    continue
                key = (ore.x + dx, ore.y + dy)
                w_val = ORE_WEIGHT_BASE + ORE_WEIGHT_FALLOFF * (ORE_PROXIMITY_RADIUS - d)
                if ore_weight.get(key, 0.0) < w_val:
                    ore_weight[key] = w_val

    # Dijkstra: cost = sum of (1 / ore_weight) so high-ore paths are preferred
    INF = float("inf")
    dist: dict[tuple[int,int], float] = {}
    parent: dict[tuple[int,int], tuple[int,int] | None] = {}

    start = (core_pos.x, core_pos.y)
    heap: list[tuple[float, tuple[int,int]]] = [(0.0, start)]
    dist[start] = 0.0
    parent[start] = None

    while heap:
        d, (cx, cy) = heapq.heappop(heap)
        if d > dist.get((cx, cy), INF):
            continue
        for direction in _CARDINAL:
            ddx, ddy = _DIR_DXY[direction]
            nx, ny = cx + ddx, cy + ddy
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            npos = Position(nx, ny)
            try:
                if not ct.is_tile_passable(npos):
                    # walls block the field — skip
                    # (conveyors on impassable tiles shouldn't exist anyway)
                    env = ct.get_tile_env(npos)
                    if env == Environment.WALL:
                        continue
            except Exception:
                continue
            # Edge cost: lower near ore (encourage routing through ore-rich areas)
            w_val = ore_weight.get((nx, ny), 1.0)
            edge_cost = 1.0 / max(w_val, 0.1)
            nd = d + edge_cost
            if nd < dist.get((nx, ny), INF):
                dist[(nx, ny)] = nd
                parent[(nx, ny)] = (cx, cy)
                heapq.heappush(heap, (nd, (nx, ny)))

    # Build flow vectors: each tile points toward its parent (toward core)
    flow: dict[tuple[int,int], tuple[float,float]] = {}
    for (x, y), par in parent.items():
        if par is None:
            flow[(x, y)] = (0.0, 0.0)  # core itself
            continue
        px, py = par
        fx, fy = px - x, py - y          # raw vector toward core
        mag = math.sqrt(fx * fx + fy * fy)
        if mag > 0:
            flow[(x, y)] = (fx / mag, fy / mag)
        else:
            flow[(x, y)] = (0.0, 0.0)

    return dist, flow


# ---------------------------------------------------------------------------
# 2. Connectivity BFS over friendly conveyor graph
# ---------------------------------------------------------------------------

def _connected_conveyors(
    ct: Controller,
    team,
    core_pos: Position,
) -> tuple[set[tuple[int,int]], dict[tuple[int,int], list[tuple[int,int]]]]:
    """
    BFS from core through adjacent friendly conveyors.

    Returns:
        connected : set of (x,y) keys reachable from core via conveyor chain
        adj       : adjacency list for all friendly conveyors (for disconnect dist)
    """
    w, h = ct.get_map_width(), ct.get_map_height()

    # Collect all friendly conveyors by scanning the full map
    # (end-of-game, so we can afford a full sweep)
    all_conv: set[tuple[int,int]] = set()
    adj: dict[tuple[int,int], list[tuple[int,int]]] = {}

    for y in range(h):
        for x in range(w):
            pos = Position(x, y)
            try:
                bid = ct.get_tile_building_id(pos)
                if bid is None:
                    continue
                btype = ct.get_entity_type(bid)
                if btype not in _CONV_TYPES:
                    continue
                if ct.get_team(bid) != team:
                    continue
                key = (x, y)
                all_conv.add(key)
                adj[key] = []
            except Exception:
                continue

    # Build adjacency (cardinal neighbours that are also friendly conveyors)
    for (x, y) in all_conv:
        for direction in _CARDINAL:
            ddx, ddy = _DIR_DXY[direction]
            nk = (x + ddx, y + ddy)
            if nk in all_conv:
                adj[(x, y)].append(nk)

    # BFS from core's cardinal neighbours
    connected: set[tuple[int,int]] = set()
    queue: list[tuple[int,int]] = []
    for direction in _CARDINAL:
        ddx, ddy = _DIR_DXY[direction]
        nk = (core_pos.x + ddx, core_pos.y + ddy)
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

    return connected, adj


def _disconnect_distances(
    connected: set[tuple[int,int]],
    adj: dict[tuple[int,int], list[tuple[int,int]]],
    all_conv: set[tuple[int,int]],
) -> dict[tuple[int,int], int]:
    """
    BFS from all connected conveyors outward to find minimum hop distance
    to the connected chain for every disconnected conveyor.
    Returns dist map (0 = in chain, >=1 = hops away).
    """
    dist: dict[tuple[int,int], int] = {k: 0 for k in connected}
    queue = list(connected)
    head = 0
    while head < len(queue):
        cur = queue[head]; head += 1
        for nb in adj.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    # Any conveyor not reached gets max distance
    for k in all_conv:
        if k not in dist:
            dist[k] = len(all_conv) + 1
    return dist


# ---------------------------------------------------------------------------
# 3. Main scorer
# ---------------------------------------------------------------------------

def score_conveyor_flow(
    ct: Controller,
    team,
    core_pos: Position,
    ore_positions: list[Position],
) -> float:
    """
    Compute post-mortem flow field alignment score for all friendly conveyors.

    Args:
        ct           : Controller (end-of-game state)
        team         : the bot's team identifier
        core_pos     : Position of the friendly core
        ore_positions: list of known ore tile positions (from Player._all_ore)

    Returns:
        float bonus score (unnormalised sum, ready to scale before injecting
        into the terminal trajectory step)
    """
    if not ore_positions:
        return 0.0

    # 1. Build flow field
    _, flow = _build_flow_field(ct, core_pos, ore_positions)

    # 2. Build conveyor connectivity
    w, h = ct.get_map_width(), ct.get_map_height()
    all_conv: set[tuple[int,int]] = set()
    conv_dirs: dict[tuple[int,int], tuple[float,float]] = {}
    adj: dict[tuple[int,int], list[tuple[int,int]]] = {}

    for y in range(h):
        for x in range(w):
            pos = Position(x, y)
            try:
                bid = ct.get_tile_building_id(pos)
                if bid is None:
                    continue
                btype = ct.get_entity_type(bid)
                if btype not in _CONV_TYPES:
                    continue
                if ct.get_team(bid) != team:
                    continue
                key = (x, y)
                all_conv.add(key)
                adj[key] = []
                try:
                    cdir = ct.get_direction(bid)
                    ddx, ddy = _DIR_DXY.get(cdir, (0, 0))
                    mag = math.sqrt(ddx*ddx + ddy*ddy)
                    conv_dirs[key] = (ddx/mag, ddy/mag) if mag > 0 else (0.0, 0.0)
                except Exception:
                    conv_dirs[key] = (0.0, 0.0)
            except Exception:
                continue

    if not all_conv:
        return 0.0

    # Build adjacency
    for (x, y) in all_conv:
        for direction in _CARDINAL:
            ddx, ddy = _DIR_DXY[direction]
            nk = (x + ddx, y + ddy)
            if nk in all_conv:
                adj[(x, y)].append(nk)

    # BFS connectivity from core
    connected: set[tuple[int,int]] = set()
    queue: list[tuple[int,int]] = []
    for direction in _CARDINAL:
        ddx, ddy = _DIR_DXY[direction]
        nk = (core_pos.x + ddx, core_pos.y + ddy)
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

    # Precompute ore weight per tile (reuse same logic as flow field)
    ore_weight_map: dict[tuple[int,int], float] = {}
    for ore in ore_positions:
        for dy in range(-ORE_PROXIMITY_RADIUS, ORE_PROXIMITY_RADIUS + 1):
            for dx in range(-ORE_PROXIMITY_RADIUS, ORE_PROXIMITY_RADIUS + 1):
                d = abs(dx) + abs(dy)
                if d > ORE_PROXIMITY_RADIUS:
                    continue
                key = (ore.x + dx, ore.y + dy)
                w_val = ORE_WEIGHT_BASE + ORE_WEIGHT_FALLOFF * (ORE_PROXIMITY_RADIUS - d)
                if ore_weight_map.get(key, 0.0) < w_val:
                    ore_weight_map[key] = w_val

    # 3. Score each conveyor
    total = 0.0
    for key, (cdx, cdy) in conv_dirs.items():
        fvec = flow.get(key)
        if fvec is None:
            continue  # tile unreachable from core — no reward

        fx, fy = fvec
        # Dot product: +1 = perfectly aligned toward core, -1 = facing away
        alignment = cdx * fx + cdy * fy

        # Ore proximity weight
        ore_w = ore_weight_map.get(key, 1.0)

        # Connectivity discount
        hops = disconnect_dist.get(key, len(all_conv) + 1)
        connectivity = 1.0 / (1.0 + DISCONNECT_SCALE * hops)

        contribution = min(ore_w * alignment * connectivity, MAX_SCORE_PER_CONV)
        total += contribution

    # Incentivise more conveyors
    return math.log1p(total)
