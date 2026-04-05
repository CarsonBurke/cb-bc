# shared.py — Constants, obs encoding, masks, rewards, and utilities shared
# between tensor_proxy (training) and tensor_inference (deploy).
# No tensor_lib dependency — stdlib only.

import math
import os
import random
import sys

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position, ResourceType

# Keep actor architecture constants sourced from the deploy model constants.
_INFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor_inference")
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)
from model_constants import (
    ACTOR_CLS_TOKENS,
    ACTOR_CONTEXT_CHUNKS,
    ACTOR_LATENT_DIM,
    COMPRESS_DIM,
    GLOBAL_FEATURES,
    GRID_HALF,
    GRID_SIZE,
    HIDDEN_SIZES,
    H_DIM,
    INPUT_SIZE,
    RMSNORM_LAYERS,
    TILE_FEATURES,
)

# -- Layout constants --

DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]

BUILD_ACTIONS = [
    "none",
    "harvester",
    "conveyor",
    "road",
    "gunner",
    "sentinel",
    "breach",
    "splitter",
    "armoured_conveyor",
    "barrier",
    "foundry",
    "launcher",
    "bridge",
]

DIR_MAP_9 = [
    Direction.NORTHWEST,
    Direction.NORTH,
    Direction.NORTHEAST,
    Direction.WEST,
    Direction.CENTRE,
    Direction.EAST,
    Direction.SOUTHWEST,
    Direction.SOUTH,
    Direction.SOUTHEAST,
]
CONV_DIR_4 = [Direction.NORTH, Direction.WEST, Direction.EAST, Direction.SOUTH]
_CONV_DIR_MASK = [True, True, True, True]  # all 4 cardinal dirs always valid
_HARVESTER_BUILD_TYPE = 1

ENV_IDX = {
    Environment.EMPTY: 0,
    Environment.WALL: 1,
    Environment.ORE_TITANIUM: 2,
    Environment.ORE_AXIONITE: 3,
}

BLD_IDX = {
    EntityType.HARVESTER: 1,
    EntityType.CONVEYOR: 2,
    EntityType.ARMOURED_CONVEYOR: 2,
    EntityType.SPLITTER: 3,
    EntityType.ROAD: 4,
    EntityType.BARRIER: 5,
    EntityType.BRIDGE: 6,
    EntityType.FOUNDRY: 7,
    EntityType.GUNNER: 8,
    EntityType.SENTINEL: 8,
    EntityType.BREACH: 8,
    EntityType.LAUNCHER: 8,
    EntityType.CORE: 9,
}
N_BLD_TYPES = 10

DIR_DXY = {
    Direction.NORTHWEST: (-1, -1),
    Direction.NORTH: (0, -1),
    Direction.NORTHEAST: (1, -1),
    Direction.WEST: (-1, 0),
    Direction.CENTRE: (0, 0),
    Direction.EAST: (1, 0),
    Direction.SOUTHWEST: (-1, 1),
    Direction.SOUTH: (0, 1),
    Direction.SOUTHEAST: (1, 1),
}

# Base build costs (titanium, axionite) indexed by BUILD_ACTIONS, derived from GameConstants
_GC = GameConstants
BUILD_COSTS = (
    (0, 0),                           # none
    _GC.HARVESTER_BASE_COST,          # harvester
    _GC.CONVEYOR_BASE_COST,           # conveyor
    _GC.ROAD_BASE_COST,               # road
    _GC.GUNNER_BASE_COST,             # gunner
    _GC.SENTINEL_BASE_COST,           # sentinel
    _GC.BREACH_BASE_COST,             # breach
    _GC.SPLITTER_BASE_COST,           # splitter
    _GC.ARMOURED_CONVEYOR_BASE_COST,  # armoured_conveyor
    _GC.BARRIER_BASE_COST,            # barrier
    _GC.FOUNDRY_BASE_COST,            # foundry
    _GC.LAUNCHER_BASE_COST,           # launcher
    _GC.BRIDGE_BASE_COST,             # bridge
)

# Bridge target offsets: all (dx, dy) with 0 < dx²+dy² ≤ 9, row-major order.
# Matches the sequential index from replay_parser's bridge_target_to_idx.
BRIDGE_OFFSETS = tuple(
    (dx, dy)
    for dy in range(-3, 4)
    for dx in range(-3, 4)
    if 0 < dx * dx + dy * dy <= 9
)

_BUILD_DIR_TILE_INDICES = tuple(
    (GRID_HALF + int(DIR_DXY[d][1])) * GRID_SIZE + (GRID_HALF + int(DIR_DXY[d][0]))
    for d in DIR_MAP_9
)

_GLOBAL_OBS_START = GRID_SIZE * GRID_SIZE * TILE_FEATURES

_ZERO_TILE = (0.0,) * TILE_FEATURES
_NO_BUILD_TILE = (1.0,) + (0.0,) * (N_BLD_TYPES - 1) + (0.0, 0.0, 0.0, 0.0, 0.0)
_BLD_ONE_HOTS = tuple(
    tuple(1.0 if i == idx else 0.0 for i in range(N_BLD_TYPES))
    for idx in range(N_BLD_TYPES)
)


# ---------------------------------------------------------------------------
# GameState — shared mutable state across all units (replaces Player class vars)
# ---------------------------------------------------------------------------

class GameState:
    """Module-level shared state for a single game. Call reset() between games."""
    conv_count: int = 0
    all_ore: list[Position] | None = None
    all_ore_keys: set[tuple[int, int]] = set()
    harv_proximity: dict[tuple[int, int], float] = {}
    harv_placed_ores: set[tuple[int, int]] = set()
    conv_proximity: dict[tuple[int, int], float] = {}  # ore_key → conv proximity reward given

    working_harv_positions: set[tuple[int, int]] = set()
    working_harv_round: int = -1
    working_harv_count: int = 0
    enemy_core_hp: float = 500.0
    own_core_hp: float = 500.0
    enemy_core_dmg: float = 0.0
    enemy_core_dmg_round: int = -1
    own_core_dmg: float = 0.0
    own_core_dmg_round: int = -1

    @classmethod
    def reset(cls) -> None:
        cls.conv_count = 0
        cls.all_ore = None
        cls.all_ore_keys = set()
        cls.harv_proximity = {}
        cls.harv_placed_ores = set()
        cls.conv_proximity = {}

        cls.working_harv_positions = set()
        cls.working_harv_round = -1
        cls.working_harv_count = 0
        cls.enemy_core_hp = 500.0
        cls.own_core_hp = 500.0
        cls.enemy_core_dmg = 0.0
        cls.enemy_core_dmg_round = -1
        cls.own_core_dmg = 0.0
        cls.own_core_dmg_round = -1


# ---------------------------------------------------------------------------
# State scanning functions (operate on GameState class attrs)
# ---------------------------------------------------------------------------

def discover_ore(ct: Controller) -> None:
    if GameState.all_ore is None:
        GameState.all_ore = []
        GameState.all_ore_keys = set()
    for tile_pos in ct.get_nearby_tiles():
        key = (tile_pos.x, tile_pos.y)
        if key in GameState.all_ore_keys:
            continue
        try:
            env = ct.get_tile_env(tile_pos)
            if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                GameState.all_ore.append(tile_pos)
                GameState.all_ore_keys.add(key)
        except Exception:
            pass


def scan_enemy_core_dmg(ct: Controller) -> None:
    rnd = ct.get_current_round()
    if GameState.enemy_core_dmg_round != rnd:
        GameState.enemy_core_dmg = 0.0
        GameState.enemy_core_dmg_round = rnd
    team = ct.get_team()
    for eid in ct.get_nearby_entities():
        try:
            if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) != team:
                hp = ct.get_hp(eid)
                dmg = max(0.0, GameState.enemy_core_hp - hp)
                if dmg > GameState.enemy_core_dmg:
                    GameState.enemy_core_dmg = dmg
                GameState.enemy_core_hp = hp
                break
        except Exception:
            pass


def scan_own_core_dmg(ct: Controller) -> None:
    rnd = ct.get_current_round()
    if GameState.own_core_dmg_round != rnd:
        GameState.own_core_dmg = 0.0
        GameState.own_core_dmg_round = rnd
    team = ct.get_team()
    for eid in ct.get_nearby_entities():
        try:
            if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) == team:
                hp = ct.get_hp(eid)
                dmg = max(0.0, GameState.own_core_hp - hp)
                if dmg > GameState.own_core_dmg:
                    GameState.own_core_dmg = dmg
                GameState.own_core_hp = hp
                break
        except Exception:
            pass


_CONV_TYPES = (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.SPLITTER)


def scan_working_harvesters(ct: Controller) -> None:
    rnd = ct.get_current_round()
    if GameState.working_harv_round != rnd:
        GameState.working_harv_count = len(GameState.working_harv_positions)
        GameState.working_harv_positions = set()
        GameState.working_harv_round = rnd
    team = ct.get_team()
    for tile_pos in ct.get_nearby_tiles():
        try:
            bid = ct.get_tile_building_id(tile_pos)
            if bid is None or ct.get_entity_type(bid) != EntityType.HARVESTER \
               or ct.get_team(bid) != team:
                continue
            hkey = (tile_pos.x, tile_pos.y)
            if hkey in GameState.working_harv_positions:
                continue
            for d in (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
                adj = tile_pos.add(d)
                try:
                    cbid = ct.get_tile_building_id(adj)
                    if cbid is None:
                        continue
                    ctype = ct.get_entity_type(cbid)
                    if ctype not in _CONV_TYPES or ct.get_team(cbid) != team:
                        continue
                    cdir = ct.get_direction(cbid)
                    if cdir == d.opposite():
                        continue
                except Exception:
                    continue
                GameState.working_harv_positions.add(hkey)
                break
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Obs encoding
# ---------------------------------------------------------------------------

def encode_state_into(ct: Controller, features, core_center, enemy_core,
                      my_team, pos_encoder, prev_collected, spawn_round,
                      n_builders_out: list) -> None:
    """Append encoded state into a list-like float buffer.

    n_builders_out is a single-element list that receives the builder count.
    """
    pos = ct.get_position()
    append = features.append
    extend = features.extend
    get_tile_env = ct.get_tile_env
    get_tile_building_id = ct.get_tile_building_id
    get_tile_builder_bot_id = ct.get_tile_builder_bot_id
    get_entity_type = ct.get_entity_type
    get_team = ct.get_team
    get_direction = ct.get_direction

    w, h = ct.get_map_width(), ct.get_map_height()
    for dy in range(-GRID_HALF, GRID_HALF + 1):
        for dx in range(-GRID_HALF, GRID_HALF + 1):
            tx, ty = pos.x + dx, pos.y + dy
            if tx < 0 or tx >= w or ty < 0 or ty >= h or not ct.is_in_vision(Position(tx, ty)):
                extend(_ZERO_TILE)
                continue
            tp = Position(tx, ty)

            env = get_tile_env(tp)
            append(1.0 if env == Environment.WALL else 0.0)
            append(1.0 if env == Environment.ORE_TITANIUM else 0.0)
            append(1.0 if env == Environment.ORE_AXIONITE else 0.0)
            append(0.0 if ct.is_tile_passable(tp) else 1.0)

            bid = get_tile_building_id(tp)
            if bid is not None:
                btype = get_entity_type(bid)
                bidx = BLD_IDX.get(btype, 3)
                extend(_BLD_ONE_HOTS[bidx])
                append(1.0 if get_team(bid) == my_team else -1.0)
                try:
                    bdir = get_direction(bid)
                    bdx, bdy = DIR_DXY.get(bdir, (0, 0))
                    append(float(bdx))
                    append(float(bdy))
                except Exception:
                    extend((0.0, 0.0))
                append(0.0)  # bld_cooldown: populated by replay parser during training
                try:
                    sr = ct.get_stored_resource(bid)
                    append(0.0 if sr is None else 1.0 if sr == ResourceType.TITANIUM else -1.0)
                except Exception:
                    append(0.0)
            else:
                extend(_NO_BUILD_TILE)

            uid = get_tile_builder_bot_id(tp)
            if uid is not None:
                append(1.0)
                append(1.0 if get_team(uid) == my_team else -1.0)
            else:
                extend((0.0, 0.0))

    rnd = ct.get_current_round() / 2000.0
    x_norm = pos.x / max(w, 1)
    y_norm = pos.y / max(h, 1)
    ti, ax = ct.get_global_resources()
    scale = ct.get_scale_percent() / 500.0

    n_builders = 0
    own_hp, enemy_hp = GameState.own_core_hp, GameState.enemy_core_hp
    _enemy_core = enemy_core
    _core_center = core_center
    for eid in ct.get_nearby_entities():
        try:
            etype = get_entity_type(eid)
            team = get_team(eid)
            if etype == EntityType.BUILDER_BOT and team == my_team:
                n_builders += 1
            elif etype == EntityType.CORE:
                if team == my_team:
                    if _core_center is None:
                        _core_center = ct.get_position(eid)
                    own_hp = ct.get_hp(eid)
                    GameState.own_core_hp = own_hp
                else:
                    if _enemy_core is None:
                        _enemy_core = ct.get_position(eid)
                    hp = ct.get_hp(eid)
                    GameState.enemy_core_hp = hp
                    enemy_hp = hp
        except Exception:
            pass
    n_builders_out[0] = n_builders

    core_dx, core_dy, core_known = 0.0, 0.0, 0.0
    if _core_center is not None:
        core_dx = float(_core_center.x - pos.x)
        core_dy = float(_core_center.y - pos.y)
        core_known = 1.0

    ecore_dx, ecore_dy, ecore_known = 0.0, 0.0, 0.0
    if _enemy_core is not None:
        ecore_dx = float(_enemy_core.x - pos.x)
        ecore_dy = float(_enemy_core.y - pos.y)
        ecore_known = 1.0

    ti_income, ax_income = 0.0, 0.0
    if prev_collected is not None:
        prev_ti, prev_ax = prev_collected
        ti_income = (ti - prev_ti) / 500.0
        ax_income = (ax - prev_ax) / 500.0

    hp_advantage = (own_hp - enemy_hp) / 500.0

    scan_working_harvesters(ct)

    extend([
        rnd, x_norm, y_norm, ti / 500.0, ax / 500.0, scale,
        core_dx, core_dy, ecore_dx, ecore_dy,
        n_builders / 10.0,
        ti_income, ax_income, hp_advantage,
        spawn_round / 2000.0 if spawn_round >= 0 else 0.0,
        core_known, ecore_known,
        w / 50.0, h / 50.0,
        GameState.working_harv_count / 20.0,
    ])

    return _core_center, _enemy_core, (ti, ax)


def encode_state(ct: Controller, core_center, enemy_core, my_team,
                 pos_encoder, prev_collected, spawn_round,
                 n_builders_out: list) -> list[float]:
    features: list[float] = []
    updated = encode_state_into(ct, features, core_center, enemy_core,
                                my_team, pos_encoder, prev_collected,
                                spawn_round, n_builders_out)
    return features, updated


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def compute_masks_from_obs(obs: list[float]) -> tuple[list[bool], list[bool]]:
    """Returns (bt_mask, bd_mask)."""
    g = _GLOBAL_OBS_START
    ti = obs[g + 3] * 500.0
    ax = obs[g + 4] * 500.0
    scale = obs[g + 5] * 5.0

    bt_mask = [True]
    any_affordable = False
    for i in range(1, len(BUILD_COSTS)):
        base_ti, base_ax = BUILD_COSTS[i]
        ok = ti >= int(scale * base_ti) and ax >= int(scale * base_ax)
        bt_mask.append(ok)
        any_affordable = any_affordable or ok

    bd_mask = [False] * 9
    any_dir = False
    for d_idx in range(9):
        ts = _BUILD_DIR_TILE_INDICES[d_idx] * TILE_FEATURES
        not_wall = obs[ts] == 0.0
        no_unit = obs[ts + 19] == 0.0
        empty = obs[ts + 4] == 1.0
        replaceable = obs[ts + 4] == 0.0 and obs[ts + 14] == 1.0 and obs[ts + 13] == 0.0
        if not_wall and no_unit and (empty or replaceable):
            bd_mask[d_idx] = True
            any_dir = True

    if not any_affordable or not any_dir:
        return [True] + [False] * (len(BUILD_COSTS) - 1), [True] * 9

    return bt_mask, bd_mask


def compute_harvester_build_dir_mask(obs: list[float], bd_mask: list[bool]) -> list[bool]:
    """Harvester placement is legal on ore tiles that are empty or have a friendly replaceable building."""
    harv_mask = [False] * 9
    for d_idx, allowed in enumerate(bd_mask):
        if not allowed:
            continue
        ts = _BUILD_DIR_TILE_INDICES[d_idx] * TILE_FEATURES
        ore_here = obs[ts + 1] == 1.0 or obs[ts + 2] == 1.0
        harv_mask[d_idx] = ore_here
    return harv_mask


def select_greedy_actions_from_outputs(
    obs: list[float],
    outputs: dict[str, list[float]],
) -> tuple[int, int, int, int, int]:
    """Match deploy/training greedy action selection for factorized actor heads."""
    bt_mask, bd_mask = compute_masks_from_obs(obs)
    build_dir_idx = argmax(outputs["build_dir"], bd_mask)

    # Harvester legality depends on the chosen build direction.
    ts = _BUILD_DIR_TILE_INDICES[build_dir_idx] * TILE_FEATURES
    if not (obs[ts + 1] == 1.0 or obs[ts + 2] == 1.0):
        bt_mask[_HARVESTER_BUILD_TYPE] = False

    build_type_idx = argmax(outputs["build_type"], bt_mask)
    conv_dir_idx = argmax(outputs["conv_dir"])
    bridge_target_idx = argmax(outputs["bridge_target"])
    move_dir_idx = argmax(outputs["move_dir"])
    return build_type_idx, build_dir_idx, conv_dir_idx, bridge_target_idx, move_dir_idx


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

PLACE_HARV = 5.0
PLACE_CONV = 2.5


def proximity_reward(ct: Controller, build_pos: Position, my_team) -> float:
    best_dist, best_key = None, None
    ore_tiles = GameState.all_ore or ()
    for tile_pos in ore_tiles:
        try:
            d = max(abs(tile_pos.x - build_pos.x), abs(tile_pos.y - build_pos.y))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_key = (tile_pos.x, tile_pos.y)
        except Exception:
            continue
    if best_key is None or best_dist > 10:
        return 0.0
    prev = GameState.harv_proximity.get(best_key, 0.0)
    reward = PLACE_HARV * 0.2 / (1.0 + best_dist)
    delta = reward - prev
    if delta <= 0:
        return 0.0
    GameState.harv_proximity[best_key] = reward
    return delta


def should_reward_harvester_proximity(
    ct: Controller, build_pos: Position, my_team,
    tile_building_id=None, tile_building_type=None,
) -> bool:
    try:
        env = ct.get_tile_env(build_pos)
        if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
            return False
        if env == Environment.WALL:
            return False
        if ct.get_tile_builder_bot_id(build_pos) is not None:
            return False
        bid = tile_building_id
        btype = tile_building_type
        if bid is None:
            bid = ct.get_tile_building_id(build_pos)
            if bid is not None:
                btype = ct.get_entity_type(bid)
        if bid is None:
            return True
        return ct.get_team(bid) == my_team and btype != EntityType.CORE
    except Exception:
        return False



def conveyor_reward(ct: Controller, pos: Position, facing: Direction) -> float:
    """Reward for successful conveyor placement based on manhattan proximity to ore.
    Target distance is 1 (cardinally adjacent). Zero-sum with proximity advances.
    Always matches closest ore regardless of fulfillment — no reward if already claimed."""
    best_dist, best_key = None, None
    fdx, fdy = DIR_DXY.get(facing, (0, 0))
    for ore_pos in (GameState.all_ore or []):
        dx, dy = ore_pos.x - pos.x, ore_pos.y - pos.y
        d = abs(dx) + abs(dy)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_key = (ore_pos.x, ore_pos.y)
    if best_key is None:
        return 0.0
    # Already finalized — no reward
    prev = GameState.conv_proximity.get(best_key)
    if prev is not None and prev < 0:
        return 0.0
    # Cardinal adjacent: check facing doesn't point at ore
    dx, dy = best_key[0] - pos.x, best_key[1] - pos.y
    if best_dist == 1 and fdx == dx and fdy == dy:
        return 0.0  # facing into the ore — useless
    best_given = prev or 0.0
    effective_dist = max(0, best_dist - 1)  # target is cardinal adjacent (d=1)
    reward = PLACE_CONV / (1.0 + effective_dist)
    GameState.conv_proximity[best_key] = -1.0  # mark finalized
    return reward - best_given


def proximity_reward_conv(ct: Controller) -> float:
    """Proximity reward for failed conveyor placement.
    Returns delta improvement: new_advance - old_advance. Improvable on closer approach.
    Always matches closest ore — no reward if already finalized."""
    pos = ct.get_position()
    best_dist, best_key = None, None
    for tile_pos in ct.get_nearby_tiles():
        try:
            env = ct.get_tile_env(tile_pos)
            if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                continue
            d = abs(tile_pos.x - pos.x) + abs(tile_pos.y - pos.y)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_key = (tile_pos.x, tile_pos.y)
        except Exception:
            continue
    if best_key is None:
        return 0.0
    prev = GameState.conv_proximity.get(best_key)
    if prev is not None and prev < 0:
        return 0.0  # finalized by successful placement
    prev = prev or 0.0
    effective_dist = max(0, best_dist - 1)
    reward = min(PLACE_CONV / (1.0 + effective_dist), PLACE_CONV * 0.1)
    delta = reward - prev
    if delta <= 0:
        return 0.0
    GameState.conv_proximity[best_key] = reward
    return delta


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def argmax(data: list[float], mask: list[bool] | None = None) -> int:
    if mask is not None:
        masked = [d if m else float('-inf') for d, m in zip(data, mask)]
        return max(range(len(masked)), key=masked.__getitem__)
    return max(range(len(data)), key=data.__getitem__)


def sample_from_logits(
    logits: list[float], mask: list[bool] | None = None,
) -> tuple[int, float]:
    if mask is not None:
        logits = [l if m else -1e9 for l, m in zip(logits, mask)]
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    r = random.random() * s
    chosen = len(logits) - 1
    cumulative = 0.0
    for i, e in enumerate(exps):
        cumulative += e
        if r <= cumulative:
            chosen = i
            break
    log_prob = math.log(max(exps[chosen] / s, 1e-12))
    return chosen, log_prob


# ---------------------------------------------------------------------------
# Execute actions
# ---------------------------------------------------------------------------

_ZERO_DBG = [0, 0, 0, 0, 0, 0, 0, 0]


def execute_build(ct: Controller, action: str, direction: Direction,
                  conv_direction: Direction, my_team, bridge_target_idx: int = 0) -> tuple[float, float, list]:
    """Execute build action. Returns (harv_reward, conv_reward, dbg)."""
    if action == "none":
        return 0.0, 0.0, _ZERO_DBG

    pos = ct.get_position()
    build_pos = pos.add(direction)

    w, h = ct.get_map_width(), ct.get_map_height()
    if build_pos.x < 0 or build_pos.x >= w or build_pos.y < 0 or build_pos.y >= h:
        return 0.0, 0.0, _ZERO_DBG

    bid = ct.get_tile_building_id(build_pos)
    btype = None
    if bid is not None:
        try:
            btype = ct.get_entity_type(bid)
            if ct.get_team(bid) == my_team:
                if action == "conveyor" and btype == EntityType.CONVEYOR and ct.get_direction(bid) == conv_direction:
                    return 0.0, 0.0, _ZERO_DBG
                if action == "armoured_conveyor" and btype == EntityType.ARMOURED_CONVEYOR and ct.get_direction(bid) == conv_direction:
                    return 0.0, 0.0, _ZERO_DBG
                if action == "splitter" and btype == EntityType.SPLITTER and ct.get_direction(bid) == conv_direction:
                    return 0.0, 0.0, _ZERO_DBG
            if ct.get_team(bid) == my_team and btype != EntityType.CORE:
                idx = BUILD_ACTIONS.index(action)
                base_ti, base_ax = BUILD_COSTS[idx]
                scale = ct.get_scale_percent() / 100.0
                ti, ax = ct.get_global_resources()
                if ti >= int(scale * base_ti) and ax >= int(scale * base_ax):
                    ct.destroy(build_pos)
                    bid, btype = None, None
                else:
                    return 0.0, 0.0, _ZERO_DBG
        except Exception:
            pass

    dbg = [0, 0, 0, 0, 0, 0, 0, 0]  # harv_att, harv_fail, harv_gate, harv_paid, conv_att, conv_fail, conv_gate, conv_paid

    if action == "conveyor":
        dbg[4] = 1
        if ct.can_build_conveyor(build_pos, conv_direction):
            ct.build_conveyor(build_pos, conv_direction)
            GameState.conv_count += 1
            cr = conveyor_reward(ct, build_pos, conv_direction)
            dbg[6] = 1
            if cr > 0:
                dbg[7] = 1
            return 0.0, cr, dbg
        else:
            dbg[5] = 1
        cr = proximity_reward_conv(ct)
        dbg[6] = 1
        if cr > 0:
            dbg[7] = 1
        return 0.0, cr, dbg
    elif action == "harvester":
        dbg[0] = 1
        if ct.can_build_harvester(build_pos):
            ct.build_harvester(build_pos)
            ore_key = (build_pos.x, build_pos.y)
            if ore_key in GameState.harv_placed_ores:
                dbg[2] = 1
                return 0.0, 0.0, dbg
            GameState.harv_placed_ores.add(ore_key)
            dbg[2] = 1
            dbg[3] = 1
            return PLACE_HARV, 0.0, dbg
        dbg[1] = 1
        if should_reward_harvester_proximity(ct, build_pos, my_team, bid, btype):
            dbg[2] = 1
            reward = proximity_reward(ct, build_pos, my_team)
            return reward, 0.0, dbg
        return 0.0, 0.0, dbg
    elif action == "road":
        if ct.can_build_road(build_pos):
            ct.build_road(build_pos)
    elif action == "gunner":
        if ct.can_build_gunner(build_pos, direction):
            ct.build_gunner(build_pos, direction)
    elif action == "sentinel":
        if ct.can_build_sentinel(build_pos, direction):
            ct.build_sentinel(build_pos, direction)
    elif action == "breach":
        if ct.can_build_breach(build_pos, direction):
            ct.build_breach(build_pos, direction)
    elif action == "splitter":
        if ct.can_build_splitter(build_pos, conv_direction):
            ct.build_splitter(build_pos, conv_direction)
    elif action == "armoured_conveyor":
        dbg[4] = 1
        if ct.can_build_armoured_conveyor(build_pos, conv_direction):
            ct.build_armoured_conveyor(build_pos, conv_direction)
            GameState.conv_count += 1
            cr = conveyor_reward(ct, build_pos, conv_direction)
            dbg[6] = 1
            if cr > 0:
                dbg[7] = 1
            return 0.0, cr, dbg
        else:
            dbg[5] = 1
        cr = proximity_reward_conv(ct)
        dbg[6] = 1
        if cr > 0:
            dbg[7] = 1
        return 0.0, cr, dbg
    elif action == "barrier":
        if ct.can_build_barrier(build_pos):
            ct.build_barrier(build_pos)
    elif action == "foundry":
        if ct.can_build_foundry(build_pos):
            ct.build_foundry(build_pos)
    elif action == "launcher":
        if ct.can_build_launcher(build_pos):
            ct.build_launcher(build_pos)
    elif action == "bridge":
        if bridge_target_idx < len(BRIDGE_OFFSETS):
            bdx, bdy = BRIDGE_OFFSETS[bridge_target_idx]
            target = Position(build_pos.x + bdx, build_pos.y + bdy)
            if ct.can_build_bridge(build_pos, target):
                ct.build_bridge(build_pos, target)
    return 0.0, 0.0, dbg


def execute_move(ct: Controller, direction: Direction) -> None:
    if direction == Direction.CENTRE:
        return
    pos = ct.get_position()
    move_pos = pos.add(direction)
    if ct.can_build_road(move_pos):
        ct.build_road(move_pos)
    if ct.can_move(direction):
        ct.move(direction)
        return
    for alt in (direction.rotate_left(), direction.rotate_right()):
        alt_pos = pos.add(alt)
        if ct.can_build_road(alt_pos):
            ct.build_road(alt_pos)
        if ct.can_move(alt):
            ct.move(alt)
            return
