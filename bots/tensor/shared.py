# shared.py — Constants, obs encoding, masks, rewards, and utilities shared
# between tensor_proxy (training) and tensor_inference (deploy).
# No tensor_lib dependency — stdlib only.

import math
import random

from cambc import Controller, Direction, EntityType, Environment, Position

# -- Layout constants --
GRID_SIZE = 5
GRID_HALF = GRID_SIZE // 2
TILE_FEATURES = 17
GLOBAL_FEATURES = 15
POS_EMBED_DIM = 8
INPUT_SIZE = GRID_SIZE * GRID_SIZE * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN_SIZES = [64, 64, 48, 48]
H_DIM = 16
RMSNORM_LAYERS = [True, True, True, True]

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
_CONV_DIR_MASK = [d != Direction.CENTRE for d in DIR_MAP_9]
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
    EntityType.BRIDGE: 5,
    EntityType.FOUNDRY: 5,
    EntityType.GUNNER: 6,
    EntityType.SENTINEL: 6,
    EntityType.BREACH: 6,
    EntityType.LAUNCHER: 6,
    EntityType.CORE: 7,
}
N_BLD_TYPES = 8

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

BUILD_COSTS = (
    (0, 0),      # none
    (80, 0),     # harvester
    (3, 0),      # conveyor
    (1, 0),      # road
    (10, 0),     # gunner
    (15, 0),     # sentinel
    (30, 10),    # breach
    (6, 0),      # splitter
    (10, 5),     # armoured_conveyor
    (3, 0),      # barrier
    (120, 0),    # foundry
    (20, 0),     # launcher
)

_BUILD_DIR_TILE_INDICES = tuple(
    (GRID_HALF + int(DIR_DXY[d][1])) * GRID_SIZE + (GRID_HALF + int(DIR_DXY[d][0]))
    for d in DIR_MAP_9
)

_GLOBAL_OBS_START = GRID_SIZE * GRID_SIZE * TILE_FEATURES

_ZERO_TILE = (0.0,) * TILE_FEATURES
_NO_BUILD_TILE = (1.0,) + (0.0,) * (N_BLD_TYPES - 1) + (0.0, 0.0, 0.0)
_BLD_ONE_HOTS = tuple(
    tuple(1.0 if i == idx else 0.0 for i in range(N_BLD_TYPES))
    for idx in range(N_BLD_TYPES)
)


# ---------------------------------------------------------------------------
# SinusoidalPositionEmbedding (stdlib-only, no tensor_lib)
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding:
    __slots__ = ("d_model", "freqs", "_quarter")

    def __init__(self, d_model: int, base: float = 10000.0):
        self.d_model = d_model
        self._quarter = d_model // 4
        self.freqs: list[float] = [
            1.0 / (base ** (2.0 * i / d_model))
            for i in range(self._quarter)
        ]

    def encode(self, x: int, y: int) -> list[float]:
        q = self._quarter
        freqs = self.freqs
        vec = [0.0] * self.d_model
        half = q * 2
        for i in range(q):
            angle = x * freqs[i]
            vec[2 * i] = math.sin(angle)
            vec[2 * i + 1] = math.cos(angle)
        for i in range(q):
            angle = y * freqs[i]
            vec[half + 2 * i] = math.sin(angle)
            vec[half + 2 * i + 1] = math.cos(angle)
        return vec


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
    conv_proximity: dict[tuple[int, int], float] = {}
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
                      my_team, pos_encoder, prev_resources, spawn_round,
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

    core_dx, core_dy = 0.0, 0.0
    if _core_center is not None:
        cdx = _core_center.x - pos.x
        cdy = _core_center.y - pos.y
        d = max(abs(cdx), abs(cdy), 1)
        core_dx = cdx / d
        core_dy = cdy / d

    ecore_dx, ecore_dy = 0.0, 0.0
    if _enemy_core is not None:
        edx = _enemy_core.x - pos.x
        edy = _enemy_core.y - pos.y
        d = max(abs(edx), abs(edy), 1)
        ecore_dx = edx / d
        ecore_dy = edy / d

    ti_delta, ax_delta = 0.0, 0.0
    if prev_resources is not None:
        prev_ti, prev_ax = prev_resources
        ti_delta = (ti - prev_ti) / 500.0
        ax_delta = (ax - prev_ax) / 500.0

    hp_advantage = (own_hp - enemy_hp) / 500.0

    extend([
        rnd, x_norm, y_norm, ti / 500.0, ax / 500.0, scale,
        core_dx, core_dy, ecore_dx, ecore_dy,
        n_builders / 10.0,
        ti_delta, ax_delta, hp_advantage,
        spawn_round / 2000.0 if spawn_round >= 0 else 0.0,
    ])

    extend(pos_encoder.encode(pos.x, pos.y))

    return _core_center, _enemy_core


def encode_state(ct: Controller, core_center, enemy_core, my_team,
                 pos_encoder, prev_resources, spawn_round,
                 n_builders_out: list) -> list[float]:
    features: list[float] = []
    updated = encode_state_into(ct, features, core_center, enemy_core,
                                my_team, pos_encoder, prev_resources,
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
        no_unit = obs[ts + 15] == 0.0
        empty = obs[ts + 4] == 1.0
        replaceable = obs[ts + 4] == 0.0 and obs[ts + 12] == 1.0 and obs[ts + 11] == 0.0
        if not_wall and no_unit and (empty or replaceable):
            bd_mask[d_idx] = True
            any_dir = True

    if not any_affordable or not any_dir:
        return [True] + [False] * (len(BUILD_COSTS) - 1), [True] * 9

    return bt_mask, bd_mask


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

PLACE_HARV = 5.0
PLACE_CONV = 5.0


def proximity_reward(ct: Controller, build_pos: Position, my_team) -> float:
    best_delta, best_key, best_reward = 0.0, None, 0.0
    ore_tiles = GameState.all_ore or ()
    for tile_pos in ore_tiles:
        try:
            key = (tile_pos.x, tile_pos.y)
            prev = GameState.harv_proximity.get(key, 0.0)
            d = max(abs(tile_pos.x - build_pos.x), abs(tile_pos.y - build_pos.y))
            if d > 10:
                continue
            reward = PLACE_HARV * 0.2 / (1.0 + d)
            delta = reward - prev
            if delta > best_delta:
                best_delta, best_key, best_reward = delta, key, reward
        except Exception:
            continue
    if best_key is None or best_delta <= 0:
        return 0.0
    GameState.harv_proximity[best_key] = best_reward
    return best_delta


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


def conv_proximity_reward(ct: Controller, build_pos: Position, conv_direction: Direction) -> float:
    best_delta, best_key, best_reward = 0.0, None, 0.0
    ore_tiles = GameState.all_ore or ()
    fdx, fdy = DIR_DXY.get(conv_direction, (0, 0))
    for tile_pos in ore_tiles:
        try:
            key = (tile_pos.x, tile_pos.y)
            prev = GameState.conv_proximity.get(key, 0.0)
            dx = tile_pos.x - build_pos.x
            dy = tile_pos.y - build_pos.y
            d = abs(dx) + abs(dy)
            if d > 10:
                continue
            if d == 1 and fdx == dx and fdy == dy:
                continue
            if d <= 1:
                reward = PLACE_CONV
            else:
                reward = PLACE_CONV * 0.1 / (d - 1)
            delta = reward - prev
            if delta > best_delta:
                best_delta, best_key, best_reward = delta, key, reward
        except Exception:
            continue
    if best_key is None or best_delta <= 0:
        return 0.0
    GameState.conv_proximity[best_key] = best_reward
    return best_delta


def should_reward_conv_proximity(
    ct: Controller, build_pos: Position, conv_direction: Direction, my_team,
    tile_building_id=None, tile_building_type=None,
) -> bool:
    if conv_direction == Direction.CENTRE:
        return False
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
                  conv_direction: Direction, my_team) -> tuple[float, float, list]:
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
        else:
            dbg[5] = 1
    elif action == "harvester":
        dbg[0] = 1
        if ct.can_build_harvester(build_pos):
            ct.build_harvester(build_pos)
            ore_key = (build_pos.x, build_pos.y)
            if ore_key in GameState.harv_placed_ores:
                return 0.0, 0.0, dbg
            GameState.harv_placed_ores.add(ore_key)
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
        else:
            dbg[5] = 1
    elif action == "barrier":
        if ct.can_build_barrier(build_pos):
            ct.build_barrier(build_pos)
    elif action == "foundry":
        if ct.can_build_foundry(build_pos):
            ct.build_foundry(build_pos)
    elif action == "launcher":
        if ct.can_build_launcher(build_pos):
            ct.build_launcher(build_pos)
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
