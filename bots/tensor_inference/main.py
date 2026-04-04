# tensor_inference: production deploy bot with bundled stdlib-only weights.
import gc
import json
import math
import os
import random

# Disable cyclic GC — the bot creates no reference cycles (only lists,
# tuples, floats, and a plain MLP). Avoids unpredictable collection
# pauses within the 2ms-per-turn CPU budget.
gc.disable()

from cambc import Controller, Direction, EntityType, Environment, GameConstants, Position, ResourceType

import sys

def _resolve_bot_dir() -> str:
    file_path = globals().get("__file__")
    if file_path:
        return os.path.dirname(os.path.abspath(file_path))
    cwd = os.getcwd()
    if cwd:
        return cwd
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        return os.path.dirname(os.path.abspath(argv0))
    return "."


BOT_DIR = _resolve_bot_dir()
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)
from tensor_lib import MLP
from model_constants import (
    GRID_SIZE, GRID_HALF, TILE_FEATURES, GLOBAL_FEATURES,
    H_DIM, HIDDEN_SIZES, COMPRESS_DIM, RMSNORM_LAYERS,
    TILE_EMBED_DIM, TILE_INPUT_DIM, TILE_GRID_N,
    INPUT_SIZE, OBS_SIZE, TRUNK_INPUT_SIZE,
)

DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]

# Build actions the network can choose from
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
_ACTION_NONE = 0
_ACTION_HARVESTER = 1
_ACTION_CONVEYOR = 2
_ACTION_ROAD = 3
_ACTION_GUNNER = 4
_ACTION_SENTINEL = 5
_ACTION_BREACH = 6
_ACTION_SPLITTER = 7
_ACTION_ARMOURED_CONVEYOR = 8
_ACTION_BARRIER = 9
_ACTION_FOUNDRY = 10
_ACTION_LAUNCHER = 11
_ACTION_BRIDGE = 12

# Maps network output index -> Direction (3x3 grid layout)
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

# Environment type -> one-hot index
ENV_IDX = {
    Environment.EMPTY: 0,
    Environment.WALL: 1,
    Environment.ORE_TITANIUM: 2,
    Environment.ORE_AXIONITE: 3,
}

# Building type -> one-hot index (4 categories)
# 0=none, 1=economy(harvester/conveyor-family), 2=road, 3=military(turrets/barrier/foundry/launcher)
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

# Direction -> (dx, dy) normalized offsets
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
BRIDGE_OFFSETS = tuple(
    (dx, dy)
    for dy in range(-3, 4)
    for dx in range(-3, 4)
    if 0 < dx * dx + dy * dy <= 9
)
N_BRIDGE_TARGETS = len(BRIDGE_OFFSETS)  # 28

# Tile indices in 5x5 grid for each build direction (matches DIR_MAP_9 order)
_BUILD_DIR_TILE_INDICES = tuple(
    (GRID_HALF + int(DIR_DXY[d][1])) * GRID_SIZE + (GRID_HALF + int(DIR_DXY[d][0]))
    for d in DIR_MAP_9
)
_LOCAL_TILE_OFFSETS = tuple(
    (dx, dy)
    for dy in range(-GRID_HALF, GRID_HALF + 1)
    for dx in range(-GRID_HALF, GRID_HALF + 1)
)

_ZERO_TILE = (0.0,) * TILE_FEATURES
_NO_BUILD_TILE = (1.0,) + (0.0,) * (N_BLD_TYPES - 1) + (0.0, 0.0, 0.0, 0.0, 0.0)
_BLD_ONE_HOTS = tuple(
    tuple(1.0 if i == idx else 0.0 for i in range(N_BLD_TYPES))
    for idx in range(N_BLD_TYPES)
)
# Weights file path (adjacent to main.py)
WEIGHT_PATH = os.path.join(BOT_DIR, "weights.json")
_HARVESTER_BUILD_TYPE = 1
_ALL_DIRS_MASK = [True] * 9
_NONE_ONLY_BUILD_MASK = [True] + [False] * (len(BUILD_COSTS) - 1)

_SHARED_MODEL = None
_SHARED_MODEL_KEY = None


def _weight_cache_key(path: str) -> tuple[str, int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return path, st.st_mtime_ns, st.st_size


_GLOBAL_OBS_START = TILE_GRID_N * TILE_FEATURES
_ZERO_OBS = (0.0,) * OBS_SIZE


def _build_model() -> MLP:
    model = MLP(
        input_size=COMPRESS_DIM + GLOBAL_FEATURES + H_DIM,
        hidden_sizes=HIDDEN_SIZES,
        output_heads={
            "build_type": len(BUILD_ACTIONS),
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": N_BRIDGE_TARGETS,
            "move_dir": 9,
            "logit_scale": 5,
            "h_gate": H_DIM,
            "h_cand": H_DIM,
        },
        activation="silu",
        rmsnorm_layers=RMSNORM_LAYERS,
        tile_encoder_sizes=(TILE_INPUT_DIM, TILE_EMBED_DIM),
        tile_grid_n=GRID_SIZE * GRID_SIZE,
        tile_conv=True,
        tile_compress_size=COMPRESS_DIM,
    )
    model.eval()
    return model


def _get_shared_model() -> MLP:
    global _SHARED_MODEL, _SHARED_MODEL_KEY

    cache_key = _weight_cache_key(WEIGHT_PATH)
    if _SHARED_MODEL is not None and cache_key == _SHARED_MODEL_KEY:
        return _SHARED_MODEL

    model = _build_model()
    if cache_key is not None:
        try:
            with open(WEIGHT_PATH) as f:
                model.load(json.load(f))
        except Exception:
            pass

    _SHARED_MODEL = model
    _SHARED_MODEL_KEY = cache_key
    return model



class Player:
    _conv_count: int = 0  # global conveyor count across all units
    _all_ore: list[Position] | None = None  # discovered ore tiles (accumulated)
    _all_ore_keys: set[tuple[int, int]] = set()
    _harv_proximity: dict[tuple[int, int], float] = {}  # ore_key → proximity reward given
    _working_harv_positions: set[tuple[int, int]] = set()  # accumulated per round
    _working_harv_round: int = -1
    _working_harv_count: int = 0  # final count from previous round
    _enemy_core_hp: float = 500.0  # shared enemy core HP tracking
    _own_core_hp: float = 500.0  # shared own core HP tracking
    _enemy_core_dmg: float = 0.0  # damage dealt this round
    _enemy_core_dmg_round: int = -1

    def __init__(self):
        self.core_center: Position | None = None
        self.enemy_core: Position | None = None
        self.my_team = None
        self.num_spawned = 0
        self.num_builders = 0

        # Share weights/process-local inference cache across all unit instances.
        self.model = _get_shared_model()
        self.h_prev: list[float] = [0.0] * H_DIM

        self._prev_collected: tuple[int, int] | None = None
        self._spawn_round: int = -1
        self._obs_buf: list[float] = []
        self._feature_buf: list[float] = []

    def run(self, ct: Controller) -> None:
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    # ---- Core: spawn builders ----

    def _run_core(self, ct: Controller) -> None:
        if self.core_center is None:
            self.core_center = ct.get_position()
        self._discover_ore(ct)
        self._scan_enemy_core_dmg(ct)

        ti, ax = ct.get_global_resources()

        # Convert all available axionite into titanium (4:1 rate)
        if ax > 0:
            ct.convert(ax)
            ti += ax * 4
            ax = 0

        rnd = ct.get_current_round()

        # Reserve floor: each builder needs ~150 ti to build infrastructure
        # (harvester=80 + conveyors + roads). Keep enough for all existing builders.
        reserve_per_builder = 150
        reserve_floor = reserve_per_builder * (self.num_spawned + 1)
        if ti < reserve_floor:
            return

        # Scale hard cap — don't let costs spiral
        if ct.get_scale_percent() > 300:
            return

        pos = ct.get_position()
        dirs = list(DIRECTIONS)
        random.shuffle(dirs)
        for d in dirs:
            sp = pos.add(d)
            if ct.can_spawn(sp):
                ct.spawn_builder(sp)
                self.num_spawned += 1
                return

    # ---- Builder: neural network decisions ----

    def _run_builder(self, ct: Controller) -> None:
        if self._spawn_round < 0:
            self._spawn_round = ct.get_current_round()
        if self.my_team is None:
            self.my_team = ct.get_team()
        nearby_tiles = ct.get_nearby_tiles()
        nearby_entities = ct.get_nearby_entities()
        if self.core_center is None or self.enemy_core is None:
            self._find_cores_in_entities(ct, nearby_entities)

        # Encode game state + recurrent hidden -> feature vector
        obs = self._encode_state(ct, nearby_tiles, nearby_entities)

        h_in = self.h_prev
        features = self._feature_buf
        features.clear()
        features.extend(obs)
        features.extend(h_in)
        outputs = self.model.infer(features)

        # minGRU: gated recurrent update
        h_gate_raw = outputs["h_gate"]
        h_cand = outputs["h_cand"]
        h_in = self.h_prev
        exp = math.exp
        h_new = [0.0] * len(h_in)
        for i, g in enumerate(h_gate_raw):
            if g >= 0:
                z = 1.0 / (1.0 + exp(-g))
            else:
                e = exp(g)
                z = e / (1.0 + e)
            h_new[i] = h_in[i] + z * (h_cand[i] - h_in[i])
        self.h_prev = h_new

        # Compute action masks (which builds are affordable, which tiles are free)
        bt_mask, bd_mask = _compute_masks_from_obs(obs)

        # Sample direction first, then condition build-type on the target tile.
        build_dir_idx = _argmax(outputs["build_dir"], bd_mask)
        # Harvester only legal when chosen direction has ore.
        ts = _BUILD_DIR_TILE_INDICES[build_dir_idx] * TILE_FEATURES
        if not (obs[ts + 1] == 1.0 or obs[ts + 2] == 1.0):
            bt_mask[_HARVESTER_BUILD_TYPE] = False
        build_type_idx = _argmax(outputs["build_type"], bt_mask)
        conv_dir_idx = _argmax(outputs["conv_dir"], None)
        bridge_target_idx = _argmax(outputs["bridge_target"], None)
        move_dir_idx = _argmax(outputs["move_dir"], None)

        # Execute actions
        build_dir = DIR_MAP_9[build_dir_idx]
        conv_dir = CONV_DIR_4[conv_dir_idx]
        self._execute_build(ct, build_type_idx, build_dir, conv_dir, bridge_target_idx)

        move_dir = DIR_MAP_9[move_dir_idx]
        self._execute_move(ct, move_dir)


    @staticmethod
    def _discover_ore(ct: Controller, nearby_tiles=None) -> None:
        """Add any newly visible ore tiles to the shared set."""
        if Player._all_ore is None:
            Player._all_ore = []
            Player._all_ore_keys = set()
        if nearby_tiles is None:
            nearby_tiles = ct.get_nearby_tiles()
        for tile_pos in nearby_tiles:
            key = (tile_pos.x, tile_pos.y)
            if key in Player._all_ore_keys:
                continue
            try:
                env = ct.get_tile_env(tile_pos)
                if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    Player._all_ore.append(tile_pos)
                    Player._all_ore_keys.add(key)
            except Exception:
                pass

    _CONV_TYPES = (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.SPLITTER)

    @staticmethod
    def _scan_working_harvesters(ct: Controller) -> None:
        """Accumulate working harvester positions from this unit's vision.
        A harvester is 'working' if it has a cardinally adjacent conveyor/splitter
        that accepts from its side (i.e. not facing toward the harvester).
        Resets each round; count finalizes after all units have run."""
        rnd = ct.get_current_round()
        if Player._working_harv_round != rnd:
            Player._working_harv_count = len(Player._working_harv_positions)
            Player._working_harv_positions = set()
            Player._working_harv_round = rnd
        team = ct.get_team()
        for tile_pos in ct.get_nearby_tiles():
            try:
                bid = ct.get_tile_building_id(tile_pos)
                if bid is None or ct.get_entity_type(bid) != EntityType.HARVESTER \
                   or ct.get_team(bid) != team:
                    continue
                hkey = (tile_pos.x, tile_pos.y)
                if hkey in Player._working_harv_positions:
                    continue
                # Check 4 cardinal neighbors for accepting conveyor
                for d in (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST):
                    adj = tile_pos.add(d)
                    try:
                        cbid = ct.get_tile_building_id(adj)
                        if cbid is None:
                            continue
                        ctype = ct.get_entity_type(cbid)
                        if ctype not in Player._CONV_TYPES or ct.get_team(cbid) != team:
                            continue
                        # Conveyor accepts from 3 non-output sides.
                        # Harvester is in direction d.opposite() from conveyor.
                        # Conveyor rejects if output (cdir) faces harvester.
                        cdir = ct.get_direction(cbid)
                        if cdir == d.opposite():
                            continue  # conveyor output faces harvester — can't accept
                    except Exception:
                        continue
                    Player._working_harv_positions.add(hkey)
                    break
            except Exception:
                continue

    @staticmethod
    def _scan_enemy_core_dmg(ct: Controller, nearby_entities=None) -> None:
        """Track enemy core HP changes. Any unit seeing the enemy core updates shared state."""
        rnd = ct.get_current_round()
        if Player._enemy_core_dmg_round != rnd:
            Player._enemy_core_dmg = 0.0
            Player._enemy_core_dmg_round = rnd
        team = ct.get_team()
        if nearby_entities is None:
            nearby_entities = ct.get_nearby_entities()
        for eid in nearby_entities:
            try:
                if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) != team:
                    hp = ct.get_hp(eid)
                    dmg = max(0.0, Player._enemy_core_hp - hp)
                    if dmg > Player._enemy_core_dmg:
                        Player._enemy_core_dmg = dmg
                    Player._enemy_core_hp = hp
                    break
            except Exception:
                pass

    def _proximity_reward(self, ct: Controller) -> float:
        """Proximity reward for failed harvester placement.
        Returns delta improvement: new_advance - old_advance. Improvable on closer approach.
        Zero-sum with placement: total advance is deducted on successful build."""
        pos = ct.get_position()
        best_delta, best_key, best_reward = 0.0, None, 0.0
        ore_tiles = Player._all_ore or ()
        for tile_pos in ore_tiles:
            try:
                key = (tile_pos.x, tile_pos.y)
                prev = Player._harv_proximity.get(key, 0.0)
                d = max(abs(tile_pos.x - pos.x), abs(tile_pos.y - pos.y))
                if d > 10:
                    continue
                reward = self._PLACE_HARV * 0.2 / (1.0 + d)
                delta = reward - prev
                if delta > best_delta:
                    best_delta, best_key, best_reward = delta, key, reward
            except Exception:
                continue
        if best_key is None or best_delta <= 0:
            return 0.0
        Player._harv_proximity[best_key] = best_reward
        return best_delta


    def _find_core(self, ct: Controller) -> Position:
        if self.my_team is None:
            self.my_team = ct.get_team()
        for eid in ct.get_nearby_entities():
            try:
                if ct.get_entity_type(eid) == EntityType.CORE:
                    if ct.get_team(eid) == self.my_team:
                        self.core_center = ct.get_position(eid)
                    else:
                        self.enemy_core = ct.get_position(eid)
            except Exception:
                continue
        return self.core_center or ct.get_position()

    def _find_cores_in_entities(self, ct: Controller, nearby_entities) -> Position:
        if self.my_team is None:
            self.my_team = ct.get_team()
        get_entity_type = ct.get_entity_type
        get_team = ct.get_team
        get_position = ct.get_position
        for eid in nearby_entities:
            try:
                if get_entity_type(eid) != EntityType.CORE:
                    continue
                if get_team(eid) == self.my_team:
                    self.core_center = get_position(eid)
                else:
                    self.enemy_core = get_position(eid)
            except Exception:
                continue
        return self.core_center or ct.get_position()

    # ---- State encoding ----

    def _encode_state_into(self, ct: Controller, features, nearby_tiles=None, nearby_entities=None) -> None:
        """Append encoded state into a list-like float buffer."""
        if len(features) != OBS_SIZE:
            features.clear()
            features.extend(_ZERO_OBS)
        else:
            features[:] = _ZERO_OBS
        pos = ct.get_position()
        pos_x = pos.x
        pos_y = pos.y
        my_team = self.my_team
        get_tile_env = ct.get_tile_env
        is_tile_passable = ct.is_tile_passable
        get_tile_building_id = ct.get_tile_building_id
        get_tile_builder_bot_id = ct.get_tile_builder_bot_id
        get_entity_type = ct.get_entity_type
        get_team = ct.get_team
        get_direction = ct.get_direction
        get_stored_resource = ct.get_stored_resource
        get_position = ct.get_position
        get_hp = ct.get_hp

        # 5x5 local grid
        w, h = ct.get_map_width(), ct.get_map_height()
        ore_keys = Player._all_ore_keys
        ore_tiles = Player._all_ore
        if ore_tiles is None:
            ore_tiles = []
            Player._all_ore = ore_tiles
            Player._all_ore_keys = ore_keys = set()
        if nearby_tiles is None:
            nearby_tiles = ct.get_nearby_tiles()
        for tile_pos in nearby_tiles:
            tx = tile_pos.x
            ty = tile_pos.y
            lx = tx - pos_x + GRID_HALF
            ly = ty - pos_y + GRID_HALF
            if lx < 0 or lx >= GRID_SIZE or ly < 0 or ly >= GRID_SIZE:
                continue
            try:
                env = get_tile_env(tile_pos)
            except Exception:
                continue
            key = (tx, ty)
            if key not in ore_keys and env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                ore_tiles.append(tile_pos)
                ore_keys.add(key)
            base = (ly * GRID_SIZE + lx) * TILE_FEATURES
            features[base] = 1.0 if env == Environment.WALL else 0.0
            features[base + 1] = 1.0 if env == Environment.ORE_TITANIUM else 0.0
            features[base + 2] = 1.0 if env == Environment.ORE_AXIONITE else 0.0
            features[base + 3] = 0.0 if is_tile_passable(tile_pos) else 1.0
            features[base + 4] = 1.0  # default building type: none

            bid = get_tile_building_id(tile_pos)
            if bid is not None:
                try:
                    bdir = get_direction(bid)
                    sr = get_stored_resource(bid)
                except Exception:
                    bdir = Direction.CENTRE
                    sr = None
                btype = get_entity_type(bid)
                bidx = BLD_IDX.get(btype, 3)
                one_hot = _BLD_ONE_HOTS[bidx]
                features[base + 4:base + 4 + N_BLD_TYPES] = one_hot
                features[base + 14] = 1.0 if get_team(bid) == my_team else -1.0
                bdx, bdy = DIR_DXY.get(bdir, (0, 0))
                features[base + 15] = float(bdx)
                features[base + 16] = float(bdy)
                features[base + 18] = 0.0 if sr is None else 1.0 if sr == ResourceType.TITANIUM else -1.0

            uid = get_tile_builder_bot_id(tile_pos)
            if uid is not None:
                features[base + 19] = 1.0
                features[base + 20] = 1.0 if get_team(uid) == my_team else -1.0

        rnd_n = ct.get_current_round()
        rnd = rnd_n / 2000.0
        x_norm = pos_x / max(w, 1)
        y_norm = pos_y / max(h, 1)
        ti, ax = ct.get_global_resources()
        prev_collected = self._prev_collected
        self._prev_collected = (ti, ax)
        scale = ct.get_scale_percent() / 500.0

        if Player._enemy_core_dmg_round != rnd_n:
            Player._enemy_core_dmg = 0.0
            Player._enemy_core_dmg_round = rnd_n
        n_builders = 0
        own_hp, enemy_hp = Player._own_core_hp, Player._enemy_core_hp
        if nearby_entities is None:
            nearby_entities = ct.get_nearby_entities()
        for eid in nearby_entities:
            try:
                etype = get_entity_type(eid)
                team = get_team(eid)
                if etype == EntityType.BUILDER_BOT and team == my_team:
                    n_builders += 1
                elif etype == EntityType.CORE:
                    if team == my_team:
                        if self.core_center is None:
                            self.core_center = get_position(eid)
                        own_hp = get_hp(eid)
                        Player._own_core_hp = own_hp
                    else:
                        if self.enemy_core is None:
                            self.enemy_core = get_position(eid)
                        hp = get_hp(eid)
                        dmg = max(0.0, Player._enemy_core_hp - hp)
                        if dmg > Player._enemy_core_dmg:
                            Player._enemy_core_dmg = dmg
                        Player._enemy_core_hp = hp
                        enemy_hp = hp
            except Exception:
                pass
        core_dx, core_dy, core_known = 0.0, 0.0, 0.0
        if self.core_center is not None:
            core_dx = float(self.core_center.x - pos_x)
            core_dy = float(self.core_center.y - pos_y)
            core_known = 1.0

        ecore_dx, ecore_dy, ecore_known = 0.0, 0.0, 0.0
        if self.enemy_core is not None:
            ecore_dx = float(self.enemy_core.x - pos_x)
            ecore_dy = float(self.enemy_core.y - pos_y)
            ecore_known = 1.0

        ti_income, ax_income = 0.0, 0.0
        if prev_collected is not None:
            prev_ti, prev_ax = prev_collected
            ti_income = (ti - prev_ti) / 500.0
            ax_income = (ax - prev_ax) / 500.0

        hp_advantage = (own_hp - enemy_hp) / 500.0

        Player._scan_working_harvesters(ct)

        features[_GLOBAL_OBS_START:_GLOBAL_OBS_START + GLOBAL_FEATURES] = (
            rnd, x_norm, y_norm, ti / 500.0, ax / 500.0, scale,
            core_dx, core_dy, ecore_dx, ecore_dy,
            n_builders / 10.0,
            ti_income, ax_income, hp_advantage,
            self._spawn_round / 2000.0 if self._spawn_round >= 0 else 0.0,
            core_known, ecore_known,
            w / 50.0, h / 50.0,
            Player._working_harv_count / 20.0,
        )

    def _encode_state(self, ct: Controller, nearby_tiles=None, nearby_entities=None) -> list[float]:
        """Encode 5x5 grid (21 feat/tile) + global (19)."""
        features = self._obs_buf
        features.clear()
        self._encode_state_into(ct, features, nearby_tiles, nearby_entities)
        return features

    # ---- Action execution ----

    _PLACE_HARV = 5.0

    def _execute_build(
        self, ct: Controller, action_idx: int, direction: Direction,
        conv_direction: Direction | None = None, bridge_target_idx: int = 0,
    ) -> tuple[float, float]:
        """Execute build action. Returns (harv_reward, conv_reward).

        If the target tile has a friendly non-core building, auto-destroys it
        first (destroy has no cooldown cost). Skips if we can't afford the build.
        """
        if action_idx == _ACTION_NONE:
            return 0.0, 0.0

        build_pos = ct.get_position().add(direction)

        # Bounds check — build_pos can be out of map when near edges
        w, h = ct.get_map_width(), ct.get_map_height()
        if build_pos.x < 0 or build_pos.x >= w or build_pos.y < 0 or build_pos.y >= h:
            return 0.0, 0.0

        # Auto-destroy friendly non-core building to make room.
        # Check affordability first to avoid destroying for nothing.
        bid = ct.get_tile_building_id(build_pos)
        if bid is not None:
            try:
                if ct.get_team(bid) == self.my_team:
                    btype_existing = ct.get_entity_type(bid)
                    if btype_existing == EntityType.CORE:
                        return 0.0, 0.0
                    # Skip rebuilding identical conveyor-family buildings
                    conv_dir = conv_direction if conv_direction is not None else direction
                    if action_idx == _ACTION_CONVEYOR and btype_existing == EntityType.CONVEYOR and ct.get_direction(bid) == conv_dir:
                        return 0.0, 0.0
                    if action_idx == _ACTION_ARMOURED_CONVEYOR and btype_existing == EntityType.ARMOURED_CONVEYOR and ct.get_direction(bid) == conv_dir:
                        return 0.0, 0.0
                    if action_idx == _ACTION_SPLITTER and btype_existing == EntityType.SPLITTER and ct.get_direction(bid) == conv_dir:
                        return 0.0, 0.0
                    base_ti, base_ax = BUILD_COSTS[action_idx]
                    scale = ct.get_scale_percent() / 100.0
                    ti, ax = ct.get_global_resources()
                    if ti >= int(scale * base_ti) and ax >= int(scale * base_ax):
                        ct.destroy(build_pos)
                    else:
                        return 0.0, 0.0
            except Exception:
                pass

        if action_idx == _ACTION_CONVEYOR:
            conv_dir = conv_direction if conv_direction is not None else direction
            if ct.can_build_conveyor(build_pos, conv_dir):
                ct.build_conveyor(build_pos, conv_dir)
                Player._conv_count += 1
        elif action_idx == _ACTION_HARVESTER:
            if ct.can_build_harvester(build_pos):
                ct.build_harvester(build_pos)
                ore_key = (build_pos.x, build_pos.y)
                given = Player._harv_proximity.get(ore_key, 0.0)
                Player._harv_proximity[ore_key] = self._PLACE_HARV
                return self._PLACE_HARV - given, 0.0
            # Failed — proximity reward toward nearest ore (one-time per ore)
            return self._proximity_reward(ct), 0.0
        elif action_idx == _ACTION_ROAD:
            if ct.can_build_road(build_pos):
                ct.build_road(build_pos)
        elif action_idx == _ACTION_GUNNER:
            if ct.can_build_gunner(build_pos, direction):
                ct.build_gunner(build_pos, direction)
        elif action_idx == _ACTION_SENTINEL:
            if ct.can_build_sentinel(build_pos, direction):
                ct.build_sentinel(build_pos, direction)
        elif action_idx == _ACTION_BREACH:
            if ct.can_build_breach(build_pos, direction):
                ct.build_breach(build_pos, direction)
        elif action_idx == _ACTION_SPLITTER:
            sp_dir = conv_direction if conv_direction is not None else direction
            if ct.can_build_splitter(build_pos, sp_dir):
                ct.build_splitter(build_pos, sp_dir)
        elif action_idx == _ACTION_ARMOURED_CONVEYOR:
            ac_dir = conv_direction if conv_direction is not None else direction
            if ct.can_build_armoured_conveyor(build_pos, ac_dir):
                ct.build_armoured_conveyor(build_pos, ac_dir)
                Player._conv_count += 1
        elif action_idx == _ACTION_BARRIER:
            if ct.can_build_barrier(build_pos):
                ct.build_barrier(build_pos)
        elif action_idx == _ACTION_FOUNDRY:
            if ct.can_build_foundry(build_pos):
                ct.build_foundry(build_pos)
        elif action_idx == _ACTION_LAUNCHER:
            if ct.can_build_launcher(build_pos):
                ct.build_launcher(build_pos)
        elif action_idx == _ACTION_BRIDGE:
            if bridge_target_idx < len(BRIDGE_OFFSETS):
                bdx, bdy = BRIDGE_OFFSETS[bridge_target_idx]
                target = Position(build_pos.x + bdx, build_pos.y + bdy)
                if ct.can_build_bridge(build_pos, target):
                    ct.build_bridge(build_pos, target)
        # Punish placing non-harvester on ore (wasted tile)
        ore_key = (build_pos.x, build_pos.y)
        if ore_key in Player._all_ore_keys:
            return -self._PLACE_HARV, 0.0
        return 0.0, 0.0

    def _execute_move(self, ct: Controller, direction: Direction) -> None:
        if direction == Direction.CENTRE:
            return

        pos = ct.get_position()
        move_pos = pos.add(direction)

        # Build road if needed to make tile passable
        if ct.can_build_road(move_pos):
            ct.build_road(move_pos)

        if ct.can_move(direction):
            ct.move(direction)
            return

        # Try adjacent directions as fallback
        for alt in (direction.rotate_left(), direction.rotate_right()):
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return
def _compute_masks_from_obs(obs: list[float]) -> tuple[list[bool], list[bool]]:
    """Compute build_type and build_dir action masks.

    Build directions allow both empty tiles and tiles with friendly non-core
    buildings (the execution layer auto-destroys before building).

    Returns (bt_mask, bd_mask) as lists of booleans (True = allowed).
    Harvester gating is done after direction sampling (caller checks ore).
    """
    g = _GLOBAL_OBS_START
    ti = obs[g + 3] * 500.0
    ax = obs[g + 4] * 500.0
    scale = obs[g + 5] * 5.0  # scale_pct / 100

    # Build type mask: can we afford it?
    bt_mask = [True]  # "none" always allowed
    any_affordable = False
    for i in range(1, len(BUILD_COSTS)):
        base_ti, base_ax = BUILD_COSTS[i]
        ok = ti >= int(scale * base_ti) and ax >= int(scale * base_ax)
        bt_mask.append(ok)
        any_affordable = any_affordable or ok

    # Build dir mask: empty tile OR friendly non-core building (auto-destroyed).
    # Must not be a wall and must not have a unit.
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

    # If nothing affordable or nowhere to build, force "none" (dir irrelevant)
    if not any_affordable or not any_dir:
        return _NONE_ONLY_BUILD_MASK, _ALL_DIRS_MASK

    return bt_mask, bd_mask


def _argmax(data: list[float], mask: list[bool] | None = None) -> int:
    """Argmax without allocating a masked copy."""
    best_idx = 0
    best_val = float("-inf")
    if mask is None:
        for i, value in enumerate(data):
            if value > best_val:
                best_idx = i
                best_val = value
        return best_idx
    for i, value in enumerate(data):
        if mask[i] and value > best_val:
            best_idx = i
            best_val = value
    return best_idx
