# tensor: Neural network-powered builder bot.
# Uses a lightweight MLP (64->48 hidden) to decide what to build, where to build,
# and where to move. Forward pass fits within ~0.7ms of the 2ms CPU budget.
# Supports online REINFORCE learning and offline weight loading.

import atexit
import gc
import json
import math
import os
import random

# Disable cyclic GC — the bot creates no reference cycles (only lists,
# tuples, floats, and a plain MLP). Avoids unpredictable collection
# pauses within the 2ms-per-turn CPU budget.
gc.disable()

from cambc import Controller, Direction, EntityType, Environment, Position

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tensor_lib import MLP, SinusoidalPositionEmbedding

# -- Layout constants --
GRID_SIZE = 5
GRID_HALF = GRID_SIZE // 2
# Per tile: is_wall(1) + ore_ti(1) + ore_ax(1) + is_impassable(1)
#         + bld_type(8: none/harvester/conveyor/splitter/road/support/military/core)
#         + bld_friendly(1) + bld_dir(2)
#         + has_unit(1) + unit_friendly(1) = 17
TILE_FEATURES = 17
# Global: round, x, y, ti, ax, scale, core_dx, core_dy, enemy_core_dx, enemy_core_dy,
#         n_builders, ti_delta, ax_delta, hp_advantage, spawn_round = 15
GLOBAL_FEATURES = 15
POS_EMBED_DIM = 8
INPUT_SIZE = GRID_SIZE * GRID_SIZE * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN_SIZES = [64, 64, 48, 48]
H_DIM = 16  # Recurrent hidden state size
RMSNORM_LAYERS = [True, True, True, True]

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
]

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
_CONV_DIR_MASK = [d != Direction.CENTRE for d in DIR_MAP_9]
_HARVESTER_BUILD_TYPE = 1

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
    EntityType.BRIDGE: 5,
    EntityType.FOUNDRY: 5,
    EntityType.GUNNER: 6,
    EntityType.SENTINEL: 6,
    EntityType.BREACH: 6,
    EntityType.LAUNCHER: 6,
    EntityType.CORE: 7,
}
N_BLD_TYPES = 8

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

# Base build costs (titanium, axionite) indexed by BUILD_ACTIONS
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

# Tile indices in 5x5 grid for each build direction (matches DIR_MAP_9 order)
_BUILD_DIR_TILE_INDICES = tuple(
    (GRID_HALF + int(DIR_DXY[d][1])) * GRID_SIZE + (GRID_HALF + int(DIR_DXY[d][0]))
    for d in DIR_MAP_9
)

# Offset where global features start in observation vector
_GLOBAL_OBS_START = GRID_SIZE * GRID_SIZE * TILE_FEATURES

_ZERO_TILE = (0.0,) * TILE_FEATURES
_NO_BUILD_TILE = (1.0,) + (0.0,) * (N_BLD_TYPES - 1) + (0.0, 0.0, 0.0)
_BLD_ONE_HOTS = tuple(
    tuple(1.0 if i == idx else 0.0 for i in range(N_BLD_TYPES))
    for idx in range(N_BLD_TYPES)
)

# Weights file path (adjacent to main.py)
WEIGHT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")

# Training mode: set TENSOR_TRAIN_LOG to a file path to log trajectories
_TRAIN_LOG = os.environ.get("TENSOR_TRAIN_LOG")
_SHARED_MODEL = None
_SHARED_MODEL_KEY = None
_LIVE_LOG_PLAYERS: set["Player"] = set()


def _flush_live_logs() -> None:
    for player in list(_LIVE_LOG_PLAYERS):
        try:
            player._flush_log()
        except Exception:
            pass


atexit.register(_flush_live_logs)


def _weight_cache_key(path: str) -> tuple[str, int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return path, st.st_mtime_ns, st.st_size


def _build_model() -> MLP:
    model = MLP(
        input_size=INPUT_SIZE + H_DIM,
        hidden_sizes=HIDDEN_SIZES,
        output_heads={
            "build_type": len(BUILD_ACTIONS),
            "build_dir": 9,
            "conv_dir": 9,
            "move_dir": 9,
            "h_next": H_DIM,
        },
        activation="silu",
        rmsnorm_layers=RMSNORM_LAYERS,
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
    _harv_placed_ores: set[tuple[int, int]] = set()  # ore tiles already rewarded for placement
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

        # Positional encoder (fixed sinusoidal)
        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)

        # Share weights/process-local inference cache across all unit instances.
        self.model = _get_shared_model()
        self.h_prev: list[float] = [0.0] * H_DIM

        # Trajectory logging for training
        self._train_log: list | None = [] if _TRAIN_LOG else None
        self._prev_resources: tuple[float, float] | None = None
        self._n_builders_cached: int = 1
        self._spawn_round: int = -1
        self._dbg_harv_attempts: int = 0
        self._dbg_harv_failures: int = 0
        self._dbg_harv_prox_gate: int = 0
        self._dbg_harv_prox_paid: int = 0
        if self._train_log is not None:
            _LIVE_LOG_PLAYERS.add(self)

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
        if self.core_center is None:
            self._find_core(ct)
        # Accumulate ore discoveries and enemy core damage.
        # Working-harvester reward is replay-derived during training, so we do
        # not spend per-turn CPU estimating it from partial vision here.
        self._discover_ore(ct)
        self._scan_enemy_core_dmg(ct)
        # Opportunistically scan for enemy core
        if self.enemy_core is None:
            for eid in ct.get_nearby_entities():
                try:
                    if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) != self.my_team:
                        self.enemy_core = ct.get_position(eid)
                        break
                except Exception:
                    continue

        # Encode game state + recurrent hidden -> feature vector
        obs = self._encode_state(ct)

        h_in = self.h_prev
        features = obs + h_in
        outputs = self.model.infer(features)

        # Update recurrent hidden state (tanh-bounded)
        h_raw = outputs["h_next"]
        self.h_prev = [math.tanh(x) for x in h_raw]

        # Compute action masks (which builds are affordable, which tiles are free)
        bt_mask, bd_mask, harv_bd_mask = _compute_masks_from_obs(obs)

        # Decode each head via argmax (sample in training mode)
        if self._train_log is not None:
            build_type_idx, bt_lp = _sample_from_logits(outputs["build_type"], bt_mask)
            build_dir_mask = _compute_conditional_build_dir_mask_from_obs(obs, build_type_idx, bd_mask, harv_bd_mask)
            build_dir_idx, bd_lp = _sample_from_logits(outputs["build_dir"], build_dir_mask)
            conv_dir_idx, cd_lp = _sample_from_logits(outputs["conv_dir"], _CONV_DIR_MASK)
            move_dir_idx, md_lp = _sample_from_logits(outputs["move_dir"], None)
            # Mask build_dir log_prob when build_type is "none" (idx 0)
            # — the direction is causally irrelevant, so exclude from policy gradient
            conv_causal = build_type_idx in (2, 8)
            old_log_prob = (
                bt_lp
                + (bd_lp if build_type_idx != 0 else 0.0)
                + (cd_lp if conv_causal else 0.0)
                + md_lp
            )
        else:
            build_type_idx = _argmax(outputs["build_type"], bt_mask)
            build_dir_mask = _compute_conditional_build_dir_mask_from_obs(obs, build_type_idx, bd_mask, harv_bd_mask)
            build_dir_idx = _argmax(outputs["build_dir"], build_dir_mask)
            conv_dir_idx = _argmax(outputs["conv_dir"], _CONV_DIR_MASK)
            move_dir_idx = _argmax(outputs["move_dir"], None)

        # Snapshot resources before actions to isolate delivery from spending
        ti_before, ax_before = ct.get_global_resources()

        # Execute actions
        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        conv_dir = DIR_MAP_9[conv_dir_idx]
        self._dbg_harv_attempts = 0
        self._dbg_harv_failures = 0
        self._dbg_harv_prox_gate = 0
        self._dbg_harv_prox_paid = 0
        harv_reward, conv_reward = self._execute_build(ct, build_action, build_dir, conv_dir)
        place_reward = harv_reward + conv_reward

        move_dir = DIR_MAP_9[move_dir_idx]
        self._execute_move(ct, move_dir)

        # Enemy core damage (class-level, accumulated from all units' vision)
        enemy_dmg = Player._enemy_core_dmg

        ti_after, ax_after = ct.get_global_resources()
        self._prev_resources = (ti_after, ax_after)

        # Log trajectory step
        if self._train_log is not None:
            # Tuple: obs, h, bt, bd, cd, md, value, n_working, conv_r, ti, harv_r, dmg, old_lp,
            #        harv_attempt, harv_fail, harv_gate, harv_paid
            self._train_log.append((
                obs,
                list(h_in),
                build_type_idx,
                build_dir_idx,
                conv_dir_idx,
                move_dir_idx,
                0.0,  # value placeholder
                0,  # replay augmentation fills in true working-harvester count
                conv_reward,
                ti_after,
                harv_reward,
                enemy_dmg,
                old_log_prob,
                self._dbg_harv_attempts,
                self._dbg_harv_failures,
                self._dbg_harv_prox_gate,
                self._dbg_harv_prox_paid,
            ))

            if ct.get_current_round() >= 1999:
                self._flush_log()

    @staticmethod
    def _discover_ore(ct: Controller) -> None:
        """Add any newly visible ore tiles to the shared set."""
        if Player._all_ore is None:
            Player._all_ore = []
            Player._all_ore_keys = set()
        for tile_pos in ct.get_nearby_tiles():
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
    def _scan_enemy_core_dmg(ct: Controller) -> None:
        """Track enemy core HP changes. Any unit seeing the enemy core updates shared state."""
        rnd = ct.get_current_round()
        if Player._enemy_core_dmg_round != rnd:
            Player._enemy_core_dmg = 0.0
            Player._enemy_core_dmg_round = rnd
        team = ct.get_team()
        for eid in ct.get_nearby_entities():
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

    def _proximity_reward(self, ct: Controller, build_pos: Position) -> float:
        """Proximity reward for failed harvester placement.
        Returns delta improvement: new_advance - old_advance. Improvable on closer approach.
        Zero-sum with placement: total advance is deducted on successful build."""
        best_delta, best_key, best_reward = 0.0, None, 0.0
        ore_tiles = Player._all_ore or ()
        for tile_pos in ore_tiles:
            try:
                key = (tile_pos.x, tile_pos.y)
                prev = Player._harv_proximity.get(key, 0.0)
                d = max(abs(tile_pos.x - build_pos.x), abs(tile_pos.y - build_pos.y))
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

    def _should_reward_harvester_proximity(
        self,
        ct: Controller,
        build_pos: Position,
        tile_building_id: int | None = None,
        tile_building_type: EntityType | None = None,
    ) -> bool:
        """Only reward harvester proximity when the attempted tile is otherwise valid.

        The shaping is meant to say "good harvester intent, wrong tile type",
        not "good job attempting an impossible build". Match build-dir validity:
        non-wall, unit-free, and either empty or occupied by a friendly non-core
        building that could be replaced. The only missing property should be that
        the tile is not an ore tile.
        """
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
            return ct.get_team(bid) == self.my_team and btype != EntityType.CORE
        except Exception:
            return False


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

    # ---- State encoding ----

    def _encode_state_into(self, ct: Controller, features) -> None:
        """Append encoded state into a list-like float buffer."""
        pos = ct.get_position()
        my_team = self.my_team
        append = features.append
        extend = features.extend
        get_tile_env = ct.get_tile_env
        get_tile_building_id = ct.get_tile_building_id
        get_tile_builder_bot_id = ct.get_tile_builder_bot_id
        get_entity_type = ct.get_entity_type
        get_team = ct.get_team
        get_direction = ct.get_direction

        # 5x5 local grid
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
        own_hp, enemy_hp = Player._own_core_hp, Player._enemy_core_hp
        for eid in ct.get_nearby_entities():
            try:
                etype = get_entity_type(eid)
                team = get_team(eid)
                if etype == EntityType.BUILDER_BOT and team == my_team:
                    n_builders += 1
                elif etype == EntityType.CORE:
                    if team == my_team:
                        if self.core_center is None:
                            self.core_center = ct.get_position(eid)
                        own_hp = ct.get_hp(eid)
                        Player._own_core_hp = own_hp
                    else:
                        if self.enemy_core is None:
                            self.enemy_core = ct.get_position(eid)
                        hp = ct.get_hp(eid)
                        Player._enemy_core_hp = hp
                        enemy_hp = hp
            except Exception:
                pass
        self._n_builders_cached = n_builders

        core_dx, core_dy = 0.0, 0.0
        if self.core_center is not None:
            cdx = self.core_center.x - pos.x
            cdy = self.core_center.y - pos.y
            d = max(abs(cdx), abs(cdy), 1)
            core_dx = cdx / d
            core_dy = cdy / d

        ecore_dx, ecore_dy = 0.0, 0.0
        if self.enemy_core is not None:
            edx = self.enemy_core.x - pos.x
            edy = self.enemy_core.y - pos.y
            d = max(abs(edx), abs(edy), 1)
            ecore_dx = edx / d
            ecore_dy = edy / d

        ti_delta, ax_delta = 0.0, 0.0
        if self._prev_resources is not None:
            prev_ti, prev_ax = self._prev_resources
            ti_delta = (ti - prev_ti) / 500.0
            ax_delta = (ax - prev_ax) / 500.0

        hp_advantage = (own_hp - enemy_hp) / 500.0

        extend([
            rnd, x_norm, y_norm, ti / 500.0, ax / 500.0, scale,
            core_dx, core_dy, ecore_dx, ecore_dy,
            n_builders / 10.0,
            ti_delta, ax_delta, hp_advantage,
            self._spawn_round / 2000.0 if self._spawn_round >= 0 else 0.0,
        ])

        extend(self.pos_encoder.encode(pos.x, pos.y))

    def _encode_state(self, ct: Controller) -> list[float]:
        """Encode 5x5 grid (16 feat/tile) + global (14) + pos embed (8)."""
        features: list[float] = []
        self._encode_state_into(ct, features)
        return features

    # ---- Action execution ----

    _PLACE_HARV = 5.0

    def _execute_build(
        self,
        ct: Controller,
        action: str,
        direction: Direction,
        conv_direction: Direction,
    ) -> tuple[float, float]:
        """Execute build action. Returns (harv_reward, conv_reward).

        If the target tile has a friendly non-core building, auto-destroys it
        first (destroy has no cooldown cost). Skips if we can't afford the build.
        """
        if action == "none":
            return 0.0, 0.0

        pos = ct.get_position()
        build_pos = pos.add(direction)

        # Bounds check — build_pos can be out of map when near edges
        w, h = ct.get_map_width(), ct.get_map_height()
        if build_pos.x < 0 or build_pos.x >= w or build_pos.y < 0 or build_pos.y >= h:
            return 0.0, 0.0

        # Auto-destroy friendly non-core building to make room.
        # Check affordability first to avoid destroying for nothing.
        bid = ct.get_tile_building_id(build_pos)
        btype = None
        if bid is not None:
            try:
                btype = ct.get_entity_type(bid)
                if ct.get_team(bid) == self.my_team:
                    if action == "conveyor" and btype == EntityType.CONVEYOR and ct.get_direction(bid) == conv_direction:
                        return 0.0, 0.0
                    if action == "armoured_conveyor" and btype == EntityType.ARMOURED_CONVEYOR and ct.get_direction(bid) == conv_direction:
                        return 0.0, 0.0
                if ct.get_team(bid) == self.my_team and btype != EntityType.CORE:
                    idx = BUILD_ACTIONS.index(action)
                    base_ti, base_ax = BUILD_COSTS[idx]
                    scale = ct.get_scale_percent() / 100.0
                    ti, ax = ct.get_global_resources()
                    if ti >= int(scale * base_ti) and ax >= int(scale * base_ax):
                        ct.destroy(build_pos)
                        bid, btype = None, None
                    else:
                        return 0.0, 0.0
            except Exception:
                pass

        if action == "conveyor":
            if ct.can_build_conveyor(build_pos, conv_direction):
                ct.build_conveyor(build_pos, conv_direction)
                Player._conv_count += 1
        elif action == "harvester":
            self._dbg_harv_attempts = 1
            if ct.can_build_harvester(build_pos):
                ct.build_harvester(build_pos)
                ore_key = (build_pos.x, build_pos.y)
                if ore_key in Player._harv_placed_ores:
                    return 0.0, 0.0
                Player._harv_placed_ores.add(ore_key)
                return self._PLACE_HARV, 0.0
            self._dbg_harv_failures = 1
            if self._should_reward_harvester_proximity(ct, build_pos, bid, btype):
                self._dbg_harv_prox_gate = 1
                reward = self._proximity_reward(ct, build_pos)
                if reward > 0.0:
                    self._dbg_harv_prox_paid = 1
                return reward, 0.0
            return 0.0, 0.0
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
            if ct.can_build_splitter(build_pos, direction):
                ct.build_splitter(build_pos, direction)
        elif action == "armoured_conveyor":
            if ct.can_build_armoured_conveyor(build_pos, conv_direction):
                ct.build_armoured_conveyor(build_pos, conv_direction)
                Player._conv_count += 1
        elif action == "barrier":
            if ct.can_build_barrier(build_pos):
                ct.build_barrier(build_pos)
        elif action == "foundry":
            if ct.can_build_foundry(build_pos):
                ct.build_foundry(build_pos)
        elif action == "launcher":
            if ct.can_build_launcher(build_pos):
                ct.build_launcher(build_pos)
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


    def _flush_log(self) -> None:
        """Append trajectory log to file specified by TENSOR_TRAIN_LOG (pickle).

        Uses append mode so multiple builder instances' trajectories
        are all preserved in the same file as separate pickle objects.
        """
        if not self._train_log or not _TRAIN_LOG:
            return
        try:
            import pickle
            with open(_TRAIN_LOG, "ab") as f:
                pickle.dump(self._train_log, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        self._train_log = []
        _LIVE_LOG_PLAYERS.discard(self)


def _compute_harvester_build_dir_mask_from_obs(obs: list[float], bd_mask: list[bool]) -> list[bool]:
    has_visible_ore = False
    for tile in range(GRID_SIZE * GRID_SIZE):
        ts = tile * TILE_FEATURES
        if obs[ts + 1] == 1.0 or obs[ts + 2] == 1.0:
            has_visible_ore = True
            break

    harv_bd_mask = [False] * 9
    for d_idx in range(9):
        if not bd_mask[d_idx]:
            continue
        ts = _BUILD_DIR_TILE_INDICES[d_idx] * TILE_FEATURES
        is_ore = obs[ts + 1] == 1.0 or obs[ts + 2] == 1.0
        empty = obs[ts + 4] == 1.0
        harv_bd_mask[d_idx] = (is_ore and empty) or ((not is_ore) and has_visible_ore)
    return harv_bd_mask


def _compute_conditional_build_dir_mask_from_obs(
    obs: list[float],
    build_type_idx: int,
    bd_mask: list[bool],
    harv_bd_mask: list[bool] | None = None,
) -> list[bool]:
    if build_type_idx != _HARVESTER_BUILD_TYPE:
        return bd_mask
    if harv_bd_mask is None:
        harv_bd_mask = _compute_harvester_build_dir_mask_from_obs(obs, bd_mask)
    return harv_bd_mask


def _compute_masks_from_obs(obs: list[float]) -> tuple[list[bool], list[bool], list[bool]]:
    """Compute build_type and build_dir action masks.

    Build directions allow both empty tiles and tiles with friendly non-core
    buildings (the execution layer auto-destroys before building).

    Returns (bt_mask, bd_mask, harv_bd_mask) as lists of booleans (True = allowed).
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
        no_unit = obs[ts + 15] == 0.0
        empty = obs[ts + 4] == 1.0
        replaceable = obs[ts + 4] == 0.0 and obs[ts + 12] == 1.0 and obs[ts + 11] == 0.0
        if not_wall and no_unit and (empty or replaceable):
            bd_mask[d_idx] = True
            any_dir = True

    harv_bd_mask = _compute_harvester_build_dir_mask_from_obs(obs, bd_mask)
    if not any(harv_bd_mask):
        bt_mask[_HARVESTER_BUILD_TYPE] = False

    # If nothing affordable or nowhere to build, force "none" (dir irrelevant)
    if not any_affordable or not any_dir:
        return [True] + [False] * (len(BUILD_COSTS) - 1), [True] * 9, [True] * 9

    return bt_mask, bd_mask, harv_bd_mask


def _argmax(data: list[float], mask: list[bool] | None = None) -> int:
    """C-friendly argmax using max() with key, with optional mask."""
    if mask is not None:
        masked = [d if m else float('-inf') for d, m in zip(data, mask)]
        return max(range(len(masked)), key=masked.__getitem__)
    return max(range(len(data)), key=data.__getitem__)


def _sample_from_logits(
    logits: list[float], mask: list[bool] | None = None,
) -> tuple[int, float]:
    """Sample action from logits using softmax. Returns (index, log_prob).

    If mask is provided, masked (False) actions get -1e9 logits.
    """
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
