# tensor: Neural network-powered builder bot.
# Uses a lightweight MLP (32->16 hidden) to decide what to build, where to build,
# and where to move. Forward pass fits within ~0.7ms of the 2ms CPU budget.
# Supports online REINFORCE learning and offline weight loading.

import json
import math
import os
import random

from cambc import Controller, Direction, EntityType, Environment, Position

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tensor_lib import MLP, SinusoidalPositionEmbedding

# -- Layout constants --
GRID_SIZE = 5
GRID_HALF = GRID_SIZE // 2
# Per tile: is_wall(1) + ore_ti(1) + ore_ax(1)
#         + bld_type(8: none/harvester/conveyor/splitter/road/support/military/core)
#         + bld_friendly(1) + bld_dir(2)
#         + has_unit(1) + unit_friendly(1) = 16
TILE_FEATURES = 16
# Global: round, x, y, ti, ax, scale, core_dx, core_dy, enemy_core_dx, enemy_core_dy, n_builders, prev_reward = 12
GLOBAL_FEATURES = 12
POS_EMBED_DIM = 8
INPUT_SIZE = GRID_SIZE * GRID_SIZE * TILE_FEATURES + GLOBAL_FEATURES + POS_EMBED_DIM
HIDDEN_SIZES = [32, 32, 32, 16]
H_DIM = 16  # Recurrent hidden state size

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

# Weights file path (adjacent to main.py)
WEIGHT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")

# Training mode: set TENSOR_TRAIN_LOG to a file path to log trajectories
_TRAIN_LOG = os.environ.get("TENSOR_TRAIN_LOG")


class Player:
    def __init__(self):
        self.core_center: Position | None = None
        self.enemy_core: Position | None = None
        self.my_team = None
        self.num_spawned = 0
        self.num_builders = 0

        # Positional encoder (fixed sinusoidal)
        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)

        # Neural network with action heads + recurrent hidden state
        self.model = MLP(
            input_size=INPUT_SIZE + H_DIM,
            hidden_sizes=HIDDEN_SIZES,
            output_heads={
                "build_type": len(BUILD_ACTIONS),  # 12
                "build_dir": 9,
                "move_dir": 9,
                "value": 1,
                "h_next": H_DIM,
            },
            activation="silu",
        )
        self.model.eval()  # no dropout during game
        self.h_prev: list[float] = [0.0] * H_DIM

        # Load pre-trained weights if available
        if os.path.exists(WEIGHT_PATH):
            try:
                with open(WEIGHT_PATH) as f:
                    self.model.load(json.load(f))
            except Exception:
                pass

        # Trajectory logging for training
        self._train_log: list | None = [] if _TRAIN_LOG else None
        self._prev_resources: tuple[float, float] | None = None
        self._prev_enemy_core_hp: float = 500.0
        self._n_builders_cached: int = 1
        self._prev_reward: float = 0.0

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

        ti, ax = ct.get_global_resources()
        rnd = ct.get_current_round()

        # Reserve floor: always keep enough ti for builders to actually build.
        # Starts high (builders need resources for harvesters/conveyors),
        # then grows further with each additional builder.
        #   builder 1: need 150 ti (harvester=80 + conveyors)
        #   builder 2: need 200 ti
        #   builder 3: need 250 ti
        #   builder 4+: need 250 + 75 * (n - 3)
        if self.num_spawned < 3:
            reserve_floor = 150 + 50 * self.num_spawned
        else:
            reserve_floor = 250 + 75 * (self.num_spawned - 3)
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
        if self.my_team is None:
            self.my_team = ct.get_team()
        if self.core_center is None:
            self._find_core(ct)
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
        features = obs + self.h_prev

        # Fast inference (raw lists, no Tensor overhead)
        outputs = self.model.infer(features)

        # Update recurrent hidden state (tanh-bounded)
        h_raw = outputs["h_next"]
        self.h_prev = [math.tanh(x) for x in h_raw]

        # Decode each head via argmax (sample in training mode)
        if self._train_log is not None:
            build_type_idx, bt_lp = _sample_from_logits(outputs["build_type"])
            build_dir_idx, bd_lp = _sample_from_logits(outputs["build_dir"])
            move_dir_idx, md_lp = _sample_from_logits(outputs["move_dir"])
            old_log_prob = bt_lp + bd_lp + md_lp
        else:
            build_type_idx = _argmax(outputs["build_type"])
            build_dir_idx = _argmax(outputs["build_dir"])
            move_dir_idx = _argmax(outputs["move_dir"])

        # Snapshot resources before actions to isolate delivery from spending
        ti_before, ax_before = ct.get_global_resources()

        # Execute actions
        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        harv_reward, conv_reward = self._execute_build(ct, build_action, build_dir)

        move_dir = DIR_MAP_9[move_dir_idx]
        self._execute_move(ct, move_dir)

        # Compute reward signals (used for both prev_reward obs and training log)
        ti_after, ax_after = ct.get_global_resources()
        if self._prev_resources is None:
            ti_delivered = 0.0
            ax_delivered = 0.0
        else:
            prev_ti, prev_ax = self._prev_resources
            n_bld = max(1, self._n_builders_cached)
            ti_delivered = max(0.0, ti_before - prev_ti) / n_bld
            ax_delivered = max(0.0, ax_before - prev_ax) / n_bld

        enemy_dmg = 0.0
        if self.enemy_core is not None:
            for eid in ct.get_nearby_entities():
                try:
                    if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) != self.my_team:
                        hp = ct.get_hp(eid)
                        enemy_dmg = max(0.0, self._prev_enemy_core_hp - hp)
                        self._prev_enemy_core_hp = hp
                        break
                except Exception:
                    pass

        # Store reward for next step's obs (same formula as training)
        self._prev_reward = harv_reward * 5.0 + conv_reward * 2.0 + max(ti_delivered, 0) * 0.2 + enemy_dmg * 1.0
        self._prev_resources = (ti_after, ax_after)

        # Log trajectory step
        if self._train_log is not None:
            # Tuple: obs, h, bt, bd, md, value, ti_del, ax_del, ti, harv, dmg, old_lp, conv
            self._train_log.append((
                obs,
                list(self.h_prev),
                build_type_idx,
                build_dir_idx,
                move_dir_idx,
                outputs["value"][0],
                ti_delivered,
                ax_delivered,
                ti_after,
                harv_reward,
                enemy_dmg,
                old_log_prob,
                conv_reward,
            ))

            if ct.get_current_round() >= 1999:
                self._flush_log()

    _CARDINAL = (Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST)

    def _conveyor_chain_reward(self, ct: Controller, pos: Position, facing: Direction) -> float:
        """Reward for a conveyor placement based on chain connectivity.

        Rules:
        - Output must connect to a conveyor accepting from this side, or to core
        - Input must come from a harvester or conveyor pointing into this one
        - Must not point at a harvester (feed out, not in)
        - No direct loops (two conveyors pointing at each other)

        Returns 1.0 if both input+output connected, 0.5 if one side, 0.0 if isolated.
        """
        my_team = self.my_team
        output_pos = pos.add(facing)

        # --- Check output connectivity ---
        output_ok = False
        try:
            out_bid = ct.get_tile_building_id(output_pos)
            if out_bid is not None:
                out_type = ct.get_entity_type(out_bid)
                out_team = ct.get_team(out_bid)
                if out_team == my_team:
                    if out_type == EntityType.CORE:
                        output_ok = True
                    elif out_type in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.SPLITTER):
                        # Target conveyor accepts from non-output sides
                        # Our conveyor is on the opposite(facing) side of the target
                        target_dir = ct.get_direction(out_bid)
                        # Loop check: target points back at us
                        if target_dir != facing.opposite():
                            # Target accepts from all sides except its output direction
                            # We arrive from opposite(facing) side of target
                            # Target rejects if our arrival side == target's output dir
                            if facing.opposite() != target_dir:
                                output_ok = True
                    elif out_type == EntityType.HARVESTER:
                        # Pointing at a harvester = bad (feeding into harvester)
                        return 0.0
        except Exception:
            pass

        # --- Check input connectivity ---
        input_ok = False
        for card_dir in self._CARDINAL:
            if card_dir == facing:
                continue  # output side, not an input
            input_pos = pos.add(card_dir)
            try:
                in_bid = ct.get_tile_building_id(input_pos)
                if in_bid is None:
                    continue
                in_type = ct.get_entity_type(in_bid)
                in_team = ct.get_team(in_bid)
                if in_team != my_team:
                    continue
                if in_type == EntityType.HARVESTER:
                    # Harvester outputs to all adjacent conveyors
                    input_ok = True
                    break
                elif in_type in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                    in_dir = ct.get_direction(in_bid)
                    # This neighbor outputs toward us if in_dir points from input_pos to pos
                    if in_dir == card_dir.opposite():
                        # Loop check: we point back at it
                        if facing != card_dir:
                            input_ok = True
                            break
                elif in_type == EntityType.SPLITTER:
                    in_dir = ct.get_direction(in_bid)
                    # Splitter outputs to 3 front directions; check if one reaches us
                    # Splitter faces in_dir, outputs to in_dir, in_dir.rotate_left(), in_dir.rotate_right()
                    # It reaches us if card_dir.opposite() is one of those
                    opp = card_dir.opposite()
                    if opp in (in_dir, in_dir.rotate_left(), in_dir.rotate_right()):
                        input_ok = True
                        break
            except Exception:
                continue

        # --- Trace forward for loops (up to 8 hops) ---
        if output_ok:
            visited = {(pos.x, pos.y)}
            cur_pos, cur_dir = output_pos, None
            is_loop = False
            for _ in range(8):
                key = (cur_pos.x, cur_pos.y)
                if key in visited:
                    is_loop = True
                    break
                visited.add(key)
                try:
                    bid = ct.get_tile_building_id(cur_pos)
                    if bid is None:
                        break
                    btype = ct.get_entity_type(bid)
                    if btype == EntityType.CORE:
                        break  # Reached core — not a loop
                    if btype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                        cur_dir = ct.get_direction(bid)
                        cur_pos = cur_pos.add(cur_dir)
                    else:
                        break
                except Exception:
                    break
            if is_loop:
                return 0.0

        if output_ok and input_ok:
            return 1.0
        elif output_ok or input_ok:
            return 0.5
        return 0.0

    def _nearest_uncontested_ore_reward(self, ct: Controller) -> float:
        """Reward based on distance to nearest ore that doesn't already have
        a friendly harvester equally close or closer.

        Returns 1.0/(1+dist) for the best uncontested ore, or 0 if none.
        """
        pos = ct.get_position()
        my_team = self.my_team

        # Collect ore tiles and friendly harvester positions in vision
        ore_tiles = []
        harv_positions = []
        for tile_pos in ct.get_nearby_tiles():
            try:
                env = ct.get_tile_env(tile_pos)
                if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    ore_tiles.append(tile_pos)
                bid = ct.get_tile_building_id(tile_pos)
                if bid is not None and ct.get_entity_type(bid) == EntityType.HARVESTER \
                   and ct.get_team(bid) == my_team:
                    harv_positions.append(tile_pos)
            except Exception:
                pass

        best_dist = None
        for ore in ore_tiles:
            my_dist = max(abs(ore.x - pos.x), abs(ore.y - pos.y))
            # Skip if a friendly harvester is already on or closer to this ore
            contested = False
            for hp in harv_positions:
                h_dist = max(abs(ore.x - hp.x), abs(ore.y - hp.y))
                if h_dist <= my_dist:
                    contested = True
                    break
            if not contested and (best_dist is None or my_dist < best_dist):
                best_dist = my_dist

        if best_dist is None:
            return 0.0
        return 1.0 / (1.0 + best_dist)

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

    def _encode_state(self, ct: Controller) -> list[float]:
        """Encode 5x5 grid (12 feat/tile) + global (11) + pos embed (8)."""
        pos = ct.get_position()
        my_team = self.my_team
        features: list[float] = []

        # 5x5 local grid: 12 features per tile
        w, h = ct.get_map_width(), ct.get_map_height()
        for dy in range(-GRID_HALF, GRID_HALF + 1):
            for dx in range(-GRID_HALF, GRID_HALF + 1):
                tx, ty = pos.x + dx, pos.y + dy
                if tx < 0 or tx >= w or ty < 0 or ty >= h or not ct.is_in_vision(Position(tx, ty)):
                    features.extend([0.0] * TILE_FEATURES)
                    continue
                tp = Position(tx, ty)

                # Environment (3): is_wall, ore_ti, ore_ax
                env = ct.get_tile_env(tp)
                features.append(1.0 if env == Environment.WALL else 0.0)
                features.append(1.0 if env == Environment.ORE_TITANIUM else 0.0)
                features.append(1.0 if env == Environment.ORE_AXIONITE else 0.0)

                # Building type one-hot (4) + friendly (1) + dir (2) = 7
                bid = ct.get_tile_building_id(tp)
                if bid is not None:
                    btype = ct.get_entity_type(bid)
                    bidx = BLD_IDX.get(btype, 3)
                    features.extend(1.0 if i == bidx else 0.0 for i in range(N_BLD_TYPES))
                    features.append(1.0 if ct.get_team(bid) == my_team else -1.0)
                    try:
                        bdir = ct.get_direction(bid)
                        bdx, bdy = DIR_DXY.get(bdir, (0, 0))
                        features.append(float(bdx))
                        features.append(float(bdy))
                    except Exception:
                        features.extend([0.0, 0.0])
                else:
                    features.append(1.0)  # none category (index 0)
                    features.extend([0.0] * (N_BLD_TYPES - 1))
                    features.extend([0.0, 0.0, 0.0])  # friendly, dir_dx, dir_dy

                # Unit (2): has_unit, unit_friendly
                uid = ct.get_tile_builder_bot_id(tp)
                if uid is not None:
                    features.append(1.0)
                    features.append(1.0 if ct.get_team(uid) == my_team else -1.0)
                else:
                    features.extend([0.0, 0.0])

        # Global features (11 values)
        rnd = ct.get_current_round() / 2000.0
        w, h = ct.get_map_width(), ct.get_map_height()
        x_norm = pos.x / max(w, 1)
        y_norm = pos.y / max(h, 1)
        ti, ax = ct.get_global_resources()
        scale = ct.get_scale_percent() / 500.0

        # Direction to own core
        core_dx, core_dy = 0.0, 0.0
        if self.core_center is not None:
            cdx = self.core_center.x - pos.x
            cdy = self.core_center.y - pos.y
            d = max(abs(cdx), abs(cdy), 1)
            core_dx = cdx / d
            core_dy = cdy / d

        # Direction to enemy core
        ecore_dx, ecore_dy = 0.0, 0.0
        if self.enemy_core is not None:
            edx = self.enemy_core.x - pos.x
            edy = self.enemy_core.y - pos.y
            d = max(abs(edx), abs(edy), 1)
            ecore_dx = edx / d
            ecore_dy = edy / d

        # Count friendly builders in vision
        n_builders = 0
        for eid in ct.get_nearby_entities():
            try:
                if ct.get_entity_type(eid) == EntityType.BUILDER_BOT and ct.get_team(eid) == my_team:
                    n_builders += 1
            except Exception:
                pass
        self._n_builders_cached = n_builders

        features.extend([
            rnd, x_norm, y_norm, ti / 500.0, ax / 500.0, scale,
            core_dx, core_dy, ecore_dx, ecore_dy,
            n_builders / 10.0,
            self._prev_reward,
        ])

        # Sinusoidal position embedding (8 values)
        features.extend(self.pos_encoder.encode(pos.x, pos.y))

        return features

    # ---- Action execution ----

    def _execute_build(
        self, ct: Controller, action: str, direction: Direction
    ) -> tuple[float, float]:
        """Execute build action. Returns (harv_reward, conv_reward)."""
        if action == "none":
            return 0.0, 0.0

        pos = ct.get_position()
        build_pos = pos.add(direction)

        # Directional builds need the direction parameter
        if action == "conveyor":
            # Point conveyor toward core
            conv_dir = direction
            if self.core_center:
                conv_dir = build_pos.direction_to(self.core_center)
                if conv_dir == Direction.CENTRE:
                    conv_dir = direction
            if ct.can_build_conveyor(build_pos, conv_dir):
                ct.build_conveyor(build_pos, conv_dir)
                return 0.0, self._conveyor_chain_reward(ct, build_pos, conv_dir)
        elif action == "harvester":
            if ct.can_build_harvester(build_pos):
                ct.build_harvester(build_pos)
                return 1.0, 0.0
            else:
                return self._nearest_uncontested_ore_reward(ct), 0.0
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
            if ct.can_build_armoured_conveyor(build_pos, direction):
                ct.build_armoured_conveyor(build_pos, direction)
                return 0.0, self._conveyor_chain_reward(ct, build_pos, direction)
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


def _argmax(data: list[float]) -> int:
    """C-friendly argmax using max() with key."""
    return max(range(len(data)), key=data.__getitem__)


def _sample_from_logits(logits: list[float]) -> tuple[int, float]:
    """Sample action from logits using softmax. Returns (index, log_prob)."""
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
