"""Proxy bot for batched GPU inference training.

Uses regular import (not importlib) to load tensor/main.py to avoid
engine module tracking issues that cause 'NoneType' has no attr 'A' panics.
"""

import atexit
from array import array
import math
import os
import pickle
import random
import struct
import sys

_TENSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor")
sys.path.insert(0, _TENSOR_DIR)
# Use importlib with explicit file path to avoid circular import when the
# engine loads this file as module "main" (same name as tensor/main.py).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "tensor_bot_main", os.path.join(_TENSOR_DIR, "main.py"),
    submodule_search_locations=[],
)
_TENSOR_MAIN = _ilu.module_from_spec(_spec)
sys.modules["tensor_bot_main"] = _TENSOR_MAIN  # register to prevent re-import
_spec.loader.exec_module(_TENSOR_MAIN)

from cambc import Controller, Direction, EntityType, Environment, Position

from tensor_lib import SinusoidalPositionEmbedding

GRID_SIZE = _TENSOR_MAIN.GRID_SIZE
GRID_HALF = _TENSOR_MAIN.GRID_HALF
TILE_FEATURES = _TENSOR_MAIN.TILE_FEATURES
GLOBAL_FEATURES = _TENSOR_MAIN.GLOBAL_FEATURES
POS_EMBED_DIM = _TENSOR_MAIN.POS_EMBED_DIM
INPUT_SIZE = _TENSOR_MAIN.INPUT_SIZE
HIDDEN_SIZES = _TENSOR_MAIN.HIDDEN_SIZES
H_DIM = _TENSOR_MAIN.H_DIM
DIRECTIONS = _TENSOR_MAIN.DIRECTIONS
BUILD_ACTIONS = _TENSOR_MAIN.BUILD_ACTIONS
DIR_MAP_9 = _TENSOR_MAIN.DIR_MAP_9
ENV_IDX = _TENSOR_MAIN.ENV_IDX
BLD_IDX = _TENSOR_MAIN.BLD_IDX
N_BLD_TYPES = _TENSOR_MAIN.N_BLD_TYPES
DIR_DXY = _TENSOR_MAIN.DIR_DXY
BUILD_COSTS = _TENSOR_MAIN.BUILD_COSTS
_argmax = _TENSOR_MAIN._argmax
_sample_from_logits = _TENSOR_MAIN._sample_from_logits
_TensorPlayer = _TENSOR_MAIN.Player

_TRAIN_LOG = os.environ.get("TENSOR_TRAIN_LOG")

# Struct-based binary IPC protocol
_OBS_FLOATS = INPUT_SIZE + H_DIM  # obs + h_prev
_OBS_BYTES = _OBS_FLOATS * 4
_ACT_HEAD = struct.Struct('=iii')
_ACT_INTS = 3  # bt, bd, md
_ACT_FLOATS = H_DIM + 1  # h_next + old_lp
_ACT_SIZE = _ACT_INTS * 4 + _ACT_FLOATS * 4
_H_BYTES = H_DIM * 4
_LP_STRUCT = struct.Struct('=f')

# Pipe connections opened lazily per-player
_opened_pipes = {}
_live_log_players = set()


def _flush_live_logs():
    for player in list(_live_log_players):
        try:
            player._flush_log()
        except Exception:
            pass


atexit.register(_flush_live_logs)


def _new_proxy_player():
    """Create a bare tensor Player without model construction or weight I/O."""
    return object.__new__(_TensorPlayer)


def _get_pipes():
    """Get or open pipe file objects from env var FDs."""
    obs_wfd = int(os.environ.get("TENSOR_OBS_WFD", "-1"))
    act_rfd = int(os.environ.get("TENSOR_ACT_RFD", "-1"))
    if obs_wfd < 0 or act_rfd < 0:
        return None, None
    key = (obs_wfd, act_rfd)
    if key not in _opened_pipes:
        _opened_pipes[key] = (obs_wfd, act_rfd)
    return obs_wfd, act_rfd


def _read_exact(fd, n):
    """Read exactly n bytes from fd."""
    data = bytearray(n)
    view = memoryview(data)
    used = 0
    while used < n:
        chunk = os.read(fd, n - used)
        if not chunk:
            return None
        end = used + len(chunk)
        view[used:end] = chunk
        used = end
    return data


def _send_obs(obs_wfd, obs, h_prev):
    """Send raw float32 observation + hidden state via pipe fd."""
    os.writev(
        obs_wfd,
        (
            memoryview(obs).cast("B"),
            memoryview(h_prev).cast("B"),
        ),
    )


def _recv_action(act_rfd):
    """Receive fixed-size action packet from inference server."""
    data = _read_exact(act_rfd, _ACT_SIZE)
    if data is None:
        return None
    bt, bd, md = _ACT_HEAD.unpack_from(data, 0)
    h_next = array("f")
    h_next.frombytes(data[12:12 + _H_BYTES])
    old_lp = _LP_STRUCT.unpack_from(data, 12 + _H_BYTES)[0]
    return bt, bd, md, 0.0, h_next, old_lp


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.num_builders = 0
        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)
        self.h_prev = array("f", [0.0] * H_DIM)
        self._obs_buf = array("f")
        self._prev_resources = None
        self._n_builders_cached = 1
        self._spawn_round: int = -1
        self._harv_rewarded_ores = set()
        self._train_log = [] if _TRAIN_LOG else None
        self._unit_id = id(self) % (2**31)
        if self._train_log is not None:
            _live_log_players.add(self)

    def run(self, ct):
        try:
            etype = ct.get_entity_type()
            if etype == EntityType.CORE:
                self._run_core(ct)
            elif etype == EntityType.BUILDER_BOT:
                self._run_builder(ct)
        except Exception:
            pass

    def _run_core(self, ct):
        if self.core_center is None:
            self.core_center = ct.get_position()
        # NOTE: _discover_ore/_scan_* not called here to avoid cross-module
        # Controller calls before ct.spawn_builder (engine module tracking bug).
        ti, ax = ct.get_global_resources()
        reserve_floor = 150 * (self.num_spawned + 1)
        if ti < reserve_floor:
            return
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

    def _run_builder(self, ct):
        if self._spawn_round < 0:
            self._spawn_round = ct.get_current_round()
        if self.my_team is None:
            self.my_team = ct.get_team()
        if self.core_center is None:
            self._find_core(ct)
        obs = self._encode_state_buf(ct)

        obs_wfd, act_rfd = _get_pipes()
        h_in = self.h_prev

        if obs_wfd is not None:
            _send_obs(obs_wfd, obs, h_in)
            result = _recv_action(act_rfd)
            if result is not None:
                build_type_idx, build_dir_idx, move_dir_idx, value, h_next, old_log_prob = result
                self.h_prev = h_next
            else:
                build_type_idx = random.randint(0, len(BUILD_ACTIONS) - 1)
                build_dir_idx = random.randint(0, 8)
                move_dir_idx = random.randint(0, 8)
                value, old_log_prob = 0.0, 0.0
        else:
            build_type_idx = random.randint(0, len(BUILD_ACTIONS) - 1)
            build_dir_idx = random.randint(0, 8)
            move_dir_idx = random.randint(0, 8)
            value, old_log_prob = 0.0, 0.0

        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        harv_reward, conv_reward = self._execute_build(ct, build_action, build_dir)

        move_dir = DIR_MAP_9[move_dir_idx]
        self._execute_move(ct, move_dir)

        ti_after, ax_after = ct.get_global_resources()
        self._prev_resources = (ti_after, ax_after)

        if self._train_log is not None:
            self._train_log.append((
                obs.tolist(), list(h_in),
                build_type_idx, build_dir_idx, move_dir_idx,
                value, 0, conv_reward, ti_after,
                harv_reward, _TensorPlayer._enemy_core_dmg, old_log_prob,
            ))
            if ct.get_current_round() >= 1999:
                self._flush_log()

    def _find_core(self, ct):
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

    def _get_proxy(self):
        if not hasattr(self, '_proxy'):
            self._proxy = _new_proxy_player()
        _p = self._proxy
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p.pos_encoder = self.pos_encoder
        _p._n_builders_cached = self._n_builders_cached
        _p._prev_resources = self._prev_resources
        _p._spawn_round = self._spawn_round
        _p._harv_rewarded_ores = self._harv_rewarded_ores
        return _p

    def _sync_from_proxy(self, _p):
        self._n_builders_cached = _p._n_builders_cached
        self._harv_rewarded_ores = _p._harv_rewarded_ores
        self.enemy_core = _p.enemy_core

    def _encode_state(self, ct):
        _TensorPlayer._discover_ore(ct)
        _TensorPlayer._scan_enemy_core_dmg(ct)
        _p = self._get_proxy()
        result = _TensorPlayer._encode_state(_p, ct)
        self._sync_from_proxy(_p)
        return result

    def _encode_state_buf(self, ct):
        _TensorPlayer._discover_ore(ct)
        _TensorPlayer._scan_enemy_core_dmg(ct)
        _p = self._get_proxy()
        del self._obs_buf[:]
        _TensorPlayer._encode_state_into(_p, ct, self._obs_buf)
        self._sync_from_proxy(_p)
        return self._obs_buf

    def _execute_build(self, ct, action, direction):
        """Execute build directly (avoids cross-module Controller calls that can panic)."""
        if action == "none":
            return 0.0, 0.0

        _p = self._get_proxy()
        pos = ct.get_position()
        build_pos = pos.add(direction)
        w, h = ct.get_map_width(), ct.get_map_height()
        if build_pos.x < 0 or build_pos.x >= w or build_pos.y < 0 or build_pos.y >= h:
            return 0.0, 0.0

        bid = ct.get_tile_building_id(build_pos)
        if bid is not None:
            try:
                if ct.get_team(bid) == self.my_team and ct.get_entity_type(bid) != EntityType.CORE:
                    idx = BUILD_ACTIONS.index(action)
                    base_ti, base_ax = BUILD_COSTS[idx]
                    scale = ct.get_scale_percent() / 100.0
                    ti, ax = ct.get_global_resources()
                    if ti >= int(scale * base_ti) and ax >= int(scale * base_ax):
                        ct.destroy(build_pos)
                    else:
                        return 0.0, 0.0
            except Exception:
                pass

        if action == "conveyor":
            conv_dir = direction
            if self.core_center:
                conv_dir = build_pos.direction_to(self.core_center)
                if conv_dir == Direction.CENTRE:
                    conv_dir = direction
            if ct.can_build_conveyor(build_pos, conv_dir):
                ct.build_conveyor(build_pos, conv_dir)
                _TensorPlayer._conv_count += 1
        if action == "harvester":
            if ct.can_build_harvester(build_pos):
                ct.build_harvester(build_pos)
                ore_key = (build_pos.x, build_pos.y)
                given = _TensorPlayer._harv_proximity.get(ore_key, 0.0)
                _TensorPlayer._harv_proximity[ore_key] = _p._PLACE_HARV
                self._sync_from_proxy(_p)
                return _p._PLACE_HARV - given, 0.0
            reward = _TensorPlayer._proximity_reward(_p, ct)
            self._sync_from_proxy(_p)
            return reward, 0.0
        if action == "road":
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
                _TensorPlayer._conv_count += 1
        elif action == "barrier":
            if ct.can_build_barrier(build_pos):
                ct.build_barrier(build_pos)
        elif action == "foundry":
            if ct.can_build_foundry(build_pos):
                ct.build_foundry(build_pos)
        elif action == "launcher":
            if ct.can_build_launcher(build_pos):
                ct.build_launcher(build_pos)
        self._sync_from_proxy(_p)
        return 0.0, 0.0

    def _execute_move(self, ct, direction):
        if direction == Direction.CENTRE:
            return
        pos = ct.get_position()
        w, h = ct.get_map_width(), ct.get_map_height()
        move_pos = pos.add(direction)
        if 0 <= move_pos.x < w and 0 <= move_pos.y < h:
            if ct.can_build_road(move_pos):
                ct.build_road(move_pos)
        if ct.can_move(direction):
            ct.move(direction)
            return
        for alt in (direction.rotate_left(), direction.rotate_right()):
            alt_pos = pos.add(alt)
            if 0 <= alt_pos.x < w and 0 <= alt_pos.y < h:
                if ct.can_build_road(alt_pos):
                    ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return

    def _ore_penalty(self, ct):
        pos = ct.get_position()
        total_ore = 0
        penalty_sum = 0.0
        n_uncapped = 0
        n_idle = 0
        for tile_pos in ct.get_nearby_tiles():
            try:
                env = ct.get_tile_env(tile_pos)
                if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    continue
                total_ore += 1
                bid = ct.get_tile_building_id(tile_pos)
                if bid is not None and ct.get_entity_type(bid) == EntityType.HARVESTER \
                   and ct.get_team(bid) == self.my_team:
                    if ct.get_stored_resource(bid) is not None:
                        penalty_sum += 0.45
                        n_idle += 1
                else:
                    d = max(abs(tile_pos.x - pos.x), abs(tile_pos.y - pos.y))
                    penalty_sum += 0.5 + 0.5 * d / (d + 1.0)
                    n_uncapped += 1
            except Exception:
                continue
        if total_ore == 0:
            return 0.0, 0.0, 0.0
        return penalty_sum / total_ore, n_uncapped / total_ore, n_idle / total_ore

    def _flush_log(self):
        if not self._train_log or not _TRAIN_LOG:
            return
        try:
            with open(_TRAIN_LOG, "ab") as f:
                pickle.dump(self._train_log, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        self._train_log = []
        _live_log_players.discard(self)
