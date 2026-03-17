"""Proxy bot for batched GPU inference training.

Instead of running the NN locally, this bot:
1. Encodes game state via Controller
2. Sends observations to the inference server via inherited pipe FDs
3. Blocks waiting for action response
4. Executes received actions
5. Logs trajectory data

Pipe FDs passed via env vars: TENSOR_OBS_WFD (write obs), TENSOR_ACT_RFD (read actions)
"""

import importlib.util
import math
import os
import pickle
import random
import struct
import sys

_TENSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor")
sys.path.insert(0, _TENSOR_DIR)
_TENSOR_MAIN_PATH = os.path.join(_TENSOR_DIR, "main.py")
_TENSOR_SPEC = importlib.util.spec_from_file_location("_tensor_main_proxy", _TENSOR_MAIN_PATH)
_TENSOR_MAIN = importlib.util.module_from_spec(_TENSOR_SPEC)
_TENSOR_SPEC.loader.exec_module(_TENSOR_MAIN)

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
_argmax = _TENSOR_MAIN._argmax
_sample_from_logits = _TENSOR_MAIN._sample_from_logits
_TensorPlayer = _TENSOR_MAIN.Player

_TRAIN_LOG = os.environ.get("TENSOR_TRAIN_LOG")

# Struct-based binary IPC protocol (replaces pickle for speed)
_OBS_FLOATS = INPUT_SIZE + H_DIM  # obs + h_prev
_OBS_STRUCT = struct.Struct(f'={_OBS_FLOATS}f')
_ACT_INTS = 3  # bt, bd, md
_ACT_FLOATS = H_DIM + 1  # h_next + old_lp
_ACT_SIZE = _ACT_INTS * 4 + _ACT_FLOATS * 4
_ACT_STRUCT = struct.Struct(f'={_ACT_INTS}i{_ACT_FLOATS}f')

# Pipe connections opened lazily per-player
_opened_pipes = {}


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
    data = b""
    while len(data) < n:
        chunk = os.read(fd, n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _send_obs(obs_wfd, game_id, unit_id, obs, h_prev):
    """Send observation as struct-packed binary via pipe fd."""
    payload = _OBS_STRUCT.pack(*obs, *h_prev)
    os.write(obs_wfd, struct.pack('=I', len(payload)) + payload)


def _recv_action(act_rfd):
    """Receive struct-packed action from inference server."""
    raw = _read_exact(act_rfd, 4)
    if raw is None:
        return None
    length = struct.unpack('=I', raw)[0]
    data = _read_exact(act_rfd, length)
    if data is None:
        return None
    vals = _ACT_STRUCT.unpack(data)
    bt, bd, md = vals[0], vals[1], vals[2]
    h_next = list(vals[3:3 + H_DIM])
    old_lp = vals[3 + H_DIM]
    return bt, bd, md, 0.0, h_next, old_lp


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.num_builders = 0
        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)
        self.h_prev = [0.0] * H_DIM
        self._prev_resources = None
        self._n_builders_cached = 1
        self._prev_reward = 0.0
        self._harv_rewarded_ores = set()
        self._train_log = [] if _TRAIN_LOG else None
        self._unit_id = id(self) % (2**31)  # Unique per player instance

    def run(self, ct):
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    def _run_core(self, ct):
        if self.core_center is None:
            self.core_center = ct.get_position()
        _TensorPlayer._discover_ore(ct)
        _TensorPlayer._scan_working_harvesters(ct)
        _TensorPlayer._scan_enemy_core_dmg(ct)
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
        if self.my_team is None:
            self.my_team = ct.get_team()
        if self.core_center is None:
            self._find_core(ct)
        if self.enemy_core is None:
            for eid in ct.get_nearby_entities():
                try:
                    if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) != self.my_team:
                        self.enemy_core = ct.get_position(eid)
                        break
                except Exception:
                    continue

        obs = self._encode_state(ct)

        obs_wfd, act_rfd = _get_pipes()
        game_id = int(os.environ.get("TENSOR_GAME_ID", "0"))

        if obs_wfd is not None:
            # Send obs to inference server, block for action
            _send_obs(obs_wfd, game_id, self._unit_id, obs, self.h_prev)
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
            # No pipe — random fallback
            build_type_idx = random.randint(0, len(BUILD_ACTIONS) - 1)
            build_dir_idx = random.randint(0, 8)
            move_dir_idx = random.randint(0, 8)
            value, old_log_prob = 0.0, 0.0

        # Execute actions
        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        harv_reward, conv_reward = self._execute_build(ct, build_action, build_dir)

        move_dir = DIR_MAP_9[move_dir_idx]
        self._execute_move(ct, move_dir)

        # Global rewards (class-level, accumulated from all units' vision)
        enemy_dmg = _TensorPlayer._enemy_core_dmg
        n_working = _TensorPlayer._working_harv_count

        self._prev_reward = harv_reward + conv_reward + enemy_dmg * 20.0 + n_working * 0.5
        ti_after, ax_after = ct.get_global_resources()
        self._prev_resources = (ti_after, ax_after)

        if self._train_log is not None:
            # Tuple: obs, h, bt, bd, md, value, n_working, conv_r, ti, harv_r, dmg, old_lp
            self._train_log.append((
                obs, list(self.h_prev),
                build_type_idx, build_dir_idx, move_dir_idx,
                value, n_working, conv_reward, ti_after,
                harv_reward, enemy_dmg, old_log_prob,
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
        """Get or create a reusable tensor Player shell for shared helper methods."""
        if not hasattr(self, '_proxy'):
            self._proxy = _new_proxy_player()
        _p = self._proxy
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p.pos_encoder = self.pos_encoder
        _p._prev_reward = self._prev_reward
        _p._n_builders_cached = self._n_builders_cached
        _p._prev_resources = self._prev_resources
        _p._harv_rewarded_ores = self._harv_rewarded_ores
        return _p

    def _sync_from_proxy(self, _p):
        self._n_builders_cached = _p._n_builders_cached
        self._harv_rewarded_ores = _p._harv_rewarded_ores

    def _encode_state(self, ct):
        _TensorPlayer._discover_ore(ct)
        _TensorPlayer._scan_working_harvesters(ct)
        _TensorPlayer._scan_enemy_core_dmg(ct)
        _p = self._get_proxy()
        result = _TensorPlayer._encode_state(_p, ct)
        self._sync_from_proxy(_p)
        return result

    def _execute_build(self, ct, action, direction):
        _p = self._get_proxy()
        result = _TensorPlayer._execute_build(_p, ct, action, direction)
        self._sync_from_proxy(_p)
        return result

    def _execute_move(self, ct, direction):
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
                pickle.dump(self._train_log, f)
        except Exception:
            pass
        self._train_log = []
