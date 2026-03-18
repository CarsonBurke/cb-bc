"""Opponent proxy bot for batched GPU inference training.

Same as tensor_proxy but:
- Reads TENSOR_OBS_WFD_B / TENSOR_ACT_RFD_B pipe env vars
- No trajectory logging (opponent doesn't train)
- Uses argmax (greedy) actions from server response
"""

from array import array
import importlib.util
import math
import os
import random
import struct
import sys

_TENSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor")
sys.path.insert(0, _TENSOR_DIR)
_TENSOR_MAIN_PATH = os.path.join(_TENSOR_DIR, "main.py")
_TENSOR_SPEC = importlib.util.spec_from_file_location("_tensor_main_proxy_opp", _TENSOR_MAIN_PATH)
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
_TensorPlayer = _TENSOR_MAIN.Player

# Struct-based binary IPC protocol
_OBS_FLOATS = INPUT_SIZE + H_DIM
_OBS_BYTES = _OBS_FLOATS * 4
_ACT_HEAD = struct.Struct('=iii')
_ACT_INTS = 3  # bt, bd, md
_ACT_B_FLOATS = H_DIM  # h_next only (no log_prob for opponent)
_ACT_B_SIZE = _ACT_INTS * 4 + _ACT_B_FLOATS * 4
_H_BYTES = H_DIM * 4

_opened_pipes = {}


def _new_proxy_player():
    """Create a bare tensor Player without model construction or weight I/O."""
    return object.__new__(_TensorPlayer)


def _get_pipes():
    obs_wfd = int(os.environ.get("TENSOR_OBS_WFD_B", "-1"))
    act_rfd = int(os.environ.get("TENSOR_ACT_RFD_B", "-1"))
    if obs_wfd < 0 or act_rfd < 0:
        return None, None
    key = (obs_wfd, act_rfd)
    if key not in _opened_pipes:
        _opened_pipes[key] = (obs_wfd, act_rfd)
    return obs_wfd, act_rfd


def _read_exact(fd, n):
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
    os.writev(
        obs_wfd,
        (
            memoryview(obs).cast("B"),
            memoryview(h_prev).cast("B"),
        ),
    )


def _recv_action(act_rfd):
    data = _read_exact(act_rfd, _ACT_B_SIZE)
    if data is None:
        return None
    bt, bd, md = _ACT_HEAD.unpack_from(data, 0)
    h_next = array("f")
    h_next.frombytes(data[12:12 + _H_BYTES])
    return bt, bd, md, h_next


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)
        self.h_prev = array("f", [0.0] * H_DIM)
        self._obs_buf = array("f")
        self._prev_resources = None
        self._n_builders_cached = 1
        self._unit_id = id(self) % (2**31)

    def run(self, ct):
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    def _run_core(self, ct):
        if self.core_center is None:
            self.core_center = ct.get_position()
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
        obs = self._encode_state_buf(ct)
        obs_wfd, act_rfd = _get_pipes()

        if obs_wfd is not None:
            _send_obs(obs_wfd, obs, self.h_prev)
            result = _recv_action(act_rfd)
            if result is not None:
                build_type_idx, build_dir_idx, move_dir_idx, h_next = result
                self.h_prev = h_next
            else:
                build_type_idx = random.randint(0, len(BUILD_ACTIONS) - 1)
                build_dir_idx = random.randint(0, 8)
                move_dir_idx = random.randint(0, 8)
        else:
            build_type_idx = random.randint(0, len(BUILD_ACTIONS) - 1)
            build_dir_idx = random.randint(0, 8)
            move_dir_idx = random.randint(0, 8)

        # Execute actions directly
        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        self._execute_build(ct, build_action, build_dir)

        move_dir = DIR_MAP_9[move_dir_idx]
        self._execute_move(ct, move_dir)

        # Update resource snapshot for delta features
        self._prev_resources = ct.get_global_resources()

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

    def _execute_build(self, ct, action, direction):
        pos = ct.get_position()
        build_pos = pos.add(direction)
        if action == "conveyor":
            conv_dir = direction
            if self.core_center:
                conv_dir = build_pos.direction_to(self.core_center)
                if conv_dir == Direction.CENTRE:
                    conv_dir = direction
            if ct.can_build_conveyor(build_pos, conv_dir):
                ct.build_conveyor(build_pos, conv_dir)
        elif action == "harvester":
            if ct.can_build_harvester(build_pos):
                ct.build_harvester(build_pos)
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
        elif action == "barrier":
            if ct.can_build_barrier(build_pos):
                ct.build_barrier(build_pos)
        elif action == "foundry":
            if ct.can_build_foundry(build_pos):
                ct.build_foundry(build_pos)
        elif action == "launcher":
            if ct.can_build_launcher(build_pos):
                ct.build_launcher(build_pos)
        elif action == "destroy":
            try:
                ct.destroy(build_pos)
            except Exception:
                pass

    def _execute_move(self, ct, direction):
        if direction == Direction.CENTRE:
            return
        if ct.can_move(direction):
            ct.move(direction)
            return
        for alt in (direction.rotate_left(), direction.rotate_right()):
            if ct.can_move(alt):
                ct.move(alt)
                return

    def _encode_state(self, ct):
        _TensorPlayer._discover_ore(ct)
        if not hasattr(self, '_proxy'):
            self._proxy = _new_proxy_player()
        _p = self._proxy
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p.pos_encoder = self.pos_encoder
        _p._n_builders_cached = self._n_builders_cached
        _p._prev_resources = self._prev_resources
        result = _TensorPlayer._encode_state(_p, ct)
        self._n_builders_cached = _p._n_builders_cached
        return result

    def _encode_state_buf(self, ct):
        _TensorPlayer._discover_ore(ct)
        _TensorPlayer._scan_working_harvesters(ct)
        _TensorPlayer._scan_enemy_core_dmg(ct)
        if not hasattr(self, '_proxy'):
            self._proxy = _new_proxy_player()
        _p = self._proxy
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p.pos_encoder = self.pos_encoder
        _p._n_builders_cached = self._n_builders_cached
        _p._prev_resources = self._prev_resources
        del self._obs_buf[:]
        _TensorPlayer._encode_state_into(_p, ct, self._obs_buf)
        self.enemy_core = _p.enemy_core
        self._n_builders_cached = _p._n_builders_cached
        return self._obs_buf
