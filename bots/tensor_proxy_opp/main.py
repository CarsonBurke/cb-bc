"""Opponent proxy bot for batched GPU inference training.

Same as tensor_proxy but:
- Reads TENSOR_OBS_WFD_B / TENSOR_ACT_RFD_B pipe env vars
- No trajectory logging (opponent doesn't train)
- Executes the exact same shared action semantics as side A
"""

from array import array
import os
import random
import struct
import sys

_TENSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor")
sys.path.insert(0, _TENSOR_DIR)

from shared import (
    GRID_SIZE, GRID_HALF, TILE_FEATURES, GLOBAL_FEATURES,
    INPUT_SIZE, HIDDEN_SIZES, H_DIM,
    DIRECTIONS, BUILD_ACTIONS, DIR_MAP_9, CONV_DIR_4, DIR_DXY, BUILD_COSTS,
    _CONV_DIR_MASK,
    GameState,
    discover_ore, scan_enemy_core_dmg, scan_own_core_dmg,
    encode_state_into,
    execute_build, execute_move,
)

from cambc import Controller, Direction, EntityType, Environment, Position

# Struct-based binary IPC protocol
_OBS_FLOATS = INPUT_SIZE + H_DIM
_OBS_BYTES = _OBS_FLOATS * 4
_ACT_HEAD = struct.Struct('=iiiii')
_ACT_INTS = 5  # bt, bd, cd, brt, md
_ACT_B_FLOATS = H_DIM  # h_next only (no log_prob for opponent)
_ACT_B_SIZE = _ACT_INTS * 4 + _ACT_B_FLOATS * 4
_H_BYTES = H_DIM * 4

_opened_pipes = {}


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
    bt, bd, cd, brt, md = _ACT_HEAD.unpack_from(data, 0)
    h_next = array("f")
    h_next.frombytes(data[_ACT_INTS * 4:_ACT_INTS * 4 + _H_BYTES])
    return bt, bd, cd, brt, md, h_next


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.pos_encoder = None
        self.h_prev = array("f", [0.0] * H_DIM)
        self._obs_buf = array("f")
        self._prev_collected = None
        self._n_builders_cached = 1
        self._spawn_round: int = -1
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
        if self._spawn_round < 0:
            self._spawn_round = ct.get_current_round()
        if self.my_team is None:
            self.my_team = ct.get_team()
        if self.core_center is None:
            self._find_core(ct)
        obs = self._encode_state_buf(ct)
        obs_wfd, act_rfd = _get_pipes()

        if obs_wfd is None or act_rfd is None:
            raise RuntimeError("tensor_proxy_opp missing inference pipes")
        _send_obs(obs_wfd, obs, self.h_prev)
        result = _recv_action(act_rfd)
        if result is None:
            raise RuntimeError("tensor_proxy_opp inference pipe closed")
        build_type_idx, build_dir_idx, conv_dir_idx, bridge_target_idx, move_dir_idx, h_next = result
        self.h_prev = h_next

        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        conv_dir = CONV_DIR_4[conv_dir_idx]
        execute_build(ct, build_action, build_dir, conv_dir, self.my_team, bridge_target_idx)

        move_dir = DIR_MAP_9[move_dir_idx]
        execute_move(ct, move_dir)


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

    def _encode_state_buf(self, ct):
        discover_ore(ct)
        scan_enemy_core_dmg(ct)
        scan_own_core_dmg(ct)
        del self._obs_buf[:]
        n_builders_out = [0]
        result = encode_state_into(
            ct, self._obs_buf, self.core_center, self.enemy_core,
            self.my_team, self.pos_encoder, self._prev_collected,
            self._spawn_round, n_builders_out,
        )
        if result is not None:
            self.core_center, self.enemy_core, self._prev_collected = result
        self._n_builders_cached = n_builders_out[0]
        return self._obs_buf
