"""Opponent proxy bot for batched GPU inference training.

Same as tensor_proxy but:
- Reads TENSOR_OBS_WFD_B / TENSOR_ACT_RFD_B pipe env vars
- No trajectory logging (opponent doesn't train)
- Uses argmax (greedy) actions from server response
"""

import math
import os
import pickle
import random
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor"))

from cambc import Controller, Direction, EntityType, Environment, Position

from main import (
    GRID_SIZE, GRID_HALF, TILE_FEATURES, GLOBAL_FEATURES, POS_EMBED_DIM,
    INPUT_SIZE, HIDDEN_SIZES, H_DIM, DIRECTIONS, BUILD_ACTIONS, DIR_MAP_9,
    ENV_IDX, BLD_IDX, N_BLD_TYPES, DIR_DXY, _argmax,
)
from tensor_lib import SinusoidalPositionEmbedding

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


def _send_obs(obs_wfd, game_id, unit_id, obs, h_prev):
    data = pickle.dumps((game_id, unit_id, obs, h_prev), protocol=pickle.HIGHEST_PROTOCOL)
    os.write(obs_wfd, struct.pack("!I", len(data)) + data)


def _recv_action(act_rfd):
    raw = b""
    while len(raw) < 4:
        chunk = os.read(act_rfd, 4 - len(raw))
        if not chunk:
            return None
        raw += chunk
    length = struct.unpack("!I", raw)[0]
    data = b""
    while len(data) < length:
        chunk = os.read(act_rfd, length - len(data))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)
        self.h_prev = [0.0] * H_DIM
        self._prev_reward = 0.0
        self._n_builders_cached = 1
        self._harv_rewarded_ores = set()
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
        if self.num_spawned < 3:
            reserve_floor = 150 + 50 * self.num_spawned
        else:
            reserve_floor = 250 + 75 * (self.num_spawned - 3)
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
            _send_obs(obs_wfd, game_id, self._unit_id, obs, self.h_prev)
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

        # Execute actions
        from main import Player as _P
        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        _p = _P.__new__(_P)
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p._n_builders_cached = self._n_builders_cached
        _p._harv_rewarded_ores = self._harv_rewarded_ores
        _P._execute_build(_p, ct, build_action, build_dir)

        move_dir = DIR_MAP_9[move_dir_idx]
        _p2 = _P.__new__(_P)
        _p2.core_center = self.core_center
        _P._execute_move(_p2, ct, move_dir)

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

    def _encode_state(self, ct):
        from main import Player as _P
        _p = _P.__new__(_P)
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p.pos_encoder = self.pos_encoder
        _p._prev_reward = self._prev_reward
        _p._n_builders_cached = self._n_builders_cached
        result = _P._encode_state(_p, ct)
        self._n_builders_cached = _p._n_builders_cached
        return result
