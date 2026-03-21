"""Proxy bot for batched GPU inference training.

Sends observations over pipes to the PyTorch inference server,
receives actions back. Uses shared.py for obs encoding, masks,
rewards, and constants — no tensor_lib dependency.
"""

import atexit
from array import array
import os
import pickle
import random
import struct
import sys

_TENSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor")
sys.path.insert(0, _TENSOR_DIR)

from shared import (
    GRID_SIZE, GRID_HALF, TILE_FEATURES, GLOBAL_FEATURES, POS_EMBED_DIM,
    INPUT_SIZE, HIDDEN_SIZES, H_DIM,
    DIRECTIONS, BUILD_ACTIONS, DIR_MAP_9, DIR_DXY, BUILD_COSTS,
    _CONV_DIR_MASK,
    SinusoidalPositionEmbedding, GameState,
    discover_ore, scan_enemy_core_dmg, scan_own_core_dmg,
    encode_state_into,
    execute_build, execute_move,
    sample_from_logits, argmax,
)

from cambc import Controller, Direction, EntityType, Environment, Position

_TRAIN_LOG = os.environ.get("TENSOR_TRAIN_LOG")

# Struct-based binary IPC protocol
_OBS_FLOATS = INPUT_SIZE + H_DIM  # obs + h_prev
_OBS_BYTES = _OBS_FLOATS * 4
_ACT_HEAD = struct.Struct('=iiii')
_ACT_INTS = 4  # bt, bd, cd, md
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
    bt, bd, cd, md = _ACT_HEAD.unpack_from(data, 0)
    h_next = array("f")
    h_next.frombytes(data[_ACT_INTS * 4:_ACT_INTS * 4 + _H_BYTES])
    old_lp = _LP_STRUCT.unpack_from(data, _ACT_INTS * 4 + _H_BYTES)[0]
    return bt, bd, cd, md, 0.0, h_next, old_lp


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

        discover_ore(ct)
        scan_enemy_core_dmg(ct)
        scan_own_core_dmg(ct)

        obs = self._encode_state_buf(ct)

        obs_wfd, act_rfd = _get_pipes()
        h_in = self.h_prev

        if obs_wfd is not None:
            _send_obs(obs_wfd, obs, h_in)
            result = _recv_action(act_rfd)
            if result is not None:
                build_type_idx, build_dir_idx, conv_dir_idx, move_dir_idx, value, h_next, old_log_prob = result
                self.h_prev = h_next
            else:
                build_type_idx = random.randint(0, len(BUILD_ACTIONS) - 1)
                build_dir_idx = random.randint(0, 8)
                conv_dir_idx = random.choice([i for i, ok in enumerate(_CONV_DIR_MASK) if ok])
                move_dir_idx = random.randint(0, 8)
                value, old_log_prob = 0.0, 0.0
        else:
            build_type_idx = random.randint(0, len(BUILD_ACTIONS) - 1)
            build_dir_idx = random.randint(0, 8)
            conv_dir_idx = random.choice([i for i, ok in enumerate(_CONV_DIR_MASK) if ok])
            move_dir_idx = random.randint(0, 8)
            value, old_log_prob = 0.0, 0.0

        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        conv_dir = DIR_MAP_9[conv_dir_idx]
        harv_reward, conv_reward, dbg = execute_build(ct, build_action, build_dir, conv_dir, self.my_team)

        move_dir = DIR_MAP_9[move_dir_idx]
        execute_move(ct, move_dir)

        ti_after, ax_after = ct.get_global_resources()
        self._prev_resources = (ti_after, ax_after)

        if self._train_log is not None:
            self._train_log.append((
                obs.tolist(), list(h_in),
                build_type_idx, build_dir_idx, conv_dir_idx, move_dir_idx,
                value, 0, conv_reward, ti_after,
                harv_reward, GameState.enemy_core_dmg, old_log_prob,
                dbg[0], dbg[1], dbg[2], dbg[3],
                dbg[4], dbg[5], dbg[6], dbg[7],
                GameState.own_core_dmg,
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

    def _encode_state_buf(self, ct):
        discover_ore(ct)
        scan_enemy_core_dmg(ct)
        scan_own_core_dmg(ct)
        del self._obs_buf[:]
        n_builders_out = [0]
        result = encode_state_into(
            ct, self._obs_buf, self.core_center, self.enemy_core,
            self.my_team, self.pos_encoder, self._prev_resources,
            self._spawn_round, n_builders_out,
        )
        if result is not None:
            self.core_center, self.enemy_core = result
        self._n_builders_cached = n_builders_out[0]
        return self._obs_buf

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
