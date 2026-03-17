"""Proxy bot for batched GPU inference training.

Instead of running the NN locally, this bot:
1. Encodes game state via Controller
2. Sends observations to the inference server via inherited pipe FDs
3. Blocks waiting for action response
4. Executes received actions
5. Logs trajectory data

Pipe FDs passed via env vars: TENSOR_OBS_WFD (write obs), TENSOR_ACT_RFD (read actions)
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
    ENV_IDX, BLD_IDX, N_BLD_TYPES, DIR_DXY, _argmax, _sample_from_logits,
)
from tensor_lib import SinusoidalPositionEmbedding

_TRAIN_LOG = os.environ.get("TENSOR_TRAIN_LOG")

# Pipe connections opened lazily per-player
_opened_pipes = {}  # (obs_wfd, act_rfd) -> (obs_file, act_file)


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


def _send_obs(obs_wfd, game_id, unit_id, obs, h_prev):
    """Send observation to inference server via pipe fd."""
    data = pickle.dumps((game_id, unit_id, obs, h_prev), protocol=pickle.HIGHEST_PROTOCOL)
    os.write(obs_wfd, struct.pack("!I", len(data)) + data)


def _recv_action(act_rfd):
    """Receive action from inference server via pipe fd."""
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
        self.num_builders = 0
        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)
        self.h_prev = [0.0] * H_DIM
        self._prev_resources = None
        self._prev_enemy_core_hp = 500.0
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

        # Snapshot resources
        ti_before, ax_before = ct.get_global_resources()

        # Execute actions (reuse tensor bot methods)
        from main import Player as _P
        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        _p = _P.__new__(_P)
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p._n_builders_cached = self._n_builders_cached
        _p._harv_rewarded_ores = self._harv_rewarded_ores
        harv_reward, conv_reward = _P._execute_build(_p, ct, build_action, build_dir)
        self._harv_rewarded_ores = _p._harv_rewarded_ores

        move_dir = DIR_MAP_9[move_dir_idx]
        _p2 = _P.__new__(_P)
        _p2.core_center = self.core_center
        _P._execute_move(_p2, ct, move_dir)

        # Compute rewards
        ti_after, ax_after = ct.get_global_resources()
        if self._prev_resources is None:
            ti_delivered, ax_delivered = 0.0, 0.0
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

        self._prev_reward = harv_reward * 5.0 + conv_reward * 2.0 + max(ti_delivered, 0) * 0.2 + enemy_dmg * 1.0
        self._prev_resources = (ti_after, ax_after)

        if self._train_log is not None:
            self._train_log.append((
                obs, list(self.h_prev),
                build_type_idx, build_dir_idx, move_dir_idx,
                value, ti_delivered, ax_delivered, ti_after,
                harv_reward, enemy_dmg, old_log_prob, conv_reward,
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

    def _flush_log(self):
        if not self._train_log or not _TRAIN_LOG:
            return
        try:
            with open(_TRAIN_LOG, "ab") as f:
                pickle.dump(self._train_log, f)
        except Exception:
            pass
        self._train_log = []
