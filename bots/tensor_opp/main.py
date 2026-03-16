"""Opponent bot for self-play training.

Same architecture as tensor bot but:
  - Loads weights from TENSOR_OPP_WEIGHTS env var (not weights.json)
  - Never logs trajectories
  - Pure inference, no training hooks
"""

import json
import math
import os
import random
import sys

# Reuse all code from tensor bot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor"))

from cambc import Controller, Direction, EntityType, Environment, Position
from tensor_lib import MLP, SinusoidalPositionEmbedding

# Import constants and helpers from tensor bot
from main import (
    GRID_SIZE, GRID_HALF, TILE_FEATURES, GLOBAL_FEATURES, POS_EMBED_DIM,
    INPUT_SIZE, HIDDEN_SIZES, H_DIM, DIRECTIONS, BUILD_ACTIONS, DIR_MAP_9,
    ENV_IDX, BLD_IDX, N_BLD_TYPES, DIR_DXY, _argmax,
)


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.num_builders = 0
        self._prev_reward = 0.0
        self._n_builders_cached = 1

        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)

        self.model = MLP(
            input_size=INPUT_SIZE + H_DIM,
            hidden_sizes=HIDDEN_SIZES,
            output_heads={
                "build_type": len(BUILD_ACTIONS),
                "build_dir": 9,
                "move_dir": 9,
                "value": 1,
                "h_next": H_DIM,
            },
            activation="silu",
        )
        self.model.eval()
        self.h_prev = [0.0] * H_DIM

        # Load opponent weights from env var
        opp_path = os.environ.get("TENSOR_OPP_WEIGHTS", "")
        if opp_path and os.path.exists(opp_path):
            try:
                with open(opp_path) as f:
                    self.model.load(json.load(f))
            except Exception:
                pass

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
        features = obs + self.h_prev
        outputs = self.model.infer(features)

        h_raw = outputs["h_next"]
        self.h_prev = [math.tanh(x) for x in h_raw]

        build_type_idx = _argmax(outputs["build_type"])
        build_dir_idx = _argmax(outputs["build_dir"])
        move_dir_idx = _argmax(outputs["move_dir"])

        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        self._execute_build(ct, build_action, build_dir)

        move_dir = DIR_MAP_9[move_dir_idx]
        self._execute_move(ct, move_dir)

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
        return self.core_center or ct.get_position()

    def _encode_state(self, ct):
        _P, _p = self._proxy()
        result = _P._encode_state(_p, ct)
        # Sync back cached values
        self._n_builders_cached = _p._n_builders_cached
        return result

    def _proxy(self):
        from main import Player as _P
        _p = _P.__new__(_P)
        _p.core_center = self.core_center
        _p.enemy_core = self.enemy_core
        _p.my_team = self.my_team
        _p.pos_encoder = self.pos_encoder
        _p._prev_reward = self._prev_reward
        _p._n_builders_cached = self._n_builders_cached
        return _P, _p

    def _execute_build(self, ct, action, direction):
        _P, _p = self._proxy()
        _P._execute_build(_p, ct, action, direction)

    def _execute_move(self, ct, direction):
        _P, _p = self._proxy()
        _P._execute_move(_p, ct, direction)
