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

# Import shared constants and encoding from tensor/shared.py
_TENSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor")
sys.path.insert(0, _TENSOR_DIR)

from shared import (
    INPUT_SIZE, HIDDEN_SIZES, H_DIM, RMSNORM_LAYERS, COMPRESS_DIM,
    DIRECTIONS, BUILD_ACTIONS, DIR_MAP_9, CONV_DIR_4,
    encode_state_into, argmax,
    execute_build,
    N_BLD_TYPES, BRIDGE_OFFSETS,
    TILE_FEATURES, GRID_SIZE,
)

# Old-architecture constants for frozen opponent weights
POS_EMBED_DIM = 8

import math as _math

class SinusoidalPositionEmbedding:
    __slots__ = ("d_model", "freqs", "_quarter")
    def __init__(self, d_model: int, base: float = 10000.0):
        self.d_model = d_model
        self._quarter = d_model // 4
        self.freqs = [1.0 / (base ** (2.0 * i / d_model)) for i in range(self._quarter)]
    def encode(self, x: int, y: int) -> list[float]:
        q, freqs = self._quarter, self.freqs
        vec = [0.0] * self.d_model
        half = q * 2
        for i in range(q):
            angle = x * freqs[i]
            vec[2 * i] = _math.sin(angle)
            vec[2 * i + 1] = _math.cos(angle)
        for i in range(q):
            angle = y * freqs[i]
            vec[half + 2 * i] = _math.sin(angle)
            vec[half + 2 * i + 1] = _math.cos(angle)
        return vec

# Import MLP from tensor_inference's tensor_lib
_INFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor_inference")
sys.path.insert(0, _INFERENCE_DIR)
from tensor_lib import ActorModel

from cambc import Controller, Direction, EntityType, Environment, Position

_MODEL_CACHE = {}


def _weight_cache_key(path: str) -> tuple[str, int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return path, st.st_mtime_ns, st.st_size


def _get_shared_model(path: str):
    cache_key = _weight_cache_key(path)
    if cache_key is not None and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model = ActorModel(
        output_heads={
            "build_type": len(BUILD_ACTIONS),
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": len(BRIDGE_OFFSETS),
            "move_dir": 9,
            "logit_scale": 5,
            "h_gate": H_DIM,
            "h_cand": H_DIM,
        },
    )
    model.eval()

    if path and os.path.exists(path):
        try:
            if path.endswith(".pkl"):
                import pickle
                with open(path, "rb") as f:
                    model.load(pickle.load(f))
            else:
                with open(path) as f:
                    model.load(json.load(f))
        except Exception as e:
            raise RuntimeError(f"tensor_opp failed to load weights from {path}: {e}") from e

    if cache_key is not None:
        _MODEL_CACHE[cache_key] = model
    return model


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.num_builders = 0
        self._prev_collected = None
        self._n_builders_cached = 1
        self._spawn_round: int = -1

        self.pos_encoder = SinusoidalPositionEmbedding(POS_EMBED_DIM)
        self.h_prev = [0.0] * H_DIM

        opp_path = os.environ.get("TENSOR_OPP_WEIGHTS", "")
        self.model = _get_shared_model(opp_path)

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

        # minGRU: gated recurrent update
        h_gate_raw = outputs["h_gate"]
        h_cand = outputs["h_cand"]
        h_in = self.h_prev
        h_new = [0.0] * len(h_in)
        for i in range(len(h_in)):
            g = h_gate_raw[i]
            if g >= 0:
                z = 1.0 / (1.0 + math.exp(-g))
            else:
                e = math.exp(g)
                z = e / (1.0 + e)
            h_new[i] = (1.0 - z) * h_in[i] + z * h_cand[i]
        self.h_prev = h_new

        build_type_idx = argmax(outputs["build_type"])
        build_dir_idx = argmax(outputs["build_dir"])
        conv_dir_idx = argmax(outputs["conv_dir"])
        bridge_target_idx = argmax(outputs["bridge_target"])
        move_dir_idx = argmax(outputs["move_dir"])

        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        self._execute_build(ct, build_action, build_dir, conv_dir_idx, bridge_target_idx)

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
        features: list[float] = []
        n_builders_out = [0]
        result = encode_state_into(
            ct, features, self.core_center, self.enemy_core,
            self.my_team, self.pos_encoder, self._prev_collected,
            self._spawn_round, n_builders_out,
        )
        if result is not None:
            self.core_center, self.enemy_core, self._prev_collected = result
        self._n_builders_cached = n_builders_out[0]
        return features

    def _execute_build(self, ct, action, direction, conv_dir_idx=0, bridge_target_idx=0):
        conv_dir = CONV_DIR_4[conv_dir_idx]
        execute_build(ct, action, direction, conv_dir, self.my_team, bridge_target_idx)

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
