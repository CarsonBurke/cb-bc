"""Opponent proxy bot for batched GPU inference training.

Same as tensor_proxy but:
- Reads shared collector env vars for side B
- No trajectory logging (opponent doesn't train)
- Executes the exact same shared action semantics as side A
"""

from array import array
import os
import random
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
from shared_collector import CollectorClient

from cambc import Controller, Direction, EntityType, Environment, Position

_collector = None
_unit_id_counter = 1


def _get_collector():
    global _collector
    if _collector is None:
        _collector = CollectorClient.from_env("b")
    return _collector


def _next_unit_id() -> int:
    global _unit_id_counter
    unit_id = _unit_id_counter
    _unit_id_counter += 1
    return unit_id


class Player:
    def __init__(self):
        self.core_center = None
        self.enemy_core = None
        self.my_team = None
        self.num_spawned = 0
        self.pos_encoder = None
        self._obs_buf = array("f")
        self._prev_collected = None
        self._n_builders_cached = 1
        self._spawn_round: int = -1
        self._unit_id = _next_unit_id()

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
        collector = _get_collector()
        if collector is None:
            raise RuntimeError("tensor_proxy_opp missing collector transport")
        result = collector.request(self._unit_id, ct.get_current_round(), obs)
        if result is None:
            raise RuntimeError("tensor_proxy_opp collector transport closed")
        build_type_idx, build_dir_idx, conv_dir_idx, bridge_target_idx, move_dir_idx = result

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
