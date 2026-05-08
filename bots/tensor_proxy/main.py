"""Proxy bot for batched GPU inference training."""

from array import array
import os
import random
import sys

_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_TENSOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tensor")
sys.path.insert(0, _TENSOR_DIR)

from shared import (
    GRID_SIZE, GRID_HALF, TILE_FEATURES, GLOBAL_FEATURES,
    INPUT_SIZE, HIDDEN_SIZES, H_DIM,
    DIRECTIONS, BUILD_ACTIONS, DIR_MAP_9, CONV_DIR_4, DIR_DXY, BUILD_COSTS,
    _CONV_DIR_MASK,
    GameState,
    discover_ore, scan_enemy_core_dmg, scan_own_core_dmg,
    compute_masks_from_obs, encode_state_into,
    execute_build, execute_move,
)
from shared_collector import CollectorClient

from cambc import Controller, Direction, EntityType, Environment, Position

_collector = None
_unit_id_counter = 1
_FALLBACK_BUILD_DIR_ORDER = (1, 5, 7, 3, 2, 8, 6, 0)
_FALLBACK_MOVE_DIR_ORDER = (1, 5, 7, 3, 2, 8, 6, 0)


def _training_turn_cap() -> int | None:
    raw = os.environ.get("TENSOR_TRAIN_MAX_TURNS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid TENSOR_TRAIN_MAX_TURNS: {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"TENSOR_TRAIN_MAX_TURNS must be > 0, got {value}")
    return value


def _rotate_order(order: tuple[int, ...], offset: int) -> tuple[int, ...]:
    if not order:
        return ()
    shift = offset % len(order)
    if shift == 0:
        return order
    return order[shift:] + order[:shift]


def _dir_tile_index(dir_idx: int) -> int:
    dx, dy = DIR_DXY[DIR_MAP_9[dir_idx]]
    return (GRID_HALF + int(dy)) * GRID_SIZE + (GRID_HALF + int(dx))


def _dir_has_ore(obs, dir_idx: int) -> bool:
    ts = _dir_tile_index(dir_idx) * TILE_FEATURES
    return obs[ts + 1] > 0.5 or obs[ts + 2] > 0.5


def _scripted_fallback_actions(obs, *, unit_id: int, round_idx: int) -> tuple[int, int, int, int, int]:
    bt_mask, bd_mask = compute_masks_from_obs(obs)
    build_order = _rotate_order(_FALLBACK_BUILD_DIR_ORDER, unit_id)
    move_order = _rotate_order(_FALLBACK_MOVE_DIR_ORDER, unit_id + round_idx)

    if bt_mask[1]:
        for dir_idx in build_order:
            if bd_mask[dir_idx] and _dir_has_ore(obs, dir_idx):
                return 1, dir_idx, 0, 0, move_order[0]

    if bt_mask[3]:
        for dir_idx in build_order:
            if bd_mask[dir_idx]:
                return 3, dir_idx, 0, 0, move_order[0]

    return 0, 4, 0, 0, move_order[0]


def _get_collector():
    global _collector
    if _collector is None:
        _collector = CollectorClient.from_env("a")
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
        self.num_builders = 0
        self.pos_encoder = None  # kept for encode_state_into signature
        self._obs_buf = array("f")
        self._prev_collected = None
        self._n_builders_cached = 1
        self._spawn_round: int = -1
        self._unit_id = _next_unit_id()
        self._dbg_flags: int = 0  # packed dbg from previous action
        self._training_turn_cap = _training_turn_cap()
        self._has_collected_step = False
        self._post_horizon_flushed = False

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
        round_idx = ct.get_current_round()
        if self._training_turn_cap is not None and round_idx >= self._training_turn_cap:
            self._run_builder_fallback(ct, round_idx)
            return

        discover_ore(ct)
        scan_enemy_core_dmg(ct)
        scan_own_core_dmg(ct)

        obs = self._encode_state_buf(ct)

        collector = _get_collector()
        if collector is None:
            raise RuntimeError("tensor_proxy missing actor collector transport")
        result = collector.request(self._unit_id, round_idx, obs, self._dbg_flags)
        if result is None:
            raise RuntimeError("tensor_proxy actor collector transport closed")
        self._has_collected_step = True
        build_type_idx, build_dir_idx, conv_dir_idx, bridge_target_idx, move_dir_idx = result

        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        conv_dir = CONV_DIR_4[conv_dir_idx]
        _harv_aux, _conv_aux, dbg = execute_build(ct, build_action, build_dir, conv_dir, self.my_team, bridge_target_idx)
        self._dbg_flags = (
            dbg[0] | (dbg[1] << 1) | (dbg[2] << 2) | (dbg[3] << 3)
            | (dbg[4] << 4) | (dbg[5] << 5) | (dbg[6] << 6) | (dbg[7] << 7)
        )

        move_dir = DIR_MAP_9[move_dir_idx]
        execute_move(ct, move_dir)

    def _run_builder_fallback(self, ct, round_idx: int) -> None:
        obs = self._encode_state_buf(ct)
        if (
            self._training_turn_cap is not None
            and round_idx == self._training_turn_cap
            and self._has_collected_step
            and not self._post_horizon_flushed
        ):
            collector = _get_collector()
            if collector is None:
                raise RuntimeError("tensor_proxy missing actor collector transport")
            result = collector.request(self._unit_id, round_idx, obs, self._dbg_flags)
            if result is None:
                raise RuntimeError("tensor_proxy actor collector transport closed")
            self._post_horizon_flushed = True
        build_type_idx, build_dir_idx, conv_dir_idx, bridge_target_idx, move_dir_idx = _scripted_fallback_actions(
            obs,
            unit_id=self._unit_id,
            round_idx=round_idx,
        )
        build_action = BUILD_ACTIONS[build_type_idx]
        build_dir = DIR_MAP_9[build_dir_idx]
        conv_dir = CONV_DIR_4[conv_dir_idx]
        _harv_aux, _conv_aux, dbg = execute_build(
            ct,
            build_action,
            build_dir,
            conv_dir,
            self.my_team,
            bridge_target_idx,
        )
        self._dbg_flags = (
            dbg[0] | (dbg[1] << 1) | (dbg[2] << 2) | (dbg[3] << 3)
            | (dbg[4] << 4) | (dbg[5] << 5) | (dbg[6] << 6) | (dbg[7] << 7)
        )
        execute_move(ct, DIR_MAP_9[move_dir_idx])

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
