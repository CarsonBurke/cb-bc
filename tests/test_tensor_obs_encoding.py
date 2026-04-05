"""Regression tests for tensor stdlib observation encoding."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tensor_main = _load_module("tensor_inference_main_for_obs_tests", "bots/tensor_inference/main.py")


Direction = tensor_main.Direction
EntityType = tensor_main.EntityType
Environment = tensor_main.Environment
Position = tensor_main.Position
ResourceType = tensor_main.ResourceType


class FakeController:
    def __init__(self):
        self.pos = Position(10, 10)
        self.team = 1
        self.round = 800
        self.width = 30
        self.height = 30
        self.ti = 200.0
        self.ax = 50.0
        self.scale = 150.0

        self.nearby_tiles = []
        self.tile_env = {}
        self.tile_passable = {}
        self.tile_building = {}
        self.tile_builder = {}
        self.entities = {}
        self.nearby_entities = []

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                pos = Position(self.pos.x + dx, self.pos.y + dy)
                self.nearby_tiles.append(pos)
                env = Environment.EMPTY
                if (dx, dy) == (-1, -1):
                    env = Environment.WALL
                elif (dx, dy) == (-2, 0):
                    env = Environment.ORE_TITANIUM
                elif (dx, dy) == (1, 0):
                    env = Environment.ORE_AXIONITE
                self.tile_env[(pos.x, pos.y)] = env
                self.tile_passable[(pos.x, pos.y)] = env != Environment.WALL and (dx, dy) != (2, -1)

        self._add_entity(
            101,
            EntityType.CONVEYOR,
            self.team,
            Direction.EAST,
            None,
            90.0,
            Position(10, 9),
            as_building=True,
        )
        self._add_entity(
            102,
            EntityType.HARVESTER,
            2,
            Direction.SOUTH,
            ResourceType.TITANIUM,
            80.0,
            Position(11, 10),
            as_building=True,
        )
        self._add_entity(
            201,
            EntityType.BUILDER_BOT,
            self.team,
            Direction.NORTH,
            None,
            50.0,
            Position(10, 10),
            as_unit=True,
        )
        self._add_entity(
            202,
            EntityType.BUILDER_BOT,
            2,
            Direction.WEST,
            None,
            50.0,
            Position(8, 11),
            as_unit=True,
        )
        self._add_entity(
            900,
            EntityType.CORE,
            self.team,
            Direction.CENTRE,
            None,
            470.0,
            Position(8, 10),
        )
        self._add_entity(
            901,
            EntityType.CORE,
            2,
            Direction.CENTRE,
            None,
            420.0,
            Position(15, 12),
        )

    def _add_entity(
        self,
        eid: int,
        etype,
        team: int,
        direction,
        stored,
        hp: float,
        pos: Position,
        *,
        as_building: bool = False,
        as_unit: bool = False,
    ) -> None:
        self.entities[eid] = {
            "etype": etype,
            "team": team,
            "direction": direction,
            "stored": stored,
            "hp": hp,
            "pos": pos,
        }
        self.nearby_entities.append(eid)
        key = (pos.x, pos.y)
        if as_building:
            self.tile_building[key] = eid
        if as_unit:
            self.tile_builder[key] = eid

    def get_position(self, eid=None):
        return self.pos if eid is None else self.entities[eid]["pos"]

    def get_team(self, eid=None):
        return self.team if eid is None else self.entities[eid]["team"]

    def get_current_round(self):
        return self.round

    def get_map_width(self):
        return self.width

    def get_map_height(self):
        return self.height

    def get_global_resources(self):
        return self.ti, self.ax

    def get_scale_percent(self):
        return self.scale

    def get_nearby_tiles(self, dist_sq=None):
        return self.nearby_tiles

    def get_nearby_entities(self, dist_sq=None):
        return self.nearby_entities

    def get_tile_env(self, pos):
        return self.tile_env[(pos.x, pos.y)]

    def is_tile_passable(self, pos):
        return self.tile_passable[(pos.x, pos.y)]

    def get_tile_building_id(self, pos):
        return self.tile_building.get((pos.x, pos.y))

    def get_tile_builder_bot_id(self, pos):
        return self.tile_builder.get((pos.x, pos.y))

    def get_entity_type(self, eid=None):
        return EntityType.BUILDER_BOT if eid is None else self.entities[eid]["etype"]

    def get_direction(self, eid):
        return self.entities[eid]["direction"]

    def get_stored_resource(self, eid):
        return self.entities[eid]["stored"]

    def get_hp(self, eid=None):
        return 50.0 if eid is None else self.entities[eid]["hp"]


def _make_player():
    tensor_main.Player._all_ore = None
    tensor_main.Player._all_ore_keys = set()
    tensor_main.Player._enemy_core_hp = 500.0
    tensor_main.Player._own_core_hp = 500.0
    tensor_main.Player._enemy_core_dmg = 0.0
    tensor_main.Player._enemy_core_dmg_round = -1
    tensor_main.Player._working_harv_positions = set()
    tensor_main.Player._working_harv_round = -1
    tensor_main.Player._working_harv_count = 0
    player = tensor_main.Player()
    player.my_team = 1
    player._prev_collected = (150.0, 20.0)
    player._spawn_round = 120
    return player


def _tile_start(dx: int, dy: int) -> int:
    return ((dy + tensor_main.GRID_HALF) * tensor_main.GRID_SIZE + (dx + tensor_main.GRID_HALF)) * tensor_main.TILE_FEATURES


def _reference_obs(player, ct: FakeController) -> list[float]:
    features: list[float] = []
    pos = ct.get_position()
    my_team = player.my_team
    visible_tiles = {(tile_pos.x, tile_pos.y): tile_pos for tile_pos in ct.get_nearby_tiles()}
    for dx, dy in tensor_main._LOCAL_TILE_OFFSETS:
        tx, ty = pos.x + dx, pos.y + dy
        if tx < 0 or tx >= ct.get_map_width() or ty < 0 or ty >= ct.get_map_height():
            features.extend(tensor_main._ZERO_TILE)
            continue
        tp = visible_tiles.get((tx, ty))
        if tp is None:
            features.extend(tensor_main._ZERO_TILE)
            continue

        env = ct.get_tile_env(tp)
        features.append(1.0 if env == Environment.WALL else 0.0)
        features.append(1.0 if env == Environment.ORE_TITANIUM else 0.0)
        features.append(1.0 if env == Environment.ORE_AXIONITE else 0.0)
        features.append(0.0 if ct.is_tile_passable(tp) else 1.0)

        bid = ct.get_tile_building_id(tp)
        if bid is not None:
            btype = ct.get_entity_type(bid)
            bidx = tensor_main.BLD_IDX.get(btype, 3)
            features.extend(tensor_main._BLD_ONE_HOTS[bidx])
            features.append(1.0 if ct.get_team(bid) == my_team else -1.0)
            try:
                bdir = ct.get_direction(bid)
                bdx, bdy = tensor_main.DIR_DXY.get(bdir, (0, 0))
                features.append(float(bdx))
                features.append(float(bdy))
            except Exception:
                features.extend((0.0, 0.0))
            features.append(0.0)
            try:
                sr = ct.get_stored_resource(bid)
                features.append(0.0 if sr is None else 1.0 if sr == ResourceType.TITANIUM else -1.0)
            except Exception:
                features.append(0.0)
        else:
            features.extend(tensor_main._NO_BUILD_TILE)

        uid = ct.get_tile_builder_bot_id(tp)
        if uid is not None:
            features.append(1.0)
            features.append(1.0 if ct.get_team(uid) == my_team else -1.0)
        else:
            features.extend((0.0, 0.0))

    rnd = ct.get_current_round() / 2000.0
    x_norm = pos.x / max(ct.get_map_width(), 1)
    y_norm = pos.y / max(ct.get_map_height(), 1)
    ti, ax = ct.get_global_resources()
    scale = ct.get_scale_percent() / 500.0

    n_builders = 0
    own_hp, enemy_hp = tensor_main.Player._own_core_hp, tensor_main.Player._enemy_core_hp
    for eid in ct.get_nearby_entities():
        try:
            etype = ct.get_entity_type(eid)
            team = ct.get_team(eid)
            if etype == EntityType.BUILDER_BOT and team == my_team:
                n_builders += 1
            elif etype == EntityType.CORE:
                if team == my_team:
                    if player.core_center is None:
                        player.core_center = ct.get_position(eid)
                    own_hp = ct.get_hp(eid)
                else:
                    if player.enemy_core is None:
                        player.enemy_core = ct.get_position(eid)
                    enemy_hp = ct.get_hp(eid)
        except Exception:
            pass

    w, h = ct.get_map_width(), ct.get_map_height()

    core_dx, core_dy, core_known = 0.0, 0.0, 0.0
    if player.core_center is not None:
        core_dx = float(player.core_center.x - pos.x)
        core_dy = float(player.core_center.y - pos.y)
        core_known = 1.0

    ecore_dx, ecore_dy, ecore_known = 0.0, 0.0, 0.0
    if player.enemy_core is not None:
        ecore_dx = float(player.enemy_core.x - pos.x)
        ecore_dy = float(player.enemy_core.y - pos.y)
        ecore_known = 1.0

    ti_delta, ax_delta = 0.0, 0.0
    if player._prev_collected is not None:
        prev_ti, prev_ax = player._prev_collected
        ti_delta = (ti - prev_ti) / 500.0
        ax_delta = (ax - prev_ax) / 500.0

    hp_advantage = (own_hp - enemy_hp) / 500.0
    features.extend((
        rnd, x_norm, y_norm, ti / 500.0, ax / 500.0, scale,
        core_dx, core_dy, ecore_dx, ecore_dy,
        n_builders / 10.0,
        ti_delta, ax_delta, hp_advantage,
        player._spawn_round / 2000.0 if player._spawn_round >= 0 else 0.0,
        core_known, ecore_known,
        w / 50.0, h / 50.0,
        tensor_main.Player._working_harv_count / 20.0,
    ))
    return features


def test_prefetched_vs_internal_nearby_match():
    ct = FakeController()

    player1 = _make_player()
    with_prefetch = list(player1._encode_state(ct, ct.nearby_tiles, ct.nearby_entities))

    player2 = _make_player()
    without_prefetch = list(player2._encode_state(ct))

    assert without_prefetch == with_prefetch


def test_obs_matches_reference_layout_and_values():
    ct = FakeController()
    player = _make_player()

    expected = _reference_obs(_make_player(), ct)
    obs = list(player._encode_state(ct, ct.nearby_tiles, ct.nearby_entities))

    assert obs == expected


def test_obs_encodes_expected_tiles_and_globals():
    ct = FakeController()
    player = _make_player()

    obs = list(player._encode_state(ct, ct.nearby_tiles, ct.nearby_entities))

    NB = tensor_main.N_BLD_TYPES  # 10

    wall_idx = _tile_start(-1, -1)
    assert obs[wall_idx:wall_idx + 4] == [1.0, 0.0, 0.0, 1.0]
    assert obs[wall_idx + 4:wall_idx + 4 + NB] == list(tensor_main._BLD_ONE_HOTS[0])

    conv_idx = _tile_start(0, -1)
    assert obs[conv_idx:conv_idx + 4] == [0.0, 0.0, 0.0, 0.0]
    assert obs[conv_idx + 4:conv_idx + 4 + NB] == list(tensor_main._BLD_ONE_HOTS[2])
    assert obs[conv_idx + 4 + NB:conv_idx + 4 + NB + 5] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert obs[conv_idx + 4 + NB + 5:conv_idx + 4 + NB + 7] == [0.0, 0.0]

    harv_idx = _tile_start(1, 0)
    assert obs[harv_idx:harv_idx + 4] == [0.0, 0.0, 1.0, 0.0]
    assert obs[harv_idx + 4:harv_idx + 4 + NB] == list(tensor_main._BLD_ONE_HOTS[1])
    assert obs[harv_idx + 4 + NB:harv_idx + 4 + NB + 5] == [-1.0, 0.0, 1.0, 0.0, 1.0]

    centre_idx = _tile_start(0, 0)
    assert obs[centre_idx + 4 + NB + 5:centre_idx + 4 + NB + 7] == [1.0, 1.0]

    global_idx = tensor_main._GLOBAL_OBS_START
    globals_slice = obs[global_idx:global_idx + tensor_main.GLOBAL_FEATURES]
    ref = _reference_obs(_make_player(), ct)
    assert globals_slice == ref[tensor_main._GLOBAL_OBS_START:]


def test_symbolic_tile_actor_matches_generic_actor_on_valid_obs():
    ct = FakeController()
    player = _make_player()
    obs = list(player._encode_state(ct, ct.nearby_tiles, ct.nearby_entities))
    random.seed(11)
    h_prev = [random.uniform(-2.0, 2.0) for _ in range(tensor_main.H_DIM)]
    packed = obs + h_prev

    generic = tensor_main.ActorModel(
        output_heads={
            "build_type": len(tensor_main.BUILD_ACTIONS),
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": tensor_main.N_BRIDGE_TARGETS,
            "move_dir": 9,
            "temperature": 5,
            "h_gate": tensor_main.H_DIM,
            "h_cand": tensor_main.H_DIM,
        },
    )
    state = generic.save()
    generic.load(state)

    symbolic = tensor_main.ActorModel(
        output_heads={
            "build_type": len(tensor_main.BUILD_ACTIONS),
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": tensor_main.N_BRIDGE_TARGETS,
            "move_dir": 9,
            "temperature": 5,
            "h_gate": tensor_main.H_DIM,
            "h_cand": tensor_main.H_DIM,
        },
        assume_symbolic_tiles=True,
    )
    symbolic.load(state)

    generic_out = generic.infer(packed)
    symbolic_out = symbolic.infer(packed)

    for key, expected in generic_out.items():
        assert symbolic_out[key] == pytest.approx(expected, abs=1e-12)
