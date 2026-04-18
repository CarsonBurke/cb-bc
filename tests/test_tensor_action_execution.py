"""Regression tests for tensor stdlib action execution."""

from __future__ import annotations

import importlib.util
import math
from array import array
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tensor_main = _load_module("tensor_inference_main_for_action_tests", "bots/tensor_inference/main.py")
<<<<<<< Updated upstream
shared_mod = _load_module("tensor_shared_for_action_tests", "bots/tensor/shared.py")
INFER_DIR = ROOT / "bots" / "tensor_inference"
if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))
from tensor_lib.actor import _l2_normalize
=======
tensor_proxy_opp = _load_module("tensor_proxy_opp_for_action_tests", "bots/tensor_proxy_opp/main.py")
tensor_shared = _load_module("tensor_shared_for_action_tests", "bots/tensor/shared.py")
>>>>>>> Stashed changes


Direction = tensor_main.Direction
EntityType = tensor_main.EntityType
Position = tensor_main.Position


class FakeActionController:
    def __init__(self):
        self.pos = Position(10, 10)
        self.team = 1
        self.width = 30
        self.height = 30
        self.scale = 100.0
        self.ti = 200
        self.ax = 50
        self.buildings = {}
        self.moves_allowed = {}
        self.road_buildable = set()
        self.build_calls = []
        self.destroy_calls = []
        self.move_calls = []

    def set_building(self, pos, eid, etype, team, direction=Direction.CENTRE):
        self.buildings[(pos.x, pos.y)] = {
            "id": eid,
            "etype": etype,
            "team": team,
            "direction": direction,
        }

    def get_position(self, eid=None):
        if eid is None:
            return self.pos
        for info in self.buildings.values():
            if info["id"] == eid:
                return Position(*next(k for k, v in self.buildings.items() if v["id"] == eid))
        return self.pos

    def get_map_width(self):
        return self.width

    def get_map_height(self):
        return self.height

    def get_tile_building_id(self, pos):
        info = self.buildings.get((pos.x, pos.y))
        return None if info is None else info["id"]

    def get_team(self, eid=None):
        if eid is None:
            return self.team
        for info in self.buildings.values():
            if info["id"] == eid:
                return info["team"]
        return self.team

    def get_entity_type(self, eid=None):
        if eid is None:
            return EntityType.BUILDER_BOT
        for info in self.buildings.values():
            if info["id"] == eid:
                return info["etype"]
        return EntityType.BUILDER_BOT

    def get_direction(self, eid):
        for info in self.buildings.values():
            if info["id"] == eid:
                return info["direction"]
        return Direction.CENTRE

    def get_scale_percent(self):
        return self.scale

    def get_global_resources(self):
        return self.ti, self.ax

    def destroy(self, pos):
        self.destroy_calls.append(pos)
        self.buildings.pop((pos.x, pos.y), None)

    def can_build_conveyor(self, pos, direction):
        return True

    def build_conveyor(self, pos, direction):
        self.build_calls.append(("conveyor", pos, direction))

    def can_build_harvester(self, pos):
        return True

    def build_harvester(self, pos):
        self.build_calls.append(("harvester", pos))

    def can_build_road(self, pos):
        return (pos.x, pos.y) in self.road_buildable

    def build_road(self, pos):
        self.build_calls.append(("road", pos))

    def can_build_gunner(self, pos, direction):
        return True

    def build_gunner(self, pos, direction):
        self.build_calls.append(("gunner", pos, direction))

    def can_build_sentinel(self, pos, direction):
        return True

    def build_sentinel(self, pos, direction):
        self.build_calls.append(("sentinel", pos, direction))

    def can_build_breach(self, pos, direction):
        return True

    def build_breach(self, pos, direction):
        self.build_calls.append(("breach", pos, direction))

    def can_build_splitter(self, pos, direction):
        return True

    def build_splitter(self, pos, direction):
        self.build_calls.append(("splitter", pos, direction))

    def can_build_armoured_conveyor(self, pos, direction):
        return True

    def build_armoured_conveyor(self, pos, direction):
        self.build_calls.append(("armoured_conveyor", pos, direction))

    def can_build_barrier(self, pos):
        return True

    def build_barrier(self, pos):
        self.build_calls.append(("barrier", pos))

    def can_build_foundry(self, pos):
        return True

    def build_foundry(self, pos):
        self.build_calls.append(("foundry", pos))

    def can_build_launcher(self, pos):
        return True

    def build_launcher(self, pos):
        self.build_calls.append(("launcher", pos))

    def can_move(self, direction):
        return self.moves_allowed.get(direction, False)

    def move(self, direction):
        self.move_calls.append(direction)


def _make_player():
    tensor_main.Player._all_ore = None
    tensor_main.Player._all_ore_keys = set()
    tensor_main.Player._harv_proximity = {}
    player = tensor_main.Player()
    player.my_team = 1
    return player


def test_execute_build_skips_identical_conveyor_rebuild():
    player = _make_player()
    ct = FakeActionController()
    target = ct.pos.add(Direction.NORTH)
    ct.set_building(target, 100, EntityType.CONVEYOR, player.my_team, Direction.EAST)

    result = player._execute_build(ct, tensor_main._ACTION_CONVEYOR, Direction.NORTH, Direction.EAST)

    assert result == (0.0, 0.0)
    assert ct.destroy_calls == []
    assert ct.build_calls == []


def test_execute_build_replaces_affordable_friendly_building():
    player = _make_player()
    ct = FakeActionController()
    target = ct.pos.add(Direction.NORTH)
    ct.set_building(target, 101, EntityType.ROAD, player.my_team)

    player._execute_build(ct, tensor_main._ACTION_BARRIER, Direction.NORTH, None)

    assert ct.destroy_calls == [target]
    assert ct.build_calls == [("barrier", target)]


def test_execute_build_does_not_destroy_when_unaffordable():
    player = _make_player()
    ct = FakeActionController()
    ct.ti = 0
    target = ct.pos.add(Direction.NORTH)
    ct.set_building(target, 102, EntityType.ROAD, player.my_team)

    result = player._execute_build(ct, tensor_main._ACTION_FOUNDRY, Direction.NORTH, None)

    assert result == (0.0, 0.0)
    assert ct.destroy_calls == []
    assert ct.build_calls == []


def test_execute_move_builds_roads_and_uses_fallback():
    player = _make_player()
    ct = FakeActionController()
    primary = ct.pos.add(Direction.NORTH)
    fallback = ct.pos.add(Direction.NORTHEAST)
    ct.road_buildable = {(primary.x, primary.y), (fallback.x, fallback.y)}
    ct.moves_allowed = {
        Direction.NORTH: False,
        Direction.NORTHWEST: False,
        Direction.NORTHEAST: True,
    }

    player._execute_move(ct, Direction.NORTH)

    assert ct.build_calls == [("road", primary), ("road", fallback)]
    assert ct.move_calls == [Direction.NORTHEAST]


<<<<<<< Updated upstream
def _empty_obs():
    obs = [0.0] * shared_mod.INPUT_SIZE
    g = shared_mod.GRID_SIZE * shared_mod.GRID_SIZE * shared_mod.TILE_FEATURES
    obs[g + 3] = 10.0  # titanium
    obs[g + 4] = 10.0  # axionite
    obs[g + 5] = 0.2   # 100% build scale
    for dir_idx in range(9):
        ts = shared_mod._BUILD_DIR_TILE_INDICES[dir_idx] * shared_mod.TILE_FEATURES
        obs[ts + 4] = 1.0
    return obs


def test_select_greedy_actions_masks_harvester_when_chosen_tile_has_no_ore():
    obs = _empty_obs()
    ore_dir_idx = 3  # west
    ore_ts = shared_mod._BUILD_DIR_TILE_INDICES[ore_dir_idx] * shared_mod.TILE_FEATURES
    obs[ore_ts + 1] = 1.0

    outputs = {
        "build_type": [0.0] * len(shared_mod.BUILD_ACTIONS),
        "build_dir": [0.0] * 9,
        "conv_dir": [0.0] * 4,
        "bridge_target": [0.0] * len(shared_mod.BRIDGE_OFFSETS),
        "move_dir": [0.0] * 9,
    }
    outputs["build_type"][1] = 10.0  # harvester
    outputs["build_type"][3] = 5.0   # road fallback
    outputs["build_dir"][1] = 20.0   # north, no ore

    build_type_idx, build_dir_idx, *_ = shared_mod.select_greedy_actions_from_outputs(obs, outputs)

    assert build_dir_idx == 1
    assert build_type_idx == 3


def test_select_greedy_actions_keeps_harvester_when_chosen_tile_has_ore():
    obs = _empty_obs()
    ore_dir_idx = 1  # north
    ore_ts = shared_mod._BUILD_DIR_TILE_INDICES[ore_dir_idx] * shared_mod.TILE_FEATURES
    obs[ore_ts + 1] = 1.0

    outputs = {
        "build_type": [0.0] * len(shared_mod.BUILD_ACTIONS),
        "build_dir": [0.0] * 9,
        "conv_dir": [0.0] * 4,
        "bridge_target": [0.0] * len(shared_mod.BRIDGE_OFFSETS),
        "move_dir": [0.0] * 9,
    }
    outputs["build_type"][1] = 10.0
    outputs["build_type"][3] = 5.0
    outputs["build_dir"][ore_dir_idx] = 20.0

    build_type_idx, build_dir_idx, *_ = shared_mod.select_greedy_actions_from_outputs(obs, outputs)

    assert build_dir_idx == ore_dir_idx
    assert build_type_idx == 1


def test_actor_l2_normalize_matches_torch_clamped_norm_behavior():
    normalized = _l2_normalize([1e-8, 0.0], eps=1e-6)

    assert normalized[0] == pytest.approx(1e-2)
    assert normalized[1] == pytest.approx(0.0)
=======
def test_unimix_sampler_can_select_non_argmax(monkeypatch):
    monkeypatch.setattr(tensor_main.random, "random", lambda: 0.99)

    idx, log_prob = tensor_main._sample_unimix(
        [30.0, -30.0, -30.0],
        [True, True, True],
        eps=0.6,
    )

    assert idx in (1, 2)
    assert log_prob < 0.0


def test_opponent_proxy_uses_shared_execute_helpers(monkeypatch):
    calls = {}

    def fake_execute_build(ct, action, build_dir, conv_dir, my_team, bridge_target_idx):
        calls["build"] = (action, build_dir, conv_dir, my_team, bridge_target_idx)
        return 0.0, 0.0, (0, 0, 0, 0, 0, 0, 0, 0)

    def fake_execute_move(ct, move_dir):
        calls["move"] = move_dir

    class FakeProxyController:
        def __init__(self):
            self.team = 1

        def get_current_round(self):
            return 0

        def get_team(self, eid=None):
            return self.team

        def get_global_resources(self):
            return 0, 0

    monkeypatch.setattr(tensor_proxy_opp, "execute_build", fake_execute_build)
    monkeypatch.setattr(tensor_proxy_opp, "execute_move", fake_execute_move)
    monkeypatch.setattr(tensor_proxy_opp, "_get_pipes", lambda: (1, 2))
    monkeypatch.setattr(tensor_proxy_opp, "_send_obs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tensor_proxy_opp,
        "_recv_action",
        lambda fd: (2, 1, 3, 5, 4, array("f", [0.0] * tensor_proxy_opp.H_DIM)),
    )

    player = tensor_proxy_opp.Player()
    player.my_team = 1
    player.core_center = object()
    player._encode_state_buf = lambda ct: array("f", [0.0] * tensor_proxy_opp.INPUT_SIZE)

    ct = FakeProxyController()
    player._run_builder(ct)

    assert calls["build"] == (
        tensor_proxy_opp.BUILD_ACTIONS[2],
        tensor_proxy_opp.DIR_MAP_9[1],
        tensor_proxy_opp.CONV_DIR_4[3],
        1,
        5,
    )
    assert calls["move"] == tensor_proxy_opp.DIR_MAP_9[4]


def test_mirror_obs_to_team_a_frame_rotates_tiles_and_globals():
    obs = array("f", [0.0] * tensor_shared.INPUT_SIZE)
    for tile in range(tensor_shared.GRID_SIZE * tensor_shared.GRID_SIZE):
        base = tile * tensor_shared.TILE_FEATURES
        obs[base] = float(tile + 1)
        obs[base + 13] = float(tile + 0.25)
        obs[base + 14] = float(-(tile + 0.5))

    g = tensor_shared._GLOBAL_OBS_START
    obs[g + 1] = 0.2
    obs[g + 2] = 0.3
    obs[g + 6] = 0.4
    obs[g + 7] = -0.5
    obs[g + 8] = 0.6
    obs[g + 9] = -0.7

    pos_encoder = tensor_shared.SinusoidalPositionEmbedding(tensor_shared.POS_EMBED_DIM)
    tensor_shared.mirror_obs_to_team_a_frame(
        obs,
        map_width=10,
        map_height=12,
        pos_x=2,
        pos_y=3,
        pos_encoder=pos_encoder,
    )

    src_tile = tensor_shared.GRID_SIZE * tensor_shared.GRID_SIZE - 1
    src_base = src_tile * tensor_shared.TILE_FEATURES
    assert obs[0] == float(src_tile + 1)
    assert obs[13] == -float(src_tile + 0.25)
    assert obs[14] == float(src_tile + 0.5)
    assert math.isclose(obs[g + 1], 0.7, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(obs[g + 2], 8.0 / 12.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(obs[g + 6], -0.4, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(obs[g + 7], 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(obs[g + 8], -0.6, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(obs[g + 9], 0.7, rel_tol=0.0, abs_tol=1e-6)

    pos_start = g + tensor_shared.GLOBAL_FEATURES
    expected_pos = pos_encoder.encode(7, 8)
    for i, expected in enumerate(expected_pos):
        assert math.isclose(obs[pos_start + i], expected, rel_tol=0.0, abs_tol=1e-6)


def test_opponent_proxy_passes_actions_unchanged(monkeypatch):
    """Opponent proxy no longer mirrors — obs encoding is core-relative and team-symmetric."""
    calls = {}

    def fake_execute_build(ct, action, build_dir, conv_dir, my_team, bridge_target_idx):
        calls["build"] = (action, build_dir, conv_dir, my_team, bridge_target_idx)
        return 0.0, 0.0, (0, 0, 0, 0, 0, 0, 0, 0)

    def fake_execute_move(ct, move_dir):
        calls["move"] = move_dir

    class FakeProxyController:
        def __init__(self):
            self.team = 2
            self.pos = Position(2, 3)

        def get_current_round(self):
            return 0

        def get_team(self, eid=None):
            return self.team

        def get_global_resources(self):
            return 0, 0

        def get_position(self, eid=None):
            return self.pos

        def get_map_width(self):
            return 10

        def get_map_height(self):
            return 12

    monkeypatch.setattr(tensor_proxy_opp, "execute_build", fake_execute_build)
    monkeypatch.setattr(tensor_proxy_opp, "execute_move", fake_execute_move)
    monkeypatch.setattr(tensor_proxy_opp, "_get_pipes", lambda: (1, 2))
    monkeypatch.setattr(tensor_proxy_opp, "_send_obs", lambda _fd, obs, _h: None)
    monkeypatch.setattr(
        tensor_proxy_opp,
        "_recv_action",
        lambda fd: (2, 1, 0, 5, 3, array("f", [0.0] * tensor_proxy_opp.H_DIM)),
    )

    player = tensor_proxy_opp.Player()
    player.my_team = 2
    player.core_center = object()
    player._encode_state_buf = lambda ct: array("f", [0.0] * tensor_proxy_opp.INPUT_SIZE)

    ct = FakeProxyController()
    player._run_builder(ct)

    # Actions passed through without mirroring
    assert calls["build"] == (
        tensor_proxy_opp.BUILD_ACTIONS[2],
        tensor_proxy_opp.DIR_MAP_9[1],
        tensor_proxy_opp.CONV_DIR_4[0],
        2,
        5,
    )
    assert calls["move"] == tensor_proxy_opp.DIR_MAP_9[3]


def test_shared_game_state_is_scoped_per_team():
    tensor_shared.GameState.reset()

    class FakeSharedController:
        def __init__(self, team, enemy_hp):
            self.team = team
            self.enemy_hp = enemy_hp

        def get_team(self, eid=None):
            if eid is None:
                return self.team
            return 2 if self.team == 1 else 1

        def get_current_round(self):
            return 0

        def get_nearby_entities(self):
            return [1]

        def get_entity_type(self, eid):
            return tensor_shared.EntityType.CORE

        def get_hp(self, eid):
            return self.enemy_hp

    tensor_shared.scan_enemy_core_dmg(FakeSharedController(team=1, enemy_hp=490.0))
    tensor_shared.scan_enemy_core_dmg(FakeSharedController(team=2, enemy_hp=500.0))

    assert tensor_shared.GameState.enemy_core_dmg_for(1) == 10.0
    assert tensor_shared.GameState.enemy_core_dmg_for(2) == 0.0
>>>>>>> Stashed changes
