"""Regression tests for tensor stdlib action execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tensor_main = _load_module("tensor_inference_main_for_action_tests", "bots/tensor_inference/main.py")


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
