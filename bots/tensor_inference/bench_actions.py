"""Microbenchmark the stdlib action execution path against the previous logic."""

from __future__ import annotations

import argparse
import time

import main as bot_main
from main import Direction, EntityType, Player, Position


class FakeActionController:
    def __init__(self):
        self.pos = Position(10, 10)
        self.team = 1
        self.width = 30
        self.height = 30
        self.scale = 120.0
        self.ti = 300
        self.ax = 80
        self.buildings = {}
        self.moves_allowed = {}
        self.road_buildable = set()

    def clone(self):
        other = FakeActionController()
        other.pos = self.pos
        other.team = self.team
        other.width = self.width
        other.height = self.height
        other.scale = self.scale
        other.ti = self.ti
        other.ax = self.ax
        other.buildings = {k: dict(v) for k, v in self.buildings.items()}
        other.moves_allowed = dict(self.moves_allowed)
        other.road_buildable = set(self.road_buildable)
        return other

    def set_building(self, pos, eid, etype, team, direction=Direction.CENTRE):
        self.buildings[(pos.x, pos.y)] = {
            "id": eid,
            "etype": etype,
            "team": team,
            "direction": direction,
        }

    def get_position(self, eid=None):
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

    def get_resources_collected(self):
        return 200, 50

    def get_global_resources_full(self):
        return self.ti, self.ax, 200, 50

    def destroy(self, pos):
        self.buildings.pop((pos.x, pos.y), None)

    def can_build_conveyor(self, pos, direction):
        return True

    def build_conveyor(self, pos, direction):
        return None

    def can_build_harvester(self, pos):
        return False

    def build_harvester(self, pos):
        return None

    def can_build_road(self, pos):
        return (pos.x, pos.y) in self.road_buildable

    def build_road(self, pos):
        return None

    def can_build_gunner(self, pos, direction):
        return True

    def build_gunner(self, pos, direction):
        return None

    def can_build_sentinel(self, pos, direction):
        return True

    def build_sentinel(self, pos, direction):
        return None

    def can_build_breach(self, pos, direction):
        return True

    def build_breach(self, pos, direction):
        return None

    def can_build_splitter(self, pos, direction):
        return True

    def build_splitter(self, pos, direction):
        return None

    def can_build_armoured_conveyor(self, pos, direction):
        return True

    def build_armoured_conveyor(self, pos, direction):
        return None

    def can_build_barrier(self, pos):
        return True

    def build_barrier(self, pos):
        return None

    def can_build_foundry(self, pos):
        return True

    def build_foundry(self, pos):
        return None

    def can_build_launcher(self, pos):
        return True

    def build_launcher(self, pos):
        return None

    def can_move(self, direction):
        return self.moves_allowed.get(direction, False)

    def move(self, direction):
        return None


def _make_player():
    bot_main.Player._all_ore = None
    bot_main.Player._all_ore_keys = set()
    bot_main.Player._harv_proximity = {}
    player = Player()
    player.my_team = 1
    return player


def _old_execute_build(player: Player, ct: FakeActionController, action: str, direction: Direction, conv_direction=None):
    if action == "none":
        return 0.0, 0.0
    pos = ct.get_position()
    build_pos = pos.add(direction)
    w, h = ct.get_map_width(), ct.get_map_height()
    if build_pos.x < 0 or build_pos.x >= w or build_pos.y < 0 or build_pos.y >= h:
        return 0.0, 0.0
    bid = ct.get_tile_building_id(build_pos)
    if bid is not None:
        try:
            if ct.get_team(bid) == player.my_team and ct.get_entity_type(bid) != EntityType.CORE:
                btype_existing = ct.get_entity_type(bid)
                conv_dir = conv_direction if conv_direction is not None else direction
                sp_dir = conv_direction if conv_direction is not None else direction
                if action == "conveyor" and btype_existing == EntityType.CONVEYOR and ct.get_direction(bid) == conv_dir:
                    return 0.0, 0.0
                if action == "armoured_conveyor" and btype_existing == EntityType.ARMOURED_CONVEYOR and ct.get_direction(bid) == conv_dir:
                    return 0.0, 0.0
                if action == "splitter" and btype_existing == EntityType.SPLITTER and ct.get_direction(bid) == sp_dir:
                    return 0.0, 0.0
                idx = bot_main.BUILD_ACTIONS.index(action)
                base_ti, base_ax = bot_main.BUILD_COSTS[idx]
                scale = ct.get_scale_percent() / 100.0
                ti, ax = ct.get_global_resources()
                if ti >= int(scale * base_ti) and ax >= int(scale * base_ax):
                    ct.destroy(build_pos)
                else:
                    return 0.0, 0.0
        except Exception:
            pass
    if action == "conveyor":
        conv_dir = conv_direction if conv_direction is not None else direction
        if ct.can_build_conveyor(build_pos, conv_dir):
            ct.build_conveyor(build_pos, conv_dir)
    elif action == "barrier":
        if ct.can_build_barrier(build_pos):
            ct.build_barrier(build_pos)
    return 0.0, 0.0


def _old_execute_move(ct: FakeActionController, direction: Direction):
    if direction == Direction.CENTRE:
        return
    pos = ct.get_position()
    move_pos = pos.add(direction)
    if ct.can_build_road(move_pos):
        ct.build_road(move_pos)
    if ct.can_move(direction):
        ct.move(direction)
        return
    for alt in (direction.rotate_left(), direction.rotate_right()):
        alt_pos = pos.add(alt)
        if ct.can_build_road(alt_pos):
            ct.build_road(alt_pos)
        if ct.can_move(alt):
            ct.move(alt)
            return


def _bench_build(iterations: int) -> tuple[float, float]:
    player = _make_player()
    target = Position(10, 9)
    template = FakeActionController()
    template.set_building(target, 100, EntityType.ROAD, player.my_team)

    for _ in range(200):
        _old_execute_build(player, template.clone(), "barrier", Direction.NORTH)
    start = time.perf_counter()
    for _ in range(iterations):
        _old_execute_build(player, template.clone(), "barrier", Direction.NORTH)
    old_us = (time.perf_counter() - start) * 1e6 / iterations

    for _ in range(200):
        player._execute_build(template.clone(), bot_main._ACTION_BARRIER, Direction.NORTH)
    start = time.perf_counter()
    for _ in range(iterations):
        player._execute_build(template.clone(), bot_main._ACTION_BARRIER, Direction.NORTH)
    new_us = (time.perf_counter() - start) * 1e6 / iterations
    return old_us, new_us


def _bench_move(iterations: int) -> tuple[float, float]:
    template = FakeActionController()
    primary = template.pos.add(Direction.NORTH)
    fallback = template.pos.add(Direction.NORTHEAST)
    template.road_buildable = {(primary.x, primary.y), (fallback.x, fallback.y)}
    template.moves_allowed = {
        Direction.NORTH: False,
        Direction.NORTHWEST: False,
        Direction.NORTHEAST: True,
    }
    player = _make_player()

    for _ in range(200):
        _old_execute_move(template.clone(), Direction.NORTH)
    start = time.perf_counter()
    for _ in range(iterations):
        _old_execute_move(template.clone(), Direction.NORTH)
    old_us = (time.perf_counter() - start) * 1e6 / iterations

    for _ in range(200):
        player._execute_move(template.clone(), Direction.NORTH)
    start = time.perf_counter()
    for _ in range(iterations):
        player._execute_move(template.clone(), Direction.NORTH)
    new_us = (time.perf_counter() - start) * 1e6 / iterations
    return old_us, new_us


def _bench_combined(iterations: int) -> tuple[float, float]:
    target = Position(10, 9)

    def make_template() -> FakeActionController:
        template = FakeActionController()
        template.set_building(target, 100, EntityType.ROAD, template.team)
        fallback = template.pos.add(Direction.NORTHEAST)
        template.road_buildable = {(fallback.x, fallback.y)}
        template.moves_allowed = {
            Direction.NORTH: False,
            Direction.NORTHWEST: False,
            Direction.NORTHEAST: True,
        }
        return template

    old_player = _make_player()
    for _ in range(200):
        ct = make_template()
        _old_execute_build(old_player, ct, "barrier", Direction.NORTH)
        _old_execute_move(ct, Direction.NORTH)
    start = time.perf_counter()
    for _ in range(iterations):
        ct = make_template()
        _old_execute_build(old_player, ct, "barrier", Direction.NORTH)
        _old_execute_move(ct, Direction.NORTH)
    old_us = (time.perf_counter() - start) * 1e6 / iterations

    new_player = _make_player()
    for _ in range(200):
        ct = make_template()
        new_player._execute_build(ct, bot_main._ACTION_BARRIER, Direction.NORTH)
        new_player._execute_move(ct, Direction.NORTH)
    start = time.perf_counter()
    for _ in range(iterations):
        ct = make_template()
        new_player._execute_build(ct, bot_main._ACTION_BARRIER, Direction.NORTH)
        new_player._execute_move(ct, Direction.NORTH)
    new_us = (time.perf_counter() - start) * 1e6 / iterations
    return old_us, new_us


def _bench_builder_tail(iterations: int) -> tuple[float, float]:
    build_type_idx = bot_main._ACTION_BARRIER
    build_dir_idx = 1
    conv_dir_idx = 0
    move_dir_idx = 1

    def make_template() -> FakeActionController:
        ct = FakeActionController()
        ct.set_building(Position(10, 9), 100, EntityType.ROAD, ct.team)
        fallback = ct.pos.add(Direction.NORTHEAST)
        ct.road_buildable = {(fallback.x, fallback.y)}
        ct.moves_allowed = {
            Direction.NORTH: False,
            Direction.NORTHWEST: False,
            Direction.NORTHEAST: True,
        }
        return ct

    old_player = _make_player()
    for _ in range(200):
        ct = make_template()
        ct.get_global_resources()
        build_action = bot_main.BUILD_ACTIONS[build_type_idx]
        build_dir = bot_main.DIR_MAP_9[build_dir_idx]
        conv_dir = bot_main.CONV_DIR_4[conv_dir_idx]
        harv_reward, conv_reward = _old_execute_build(old_player, ct, build_action, build_dir, conv_dir)
        _ = harv_reward + conv_reward
        move_dir = bot_main.DIR_MAP_9[move_dir_idx]
        _old_execute_move(ct, move_dir)
        _ = bot_main.Player._enemy_core_dmg
        old_player._prev_collected = ct.get_resources_collected()
    start = time.perf_counter()
    for _ in range(iterations):
        ct = make_template()
        ct.get_global_resources()
        build_action = bot_main.BUILD_ACTIONS[build_type_idx]
        build_dir = bot_main.DIR_MAP_9[build_dir_idx]
        conv_dir = bot_main.CONV_DIR_4[conv_dir_idx]
        harv_reward, conv_reward = _old_execute_build(old_player, ct, build_action, build_dir, conv_dir)
        _ = harv_reward + conv_reward
        move_dir = bot_main.DIR_MAP_9[move_dir_idx]
        _old_execute_move(ct, move_dir)
        _ = bot_main.Player._enemy_core_dmg
        old_player._prev_collected = ct.get_resources_collected()
    old_us = (time.perf_counter() - start) * 1e6 / iterations

    new_player = _make_player()
    for _ in range(200):
        ct = make_template()
        build_dir = bot_main.DIR_MAP_9[build_dir_idx]
        conv_dir = bot_main.CONV_DIR_4[conv_dir_idx]
        new_player._execute_build(ct, build_type_idx, build_dir, conv_dir)
        move_dir = bot_main.DIR_MAP_9[move_dir_idx]
        new_player._execute_move(ct, move_dir)
        new_player._prev_collected = ct.get_resources_collected()
    start = time.perf_counter()
    for _ in range(iterations):
        ct = make_template()
        build_dir = bot_main.DIR_MAP_9[build_dir_idx]
        conv_dir = bot_main.CONV_DIR_4[conv_dir_idx]
        new_player._execute_build(ct, build_type_idx, build_dir, conv_dir)
        move_dir = bot_main.DIR_MAP_9[move_dir_idx]
        new_player._execute_move(ct, move_dir)
        new_player._prev_collected = ct.get_resources_collected()
    new_us = (time.perf_counter() - start) * 1e6 / iterations
    return old_us, new_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50000)
    args = parser.parse_args()
    old_build_us, new_build_us = _bench_build(args.iters)
    old_move_us, new_move_us = _bench_move(args.iters)
    old_combined_us, new_combined_us = _bench_combined(args.iters)
    old_tail_us, new_tail_us = _bench_builder_tail(args.iters)
    print(f"old_build_us={old_build_us:.2f}")
    print(f"new_build_us={new_build_us:.2f}")
    print(f"old_move_us={old_move_us:.2f}")
    print(f"new_move_us={new_move_us:.2f}")
    print(f"old_combined_us={old_combined_us:.2f}")
    print(f"new_combined_us={new_combined_us:.2f}")
    print(f"old_builder_tail_us={old_tail_us:.2f}")
    print(f"new_builder_tail_us={new_tail_us:.2f}")


if __name__ == "__main__":
    main()
