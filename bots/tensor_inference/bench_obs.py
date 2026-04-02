"""Microbenchmark the stdlib observation encoder under a dense fake scene."""

from __future__ import annotations

import argparse
import time

import main as bot_main
from main import Direction, EntityType, Environment, Player, Position, ResourceType


class FakeController:
    def __init__(self):
        self.pos = Position(20, 20)
        self.team = 1
        self.round = 500
        self.width = 40
        self.height = 40
        self.ti = 180.0
        self.ax = 40.0
        self.scale = 135.0

        self.nearby_tiles = []
        self.tile_env = {}
        self.tile_passable = {}
        self.tile_building = {}
        self.tile_builder = {}
        self.entities = {}
        self.nearby_entities = []

        dirs9 = [
            Direction.NORTHWEST,
            Direction.NORTH,
            Direction.NORTHEAST,
            Direction.WEST,
            Direction.CENTRE,
            Direction.EAST,
            Direction.SOUTHWEST,
            Direction.SOUTH,
            Direction.SOUTHEAST,
        ]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                pos = Position(self.pos.x + dx, self.pos.y + dy)
                self.nearby_tiles.append(pos)
                env = Environment.EMPTY
                if (dx, dy) in {(-2, -2), (2, 1)}:
                    env = Environment.WALL
                elif (dx + dy) % 5 == 0:
                    env = Environment.ORE_TITANIUM
                elif (dx - dy) % 4 == 0:
                    env = Environment.ORE_AXIONITE
                self.tile_env[(pos.x, pos.y)] = env
                self.tile_passable[(pos.x, pos.y)] = env != Environment.WALL and (dx, dy) not in {(1, -1), (-1, 1)}
                if (dx, dy) in {(-1, -1), (0, -1), (1, 0), (0, 1), (2, 0), (-2, 1)}:
                    eid = 100 + len(self.entities)
                    etype = [
                        EntityType.HARVESTER,
                        EntityType.CONVEYOR,
                        EntityType.SPLITTER,
                        EntityType.ROAD,
                        EntityType.GUNNER,
                        EntityType.CORE,
                    ][len(self.entities) % 6]
                    team = self.team if (dx + dy) % 2 == 0 else 2
                    direction = dirs9[((dx + 2) + 5 * (dy + 2)) % 9]
                    stored = None if etype != EntityType.HARVESTER else ResourceType.TITANIUM
                    hp = 500.0 if etype == EntityType.CORE else 100.0
                    self._add_entity(eid, etype, team, direction, stored, hp, pos, as_building=True)
                if (dx, dy) in {(0, 0), (1, 1), (-2, 0), (2, -1)}:
                    uid = 200 + len(self.entities)
                    self._add_entity(uid, EntityType.BUILDER_BOT, self.team if dx <= 0 else 2, Direction.NORTH, None, 50.0, pos, as_unit=True)

        self._add_entity(900, EntityType.CORE, self.team, Direction.CENTRE, None, 470.0, Position(18, 20))
        self._add_entity(901, EntityType.CORE, 2, Direction.CENTRE, None, 420.0, Position(25, 22))

    def _add_entity(self, eid, etype, team, direction, stored, hp, pos, *, as_building=False, as_unit=False):
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
def _bench_encode(player: Player, ct: FakeController, iterations: int, prefetched: bool) -> tuple[float, int]:
    buf = []
    nearby_tiles = ct.nearby_tiles if prefetched else None
    nearby_entities = ct.nearby_entities if prefetched else None
    for _ in range(200):
        buf.clear()
        player._encode_state_into(ct, buf, nearby_tiles, nearby_entities)
    start = time.perf_counter()
    for _ in range(iterations):
        buf.clear()
        player._encode_state_into(ct, buf, nearby_tiles, nearby_entities)
    elapsed = time.perf_counter() - start
    return elapsed * 1e6 / iterations, len(buf)


def _reference_obs(player: Player, ct: FakeController) -> list[float]:
    features: list[float] = []
    pos = ct.get_position()
    my_team = player.my_team
    visible_tiles = {(tile_pos.x, tile_pos.y): tile_pos for tile_pos in ct.nearby_tiles}
    for dx, dy in bot_main._LOCAL_TILE_OFFSETS:
        tx, ty = pos.x + dx, pos.y + dy
        if tx < 0 or tx >= ct.get_map_width() or ty < 0 or ty >= ct.get_map_height():
            features.extend(bot_main._ZERO_TILE)
            continue
        tp = visible_tiles.get((tx, ty))
        if tp is None:
            features.extend(bot_main._ZERO_TILE)
            continue
        env = ct.get_tile_env(tp)
        features.append(1.0 if env == Environment.WALL else 0.0)
        features.append(1.0 if env == Environment.ORE_TITANIUM else 0.0)
        features.append(1.0 if env == Environment.ORE_AXIONITE else 0.0)
        features.append(0.0 if ct.is_tile_passable(tp) else 1.0)
        bid = ct.get_tile_building_id(tp)
        if bid is not None:
            btype = ct.get_entity_type(bid)
            bidx = bot_main.BLD_IDX.get(btype, 3)
            features.extend(bot_main._BLD_ONE_HOTS[bidx])
            features.append(1.0 if ct.get_team(bid) == my_team else -1.0)
            try:
                bdir = ct.get_direction(bid)
                bdx, bdy = bot_main.DIR_DXY.get(bdir, (0, 0))
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
            features.extend(bot_main._NO_BUILD_TILE)
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
    own_hp, enemy_hp = bot_main.Player._own_core_hp, bot_main.Player._enemy_core_hp
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
    core_dx, core_dy = 0.0, 0.0
    if player.core_center is not None:
        cdx = player.core_center.x - pos.x
        cdy = player.core_center.y - pos.y
        d = max(abs(cdx), abs(cdy), 1)
        core_dx = cdx / d
        core_dy = cdy / d
    ecore_dx, ecore_dy = 0.0, 0.0
    if player.enemy_core is not None:
        edx = player.enemy_core.x - pos.x
        edy = player.enemy_core.y - pos.y
        d = max(abs(edx), abs(edy), 1)
        ecore_dx = edx / d
        ecore_dy = edy / d
    ti_delta, ax_delta = 0.0, 0.0
    if player._prev_resources is not None:
        prev_ti, prev_ax = player._prev_resources
        ti_delta = (ti - prev_ti) / 500.0
        ax_delta = (ax - prev_ax) / 500.0
    hp_advantage = (own_hp - enemy_hp) / 500.0
    features.extend((
        rnd, x_norm, y_norm, ti / 500.0, ax / 500.0, scale,
        core_dx, core_dy, ecore_dx, ecore_dy,
        n_builders / 10.0,
        ti_delta, ax_delta, hp_advantage,
        player._spawn_round / 2000.0 if player._spawn_round >= 0 else 0.0,
    ))
    return features


def _bench_pipeline(ct: FakeController, iterations: int) -> tuple[float, float]:
    nearby_tiles = ct.nearby_tiles
    nearby_entities = ct.nearby_entities
    old_player = Player()
    old_player.my_team = ct.get_team()
    old_player._prev_resources = (160.0, 35.0)
    old_player._spawn_round = 100

    def old_pipeline() -> None:
        bot_main.Player._discover_ore(ct, nearby_tiles)
        bot_main.Player._scan_enemy_core_dmg(ct, nearby_entities)
        if old_player.enemy_core is None:
            for eid in nearby_entities:
                try:
                    if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) != old_player.my_team:
                        old_player.enemy_core = ct.get_position(eid)
                        break
                except Exception:
                    continue
        _reference_obs(old_player, ct)

    new_player = Player()
    new_player.my_team = ct.get_team()
    new_player.core_center = Position(18, 20)
    new_player.enemy_core = Position(25, 22)
    new_player._prev_resources = (160.0, 35.0)
    new_player._spawn_round = 100

    def new_pipeline() -> None:
        if new_player.core_center is None or new_player.enemy_core is None:
            new_player._find_cores_in_entities(ct, nearby_entities)
        new_player._encode_state(ct, nearby_tiles, nearby_entities)

    for _ in range(200):
        old_pipeline()
    start = time.perf_counter()
    for _ in range(iterations):
        old_pipeline()
    old_us = (time.perf_counter() - start) * 1e6 / iterations

    bot_main.Player._all_ore = None
    bot_main.Player._all_ore_keys = set()
    bot_main.Player._enemy_core_hp = 500.0
    bot_main.Player._own_core_hp = 500.0
    bot_main.Player._enemy_core_dmg = 0.0
    bot_main.Player._enemy_core_dmg_round = -1

    for _ in range(200):
        new_pipeline()
    start = time.perf_counter()
    for _ in range(iterations):
        new_pipeline()
    new_us = (time.perf_counter() - start) * 1e6 / iterations
    return old_us, new_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000, help="Number of timed encode iterations")
    args = parser.parse_args()

    ct = FakeController()
    bot_main.Player._all_ore = None
    bot_main.Player._all_ore_keys = set()
    bot_main.Player._enemy_core_hp = 500.0
    bot_main.Player._own_core_hp = 500.0
    bot_main.Player._enemy_core_dmg = 0.0
    bot_main.Player._enemy_core_dmg_round = -1
    player = Player()
    player.my_team = ct.get_team()
    player.core_center = Position(18, 20)
    player.enemy_core = Position(25, 22)
    player._prev_resources = (160.0, 35.0)
    player._spawn_round = 100

    prefetched_us, length = _bench_encode(player, ct, args.iters, prefetched=True)
    internal_us, _ = _bench_encode(player, ct, args.iters, prefetched=False)
    old_pipeline_us, new_pipeline_us = _bench_pipeline(ct, args.iters)
    print(f"encode_prefetched_us={prefetched_us:.2f} len={length}")
    print(f"encode_internal_fetch_us={internal_us:.2f}")
    print(f"old_obs_pipeline_us={old_pipeline_us:.2f}")
    print(f"fused_obs_pipeline_us={new_pipeline_us:.2f}")


if __name__ == "__main__":
    main()
