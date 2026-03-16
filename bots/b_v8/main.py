# b_v8: L-shaped conveyor chains with guaranteed connectivity. Quadrant scouting.
# 5 builders, defensive gunners. Roads for exploration, conveyors only in chains.
# Key fix: chain_back uses cardinal-only L-shaped paths (no diagonals).
# Horizontal leg then vertical leg, all conveyors face same direction per leg.

import random

from cambc import Controller, Direction, EntityType, Environment, Position

DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]

_HDIR = {1: Direction.EAST, -1: Direction.WEST}
_VDIR = {1: Direction.SOUTH, -1: Direction.NORTH}


def _dist_sq(a: Position, b: Position) -> int:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


class Player:
    __slots__ = (
        'core_center', 'core_tiles', 'target_ore', 'steps', 'phase',
        'chain_stuck', 'harvesters_built', 'num_spawned', 'explore_dir_base',
        'chain_hdir', 'chain_vdir', 'chain_target_x', 'chain_target_y',
        'chain_leg', 'chain_placed_first', 'harvester_pos', 'gunners_built',
        'builder_id_mod', 'initialized',
    )

    def __init__(self):
        self.core_center = None
        self.core_tiles = None
        self.target_ore = None
        self.steps = 0
        self.phase = "find_ore"
        self.chain_stuck = 0
        self.harvesters_built = 0
        self.num_spawned = 0
        self.explore_dir_base = None
        self.chain_hdir = None
        self.chain_vdir = None
        self.chain_target_x = 0
        self.chain_target_y = 0
        self.chain_leg = None
        self.chain_placed_first = False
        self.harvester_pos = None
        self.gunners_built = 0
        self.builder_id_mod = -1
        self.initialized = False

    def run(self, ct: Controller) -> None:
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    # -- CORE ----------------------------------------------------------------

    def _run_core(self, ct: Controller) -> None:
        scale = ct.get_scale_percent()
        # Adapt builder count to map size
        map_area = ct.get_map_width() * ct.get_map_height()
        if map_area <= 625:  # 25x25 or smaller
            max_builders = 3
        elif map_area <= 1225:  # ~35x35
            max_builders = 4
        else:
            max_builders = 5

        if self.num_spawned >= max_builders:
            return
        if scale > 250:
            return
        if scale > 180 and ct.get_current_round() % 2 != 0:
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

    # -- BUILDER BOT ----------------------------------------------------------

    def _run_builder(self, ct: Controller) -> None:
        if not self.initialized:
            self._init_builder(ct)
            self.initialized = True

        if self.phase == "find_ore":
            self._phase_find_ore(ct)
        elif self.phase == "chain_back":
            self._phase_chain_back(ct)
        elif self.phase == "patrol":
            self._phase_patrol(ct)

    def _init_builder(self, ct: Controller) -> None:
        self.core_center = self._find_core(ct)
        my_id = ct.get_id()
        self.builder_id_mod = my_id % 4
        quadrant_dirs = [Direction.NORTHEAST, Direction.NORTHWEST,
                         Direction.SOUTHEAST, Direction.SOUTHWEST]
        self.explore_dir_base = quadrant_dirs[self.builder_id_mod]
        cc = self.core_center
        if cc:
            self.core_tiles = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.core_tiles.add((cc.x + dx, cc.y + dy))

    def _find_core(self, ct: Controller) -> Position:
        for eid in ct.get_nearby_entities():
            try:
                if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) == ct.get_team():
                    return ct.get_position(eid)
            except Exception:
                continue
        return ct.get_position()

    def _is_near_core(self, pos: Position) -> bool:
        if self.core_tiles is None:
            return _dist_sq(pos, self.core_center) <= 4
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (pos.x + dx, pos.y + dy) in self.core_tiles:
                    return True
        return False

    # -- PHASE: FIND ORE ------------------------------------------------------

    def _phase_find_ore(self, ct: Controller) -> None:
        pos = ct.get_position()

        if self.harvesters_built >= 2 and self.steps > 50:
            self.phase = "patrol"
            return

        best_ore_dir = None
        best_ore_score = 99999
        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                env = ct.get_tile_env(adj)
                if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    continue
                type_penalty = 0 if env == Environment.ORE_TITANIUM else 300
                core_dist = _dist_sq(adj, self.core_center) if self.core_center else 0
                score = type_penalty + core_dist
                if score < best_ore_score:
                    best_ore_score = score
                    best_ore_dir = d

        if best_ore_dir is not None:
            adj = pos.add(best_ore_dir)
            ct.build_harvester(adj)
            self.harvester_pos = adj
            self.harvesters_built += 1
            self._setup_chain(ct)
            return

        if self.target_ore is None or self.steps > 20:
            self._scan_for_ore(ct)
            self.steps = 0

        if self.target_ore is not None:
            move_dir = pos.direction_to(self.target_ore)
            if move_dir == Direction.CENTRE:
                self.target_ore = None
                return
        else:
            move_dir = self._explore_dir(pos)

        self._move_with_road(ct, pos, move_dir)
        self.steps += 1

    def _explore_dir(self, pos: Position) -> Direction:
        base_dir = self.explore_dir_base
        if base_dir is None:
            if self.core_center is not None:
                base_dir = self.core_center.direction_to(pos)
                if base_dir == Direction.CENTRE:
                    base_dir = random.choice(DIRECTIONS)
            else:
                return random.choice(DIRECTIONS)
        if random.random() < 0.25:
            base_dir = random.choice([base_dir.rotate_left(), base_dir.rotate_right()])
        return base_dir

    def _scan_for_ore(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center
        best_score = 99999
        best = None
        for tpos in ct.get_nearby_tiles():
            env = ct.get_tile_env(tpos)
            if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                if ct.get_tile_building_id(tpos) is not None:
                    continue
                d_bot = _dist_sq(pos, tpos)
                d_core = _dist_sq(tpos, core) if core else 0
                type_penalty = 0 if env == Environment.ORE_TITANIUM else 400
                score = d_bot + d_core * 2 + type_penalty
                if score < best_score:
                    best_score = score
                    best = tpos
        if best is not None:
            self.target_ore = best

    # -- L-SHAPED CHAIN SETUP -----------------------------------------------

    def _setup_chain(self, ct: Controller) -> None:
        core = self.core_center
        if core is None:
            self.phase = "find_ore"
            return

        pos = ct.get_position()
        dx = core.x - pos.x
        dy = core.y - pos.y

        if abs(dx) <= 1 and abs(dy) <= 1:
            # Very close — no chain needed
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        sx = _sign(dx)
        sy = _sign(dy)
        self.chain_hdir = _HDIR.get(sx)
        self.chain_vdir = _VDIR.get(sy)
        self.chain_target_x = core.x
        self.chain_target_y = core.y

        # Start with horizontal leg if there's horizontal displacement
        if dx == 0:
            self.chain_leg = "vert"
        elif dy == 0:
            self.chain_leg = "horiz"
        elif abs(dx) >= abs(dy):
            self.chain_leg = "horiz"
        else:
            self.chain_leg = "vert"

        self.chain_placed_first = False
        self.phase = "chain_back"
        self.chain_stuck = 0

    # -- PHASE: BUILD L-SHAPED CONVEYOR CHAIN --------------------------------

    def _phase_chain_back(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center
        if core is None:
            self.phase = "find_ore"
            return

        if self._is_near_core(pos):
            # Place final conveyor toward core
            move_dir = pos.direction_to(core)
            if move_dir != Direction.CENTRE and ct.can_build_conveyor(pos, move_dir):
                ct.build_conveyor(pos, move_dir)
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        self.chain_stuck += 1
        if self.chain_stuck > 70:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        # === FIRST TILE: place conveyor at current pos (adjacent to harvester) ===
        if not self.chain_placed_first:
            self._chain_place_first(ct, pos)
            return

        # === NORMAL CHAIN STEP: build conveyor ahead, move ===
        move_dir = self._resolve_chain_dir(pos)
        if move_dir is None:
            self.phase = "find_ore"
            return

        next_pos = pos.add(move_dir)
        next_facing = self._get_next_facing(pos, move_dir)

        # Build conveyor ahead
        if ct.can_build_conveyor(next_pos, next_facing):
            ct.build_conveyor(next_pos, next_facing)

        if ct.can_move(move_dir):
            ct.move(move_dir)
        else:
            # Blocked — try nearby directions
            self._chain_detour(ct, pos, move_dir)

    def _chain_place_first(self, ct: Controller, pos: Position) -> None:
        """Place the first conveyor at builder's position (adjacent to harvester)."""
        facing = self._resolve_chain_dir(pos)
        if facing is None:
            self.phase = "find_ore"
            return

        # Destroy any road at current position
        if not ct.can_build_conveyor(pos, facing):
            bid = ct.get_tile_building_id(pos)
            if bid is not None:
                try:
                    ct.destroy(pos)
                except Exception:
                    pass

        if ct.can_build_conveyor(pos, facing):
            ct.build_conveyor(pos, facing)
            self.chain_placed_first = True
        else:
            # Can't build here — skip to building ahead and moving
            self.chain_placed_first = True

    def _resolve_chain_dir(self, pos: Position) -> Direction:
        """Get the current chain movement direction, updating leg as needed."""
        if self.chain_leg == "horiz":
            hdir = self.chain_hdir
            if hdir is None:
                self.chain_leg = "vert"
                return self._resolve_chain_dir(pos)
            dx = self.chain_target_x - pos.x
            if dx == 0 or _sign(dx) != _sign(hdir.delta()[0]):
                self.chain_leg = "vert"
                return self._resolve_chain_dir(pos)
            return hdir

        elif self.chain_leg == "vert":
            vdir = self.chain_vdir
            if vdir is None:
                return pos.direction_to(self.core_center) or None
            dy = self.chain_target_y - pos.y
            if dy == 0 or _sign(dy) != _sign(vdir.delta()[1]):
                return pos.direction_to(self.core_center) or None
            return vdir

        else:
            d = pos.direction_to(self.core_center)
            return d if d != Direction.CENTRE else None

    def _get_next_facing(self, pos: Position, move_dir: Direction) -> Direction:
        """Determine conveyor facing for the tile we're building AHEAD."""
        if self.chain_leg == "horiz":
            hdir = self.chain_hdir
            dx = self.chain_target_x - pos.x
            h_delta = hdir.delta()[0] if hdir else 0
            # If moving to the last tile of horizontal leg, that's the corner
            # Corner conveyor should face the vertical direction
            if dx == h_delta and self.chain_vdir is not None:
                return self.chain_vdir
            return hdir if hdir else Direction.NORTH

        elif self.chain_leg == "vert":
            return self.chain_vdir if self.chain_vdir else Direction.NORTH

        else:
            next_pos = pos.add(move_dir)
            d = next_pos.direction_to(self.core_center)
            return d if d != Direction.CENTRE else move_dir

    def _chain_detour(self, ct: Controller, pos: Position, move_dir: Direction) -> None:
        """When blocked, try alternate directions with roads."""
        for alt in (move_dir.rotate_left(), move_dir.rotate_right(),
                    move_dir.rotate_left().rotate_left(),
                    move_dir.rotate_right().rotate_right()):
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return
        for d in DIRECTIONS:
            dp = pos.add(d)
            if ct.can_build_road(dp):
                ct.build_road(dp)
            if ct.can_move(d):
                ct.move(d)
                return

    # -- PHASE: PATROL --------------------------------------------------------

    def _phase_patrol(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center

        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                env = ct.get_tile_env(adj)
                if env == Environment.ORE_TITANIUM:
                    ct.build_harvester(adj)
                    self.harvester_pos = adj
                    self.harvesters_built += 1
                    self._setup_chain(ct)
                    return

        # Only build gunners on larger maps (small maps: scale too precious)
        map_area = ct.get_map_width() * ct.get_map_height()
        if self.gunners_built < 2 and core is not None and map_area > 625:
            dist_to_core = _dist_sq(pos, core)
            if 2 <= dist_to_core <= 18:
                facing = core.direction_to(pos)
                if facing == Direction.CENTRE:
                    facing = Direction.NORTH
                for d in DIRECTIONS:
                    gp = pos.add(d)
                    if ct.can_build_gunner(gp, facing):
                        ct.build_gunner(gp, facing)
                        self.gunners_built += 1
                        break

        if ct.get_action_cooldown() == 0:
            for eid in ct.get_nearby_entities():
                try:
                    if ct.get_team(eid) == ct.get_team():
                        epos = ct.get_position(eid)
                        if _dist_sq(pos, epos) <= 2:
                            if ct.get_hp(eid) < ct.get_max_hp(eid):
                                if ct.can_heal(epos):
                                    ct.heal(epos)
                                    return
                except Exception:
                    continue

        if core is None:
            return

        dist_to_core = _dist_sq(pos, core)
        if dist_to_core > 36:
            move_dir = pos.direction_to(core)
        elif dist_to_core < 8:
            move_dir = core.direction_to(pos)
            if move_dir == Direction.CENTRE:
                move_dir = random.choice(DIRECTIONS)
        else:
            move_dir = random.choice(DIRECTIONS)

        self._move_with_road(ct, pos, move_dir)

    # -- MOVEMENT HELPER ------------------------------------------------------

    def _move_with_road(self, ct: Controller, pos: Position, move_dir: Direction) -> None:
        if move_dir == Direction.CENTRE:
            return
        move_pos = pos.add(move_dir)
        if ct.can_build_road(move_pos):
            ct.build_road(move_pos)
        if ct.can_move(move_dir):
            ct.move(move_dir)
            return
        for alt in (move_dir.rotate_left(), move_dir.rotate_right(),
                    move_dir.rotate_left().rotate_left(),
                    move_dir.rotate_right().rotate_right()):
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return
        for d in DIRECTIONS:
            dp = pos.add(d)
            if ct.can_build_road(dp):
                ct.build_road(dp)
            if ct.can_move(d):
                ct.move(d)
                return
