# b_v9: Based on a_v8. Roads for exploration (not conveyors) to save scale.
# Cardinal chain-back with corner detection from a_v8. No patrol phase — builders
# always explore+harvest. 6 builders for max scouting. Quadrant cycling for coverage.
# Also harvest axionite (tiebreaker #1: refined ax delivered).

import random

from cambc import Controller, Direction, EntityType, Environment, Position

CARDINALS = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]


def _dist_sq(a: Position, b: Position) -> int:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def _cardinal_toward(src: Position, dst: Position) -> Direction:
    dx = dst.x - src.x
    dy = dst.y - src.y
    if dx == 0 and dy == 0:
        return Direction.CENTRE
    if abs(dx) >= abs(dy):
        return Direction.EAST if dx > 0 else Direction.WEST
    return Direction.SOUTH if dy > 0 else Direction.NORTH


class Player:
    __slots__ = (
        'core_center', 'target_ore', 'steps', 'phase',
        'chain_stuck', 'harvesters_built', 'num_spawned',
        'explore_quadrant', 'quadrants_visited', 'initialized',
    )

    def __init__(self):
        self.core_center = None
        self.target_ore = None
        self.steps = 0
        self.phase = "find_ore"
        self.chain_stuck = 0
        self.harvesters_built = 0
        self.num_spawned = 0
        self.explore_quadrant = -1
        self.quadrants_visited = 0
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
        if self.num_spawned >= 6:
            return
        if scale > 300:
            return
        # Spawn first 4 fast, then slow down
        if self.num_spawned >= 4 and scale > 200:
            return
        if scale > 200 and ct.get_current_round() % 2 != 0:
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
            self.core_center = self._find_core(ct)
            self.explore_quadrant = ct.get_id() % 4
            self.initialized = True

        if self.phase == "find_ore":
            self._phase_find_ore(ct)
        elif self.phase == "chain_back":
            self._phase_chain_back(ct)

    def _find_core(self, ct: Controller) -> Position:
        for eid in ct.get_nearby_entities():
            try:
                if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) == ct.get_team():
                    return ct.get_position(eid)
            except Exception:
                continue
        return ct.get_position()

    # -- PHASE: FIND ORE ------------------------------------------------------

    def _phase_find_ore(self, ct: Controller) -> None:
        pos = ct.get_position()

        # Build harvester on adjacent ore — Ti preferred, but take Ax too
        best_ore_dir = None
        best_ore_score = 99999
        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                env = ct.get_tile_env(adj)
                if env == Environment.ORE_TITANIUM:
                    core_dist = _dist_sq(adj, self.core_center) if self.core_center else 0
                    score = core_dist
                elif env == Environment.ORE_AXIONITE:
                    core_dist = _dist_sq(adj, self.core_center) if self.core_center else 0
                    # Ax is less valuable but still worth harvesting — helps tiebreaker #1
                    score = 200 + core_dist
                else:
                    continue
                if score < best_ore_score:
                    best_ore_score = score
                    best_ore_dir = d

        if best_ore_dir is not None:
            adj = pos.add(best_ore_dir)
            ct.build_harvester(adj)
            self.target_ore = adj
            self.harvesters_built += 1
            self.phase = "chain_back"
            self.chain_stuck = 0
            return

        # No patrol phase — always keep searching
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

        # Use ROADS for exploration — save scale for chain conveyors
        self._move_with_road(ct, pos, move_dir)
        self.steps += 1

    def _explore_dir(self, pos: Position) -> Direction:
        cc = self.core_center
        if cc is None:
            return random.choice(DIRECTIONS)

        q = self.explore_quadrant
        target_dx = 20 if q in (0, 1) else -20
        target_dy = -20 if q in (0, 3) else 20
        target = Position(cc.x + target_dx, cc.y + target_dy)
        base_dir = pos.direction_to(target)

        if base_dir == Direction.CENTRE:
            # Reached quadrant target — cycle to next quadrant
            self.explore_quadrant = (self.explore_quadrant + 1) % 4
            self.quadrants_visited += 1
            return random.choice(DIRECTIONS)

        if random.random() < 0.2:
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
                type_penalty = 0 if env == Environment.ORE_TITANIUM else 300
                score = d_bot + d_core * 2 + type_penalty
                if score < best_score:
                    best_score = score
                    best = tpos
        if best is not None:
            self.target_ore = best

    # -- PHASE: CHAIN BACK TO CORE -------------------------------------------
    # Cardinal-only conveyors with corner detection (from a_v8).

    def _phase_chain_back(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center
        if core is None:
            self.phase = "find_ore"
            return

        dist = _dist_sq(pos, core)
        if dist <= 4:
            # Close enough to core — done chaining, go find more ore
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        self.chain_stuck += 1
        if self.chain_stuck > 60:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        card_dir = _cardinal_toward(pos, core)
        if card_dir == Direction.CENTRE:
            self.phase = "find_ore"
            return

        next_pos = pos.add(card_dir)

        # Corner detection: if stepping to next_pos would complete one axis,
        # face the conveyor in the other axis's direction
        conv_facing = card_dir
        ndx = core.x - next_pos.x
        ndy = core.y - next_pos.y
        if card_dir in (Direction.EAST, Direction.WEST) and ndx == 0 and ndy != 0:
            conv_facing = Direction.SOUTH if ndy > 0 else Direction.NORTH
        elif card_dir in (Direction.NORTH, Direction.SOUTH) and ndy == 0 and ndx != 0:
            conv_facing = Direction.EAST if ndx > 0 else Direction.WEST

        if ct.can_build_conveyor(next_pos, conv_facing):
            ct.build_conveyor(next_pos, conv_facing)

        if ct.can_move(card_dir):
            ct.move(card_dir)
            return

        # Blocked — try alternate cardinal toward core
        dx = core.x - pos.x
        dy = core.y - pos.y
        alt_dirs = []
        if card_dir in (Direction.EAST, Direction.WEST):
            if dy > 0:
                alt_dirs.append(Direction.SOUTH)
            elif dy < 0:
                alt_dirs.append(Direction.NORTH)
            else:
                alt_dirs = [Direction.SOUTH, Direction.NORTH]
        else:
            if dx > 0:
                alt_dirs.append(Direction.EAST)
            elif dx < 0:
                alt_dirs.append(Direction.WEST)
            else:
                alt_dirs = [Direction.EAST, Direction.WEST]

        for alt in alt_dirs:
            alt_pos = pos.add(alt)
            if ct.can_build_conveyor(alt_pos, alt):
                ct.build_conveyor(alt_pos, alt)
            if ct.can_move(alt):
                ct.move(alt)
                return

        # Remaining cardinals
        for alt in CARDINALS:
            if alt == card_dir or alt in alt_dirs:
                continue
            alt_pos = pos.add(alt)
            if ct.can_build_conveyor(alt_pos, alt):
                ct.build_conveyor(alt_pos, alt)
            elif ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return

        # Last resort: any direction
        for d in DIRECTIONS:
            dp = pos.add(d)
            if ct.can_build_road(dp):
                ct.build_road(dp)
            if ct.can_move(d):
                ct.move(d)
                return

    # -- MOVEMENT WITH ROADS (cheap exploration) ------------------------------

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
