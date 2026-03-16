# b_v11: Based on a_v8. Key changes: (1) Cap exploration conveyors to dist_sq<=200
# from core — beyond that use roads only (saves ~50% wasted conveyors). (2) Builder
# stuck detection (5 same-pos rounds) + retreat if dist_sq>625. (3) First 2 builders
# scan nearby (8 tiles) before exploring far. (4) Chain_back uses shorter-axis-first
# for straighter L-shaped paths. (5) Patrol conveyors also distance-capped.

import random

from cambc import Controller, Direction, EntityType, Environment, Position

CARDINALS = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]

# Max dist_sq from core to build conveyors during exploration
_CONV_RADIUS_SQ = 200


def _dist_sq(a: Position, b: Position) -> int:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def _cardinal_toward(src: Position, dst: Position) -> Direction:
    """Pick the cardinal direction from src toward dst with largest delta."""
    dx = dst.x - src.x
    dy = dst.y - src.y
    if dx == 0 and dy == 0:
        return Direction.CENTRE
    if abs(dx) >= abs(dy):
        return Direction.EAST if dx > 0 else Direction.WEST
    return Direction.SOUTH if dy > 0 else Direction.NORTH


def _cardinal_toward_shorter(src: Position, dst: Position) -> Direction:
    """Pick the cardinal direction along the SHORTER axis first.
    This creates L-shaped paths that maximize straight segments."""
    dx = dst.x - src.x
    dy = dst.y - src.y
    if dx == 0 and dy == 0:
        return Direction.CENTRE
    # Shorter axis first (opposite of _cardinal_toward)
    if abs(dx) <= abs(dy):
        if dx != 0:
            return Direction.EAST if dx > 0 else Direction.WEST
        return Direction.SOUTH if dy > 0 else Direction.NORTH
    if dy != 0:
        return Direction.SOUTH if dy > 0 else Direction.NORTH
    return Direction.EAST if dx > 0 else Direction.WEST


class Player:
    __slots__ = (
        'core_center', 'target_ore', 'steps', 'phase',
        'chain_stuck', 'harvesters_built', 'num_spawned',
        'explore_quadrant', 'builder_index', 'last_pos',
        'stuck_count', 'initialized',
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
        self.builder_index = -1
        self.last_pos = None
        self.stuck_count = 0
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
        if self.num_spawned >= 5:
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
            self.core_center = self._find_core(ct)
            self.explore_quadrant = ct.get_id() % 4
            self.builder_index = ct.get_id() % 6
            self.initialized = True

        # Stuck detection
        pos = ct.get_position()
        if self.last_pos is not None and pos == self.last_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_pos = pos

        # If stuck for 5+ rounds, try random movement to escape
        if self.stuck_count >= 5:
            self.stuck_count = 0
            dirs = list(DIRECTIONS)
            random.shuffle(dirs)
            for d in dirs:
                dp = pos.add(d)
                if ct.can_build_road(dp):
                    ct.build_road(dp)
                if ct.can_move(d):
                    ct.move(d)
                    return

        # Retreat if too far from core
        if self.core_center and self.phase == "find_ore":
            if _dist_sq(pos, self.core_center) > 625:
                self.phase = "find_ore"
                self.target_ore = None
                self.steps = 0
                # Head back toward core
                move_dir = pos.direction_to(self.core_center)
                if move_dir != Direction.CENTRE:
                    if ct.can_move(move_dir):
                        ct.move(move_dir)
                        return

        if self.phase == "find_ore":
            self._phase_find_ore(ct)
        elif self.phase == "chain_back":
            self._phase_chain_back(ct)
        elif self.phase == "patrol":
            self._phase_patrol(ct)

    def _find_core(self, ct: Controller) -> Position:
        my_team = ct.get_team()
        for eid in ct.get_nearby_entities():
            try:
                if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) == my_team:
                    return ct.get_position(eid)
            except Exception:
                continue
        return ct.get_position()

    # -- PHASE: FIND ORE ------------------------------------------------------

    def _phase_find_ore(self, ct: Controller) -> None:
        pos = ct.get_position()

        # Build harvester on adjacent ore
        best_ore_dir = None
        best_ore_score = 99999
        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                env = ct.get_tile_env(adj)
                if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    continue
                type_penalty = 0 if env == Environment.ORE_TITANIUM else 400
                core_dist = _dist_sq(adj, self.core_center) if self.core_center else 0
                score = type_penalty + core_dist
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

        if self.harvesters_built >= 3 and self.steps > 50:
            self.phase = "patrol"
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

        # Move with distance-capped conveyors
        self._move_with_conveyor(ct, pos, move_dir)
        self.steps += 1

    def _explore_dir(self, pos: Position) -> Direction:
        cc = self.core_center
        if cc is None:
            return random.choice(DIRECTIONS)

        q = self.explore_quadrant

        # First 2 builders scan nearby first (within ~8 tiles)
        if self.builder_index < 2 and self.steps < 12:
            target_dx = 8 if q in (0, 1) else -8
            target_dy = -8 if q in (0, 3) else 8
        else:
            target_dx = 18 if q in (0, 1) else -18
            target_dy = -18 if q in (0, 3) else 18

        target = Position(cc.x + target_dx, cc.y + target_dy)
        base_dir = pos.direction_to(target)

        if base_dir == Direction.CENTRE:
            self.explore_quadrant = (self.explore_quadrant + 1) % 4
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

    # -- PHASE: CHAIN BACK TO CORE -------------------------------------------
    # Cardinal-only conveyors, shorter-axis-first for straighter L-paths.

    def _phase_chain_back(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center
        if core is None:
            self.phase = "find_ore"
            return

        dist = _dist_sq(pos, core)
        if dist <= 4:
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

        # Use shorter-axis-first for L-shaped chain paths
        card_dir = _cardinal_toward_shorter(pos, core)
        if card_dir == Direction.CENTRE:
            self.phase = "find_ore"
            return

        next_pos = pos.add(card_dir)

        # Corner detection: if stepping completes one axis, face conveyor along other
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

        # Blocked — try alternate cardinal toward core (use longer axis)
        dx = core.x - pos.x
        dy = core.y - pos.y
        alt_dirs = []
        if card_dir in (Direction.EAST, Direction.WEST):
            if dy > 0: alt_dirs.append(Direction.SOUTH)
            elif dy < 0: alt_dirs.append(Direction.NORTH)
            else: alt_dirs = [Direction.SOUTH, Direction.NORTH]
        else:
            if dx > 0: alt_dirs.append(Direction.EAST)
            elif dx < 0: alt_dirs.append(Direction.WEST)
            else: alt_dirs = [Direction.EAST, Direction.WEST]

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

        # Opportunistic harvester building
        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                env = ct.get_tile_env(adj)
                if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    ct.build_harvester(adj)
                    self.harvesters_built += 1
                    self.target_ore = adj
                    self.phase = "chain_back"
                    self.chain_stuck = 0
                    return

        # Heal nearby friendlies
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
        if dist_to_core > 80:
            move_dir = pos.direction_to(core)
        elif dist_to_core < 8:
            move_dir = core.direction_to(pos)
            if move_dir == Direction.CENTRE:
                move_dir = random.choice(DIRECTIONS)
        else:
            move_dir = self._explore_dir(pos)

        # Patrol with distance-capped conveyors
        self._move_with_conveyor(ct, pos, move_dir)

    # -- MOVEMENT WITH DISTANCE-CAPPED CONVEYORS -----------------------------

    def _move_with_conveyor(self, ct: Controller, pos: Position, move_dir: Direction) -> None:
        """Move while building conveyors pointing toward core, but ONLY within
        _CONV_RADIUS_SQ of core. Beyond that, build roads instead."""
        if move_dir == Direction.CENTRE:
            return

        core = self.core_center
        move_pos = pos.add(move_dir)

        # Only build conveyors near core; roads elsewhere
        if core is not None:
            near_core = _dist_sq(move_pos, core) <= _CONV_RADIUS_SQ
            if near_core:
                conv_dir = _cardinal_toward(move_pos, core)
                if conv_dir != Direction.CENTRE:
                    if ct.can_build_conveyor(move_pos, conv_dir):
                        ct.build_conveyor(move_pos, conv_dir)

        if ct.can_build_road(move_pos):
            ct.build_road(move_pos)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            return

        for alt in (move_dir.rotate_left(), move_dir.rotate_right(),
                    move_dir.rotate_left().rotate_left(),
                    move_dir.rotate_right().rotate_right()):
            alt_pos = pos.add(alt)
            if core is not None:
                near_core = _dist_sq(alt_pos, core) <= _CONV_RADIUS_SQ
                if near_core:
                    conv_dir = _cardinal_toward(alt_pos, core)
                    if conv_dir != Direction.CENTRE:
                        if ct.can_build_conveyor(alt_pos, conv_dir):
                            ct.build_conveyor(alt_pos, conv_dir)
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
