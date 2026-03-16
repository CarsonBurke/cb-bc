# a_v7: Based on b_v6. Changes:
# 1. Quadrant-based exploration: each builder explores a different direction,
#    finding more ore deposits across the map.
# 2. Axionite harvesters also built (more harvesters = better tiebreaker).
# 3. Patrol continues exploring (not just orbiting core).
# 4. Slightly higher step limit before patrol (50 vs 40).

import random

from cambc import Controller, Direction, EntityType, Environment, Position

DIRECTIONS = [d for d in Direction if d != Direction.CENTRE]


def _dist_sq(a: Position, b: Position) -> int:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


class Player:
    def __init__(self):
        self.core_center: Position | None = None
        self.target_ore: Position | None = None
        self.steps: int = 0
        self.phase: str = "find_ore"
        self.chain_stuck: int = 0
        self.harvesters_built: int = 0
        self.num_spawned: int = 0
        self.patrol_dir: Direction | None = None
        self.explore_quadrant: int = -1

    def run(self, ct: Controller) -> None:
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    # -- CORE ----------------------------------------------------------------

    def _run_core(self, ct: Controller) -> None:
        scale = ct.get_scale_percent()
        if self.num_spawned >= 4:
            return
        if scale > 200:
            return
        if scale > 150 and ct.get_current_round() % 2 != 0:
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
        if self.core_center is None:
            self.core_center = self._find_core(ct)
            self.explore_quadrant = ct.get_id() % 4

        if self.phase == "find_ore":
            self._phase_find_ore(ct)
        elif self.phase == "chain_back":
            self._phase_chain_back(ct)
        elif self.phase == "patrol":
            self._phase_patrol(ct)

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

        # Build harvester on adjacent ore — prefer titanium, prefer near-core
        best_ore_dir = None
        best_ore_score = 9999
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
            self.target_ore = adj
            self.harvesters_built += 1
            self.phase = "chain_back"
            self.chain_stuck = 0
            return

        # After enough harvesters and searching, go patrol (but patrol still explores)
        if self.harvesters_built >= 3 and self.steps > 60:
            self.phase = "patrol"
            return

        # Scan for ore targets
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
        """Quadrant-based exploration so builders spread out."""
        cc = self.core_center
        if cc is None:
            return random.choice(DIRECTIONS)

        q = self.explore_quadrant
        target_dx = 18 if q in (0, 1) else -18
        target_dy = -18 if q in (0, 3) else 18
        target = Position(cc.x + target_dx, cc.y + target_dy)
        base_dir = pos.direction_to(target)

        if base_dir == Direction.CENTRE:
            self.explore_quadrant = (self.explore_quadrant + 1) % 4
            return random.choice(DIRECTIONS)

        if random.random() < 0.3:
            base_dir = random.choice([base_dir.rotate_left(), base_dir.rotate_right()])
        return base_dir

    def _scan_for_ore(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center
        best_score = 9999
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

    # -- PHASE: BUILD CONVEYOR CHAIN BACK TO CORE ----------------------------
    # (Identical to b_v6 — the zigzag chain_back works because consecutive
    # direction_to directions never face opposite, so chain connectivity holds)

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
        if self.chain_stuck > 50:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        move_dir = pos.direction_to(core)
        if move_dir == Direction.CENTRE:
            self.phase = "find_ore"
            return

        next_pos = pos.add(move_dir)

        already_passable = False
        if ct.is_in_vision(next_pos):
            already_passable = ct.is_tile_passable(next_pos)

        if already_passable:
            if ct.can_move(move_dir):
                ct.move(move_dir)
                return
        else:
            conv_dir = next_pos.direction_to(core)
            if conv_dir == Direction.CENTRE:
                conv_dir = move_dir
            if ct.can_build_conveyor(next_pos, conv_dir):
                ct.build_conveyor(next_pos, conv_dir)
            if ct.can_move(move_dir):
                ct.move(move_dir)
                return

        alternates = [
            move_dir.rotate_left(), move_dir.rotate_right(),
            move_dir.rotate_left().rotate_left(),
            move_dir.rotate_right().rotate_right(),
        ]
        for alt in alternates:
            alt_pos = pos.add(alt)
            alt_passable = False
            if ct.is_in_vision(alt_pos):
                alt_passable = ct.is_tile_passable(alt_pos)

            if alt_passable:
                if ct.can_move(alt):
                    ct.move(alt)
                    return
            else:
                alt_conv = alt_pos.direction_to(core)
                if alt_conv == Direction.CENTRE:
                    alt_conv = alt
                if ct.can_build_conveyor(alt_pos, alt_conv):
                    ct.build_conveyor(alt_pos, alt_conv)
                if ct.can_move(alt):
                    ct.move(alt)
                    return

        for alt in alternates:
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return

    # -- PHASE: PATROL --------------------------------------------------------

    def _phase_patrol(self, ct: Controller) -> None:
        """Patrol: heal + keep exploring for more ore."""
        pos = ct.get_position()
        core = self.core_center

        # Opportunistic harvester
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

        # Heal damaged nearby friendly buildings/units
        if ct.get_action_cooldown() == 0:
            for eid in ct.get_nearby_entities():
                try:
                    if ct.get_team(eid) == ct.get_team():
                        epos = ct.get_position(eid)
                        if _dist_sq(pos, epos) <= 2:
                            hp = ct.get_hp(eid)
                            max_hp = ct.get_max_hp(eid)
                            if hp < max_hp:
                                if ct.can_heal(epos):
                                    ct.heal(epos)
                                    return
                except Exception:
                    continue

        if core is None:
            return

        dist_to_core = _dist_sq(pos, core)

        # Keep exploring but don't go too far
        if dist_to_core > 80:
            move_dir = pos.direction_to(core)
        elif dist_to_core < 8:
            move_dir = core.direction_to(pos)
            if move_dir == Direction.CENTRE:
                move_dir = random.choice(DIRECTIONS)
        else:
            # Continue exploring quadrant
            move_dir = self._explore_dir(pos)

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
        for alt in [move_dir.rotate_left(), move_dir.rotate_right(),
                    move_dir.rotate_left().rotate_left(),
                    move_dir.rotate_right().rotate_right()]:
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
