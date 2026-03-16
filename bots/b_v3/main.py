# b_v3: Roads for movement, conveyors ONLY in harvester->core chains, capped builders,
# scale-aware spawning. Fixes a_v2's conveyor spam that inflated scale and lost to starter.
# Changes from a_v2:
#   - Cap builders at 6 (was unlimited). Each builder = +10% scale.
#   - Use roads for movement, not conveyors. Roads = 0.5% vs conveyor = 1%.
#   - Conveyors ONLY built during chain_back phase (harvester -> core).
#   - No opportunistic conveyor spam after 3 harvesters.
#   - Builders keep finding ore after 3 harvesters (no cap) for tiebreaker.
#   - Spawn throttling: stop at scale 200%, slow at 150%.

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
        self.phase: str = "find_ore"  # find_ore -> chain_back -> find_ore
        self.chain_stuck: int = 0
        self.harvesters_built: int = 0
        self.num_spawned: int = 0

    def run(self, ct: Controller) -> None:
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    # -- CORE ----------------------------------------------------------------

    def _run_core(self, ct: Controller) -> None:
        scale = ct.get_scale_percent()

        # Hard cap on builders - each is +10% scale
        if self.num_spawned >= 6:
            return

        # Scale-aware throttling
        if scale > 200:
            return  # stop spawning entirely
        if scale > 150:
            if ct.get_current_round() % 3 != 0:
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

        # Try to build harvester on adjacent ore
        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                ct.build_harvester(adj)
                self.target_ore = adj
                self.harvesters_built += 1
                self.phase = "chain_back"
                self.chain_stuck = 0
                return

        # Scan for ore targets
        if self.target_ore is None or self.steps > 25:
            self._scan_for_ore(ct)
            self.steps = 0

        # Move toward target or explore
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
        """Direction to explore: away from core with randomness."""
        if self.core_center is not None:
            base_dir = self.core_center.direction_to(pos)
            if base_dir == Direction.CENTRE:
                base_dir = random.choice(DIRECTIONS)
            if random.random() < 0.3:
                base_dir = random.choice([base_dir.rotate_left(), base_dir.rotate_right()])
            return base_dir
        return random.choice(DIRECTIONS)

    def _scan_for_ore(self, ct: Controller) -> None:
        pos = ct.get_position()
        best_dist = 9999
        best = None
        for tpos in ct.get_nearby_tiles():
            env = ct.get_tile_env(tpos)
            if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                if ct.get_tile_building_id(tpos) is not None:
                    continue
                d = _dist_sq(pos, tpos)
                if d < best_dist:
                    best_dist = d
                    best = tpos
        if best is not None:
            self.target_ore = best

    # -- PHASE: BUILD CONVEYOR CHAIN BACK TO CORE ----------------------------

    def _phase_chain_back(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center
        if core is None:
            self.phase = "find_ore"
            return

        dist = _dist_sq(pos, core)

        # Close enough to core - chain complete
        if dist <= 4:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        # Safety: if stuck too long, give up and go find ore
        self.chain_stuck += 1
        if self.chain_stuck > 50:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        # Strategy: build conveyor at next tile toward core, then step onto it.
        move_dir = pos.direction_to(core)
        if move_dir == Direction.CENTRE:
            self.phase = "find_ore"
            return

        next_pos = pos.add(move_dir)
        conv_dir = next_pos.direction_to(core)
        if conv_dir == Direction.CENTRE:
            conv_dir = move_dir

        # Build conveyor (this is a chain tile, so conveyor is justified)
        if ct.can_build_conveyor(next_pos, conv_dir):
            ct.build_conveyor(next_pos, conv_dir)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            return

        # Blocked - try alternates with conveyors (still part of chain)
        alternates = [
            move_dir.rotate_left(), move_dir.rotate_right(),
            move_dir.rotate_left().rotate_left(),
            move_dir.rotate_right().rotate_right(),
        ]
        for alt in alternates:
            alt_pos = pos.add(alt)
            alt_conv = alt_pos.direction_to(core)
            if alt_conv == Direction.CENTRE:
                alt_conv = alt
            if ct.can_build_conveyor(alt_pos, alt_conv):
                ct.build_conveyor(alt_pos, alt_conv)
            if ct.can_move(alt):
                ct.move(alt)
                return

        # Last resort: road to get unstuck (cheap, 0.5%)
        for alt in alternates:
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return

    # -- MOVEMENT HELPER (roads only, no conveyor spam) -----------------------

    def _move_with_road(self, ct: Controller, pos: Position, move_dir: Direction) -> None:
        """Move by building roads. Conveyors are reserved for harvester chains only."""
        if move_dir == Direction.CENTRE:
            return

        move_pos = pos.add(move_dir)

        # Build road for movement (0.5% scale vs 1% for conveyor)
        if ct.can_build_road(move_pos):
            ct.build_road(move_pos)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            return

        # Try nearby directions with roads
        for alt in [move_dir.rotate_left(), move_dir.rotate_right(),
                    move_dir.rotate_left().rotate_left(),
                    move_dir.rotate_right().rotate_right()]:
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
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
