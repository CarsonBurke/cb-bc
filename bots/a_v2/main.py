# a_v2: Fixed conveyor chains, eliminated road spam, relaxed spawn throttling.
# Changes from a_v1:
#   - Spawn every round until scale > 200 (was throttling at 130)
#   - Builders walk on conveyors not roads — conveyors toward core serve as paths
#   - Chain-back: build conveyor at next tile toward core, step onto it, repeat.
#     Each conveyor's output feeds into the next one's input side = connected chain.
#   - No road building during exploration — conveyors toward core instead
#   - Roads only as last resort when completely stuck
#   - Cap at 3 harvesters per builder, then wander dropping conveyors

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

    def run(self, ct: Controller) -> None:
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    # -- CORE ----------------------------------------------------------------

    def _run_core(self, ct: Controller) -> None:
        scale = ct.get_scale_percent()
        rnd = ct.get_current_round()

        # Spawn aggressively early, only slow down at high scale
        if scale > 300:
            if rnd % 3 != 0:
                return
        elif scale > 200:
            if rnd % 2 != 0:
                return
        # else: spawn every round

        pos = ct.get_position()
        dirs = list(DIRECTIONS)
        random.shuffle(dirs)
        for d in dirs:
            sp = pos.add(d)
            if ct.can_spawn(sp):
                ct.spawn_builder(sp)
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

        # Try to build harvester on adjacent ore (immediate priority)
        if self.harvesters_built < 3:
            for d in DIRECTIONS:
                adj = pos.add(d)
                if ct.can_build_harvester(adj):
                    ct.build_harvester(adj)
                    self.target_ore = adj
                    self.harvesters_built += 1
                    self.phase = "chain_back"
                    self.chain_stuck = 0
                    return

        # After 3 harvesters, wander dropping conveyors toward core
        if self.harvesters_built >= 3:
            self._build_opportunistic_conveyor(ct, pos)
            self._move_no_road(ct, pos, self._explore_dir(pos))
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

        self._move_no_road(ct, pos, move_dir)
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

    def _build_opportunistic_conveyor(self, ct: Controller, pos: Position) -> None:
        """After done with harvesters, drop conveyors pointing toward core."""
        if self.core_center is None:
            return
        for d in DIRECTIONS:
            cp = pos.add(d)
            conv_dir = cp.direction_to(self.core_center)
            if conv_dir == Direction.CENTRE:
                conv_dir = d
            if ct.can_build_conveyor(cp, conv_dir):
                ct.build_conveyor(cp, conv_dir)
                return

    # -- PHASE: BUILD CONVEYOR CHAIN BACK TO CORE ----------------------------

    def _phase_chain_back(self, ct: Controller) -> None:
        pos = ct.get_position()
        core = self.core_center
        if core is None:
            self.phase = "find_ore"
            return

        dist = _dist_sq(pos, core)

        # Close enough to core — chain complete
        if dist <= 4:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        # Safety: if stuck too long, give up
        self.chain_stuck += 1
        if self.chain_stuck > 50:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        # Strategy: build conveyor at next tile toward core, then step onto it.
        # Each conveyor faces toward core, so output -> next conveyor's input.
        move_dir = pos.direction_to(core)
        if move_dir == Direction.CENTRE:
            self.phase = "find_ore"
            return

        next_pos = pos.add(move_dir)
        conv_dir = next_pos.direction_to(core)
        if conv_dir == Direction.CENTRE:
            conv_dir = move_dir

        # Build conveyor (no road fallback — conveyors only for chains)
        if ct.can_build_conveyor(next_pos, conv_dir):
            ct.build_conveyor(next_pos, conv_dir)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            return

        # Blocked — try alternates with conveyors
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

        # Last resort: road to get unstuck
        for alt in alternates:
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return

    # -- MOVEMENT HELPER (no road spam) ---------------------------------------

    def _move_no_road(self, ct: Controller, pos: Position, move_dir: Direction) -> None:
        """Move by building conveyors toward core instead of roads."""
        if move_dir == Direction.CENTRE:
            return

        core = self.core_center
        move_pos = pos.add(move_dir)

        # Build conveyor facing toward core (doubles as path + resource transport)
        if core is not None:
            conv_dir = move_pos.direction_to(core)
            if conv_dir == Direction.CENTRE:
                conv_dir = move_dir
            if ct.can_build_conveyor(move_pos, conv_dir):
                ct.build_conveyor(move_pos, conv_dir)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            return

        # Try nearby directions
        for alt in [move_dir.rotate_left(), move_dir.rotate_right(),
                    move_dir.rotate_left().rotate_left(),
                    move_dir.rotate_right().rotate_right()]:
            alt_pos = pos.add(alt)
            if core is not None:
                conv_dir = alt_pos.direction_to(core)
                if conv_dir == Direction.CENTRE:
                    conv_dir = alt
                if ct.can_build_conveyor(alt_pos, conv_dir):
                    ct.build_conveyor(alt_pos, conv_dir)
            if ct.can_move(alt):
                ct.move(alt)
                return

        # Last resort: road
        for alt in [move_dir] + DIRECTIONS:
            alt_pos = pos.add(alt)
            if ct.can_build_road(alt_pos):
                ct.build_road(alt_pos)
            if ct.can_move(alt):
                ct.move(alt)
                return
