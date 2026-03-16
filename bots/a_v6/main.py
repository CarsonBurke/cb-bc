# a_v6: Smart economy bot with directional exploration and ore prioritization.
# Based on b_v3 (best win rate). Key changes:
#   - Quadrant-biased exploration: builders spread in 4 directions based on spawn position.
#   - Ore scoring: prefer titanium (+100 penalty for axionite) and ore close to core (2x weight).
#   - Boomerang exploration: builders return when >20 tiles from core to avoid getting lost.
#   - Better stuck detection: position-tracking (8 rounds same pos = give up chain).
#   - Self-destruct on enemy core if encountered.
#   - No harvester cap — keep building for tiebreaker (harvesters alive).
#   - Same scale thresholds as b_v3 (6 builders, stop at 200%, slow at 150%).

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
        # Builder identity for directional exploration
        self.builder_id: int = -1
        self.explore_bias: Direction | None = None
        # Track previous position to detect being truly stuck
        self.prev_pos: Position | None = None
        self.stuck_count: int = 0

    def run(self, ct: Controller) -> None:
        etype = ct.get_entity_type()
        if etype == EntityType.CORE:
            self._run_core(ct)
        elif etype == EntityType.BUILDER_BOT:
            self._run_builder(ct)

    # -- CORE ----------------------------------------------------------------

    def _run_core(self, ct: Controller) -> None:
        scale = ct.get_scale_percent()

        # Cap builders at 6 (same as b_v3 — enough coverage without excess)
        if self.num_spawned >= 6:
            return

        # Scale-aware throttling
        if scale > 200:
            return
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
            # Assign exploration bias based on spawn position relative to core
            pos = ct.get_position()
            if self.core_center is not None:
                dx = pos.x - self.core_center.x
                dy = pos.y - self.core_center.y
                # Pick a quadrant-based exploration direction
                if dx >= 0 and dy <= 0:
                    self.explore_bias = Direction.NORTHEAST
                elif dx < 0 and dy <= 0:
                    self.explore_bias = Direction.NORTHWEST
                elif dx >= 0 and dy > 0:
                    self.explore_bias = Direction.SOUTHEAST
                else:
                    self.explore_bias = Direction.SOUTHWEST

        # Check for enemy core to kamikaze
        self._check_enemy(ct)

        if self.phase == "find_ore":
            self._phase_find_ore(ct)
        elif self.phase == "chain_back":
            self._phase_chain_back(ct)

        # Track position for stuck detection
        self.prev_pos = ct.get_position()

    def _find_core(self, ct: Controller) -> Position:
        for eid in ct.get_nearby_entities():
            try:
                if ct.get_entity_type(eid) == EntityType.CORE and ct.get_team(eid) == ct.get_team():
                    return ct.get_position(eid)
            except Exception:
                continue
        return ct.get_position()

    def _check_enemy(self, ct: Controller) -> None:
        """Self-destruct if on enemy core tile."""
        pos = ct.get_position()
        bid = ct.get_tile_building_id(pos)
        if bid is not None:
            try:
                if ct.get_team(bid) != ct.get_team():
                    if ct.get_entity_type(bid) == EntityType.CORE:
                        ct.self_destruct()
            except Exception:
                pass

    # -- PHASE: FIND ORE ------------------------------------------------------

    def _phase_find_ore(self, ct: Controller) -> None:
        pos = ct.get_position()

        # Try to build harvester on adjacent ore — pick best one
        best_adj = None
        best_score = 99999
        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                env = ct.get_tile_env(adj)
                core_dist = _dist_sq(adj, self.core_center) if self.core_center else 0
                score = core_dist
                if env == Environment.ORE_AXIONITE:
                    score += 100  # strongly prefer titanium
                if score < best_score:
                    best_score = score
                    best_adj = adj

        if best_adj is not None:
            ct.build_harvester(best_adj)
            self.target_ore = best_adj
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
        """Biased exploration: push in assigned quadrant, with some randomness."""
        if self.explore_bias is not None and self.core_center is not None:
            # Use bias as primary direction, but if far from core, come back
            dist = _dist_sq(pos, self.core_center)
            if dist > 400:  # >20 tiles away, come back
                return pos.direction_to(self.core_center)
            base_dir = self.explore_bias
            if random.random() < 0.3:
                base_dir = random.choice([base_dir.rotate_left(), base_dir.rotate_right()])
            return base_dir
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
                score = d_bot + d_core * 2  # heavily weight proximity to core
                if env == Environment.ORE_AXIONITE:
                    score += 100
                if score < best_score:
                    best_score = score
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

        # Close enough to core — chain complete
        if dist <= 4:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            return

        # Stuck detection: if position hasn't changed
        if self.prev_pos is not None and pos.x == self.prev_pos.x and pos.y == self.prev_pos.y:
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        # Safety: if stuck too long, give up and go find ore
        self.chain_stuck += 1
        if self.chain_stuck > 40 or self.stuck_count > 8:
            self.phase = "find_ore"
            self.target_ore = None
            self.steps = 0
            self.stuck_count = 0
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

        # Last resort: any direction
        for d in DIRECTIONS:
            dp = pos.add(d)
            if ct.can_build_road(dp):
                ct.build_road(dp)
            if ct.can_move(d):
                ct.move(d)
                return
