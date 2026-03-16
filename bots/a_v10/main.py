# a_v10: Based on a_v8. Keep builders alive (retreat if far from core, unstick),
# respawn builders when they die (scale-gated), systematic sweep scouting,
# NO gunners. Axionite-aware ore priority for tiebreaker.

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
    def __init__(self):
        self.core_center: Position | None = None
        self.target_ore: Position | None = None
        self.steps: int = 0
        self.phase: str = "find_ore"
        self.chain_stuck: int = 0
        self.harvesters_built: int = 0
        # Core state
        self.num_spawned: int = 0
        self.last_builder_count: int = 0
        self.last_count_round: int = 0
        # Builder state
        self.explore_quadrant: int = -1
        self.sweep_dir: Direction | None = None
        self.sweep_steps: int = 0
        self.last_pos: Position | None = None
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
        rnd = ct.get_current_round()

        # Count alive builders every 20 rounds (cheap)
        if rnd - self.last_count_round >= 20 or self.last_count_round == 0:
            alive = 0
            for eid in ct.get_nearby_entities():
                try:
                    if (ct.get_entity_type(eid) == EntityType.BUILDER_BOT
                            and ct.get_team(eid) == ct.get_team()):
                        alive += 1
                except Exception:
                    continue
            self.last_builder_count = alive
            self.last_count_round = rnd

        # Spawn logic: initial burst of 5, then replacement if builders die
        want_spawn = False
        if self.num_spawned < 5:
            # Initial burst — same as a_v8 but with scale gate
            if scale <= 250:
                if scale <= 180 or rnd % 2 == 0:
                    want_spawn = True
        else:
            # Replacement spawning: if few builders alive and scale is OK
            if self.last_builder_count < 3 and scale < 400 and rnd % 3 == 0:
                want_spawn = True

        if not want_spawn:
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
            # Initialize sweep direction based on quadrant
            q = self.explore_quadrant
            sweep_dirs = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
            self.sweep_dir = sweep_dirs[q]

        # Track stuck detection
        pos = ct.get_position()
        if self.last_pos is not None and pos.x == self.last_pos.x and pos.y == self.last_pos.y:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_pos = pos

        # Safety: retreat if too far from core
        if self.core_center and self.phase == "find_ore":
            dist = _dist_sq(pos, self.core_center)
            if dist > 625:  # >25 tiles away
                self.phase = "chain_back"
                self.target_ore = None
                self.chain_stuck = 0

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

        # Build harvester on adjacent ore
        best_ore_dir = None
        best_ore_score = 9999
        for d in DIRECTIONS:
            adj = pos.add(d)
            if ct.can_build_harvester(adj):
                env = ct.get_tile_env(adj)
                if env not in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    continue
                # Prefer axionite (tiebreaker: refined ax delivered first)
                # then titanium close to core
                type_bonus = -200 if env == Environment.ORE_AXIONITE else 0
                core_dist = _dist_sq(adj, self.core_center) if self.core_center else 0
                score = type_bonus + core_dist
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

        self._move_with_conveyor(ct, pos, move_dir)
        self.steps += 1

    def _explore_dir(self, pos: Position) -> Direction:
        cc = self.core_center
        if cc is None:
            return random.choice(DIRECTIONS)

        # Systematic sweep: walk in sweep_dir for 30 steps, then rotate 90
        if self.stuck_count >= 5:
            # Stuck — pick a new random direction
            self.sweep_dir = random.choice(DIRECTIONS)
            self.sweep_steps = 0
            self.stuck_count = 0

        if self.sweep_steps >= 30:
            # Rotate 90 degrees and sweep back
            self.sweep_dir = self.sweep_dir.rotate_left().rotate_left()  # 90 deg
            self.sweep_steps = 0

        self.sweep_steps += 1

        # Add slight randomness to avoid walls
        if random.random() < 0.15:
            return random.choice([self.sweep_dir.rotate_left(), self.sweep_dir.rotate_right()])
        return self.sweep_dir

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
                # Prefer axionite for tiebreaker advantage
                type_bonus = -200 if env == Environment.ORE_AXIONITE else 0
                score = d_bot + d_core * 2 + type_bonus
                if score < best_score:
                    best_score = score
                    best = tpos
        if best is not None:
            self.target_ore = best

    # -- PHASE: CHAIN BACK TO CORE -------------------------------------------

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

        card_dir = _cardinal_toward(pos, core)
        if card_dir == Direction.CENTRE:
            self.phase = "find_ore"
            return

        next_pos = pos.add(card_dir)

        # Corner detection
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

        self._move_with_conveyor(ct, pos, move_dir)

    # -- MOVEMENT WITH CONVEYORS ---------------------------------------------

    def _move_with_conveyor(self, ct: Controller, pos: Position, move_dir: Direction) -> None:
        if move_dir == Direction.CENTRE:
            return

        core = self.core_center
        move_pos = pos.add(move_dir)

        if core is not None:
            conv_dir = _cardinal_toward(move_pos, core)
            if conv_dir != Direction.CENTRE:
                if ct.can_build_conveyor(move_pos, conv_dir):
                    ct.build_conveyor(move_pos, conv_dir)

        if ct.can_build_road(move_pos):
            ct.build_road(move_pos)

        if ct.can_move(move_dir):
            ct.move(move_dir)
            return

        for alt in [move_dir.rotate_left(), move_dir.rotate_right(),
                    move_dir.rotate_left().rotate_left(),
                    move_dir.rotate_right().rotate_right()]:
            alt_pos = pos.add(alt)
            if core is not None:
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
