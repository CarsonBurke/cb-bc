# turrets.py: combat logic for gunner, sentinel, breach, and launcher turrets.
# Stdlib-only — no torch or third-party packages.

from cambc import Controller, EntityType

# Entities worth spending turret ammo on, by priority (lower = better).
_TARGET_PRIORITY = {
    EntityType.GUNNER: 0,
    EntityType.SENTINEL: 0,
    EntityType.BREACH: 0,
    EntityType.LAUNCHER: 0,
    EntityType.CORE: 1,
    EntityType.BUILDER_BOT: 2,
    EntityType.HARVESTER: 3,
    EntityType.FOUNDRY: 3,
    # Junk: low-value but may block line of fire (especially for gunner).
    EntityType.BARRIER: 4,
    EntityType.CONVEYOR: 4,
    EntityType.ARMOURED_CONVEYOR: 4,
    EntityType.SPLITTER: 4,
    EntityType.BRIDGE: 4,
    EntityType.ROAD: 4,
}
# Breach uses refined axionite — only fire at high-value targets (turrets/core/bots).
_BREACH_MAX_PRIORITY = 2
# Sentinel is expensive (10 Ti + 4-round CD) — skip junk.
_SENTINEL_MAX_PRIORITY = 3
# Gunner is cheap (2 Ti) — clear junk blocking line of fire.
_GUNNER_MAX_PRIORITY = 4


def run_turret(ct: Controller, etype: EntityType) -> None:
    if ct.get_action_cooldown() > 0:
        return
    if ct.get_ammo_amount() <= 0:
        return

    if etype == EntityType.BREACH:
        max_pri = _BREACH_MAX_PRIORITY
    elif etype == EntityType.SENTINEL:
        max_pri = _SENTINEL_MAX_PRIORITY
    else:
        max_pri = _GUNNER_MAX_PRIORITY
    my_team = ct.get_team()

    # Gunner fast path: built-in line-of-sight targeting.
    if etype == EntityType.GUNNER:
        target_pos = ct.get_gunner_target()
        if target_pos is None:
            return
        # Find entity at target — tile queries miss units (turrets/cores),
        # so fall back to scanning nearby entities by position.
        eid = ct.get_tile_building_id(target_pos)
        if eid is None:
            eid = ct.get_tile_builder_bot_id(target_pos)
        if eid is None:
            for candidate in ct.get_nearby_entities():
                if ct.get_position(candidate) == target_pos:
                    eid = candidate
                    break
        if eid is None:
            return
        if ct.get_team(eid) == my_team:
            return
        et = ct.get_entity_type(eid)
        pri = _TARGET_PRIORITY.get(et, 99)
        if pri <= max_pri:
            ct.fire(target_pos)
        return

    # Sentinel / Breach: scan nearby entities for best target.
    best_pos = None
    best_pri = max_pri + 1
    # Breach splash can hit friendlies adjacent to target but outside vision r²=10.
    # Use wider radius (18) to catch them: (sqrt(5) + sqrt(2))² ≈ 13.4.
    nearby = ct.get_nearby_entities(18 if etype == EntityType.BREACH else None)

    for eid in nearby:
        if ct.get_team(eid) == my_team:
            continue
        et = ct.get_entity_type(eid)
        pri = _TARGET_PRIORITY.get(et, 99)
        if pri >= best_pri:
            continue
        pos = ct.get_position(eid)
        if not ct.can_fire(pos):
            continue
        # Breach friendly-fire check: skip if any friendly entity adjacent to target.
        if etype == EntityType.BREACH:
            skip = False
            for nid in nearby:
                if ct.get_team(nid) != my_team:
                    continue
                npos = ct.get_position(nid)
                if npos.distance_squared(pos) <= 2:
                    skip = True
                    break
            if skip:
                continue
        best_pos = pos
        best_pri = pri

    if best_pos is not None:
        ct.fire(best_pos)


def run_launcher(ct: Controller) -> None:
    if ct.get_action_cooldown() > 0:
        return

    my_team = ct.get_team()
    my_pos = ct.get_position()
    nearby = ct.get_nearby_entities()

    # Find adjacent friendly builder bots to throw.
    bots = []
    for eid in nearby:
        if ct.get_team(eid) != my_team:
            continue
        if ct.get_entity_type(eid) != EntityType.BUILDER_BOT:
            continue
        bpos = ct.get_position(eid)
        if bpos.distance_squared(my_pos) <= 2:
            bots.append(bpos)

    if not bots:
        return

    # Find best enemy target we can launch a bot at.
    best_target = None
    best_pri = _SENTINEL_MAX_PRIORITY + 1
    for eid in nearby:
        if ct.get_team(eid) == my_team:
            continue
        et = ct.get_entity_type(eid)
        pri = _TARGET_PRIORITY.get(et, 99)
        if pri >= best_pri:
            continue
        pos = ct.get_position(eid)
        for bpos in bots:
            if ct.can_launch(bpos, pos):
                best_target = (bpos, pos)
                best_pri = pri
                break

    if best_target is not None:
        ct.launch(*best_target)
