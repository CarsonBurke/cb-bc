# API Reference

All interactions go through `Controller` passed to `Player.run(c: Controller)`.

Import: `from cambc import Controller, Direction, EntityType, Environment, Position, ResourceType`

## Sensing

- `get_position(id?)`, `get_team(id?)`, `get_id()`, `get_hp(id?)`, `get_max_hp(id?)`, `get_entity_type(id?)`, `get_direction(id?)`, `get_vision_radius_sq(id?)`
- `get_action_cooldown()`, `get_move_cooldown()`
- `get_tile_env(pos)` → Environment (`EMPTY`, `WALL`, `ORE_TITANIUM`, `ORE_AXIONITE`)
- `is_tile_empty(pos)`, `is_tile_passable(pos)`, `is_in_vision(pos)`
- `get_tile_building_id(pos)`, `get_tile_builder_bot_id(pos)`
- `get_nearby_tiles(dist_sq?)`, `get_nearby_entities(dist_sq?)`, `get_nearby_buildings(dist_sq?)`, `get_nearby_units(dist_sq?)`
- `get_stored_resource(id?)` → ResourceType | None
- `get_marker_value(id)` → int

## Turret-Specific

- `get_ammo_amount()`, `get_ammo_type()` → ResourceType | None
- `get_gunner_target()` → Position | None

## Building-Specific

- `get_bridge_target(id)` → Position

## Actions

Every action has a `can_*()` → bool check variant.

- **Move:** `move(dir)` / `can_move(dir)`
- **Spawn (core):** `spawn_builder(pos)` / `can_spawn(pos)`
- **Build (directional):** `build_conveyor(pos, dir)`, `build_splitter(pos, dir)`, `build_armoured_conveyor(pos, dir)`, `build_gunner(pos, dir)`, `build_sentinel(pos, dir)`, `build_breach(pos, dir)`
- **Build (bridge):** `build_bridge(pos, target)` — target within r²≤9
- **Build (positional):** `build_harvester(pos)`, `build_road(pos)`, `build_barrier(pos)`, `build_foundry(pos)`, `build_launcher(pos)`
- **Heal:** `heal(pos)` — 10 HP to all friendly entities on tile
- **Destroy:** `destroy(pos)` — allied buildings, no cooldown cost
- **Self-destruct:** `self_destruct()` — 20 dmg to buildings on tile
- **Combat:** `fire(target)`, `launch(bot_pos, target)`
- **Markers:** `place_marker(pos, value)` — one per round per unit

## Costs

`get_*_cost()` → (ti, ax) for: conveyor, splitter, bridge, armoured_conveyor, harvester, road, barrier, gunner, sentinel, breach, launcher, foundry, builder_bot

## State

- `get_global_resources()` → (ti, ax)
- `get_scale_percent()` → float
- `get_current_round()`, `get_map_width()`, `get_map_height()`
- `get_cpu_time_elapsed()` → int

## Debug

- `draw_indicator_line(a, b, r, g, b)`, `draw_indicator_dot(pos, r, g, b)`

## Types

- **Direction:** 8 cardinal/ordinal + `CENTRE`. Methods: `delta()`, `rotate_left()`, `rotate_right()`, `opposite()`.
- **Position:** Named tuple (x, y). Methods: `add(dir)`, `distance_squared(other)`, `direction_to(other)`.
- **EntityType:** `BUILDER_BOT`, `CORE`, `GUNNER`, `SENTINEL`, `BREACH`, `LAUNCHER`, `CONVEYOR`, `SPLITTER`, `ARMOURED_CONVEYOR`, `BRIDGE`, `HARVESTER`, `FOUNDRY`, `ROAD`, `BARRIER`, `MARKER`
- **ResourceType:** `TITANIUM`, `RAW_AXIONITE`, `REFINED_AXIONITE`
- **Environment:** `EMPTY`, `WALL`, `ORE_TITANIUM`, `ORE_AXIONITE`
