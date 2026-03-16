# Core

Your team's primary structure. **If destroyed, you lose immediately.**

| Attribute | Value |
|-|-|
| HP | 500 |
| Size | 3x3 tiles |
| Vision r² | 36 |
| Action r² | 8 (from center) |

## Spawning

- Spawns **one builder bot per round** on any of the 9 tiles it occupies.
- Costs one action cooldown.
- Use `can_spawn(pos)` to check, `spawn_builder(pos)` to execute.

## Resource Hub

Resources delivered to the core via conveyors enter your team's global pool for building.
