# Builder Bot

The **only mobile unit**. Responsible for construction, healing, demolition, and scouting.

| Attribute | Value |
|-|-|
| HP | 30 |
| Base Cost | 10 Ti |
| Scale Contribution | +10% |
| Vision r² | 20 |
| Action r² | 2 |

## Movement

- 8-directional (including diagonals). Move cooldown +1 per move.
- **Walkable surfaces only:** conveyors (either team, any direction), roads (either team), allied core.
- Resource presence and conveyor direction do NOT restrict movement.

## Actions (when action cooldown = 0)

- **Build:** Construct any building/turret on unoccupied tile within action radius. Only conveyors and roads may be built on tiles containing builder bots.
- **Heal:** Restore 10 HP to all friendly entities on a tile within action radius.
- **Destroy:** Demolish allied buildings within action radius. No cooldown cost — unlimited per round.

## Self-Destruct

Available at any time. Deals **20 damage** to buildings on the current tile. (Core has 500 HP → 25 kamikazes to kill.)
