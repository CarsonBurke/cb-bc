# Game Overview

Two teams on a symmetric grid (20x20 to 50x50). Position (0,0) is northwest corner. Symmetry is either reflection or rotation.

Each team starts with a 3x3 **Core** (500 HP) and 1000 titanium.

**Objective:** Destroy the enemy core before round 2000.

**Tiebreakers (in order):** refined axionite delivered > titanium delivered > harvesters alive > axionite stored > titanium stored > coinflip.

## Turn Structure

- Units take turns in spawn order each round.
- Action and move cooldowns decrease by 1 per round.
- All units may place one marker per round within action radius.
- Each unit gets 2ms CPU time per round plus 5% refillable buffer.
- Resource distribution happens at end of round after all units act.

## Entity Classification

- **Units** (core, builder bots, turrets): execute independent code via `Player.run()`. Each gets its own `Player` instance — state is NOT shared between units.
- **Buildings** (harvesters, conveyors, roads, barriers, foundries): immovable, no code. Harvesters and foundries operate automatically.
- Inter-unit communication uses **markers** only (integer value, one per unit per round).
