# Ablation Report

## Current Rankings

Round 7 (2026-03-16T21:53:20+00:00)

| Rank | Bot | ELO | W | L | D |
|-|-|-|-|-|-|
| 1 | a_v10 | 27 | 47 | 13 | 0 |
| 2 | b_v11 | 27 | 42 | 18 | 0 |
| 3 | a_v8 | 25 | 40 | 20 | 0 |
| 4 | a_v12 | 20 | 20 | 40 | 0 |
| 5 | b_v12 | 19 | 17 | 43 | 0 |
| 6 | starter | 19 | 14 | 46 | 0 |

## ELO Progression

| Round | a_v2 | a_v6 | b_v3 | b_v6 | starter | a_v7 | b_v7 | a_v8 | b_v8 | a_v9 | b_v9 | a_v10 | b_v10 | a_v11 | b_v11 | a_v12 | b_v12 |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 1 | 20 | 19 | 19 | 21 | 18 | - | - | - | - | - | - | - | - | - | - | - | - |
| 2 | 21 | - | - | 20 | 18 | 20 | 19 | - | - | - | - | - | - | - | - | - | - |
| 3 | 21 | - | - | 20 | 19 | 21 | - | 25 | 22 | - | - | - | - | - | - | - | - |
| 4 | - | - | - | - | 19 | - | - | 25 | 21 | 25 | 17 | - | - | - | - | - | - |
| 5 | - | - | - | - | 19 | - | - | 25 | 22 | - | - | 24 | 23 | - | - | - | - |
| 6 | - | - | - | - | 18 | - | - | 25 | - | - | - | 26 | - | 21 | 26 | - | - |
| 7 | - | - | - | - | 19 | - | - | 25 | - | - | - | 27 | - | - | 27 | 20 | 19 |

## Win Rate Matrix (Latest Round)

| | a_v10 | a_v12 | a_v8 | b_v11 | b_v12 | starter |
|-|-|-|-|-|-|-|
| a_v10 | - | 83% | 58% | 67% | 92% | 92% |
| a_v12 | 17% | - | 8% | 25% | 50% | 67% |
| a_v8 | 42% | 92% | - | 42% | 75% | 83% |
| b_v11 | 33% | 75% | 58% | - | 92% | 92% |
| b_v12 | 8% | 50% | 25% | 8% | - | 50% |
| starter | 8% | 33% | 17% | 8% | 50% | - |

## Bot Lineage

```
starter  (Starter bot - a simple example to demonstrate usage of the Controller API.)
|-- a_v1  (a_v1: Economy-focused bot — continuous builder spawning, systematic ore search,)
|   `-- a_v2  (a_v2: Fixed conveyor chains, eliminated road spam, relaxed spawn throttling.)
|       `-- a_v3  (a_v3: Smarter early game — harvest near core first, roads for movement, conveyors only for transport chains.)
|           `-- a_v4  (a_v4: Based on b_v3 (best win rate). Changes:)
|               `-- a_v5  (a_v5: BFS pathfinding for conveyor chains. Instead of direction_to(core) zigzag,)
|                   `-- a_v6  (a_v6: Smart economy bot with directional exploration and ore prioritization.)
|                       `-- a_v7  (a_v7: Based on b_v6. Changes:)
|                           `-- a_v8  (a_v8: Hybrid approach — conveyor-dense exploration (from a_v2) + quadrant scouting)
|                               `-- a_v9  (a_v9: Never-stop exploring (remove patrol phase + harvester cap) + defensive gunners)
|                                   `-- a_v10  (a_v10: Based on a_v8. Keep builders alive (retreat if far from core, unstick),)
|                                       `-- a_v11  (a_v11: Based on a_v8. Builder survival (stuck detect + retreat), minimal respawn)
|                                           `-- a_v12  (a_v12: Roads-only exploration, conveyors only in chain_back + 10 post-chain steps.)
|-- b_v1  (b_v1: Rapid expansion + military aggression. Aggressive spawning, economy)
|   `-- b_v2  (b_v2: Fixed economy pipeline — builders walk back to core placing conveyors,)
|       `-- b_v3  (b_v3: Roads for movement, conveyors ONLY in harvester->core chains, capped builders,)
|           `-- b_v4  (b_v4: Fixed conveyor chain connectivity via deferred placement.)
|               `-- b_v5  (b_v5: Economy + defense gunners + late-game launcher raids.)
|                   `-- b_v6  (b_v6: Ultra-lean economy. Only 4 builders, near-core ore priority, titanium-first,)
|                       `-- b_v7  (b_v7: b_v6 exact chain logic + quadrant scouting + axionite harvesting in patrol.)
|                           `-- b_v8  (b_v8: L-shaped conveyor chains with guaranteed connectivity. Quadrant scouting.)
|                               `-- b_v9  (b_v9: Based on a_v8. Roads for exploration (not conveyors) to save scale.)
|                                   `-- b_v10  (b_v10: Based on a_v8. Key changes: (1) Opposite-of-movement conveyor facing)
|                                       `-- b_v11  (b_v11: Based on a_v8. Key changes: (1) Cap exploration conveyors to dist_sq<=200)
|                                           `-- b_v12  (b_v12: Fix conveyor placement. (1) Exploration builds roads only (no conveyors).)
`-- c_v1  (c_v1: Planned base layout + efficient expansion.)
```

## Economy Stats (avg end-game)

| Bot | Ti | Scale | Harvesters | Conveyors | Roads | Builders | Core HP |
|-|-|-|-|-|-|-|-|
| a_v10 | 4595 | 504.6% | 1.4 | 41.1 | 8.3 | 3.1 | 500 |
| b_v11 | 6208 | 294.8% | 1.5 | 29.9 | 0.7 | 1.7 | 500 |
| a_v8 | 5930 | 291.3% | 1.5 | 27.9 | 0.3 | 1.6 | 500 |
| a_v12 | 1027 | 253.3% | 1.5 | 12.9 | 17.7 | 1.5 | 500 |
| b_v12 | 186 | 245.6% | 1.5 | 10.8 | 17.8 | 1.8 | 500 |
| starter | 9 | 271.1% | 1.2 | 0.0 | 55.0 | 0.8 | 500 |

## Key Stats

- **Best bot**: a_v10 (ELO 27)
- **Biggest ELO jump**: a_v10 (+1)
- **Matches in latest round**: 180

