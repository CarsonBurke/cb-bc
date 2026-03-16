# Turrets

All turrets are units (run code). Most face a direction and need ammo delivered via conveyors from non-facing sides.

## Gunner

| Attribute | Value |
|-|-|
| HP | 40 |
| Base Cost | 10 Ti |
| Damage | 10 (20 with refined axionite ammo) |
| Ammo Cost | 2 per shot |
| Cooldown | 1 round |
| Vision/Attack r² | 13 |

Fires at the **closest non-empty tile** in its facing direction.

## Sentinel

| Attribute | Value |
|-|-|
| HP | 30 |
| Base Cost | 15 Ti |
| Damage | 20 |
| Ammo Cost | 10 per shot |
| Cooldown | 4 rounds |
| Vision r² | 32 |

Hits tiles within one king's move of its facing line. With refined axionite ammo: **stuns** targets (+3 action and move cooldown).

## Breach

| Attribute | Value |
|-|-|
| HP | 60 |
| Base Cost | 30 Ti + 10 Ax |
| Damage | 40 direct + 20 splash |
| Ammo Cost | 5 (refined axionite ONLY) |
| Cooldown | 1 round |
| Vision r² | 10, Attack r² | 5 |

Attacks in a **180° cone** in facing direction. **Has friendly fire** on splash damage.

## Launcher

| Attribute | Value |
|-|-|
| HP | 30 |
| Base Cost | 20 Ti |
| Cooldown | 1 round |
| Vision/Attack r² | 26 |

**No facing direction. No ammo required.** Throws adjacent builder bots to target tile within range. Use for deep aggression or kamikaze strikes.
