# Resources

## Types

- **Titanium (Ti):** Main currency. Harvested from titanium ore tiles. Starting amount: 1000.
- **Raw Axionite:** Harvested from axionite ore tiles. Decays into titanium when consumed directly. Must be refined by a Foundry to be useful.
- **Refined Axionite:** Produced by foundries (raw axionite + titanium input). Powers advanced units (breach turrets, armoured conveyors) and buffs (gunner 2x damage, sentinel stun).

## Transport

Resources move in **stacks of 10** at end of round. Buildings that produce resources send outputs to adjacent accepting buildings.

**WARNING:** Resources CAN be sent to enemy buildings. Careful with conveyor placement near opponents.

## Delivery

Resources must reach your Core via conveyor chains to enter your global resource pool. Without delivery infrastructure, harvested resources sit idle.

## Cost Scaling

Every entity built increases a global cost multiplier. Formula: `cost = floor(scale × base_cost)`

| Entity Type | Scale Contribution |
|-|-|
| Roads | +0.5% |
| Conveyors, barriers, bridges | +1% |
| Combat units, harvesters | +10% |
| Foundries | +100% |

Destroying entities removes their scaling contribution. Query current scale: `c.get_scale_percent()`.
