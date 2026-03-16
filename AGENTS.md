# AGENTS.md — Cambridge Battlecode Bot Engineer

You are an expert competitive game bot engineer. Your job is to iteratively improve bots for Cambridge Battlecode, run ablations, and maximize match performance.

## Environment Constraints

- **Python stdlib only.** The engine runs sandboxed CPython — no numpy, no pip packages, nothing outside the standard library.
- **2ms CPU per unit per round** with a 5% refillable buffer. Local runs (`cambc run`) have no time limit — always validate with `cambc test-run` for real perf.
- **Each unit gets its own `Player` instance.** Instance variables persist across rounds for that unit but are NOT shared between units. Inter-unit communication uses **markers** only.

## Performance

**2ms per unit per round is tight. Performance matters.** Prefer C-backed stdlib over pure Python loops wherever reasonable:

- **`itertools`** — `product`, `chain`, `islice`, `filterfalse` are C-implemented. Use instead of nested loops/list concatenation.
- **`min()`/`max()`/`sorted()` with `key=`** — C-level iteration, faster than manual loops to find best candidate.
- **`collections.deque`** for BFS queues instead of `list.pop(0)` (O(1) vs O(n)).
- **`set` lookups** over `list` membership checks. Pre-build sets for repeated `in` checks.
- **`sum()`, `any()`, `all()`** with generators — single C-level pass, no intermediate list.
- **`dict`/`defaultdict`** for O(1) spatial lookups (e.g., position → entity) instead of scanning lists.
- **Precompute at `__init__`** — direction lists, offset tables, anything that doesn't change per round.
- **Avoid** deep recursion, repeated `get_nearby_*` calls with same args, string operations in hot paths, and building large intermediate lists.

## Game Mechanics

Read `mechanics/` files for full details as needed. Start with `mechanics/api.md` for the Controller API.

**Quick summary:** Two teams, symmetric grid, each has a 3x3 Core (500 HP). Harvest ore → convey to core → build army. Destroy enemy core or win on tiebreakers (refined ax delivered > ti delivered > harvesters > stored resources > coinflip) at round 2000.

## Bot Structure

Each bot lives in `bots/<bot_name>/` with `main.py` as the entry point. The engine looks for a `Player` class with a `run(self, c: Controller)` method — one instance per unit, called each round. Beyond that, organize freely: multiple modules, helper files, shared utilities, etc. See `bots/starter/main.py` for a minimal example.

## CLI

All commands run via the project venv: `.venv/bin/cambc`. Map names require the `.map26` extension.

```bash
.venv/bin/cambc run <bot_a> <bot_b> [map.map26]       # Local match (no CPU limit)
.venv/bin/cambc run <a> <b> [map.map26] --watch        # Local match + open visualizer
.venv/bin/cambc test-run <a> <b> [map.map26]           # Remote match with 2ms CPU enforcement
.venv/bin/cambc watch [replay]                         # Open replay in browser
.venv/bin/cambc test-matches                           # Show remote test run history
.venv/bin/cambc submit <path>                          # Upload bot to ladder
```

Replays are written to `replays/`.

## Ablations

### Tooling

- **ELO tournament:** `python ablations/elo.py` — round-robin (both sides, all 6 maps), parallel. Writes `ablations/elo_results.json`.
- **Selective:** `python ablations/elo.py bot_a bot_b` — specific bots only.
- **History:** `python ablations/history.py` — appends ELO results to `ablations/history.json` with timestamps, bot lineage, and changelogs.
- **Report:** `python ablations/report.py` — generates `ablations/REPORT.md` with rankings, ELO progression, win rate matrix, lineage tree, and key stats.
- **Viewer:** `python ablations/viewer.py` — generates `ablations/viewer/dashboard.html` from template and opens in browser. Dashboard has sidebar with round selection, ELO progression chart, lineage node graph, rankings with deltas, win rate heatmap, ELO distribution bars, and collapsible match results grouped by map.
- **Full pipeline:** `python ablations/elo.py && python ablations/history.py && python ablations/report.py && python ablations/viewer.py`
- **Competitive ablation process:** Read and follow `ablations/ABLATION_PROMPT.md` — overseer loop with two parallel agent tracks (A and B), iterative improvement, ELO evaluation. **This is the default task when asked to run ablations.**

### Workflow

1. **Version, don't mutate.** `cp -r bots/v3 bots/v4`. Always keep the previous version for A/B comparison.
2. **One change per version.** Isolate variables for clear ablation signal.
3. **Add a changelog comment** at the top of `main.py`: `# v2: improved economy`. The history tracker reads this.
4. **Evaluate.** `python ablations/elo.py && python ablations/history.py && python ablations/report.py && python ablations/viewer.py`
5. **Remote validate.** `.venv/bin/cambc test-run` to confirm no CPU timeouts before submitting.
6. **Submit winners.** `.venv/bin/cambc submit bots/<best_bot>`

### Maps

`default_small1.map26` through `default_large2.map26` — all in `maps/`.

## Strategy

Read `mechanics/` files to understand the game deeply. Think about what wins: resource delivery, map control, cost efficiency, tiebreakers, combat. Study the starter bot and existing bots in `bots/` to understand what's been tried.
