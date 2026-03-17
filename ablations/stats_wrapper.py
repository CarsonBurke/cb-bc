"""Stats-collecting bot wrapper for tournament matches.

Creates a temporary bot directory that wraps an existing bot's Player class,
collecting per-turn metrics from the core's Controller. After the match,
stats are written to a JSON sidecar file.

Tracked metrics per snapshot (every 100 rounds):
  - global resources (titanium/axionite at core, available to spend)
  - cumulative income: total resources that arrived at the core pool
  - cumulative spent: total resources spent (building, spawning, etc.)
  - resources in transit: count of conveyor-type entities carrying resources
  - harvester count: proxy for mining throughput
  - unit/building counts by type (own and enemy visible)
  - core HP, cost scale %
"""

import json
import tempfile
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = PROJECT_ROOT / "replays" / "stats"


def create_wrapper(bot_main_py: str, team: str, match_id: int) -> str:
    """Create a temporary wrapper main.py that instruments the given bot.

    Returns path to the temporary main.py file.
    """
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = str(STATS_DIR / f"match_{match_id}_{team}.json")

    wrapper_code = textwrap.dedent(f'''\
        import sys
        import json
        import importlib.util
        from pathlib import Path

        _spec = importlib.util.spec_from_file_location("_orig_bot", {bot_main_py!r})
        _orig = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_orig)

        from cambc import Controller, EntityType, ResourceType, Environment

        SAMPLE_INTERVAL = 100
        STATS_PATH = {stats_path!r}

        # Conveyor-family types that can carry resources
        CARRIER_TYPES = {{
            EntityType.CONVEYOR, EntityType.SPLITTER,
            EntityType.ARMOURED_CONVEYOR, EntityType.BRIDGE,
            EntityType.FOUNDRY,
        }}

        _stats = {{
            "snapshots": [],
            "final": None,
        }}
        _last_round_sampled = {{}}
        _prev_resources = {{}}  # core_id -> (ti, ax) from previous sample
        _cumulative_income = {{"titanium": 0, "axionite": 0}}
        _cumulative_spent = {{"titanium": 0, "axionite": 0}}

        def _count_nearby(ct):
            """Count visible entities by type/team, plus resources in transit."""
            counts = {{"own": {{}}, "enemy": {{}}}}
            my_team = ct.get_team()
            in_transit = {{"titanium": 0, "raw_axionite": 0, "refined_axionite": 0}}
            for eid in ct.get_nearby_entities():
                try:
                    team = ct.get_team(eid)
                    etype = ct.get_entity_type(eid)
                    bucket = "own" if team == my_team else "enemy"
                    counts[bucket][etype.value] = counts[bucket].get(etype.value, 0) + 1
                    if team == my_team and etype in CARRIER_TYPES:
                        res = ct.get_stored_resource(eid)
                        if res is not None:
                            in_transit[res.value] = in_transit.get(res.value, 0) + 1
                except Exception:
                    pass
            return counts, in_transit

        def _count_ore_tiles(ct):
            """Count visible ore tiles and how many have harvesters (are being mined)."""
            total_ore = 0
            mined_ore = 0
            for pos in ct.get_nearby_tiles():
                env = ct.get_tile_env(pos)
                if env in (Environment.ORE_TITANIUM, Environment.ORE_AXIONITE):
                    total_ore += 1
                    bid = ct.get_tile_building_id(pos)
                    if bid is not None:
                        try:
                            if ct.get_entity_type(bid) == EntityType.HARVESTER:
                                mined_ore += 1
                        except Exception:
                            pass
            return total_ore, mined_ore

        def _sample(ct, core_id):
            """Collect a stats snapshot from the core's perspective."""
            rnd = ct.get_current_round()

            last = _last_round_sampled.get(core_id, -SAMPLE_INTERVAL)
            if rnd - last < SAMPLE_INTERVAL and rnd != 1:
                return
            _last_round_sampled[core_id] = rnd

            ti, ax = ct.get_global_resources()
            scale = ct.get_scale_percent()
            counts, in_transit = _count_nearby(ct)
            visible_ore, mined_ore = _count_ore_tiles(ct)

            # Track income vs spending using resource pool deltas.
            # Between two samples: if pool increased, income >= increase + any spending.
            # We can't see spending directly, but we can compute:
            #   income = max(0, current - prev + estimated_spend)
            # Simplification: track pool value and let the dashboard compute deltas.
            # Also track cumulative: income = sum of positive deltas, spent = sum of negative deltas
            prev = _prev_resources.get(core_id)
            if prev is not None:
                ti_delta = ti - prev[0]
                ax_delta = ax - prev[1]
                if ti_delta > 0:
                    _cumulative_income["titanium"] += ti_delta
                else:
                    _cumulative_spent["titanium"] += abs(ti_delta)
                if ax_delta > 0:
                    _cumulative_income["axionite"] += ax_delta
                else:
                    _cumulative_spent["axionite"] += abs(ax_delta)
            _prev_resources[core_id] = (ti, ax)

            snapshot = {{
                "round": rnd,
                "titanium": ti,
                "axionite": ax,
                "cumulative_income_ti": _cumulative_income["titanium"],
                "cumulative_income_ax": _cumulative_income["axionite"],
                "cumulative_spent_ti": _cumulative_spent["titanium"],
                "cumulative_spent_ax": _cumulative_spent["axionite"],
                "in_transit_ti": in_transit.get("titanium", 0),
                "in_transit_ax": in_transit.get("raw_axionite", 0) + in_transit.get("refined_axionite", 0),
                "scale_percent": scale,
                "own_units": counts["own"],
                "enemy_units": counts["enemy"],
                "core_hp": ct.get_hp(),
                "visible_ore": visible_ore,
                "mined_ore": mined_ore,
                "unmined_ore": visible_ore - mined_ore,
            }}
            _stats["snapshots"].append(snapshot)

        def _save():
            try:
                Path(STATS_PATH).write_text(json.dumps(_stats, separators=(",", ":")))
            except Exception:
                pass

        class Player:
            def __init__(self):
                self._inner = _orig.Player()

            def run(self, ct: Controller) -> None:
                etype = ct.get_entity_type()
                if etype == EntityType.CORE:
                    rnd = ct.get_current_round()
                    _sample(ct, ct.get_id())

                    if rnd >= 1999:
                        ti, ax = ct.get_global_resources()
                        _stats["final"] = {{
                            "round": rnd,
                            "titanium": ti,
                            "axionite": ax,
                            "core_hp": ct.get_hp(),
                            "cumulative_income_ti": _cumulative_income["titanium"],
                            "cumulative_income_ax": _cumulative_income["axionite"],
                            "cumulative_spent_ti": _cumulative_spent["titanium"],
                            "cumulative_spent_ax": _cumulative_spent["axionite"],
                        }}
                        _save()

                self._inner.run(ct)

            def __del__(self):
                if _stats["snapshots"]:
                    _save()
    ''')

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"stats_{team}_{match_id}_"))
    wrapper_path = tmp_dir / "main.py"
    wrapper_path.write_text(wrapper_code)
    return str(wrapper_path)


def read_stats(match_id: int, team: str) -> dict | None:
    """Read collected stats for a match/team. Returns None if not found."""
    stats_file = STATS_DIR / f"match_{match_id}_{team}.json"
    if stats_file.exists():
        try:
            return json.loads(stats_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


def aggregate_match_stats(match_id: int) -> dict | None:
    """Aggregate stats from both teams into a single match stats dict."""
    a = read_stats(match_id, "a")
    b = read_stats(match_id, "b")
    if not a and not b:
        return None

    def summarize_team(stats):
        if not stats or not stats.get("snapshots"):
            return None
        snaps = stats["snapshots"]
        final = stats.get("final") or snaps[-1]
        return {
            "snapshots": snaps,
            "final": stats.get("final"),
            "peak_titanium": max(s["titanium"] for s in snaps),
            "peak_axionite": max(s["axionite"] for s in snaps),
            "total_income_ti": final.get("cumulative_income_ti", 0),
            "total_income_ax": final.get("cumulative_income_ax", 0),
            "total_spent_ti": final.get("cumulative_spent_ti", 0),
            "total_spent_ax": final.get("cumulative_spent_ax", 0),
            "final_core_hp": final.get("core_hp"),
        }

    return {
        "team_a": summarize_team(a),
        "team_b": summarize_team(b),
    }
