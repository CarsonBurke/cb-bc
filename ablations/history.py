#!/usr/bin/env python3
"""Append current ELO results to ablations/history.json with bot lineage tracking.

Usage:
    python ablations/history.py
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = PROJECT_ROOT / "ablations"
ELO_RESULTS = ABLATIONS_DIR / "elo_results.json"
HISTORY_FILE = ABLATIONS_DIR / "history.json"
BOTS_DIR = PROJECT_ROOT / "bots"


def read_changelog(bot_name: str) -> str:
    """Extract the first comment line from a bot's main.py as its changelog."""
    main_py = BOTS_DIR / bot_name / "main.py"
    if not main_py.exists():
        return ""
    try:
        for line in main_py.read_text().splitlines():
            stripped = line.strip()
            # Match lines like: # v2: improved economy
            if stripped.startswith("#") and not stripped.startswith("#!"):
                # Strip the leading # and whitespace
                comment = stripped.lstrip("#").strip()
                if comment:
                    return comment
            # Also match first line of a docstring
            if stripped.startswith('"""') or stripped.startswith("'''"):
                content = stripped.strip("\"'").strip()
                if content:
                    return content
            break
    except OSError:
        pass
    return ""


def detect_parent(bot_name: str) -> str:
    """Auto-detect parent bot from version naming convention.

    a_v3 -> a_v2, a_v1 -> starter, starter -> None
    """
    m = re.match(r"^(.+)_v(\d+)$", bot_name)
    if m:
        prefix, version = m.group(1), int(m.group(2))
        if version > 1:
            return f"{prefix}_v{version - 1}"
        return "starter"
    return ""


def _aggregate_match_stats(elo_data: dict) -> dict[str, dict]:
    """Extract average end-game stats per bot from match snapshots."""
    from collections import defaultdict

    totals: dict[str, dict] = defaultdict(lambda: {
        "games": 0, "titanium": 0, "scale": 0.0,
        "harvesters": 0, "conveyors": 0, "roads": 0, "builders": 0,
        "core_hp": 0,
    })

    for match in elo_data.get("matches", []):
        stats = match.get("stats")
        if not stats:
            continue
        for team_key, bot_name in [("team_a", match["bot_a"]), ("team_b", match["bot_b"])]:
            team_stats = stats.get(team_key)
            if not team_stats:
                continue
            snapshots = team_stats.get("snapshots", [])
            if not snapshots:
                continue
            final = snapshots[-1]
            t = totals[bot_name]
            t["games"] += 1
            t["titanium"] += final.get("titanium", 0)
            t["scale"] += final.get("scale_percent", 100.0)
            t["core_hp"] += final.get("core_hp", 500)
            units = final.get("own_units", {})
            t["harvesters"] += units.get("harvester", 0)
            t["conveyors"] += units.get("conveyor", 0)
            t["roads"] += units.get("road", 0)
            t["builders"] += units.get("builder_bot", 0)

    result = {}
    for bot, t in totals.items():
        n = t["games"]
        if n == 0:
            continue
        result[bot] = {
            "avg_final_titanium": round(t["titanium"] / n),
            "avg_final_scale": round(t["scale"] / n, 1),
            "avg_final_core_hp": round(t["core_hp"] / n),
            "avg_harvesters": round(t["harvesters"] / n, 1),
            "avg_conveyors": round(t["conveyors"] / n, 1),
            "avg_roads": round(t["roads"] / n, 1),
            "avg_builders": round(t["builders"] / n, 1),
        }
    return result


def load_history() -> dict:
    """Load existing history or create empty structure."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"rounds": [], "lineage": {}}


def main():
    if not ELO_RESULTS.exists():
        print(f"No ELO results found at {ELO_RESULTS}. Run elo.py first.")
        return

    elo_data = json.loads(ELO_RESULTS.read_text())
    history = load_history()

    # Determine next round ID
    round_id = len(history["rounds"]) + 1
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Build bots dict for this round
    bots_round: dict[str, dict] = {}
    for entry in elo_data["leaderboard"]:
        bot_name = entry["bot"]
        parent = detect_parent(bot_name)
        changelog = read_changelog(bot_name)
        bots_round[bot_name] = {
            "elo": entry["elo"],
            "mu": entry.get("mu"),
            "sigma": entry.get("sigma"),
            "wins": entry["wins"],
            "losses": entry["losses"],
            "draws": entry["draws"],
            "parent": parent,
            "changelog": changelog,
        }
        # Update cumulative lineage
        if bot_name not in history["lineage"]:
            history["lineage"][bot_name] = {
                "parent": parent,
                "changelog": changelog,
            }

    # Extract per-bot aggregate stats from match snapshots if available
    bot_stats = _aggregate_match_stats(elo_data)
    for bot_name, stats in bot_stats.items():
        if bot_name in bots_round:
            bots_round[bot_name]["stats"] = stats

    # Append round
    history["rounds"].append({
        "round_id": round_id,
        "timestamp": timestamp,
        "bots": bots_round,
        "matches": elo_data.get("matches", []),
    })

    HISTORY_FILE.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Round {round_id} appended to {HISTORY_FILE}")
    print(f"  Bots: {', '.join(sorted(bots_round))}")
    print(f"  Lineage entries: {len(history['lineage'])}")


if __name__ == "__main__":
    main()
