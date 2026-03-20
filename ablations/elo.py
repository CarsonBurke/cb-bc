#!/usr/bin/env python3
"""Round-robin TrueSkill tournament for Cambridge Battlecode bots.

Usage:
    python ablations/elo.py [--rounds N] [--maps MAP1,MAP2,...] [bot1 bot2 ...]

If no bots specified, discovers all bots in bots/ directory.
Runs all pairwise matches in parallel, computes TrueSkill ratings,
prints leaderboard, appends to history.json, and generates REPORT.md.

Use --no-history to skip history/report generation.
Use --view to open the dashboard after generating the report.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import trueskill

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = PROJECT_ROOT / "ablations"
VENV_CAMBC = str(PROJECT_ROOT / ".venv" / "bin" / "cambc")
MAPS_DIR = PROJECT_ROOT / "maps"
BOTS_DIR = PROJECT_ROOT / "bots"
REPLAYS_DIR = PROJECT_ROOT / "replays"
HISTORY_FILE = ABLATIONS_DIR / "history.json"
RESULTS_FILE = ABLATIONS_DIR / "elo_results.json"
REPORT_FILE = ABLATIONS_DIR / "REPORT.md"

DEFAULT_MAPS = [
    "default_small1.map26", "default_small2.map26",
    "default_medium1.map26", "default_medium2.map26",
    "default_large1.map26", "default_large2.map26",
]

# TrueSkill environment — default mu=25, sigma=25/3
TS_ENV = trueskill.TrueSkill(draw_probability=0.02)


# ---------------------------------------------------------------------------
# Match running
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    bot_a: str
    bot_b: str
    map_name: str
    winner: str  # bot name or "draw"
    match_id: int = -1


def parse_replay_winner(replay_path: str, bot_a: str, bot_b: str) -> str:
    """Parse the replay protobuf to extract the winner."""
    from replay_parser import parse_replay_working
    result = parse_replay_working(replay_path)
    winner_val = result.get("winner")
    if winner_val == 0:
        return bot_a
    elif winner_val == 1:
        return bot_b
    return "unknown"


def run_match(bot_a: str, bot_b: str, map_name: str, match_id: int, collect_stats: bool = False) -> MatchResult:
    """Run a single cambc match and parse the winner from the replay."""
    replay_path = str(REPLAYS_DIR / f"match_{match_id}.replay26")

    run_a, run_b = bot_a, bot_b
    wrapper_paths = []
    if collect_stats:
        from ablations.stats_wrapper import create_wrapper
        bot_a_main = str(BOTS_DIR / bot_a / "main.py")
        bot_b_main = str(BOTS_DIR / bot_b / "main.py")
        run_a = create_wrapper(bot_a_main, "a", match_id)
        run_b = create_wrapper(bot_b_main, "b", match_id)
        wrapper_paths = [run_a, run_b]

    try:
        subprocess.run(
            [VENV_CAMBC, "run", run_a, run_b, map_name, "--replay", replay_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        winner = parse_replay_winner(replay_path, bot_a, bot_b)
        return MatchResult(bot_a, bot_b, map_name, winner, match_id)
    except subprocess.TimeoutExpired:
        return MatchResult(bot_a, bot_b, map_name, "timeout", match_id)
    except Exception as e:
        return MatchResult(bot_a, bot_b, map_name, "error", match_id)
    finally:
        import shutil
        for wp in wrapper_paths:
            try:
                shutil.rmtree(Path(wp).parent)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TrueSkill ratings
# ---------------------------------------------------------------------------

def load_prior_ratings() -> dict[str, trueskill.Rating]:
    """Load ratings from the most recent round in history.json."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        history = json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    rounds = history.get("rounds", [])
    if not rounds:
        return {}
    latest = rounds[-1]
    ratings = {}
    for bot_name, data in latest.get("bots", {}).items():
        mu = data.get("mu")
        sigma = data.get("sigma")
        if mu is not None and sigma is not None:
            ratings[bot_name] = TS_ENV.create_rating(mu=mu, sigma=sigma)
    return ratings


def compute_trueskill(
    results: list[MatchResult],
    bots: list[str],
    prior: dict[str, trueskill.Rating] | None = None,
) -> dict[str, trueskill.Rating]:
    """Compute TrueSkill ratings from match results."""
    prior = prior or {}

    if prior:
        mu_mean = sum(r.mu for r in prior.values()) / len(prior)
        default = TS_ENV.create_rating(mu=mu_mean)
    else:
        default = TS_ENV.create_rating()

    ratings = {}
    for b in bots:
        ratings[b] = prior.get(b, default)

    for r in results:
        if r.winner in ("timeout", "error", "unknown"):
            continue
        if r.bot_a not in ratings or r.bot_b not in ratings:
            continue

        ra, rb = ratings[r.bot_a], ratings[r.bot_b]

        if r.winner == r.bot_a:
            new_a, new_b = TS_ENV.rate_1vs1(ra, rb)
        elif r.winner == r.bot_b:
            new_b, new_a = TS_ENV.rate_1vs1(rb, ra)
        else:  # draw
            new_a, new_b = TS_ENV.rate_1vs1(ra, rb, drawn=True)

        ratings[r.bot_a] = new_a
        ratings[r.bot_b] = new_b

    return ratings


def conservative_rating(r: trueskill.Rating) -> float:
    return r.mu - 3 * r.sigma


def discover_bots() -> list[str]:
    return sorted(
        e.name for e in BOTS_DIR.iterdir()
        if e.is_dir() and (e / "main.py").exists()
    )


# ---------------------------------------------------------------------------
# Leaderboard + JSON output
# ---------------------------------------------------------------------------

def build_leaderboard(
    ratings: dict[str, trueskill.Rating],
    results: list[MatchResult],
    collect_stats: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Build leaderboard and match data. Returns (leaderboard, matches)."""
    bots = sorted(ratings.keys(), key=lambda b: conservative_rating(ratings[b]), reverse=True)

    record: dict[str, dict[str, int]] = {b: {"w": 0, "l": 0, "d": 0} for b in bots}
    for r in results:
        if r.winner == r.bot_a:
            record[r.bot_a]["w"] += 1
            record[r.bot_b]["l"] += 1
        elif r.winner == r.bot_b:
            record[r.bot_b]["w"] += 1
            record[r.bot_a]["l"] += 1
        elif r.winner == "draw":
            record[r.bot_a]["d"] += 1
            record[r.bot_b]["d"] += 1

    print("\n=== TRUESKILL LEADERBOARD ===\n")
    print(f"{'Rank':<5} {'Bot':<20} {'Rating':<8} {'mu':<8} {'sigma':<8} {'W':<4} {'L':<4} {'D':<4}")
    print("-" * 65)
    for i, b in enumerate(bots, 1):
        r = record[b]
        ts = ratings[b]
        cr = conservative_rating(ts)
        print(f"{i:<5} {b:<20} {cr:<8.0f} {ts.mu:<8.1f} {ts.sigma:<8.1f} {r['w']:<4} {r['l']:<4} {r['d']:<4}")

    leaderboard = []
    for i, b in enumerate(bots, 1):
        ts = ratings[b]
        leaderboard.append({
            "rank": i,
            "bot": b,
            "elo": round(conservative_rating(ts)),
            "mu": round(ts.mu, 2),
            "sigma": round(ts.sigma, 2),
            "wins": record[b]["w"],
            "losses": record[b]["l"],
            "draws": record[b]["d"],
        })

    match_data = []
    aggregate_match_stats = None
    if collect_stats:
        from ablations.stats_wrapper import aggregate_match_stats as _aggregate_match_stats
        aggregate_match_stats = _aggregate_match_stats
    for r in results:
        entry = {"bot_a": r.bot_a, "bot_b": r.bot_b, "map": r.map_name, "winner": r.winner, "match_id": r.match_id}
        if aggregate_match_stats is not None:
            stats = aggregate_match_stats(r.match_id)
            if stats:
                entry["stats"] = stats
        match_data.append(entry)

    return leaderboard, match_data


# ---------------------------------------------------------------------------
# History (formerly history.py)
# ---------------------------------------------------------------------------

def read_changelog(bot_name: str) -> str:
    """Extract the first comment line from a bot's main.py as its changelog."""
    main_py = BOTS_DIR / bot_name / "main.py"
    if not main_py.exists():
        return ""
    try:
        for line in main_py.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("#!"):
                comment = stripped.lstrip("#").strip()
                if comment:
                    return comment
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


def _aggregate_match_stats_for_history(elo_data: dict) -> dict[str, dict]:
    """Extract average end-game stats per bot from match snapshots."""
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
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"rounds": [], "lineage": {}}


def append_history(elo_data: dict) -> dict:
    """Append current results to history.json. Returns updated history."""
    history = load_history()

    round_id = len(history["rounds"]) + 1
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

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
        if bot_name not in history["lineage"]:
            history["lineage"][bot_name] = {
                "parent": parent,
                "changelog": changelog,
            }

    bot_stats = _aggregate_match_stats_for_history(elo_data)
    for bot_name, stats in bot_stats.items():
        if bot_name in bots_round:
            bots_round[bot_name]["stats"] = stats

    compact_matches = []
    for match in elo_data.get("matches", []):
        compact = {
            "bot_a": match["bot_a"],
            "bot_b": match["bot_b"],
            "map": match["map"],
            "winner": match["winner"],
            "match_id": match["match_id"],
        }
        stats = match.get("stats")
        if stats:
            compact["stats"] = {
                team_key: {
                    k: v for k, v in team_stats.items() if k != "snapshots"
                } if team_stats else None
                for team_key, team_stats in stats.items()
            }
        compact_matches.append(compact)

    history["rounds"].append({
        "round_id": round_id,
        "timestamp": timestamp,
        "bots": bots_round,
        "matches": compact_matches,
    })

    HISTORY_FILE.write_text(json.dumps(history, separators=(",", ":")) + "\n")
    print(f"\nRound {round_id} appended to {HISTORY_FILE}")
    print(f"  Bots: {', '.join(sorted(bots_round))}")
    print(f"  Lineage entries: {len(history['lineage'])}")
    return history


# ---------------------------------------------------------------------------
# Report generation (formerly report.py)
# ---------------------------------------------------------------------------

def _section_rankings(history: dict) -> str:
    latest = history["rounds"][-1]
    bots = sorted(latest["bots"].items(), key=lambda x: x[1]["elo"], reverse=True)

    lines = ["## Current Rankings", ""]
    lines.append(f"Round {latest['round_id']} ({latest['timestamp']})")
    lines.append("")
    lines.append("| Rank | Bot | ELO | W | L | D |")
    lines.append("|-|-|-|-|-|-|")
    for i, (name, data) in enumerate(bots, 1):
        lines.append(f"| {i} | {name} | {data['elo']} | {data['wins']} | {data['losses']} | {data['draws']} |")
    return "\n".join(lines)


def _section_elo_progression(history: dict) -> str:
    if len(history["rounds"]) < 1:
        return ""

    all_bots: list[str] = []
    seen = set()
    for r in history["rounds"]:
        for b in sorted(r["bots"]):
            if b not in seen:
                all_bots.append(b)
                seen.add(b)

    lines = ["## ELO Progression", ""]
    header = "| Round | " + " | ".join(all_bots) + " |"
    sep = "|-" + "|-" * len(all_bots) + "|"
    lines.append(header)
    lines.append(sep)

    for r in history["rounds"]:
        cells = []
        for b in all_bots:
            if b in r["bots"]:
                cells.append(str(r["bots"][b]["elo"]))
            else:
                cells.append("-")
        lines.append(f"| {r['round_id']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _section_lineage(history: dict) -> str:
    lineage = history.get("lineage", {})
    if not lineage:
        return ""

    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for bot, info in lineage.items():
        parent = info.get("parent", "")
        if not parent or parent not in lineage:
            roots.append(bot)
        else:
            children[parent].append(bot)

    roots.sort()
    for k in children:
        children[k].sort()

    lines = ["## Bot Lineage", ""]
    lines.append("```")

    def render(name: str, prefix: str, is_last: bool):
        connector = "`-- " if is_last else "|-- "
        changelog = lineage.get(name, {}).get("changelog", "")
        annotation = f"  ({changelog})" if changelog else ""
        lines.append(f"{prefix}{connector}{name}{annotation}")
        child_prefix = prefix + ("    " if is_last else "|   ")
        kids = children.get(name, [])
        for i, child in enumerate(kids):
            render(child, child_prefix, i == len(kids) - 1)

    for i, root in enumerate(roots):
        changelog = lineage.get(root, {}).get("changelog", "")
        annotation = f"  ({changelog})" if changelog else ""
        lines.append(f"{root}{annotation}")
        kids = children.get(root, [])
        for j, child in enumerate(kids):
            render(child, "", j == len(kids) - 1)

    lines.append("```")
    return "\n".join(lines)


def _section_winrate_matrix(history: dict) -> str:
    latest = history["rounds"][-1]
    matches = latest.get("matches", [])
    bots = sorted(latest["bots"])

    wins: dict[str, dict[str, int]] = {b: defaultdict(int) for b in bots}
    games: dict[str, dict[str, int]] = {b: defaultdict(int) for b in bots}

    for m in matches:
        a, b, winner = m["bot_a"], m["bot_b"], m["winner"]
        if a in latest["bots"] and b in latest["bots"]:
            games[a][b] += 1
            games[b][a] += 1
            if winner == a:
                wins[a][b] += 1
            elif winner == b:
                wins[b][a] += 1

    lines = ["## Win Rate Matrix (Latest Round)", ""]
    header = "| | " + " | ".join(bots) + " |"
    sep = "|-" + "|-" * len(bots) + "|"
    lines.append(header)
    lines.append(sep)

    for b in bots:
        cells = []
        for opp in bots:
            if b == opp:
                cells.append("-")
            elif games[b][opp] > 0:
                rate = wins[b][opp] / games[b][opp]
                cells.append(f"{rate:.0%}")
            else:
                cells.append("-")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _section_economy_stats(history: dict) -> str:
    latest = history["rounds"][-1]
    bots = sorted(latest["bots"].items(), key=lambda x: x[1]["elo"], reverse=True)

    if not any("stats" in d for _, d in bots):
        return ""

    lines = ["## Economy Stats (avg end-game)", ""]
    lines.append("| Bot | Ti | Scale | Harvesters | Conveyors | Roads | Builders | Core HP |")
    lines.append("|-|-|-|-|-|-|-|-|")
    for name, data in bots:
        s = data.get("stats", {})
        if not s:
            continue
        lines.append(
            f"| {name} | {s.get('avg_final_titanium', '-')} "
            f"| {s.get('avg_final_scale', '-')}% "
            f"| {s.get('avg_harvesters', '-')} "
            f"| {s.get('avg_conveyors', '-')} "
            f"| {s.get('avg_roads', '-')} "
            f"| {s.get('avg_builders', '-')} "
            f"| {s.get('avg_final_core_hp', '-')} |"
        )
    return "\n".join(lines)


def _section_key_stats(history: dict) -> str:
    latest = history["rounds"][-1]
    bots = latest["bots"]

    lines = ["## Key Stats", ""]

    best = max(bots.items(), key=lambda x: x[1]["elo"])
    lines.append(f"- **Best bot**: {best[0]} (ELO {best[1]['elo']})")

    if len(history["rounds"]) >= 2:
        prev = history["rounds"][-2]
        jumps: list[tuple[str, int]] = []
        for name, data in bots.items():
            if name in prev["bots"]:
                delta = data["elo"] - prev["bots"][name]["elo"]
                jumps.append((name, delta))

        if jumps:
            biggest = max(jumps, key=lambda x: x[1])
            biggest_drop = min(jumps, key=lambda x: x[1])
            lines.append(f"- **Biggest ELO jump**: {biggest[0]} (+{biggest[1]})" if biggest[1] > 0 else "- **Biggest ELO jump**: none")
            if biggest_drop[1] < 0:
                lines.append(f"- **Biggest ELO drop**: {biggest_drop[0]} ({biggest_drop[1]})")
    else:
        lines.append("- *Only one round recorded; trends available after round 2.*")

    total = len(latest.get("matches", []))
    lines.append(f"- **Matches in latest round**: {total}")

    return "\n".join(lines)


def generate_report(history: dict) -> str:
    """Generate REPORT.md from history data. Returns report text."""
    if not history["rounds"]:
        return ""

    sections = [
        "# Ablation Report",
        "",
        _section_rankings(history),
        "",
        _section_elo_progression(history),
        "",
        _section_winrate_matrix(history),
        "",
        _section_lineage(history),
        "",
        _section_economy_stats(history),
        "",
        _section_key_stats(history),
        "",
    ]

    report = "\n".join(sections) + "\n"
    REPORT_FILE.write_text(report)
    print(report)
    print(f"Report written to {REPORT_FILE}", file=sys.stderr)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Round-robin TrueSkill for cambc bots")
    parser.add_argument("bots", nargs="*", help="Bot names (default: all in bots/)")
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS), help="Comma-separated map list")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds of round-robin")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers")
    parser.add_argument("--stats", action="store_true", help="Collect per-turn stats via wrapper bots")
    parser.add_argument("--no-carry", action="store_true", help="Don't carry ratings from previous round")
    parser.add_argument("--no-history", action="store_true", help="Skip history/report generation")
    parser.add_argument("--view", action="store_true", help="Open dashboard after report")
    args = parser.parse_args()

    bots = args.bots or discover_bots()
    maps = args.maps.split(",")

    if len(bots) < 2:
        print("Need at least 2 bots to run matches.", file=sys.stderr)
        sys.exit(1)

    # Load prior ratings for carry-over
    prior = {} if args.no_carry else load_prior_ratings()
    if prior:
        returning = [b for b in bots if b in prior]
        new = [b for b in bots if b not in prior]
        print(f"Carrying ratings for {len(returning)} bots, {len(new)} new")
    else:
        print("Starting fresh (no prior ratings)")

    REPLAYS_DIR.mkdir(exist_ok=True)

    # Generate all matchups — both sides (A vs B and B vs A) per map
    matchups = []
    for _ in range(args.rounds):
        for i, a in enumerate(bots):
            for b in bots[i + 1:]:
                for m in maps:
                    matchups.append((a, b, m))
                    matchups.append((b, a, m))

    print(f"Running {len(matchups)} matches across {len(bots)} bots, {len(maps)} maps, {args.rounds} round(s)")
    print(f"Using {args.workers} parallel workers\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_match, a, b, m, idx, args.stats): (a, b, m)
            for idx, (a, b, m) in enumerate(matchups)
        }
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            tag = r.winner if r.winner not in (r.bot_a, r.bot_b) else f"{r.winner} wins"
            print(f"  [{i}/{len(matchups)}] {r.bot_a} vs {r.bot_b} on {r.map_name}: {tag}")
            results.append(r)

    ratings = compute_trueskill(results, bots, prior)
    leaderboard, match_data = build_leaderboard(ratings, results, args.stats)

    elo_data = {"leaderboard": leaderboard, "matches": match_data}
    RESULTS_FILE.write_text(json.dumps(elo_data, separators=(",", ":")))
    print(f"\nDetailed results written to {RESULTS_FILE}")

    # History + Report
    if not args.no_history:
        history = append_history(elo_data)
        generate_report(history)

    # Open dashboard
    if args.view:
        import subprocess
        subprocess.run(
            [sys.executable, str(ABLATIONS_DIR / "viewer.py")],
            cwd=str(PROJECT_ROOT),
        )


if __name__ == "__main__":
    main()
