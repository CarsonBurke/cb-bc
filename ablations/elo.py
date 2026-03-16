#!/usr/bin/env python3
"""Round-robin TrueSkill rating system for Cambridge Battlecode bots.

Usage:
    python ablations/elo.py [--rounds N] [--maps MAP1,MAP2,...] [bot1 bot2 ...]

If no bots specified, discovers all bots in bots/ directory.
Runs all pairwise matches in parallel, computes TrueSkill ratings,
prints leaderboard. Results written to ablations/elo_results.json.

Ratings carry over between tournament rounds via history.json.
New bots initialize at the pool mean. The "elo" field in output is
the conservative rating estimate (mu - 3*sigma) for backwards compat.
"""

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import trueskill

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_CAMBC = str(PROJECT_ROOT / ".venv" / "bin" / "cambc")
MAPS_DIR = PROJECT_ROOT / "maps"
BOTS_DIR = PROJECT_ROOT / "bots"
REPLAYS_DIR = PROJECT_ROOT / "replays"
HISTORY_FILE = PROJECT_ROOT / "ablations" / "history.json"

DEFAULT_MAPS = [
    "default_small1.map26", "default_small2.map26",
    "default_medium1.map26", "default_medium2.map26",
    "default_large1.map26", "default_large2.map26",
]

# TrueSkill environment — default mu=25, sigma=25/3
TS_ENV = trueskill.TrueSkill(draw_probability=0.02)


@dataclass
class MatchResult:
    bot_a: str
    bot_b: str
    map_name: str
    winner: str  # bot name or "draw"
    match_id: int = -1


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def parse_replay_winner(replay_path: str, bot_a: str, bot_b: str) -> str:
    """Parse the replay protobuf to extract the winner.

    Field 4 (varint) at the top level: 0 = Team A wins, 1 = Team B wins.
    """
    data = Path(replay_path).read_bytes()
    pos = 0
    end = len(data)
    winner_val = None
    while pos < end:
        tag, pos = _decode_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:  # varint
            val, pos = _decode_varint(data, pos)
            if field_num == 4:
                winner_val = val
        elif wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            pos += length
        elif wire_type == 5:  # fixed32
            pos += 4
        elif wire_type == 1:  # fixed64
            pos += 8
        else:
            break
    if winner_val == 0:
        return bot_a
    elif winner_val == 1:
        return bot_b
    return "unknown"


def run_match(bot_a: str, bot_b: str, map_name: str, match_id: int, collect_stats: bool = False) -> MatchResult:
    """Run a single cambc match and parse the winner from the replay."""
    import subprocess

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
            capture_output=True, text=True, timeout=300,
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
    """Compute TrueSkill ratings from match results.

    Returning bots carry their prior rating. New bots start at the pool mean
    (or default if no prior exists).
    """
    prior = prior or {}

    # Initialize: returning bots keep their rating, new bots get pool mean
    if prior:
        mu_mean = sum(r.mu for r in prior.values()) / len(prior)
        default = TS_ENV.create_rating(mu=mu_mean)
    else:
        default = TS_ENV.create_rating()

    ratings = {}
    for b in bots:
        ratings[b] = prior.get(b, default)

    # Process all match results
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
    """Conservative rating estimate: mu - 3*sigma."""
    return r.mu - 3 * r.sigma


def discover_bots() -> list[str]:
    """Find all bot directories in bots/."""
    return sorted(
        e.name for e in BOTS_DIR.iterdir()
        if e.is_dir() and (e / "main.py").exists()
    )


def print_leaderboard(ratings: dict[str, trueskill.Rating], results: list[MatchResult]):
    """Print leaderboard and write JSON results."""
    bots = sorted(ratings.keys(), key=lambda b: conservative_rating(ratings[b]), reverse=True)

    # Win/loss record
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

    # JSON output
    leaderboard = []
    for i, b in enumerate(bots, 1):
        ts = ratings[b]
        leaderboard.append({
            "rank": i,
            "bot": b,
            "elo": round(conservative_rating(ts)),  # backwards compat field name
            "mu": round(ts.mu, 2),
            "sigma": round(ts.sigma, 2),
            "wins": record[b]["w"],
            "losses": record[b]["l"],
            "draws": record[b]["d"],
        })

    from ablations.stats_wrapper import aggregate_match_stats
    match_data = []
    for r in results:
        entry = {"bot_a": r.bot_a, "bot_b": r.bot_b, "map": r.map_name, "winner": r.winner, "match_id": r.match_id}
        stats = aggregate_match_stats(r.match_id)
        if stats:
            entry["stats"] = stats
        match_data.append(entry)

    results_file = PROJECT_ROOT / "ablations" / "elo_results.json"
    results_file.write_text(json.dumps({"leaderboard": leaderboard, "matches": match_data}, indent=2))
    print(f"\nDetailed results written to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Round-robin TrueSkill for cambc bots")
    parser.add_argument("bots", nargs="*", help="Bot names (default: all in bots/)")
    parser.add_argument("--maps", default=",".join(DEFAULT_MAPS), help="Comma-separated map list")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds of round-robin")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers")
    parser.add_argument("--stats", action="store_true", help="Collect per-turn stats via wrapper bots")
    parser.add_argument("--no-carry", action="store_true", help="Don't carry ratings from previous round")
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
    print_leaderboard(ratings, results)


if __name__ == "__main__":
    main()
