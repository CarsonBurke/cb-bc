#!/usr/bin/env python3
"""Download replay files from the Cambridge Battlecode ladder.

Usage:
    python scripts/download_replays.py [options]

Reads BATTLECODE_SESSION_COOKIE from .env (or --cookie override).
The cookie is the value of __Secure-better-auth.session_token from your browser.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> None:
    """Load .env file from project root into os.environ."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())

BASE = "https://game.battlecode.cam"
GAMES_FN = "/_serverFn/8450fe0d23f291c5258bcef199bba0e39aa04eec125352ed8d454d31941d17b3"
REPLAY_URL_FN = "/_serverFn/edfcd6acad8e3774d585557f54ff61bbd535b93fc8e474ba78505d9aed088687"

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replay_data"


def seroval_str(s: str, idx: int) -> dict:
    return {"t": 1, "s": s}


def seroval_num(n: int | float, idx: int) -> dict:
    return {"t": 0, "s": n}


def seroval_obj(keys: list[str], vals: list[dict], idx: int) -> dict:
    return {"t": 10, "i": idx, "p": {"k": keys, "v": vals}, "o": 0}


def make_payload(data_keys: list[str], data_vals: list[dict]) -> str:
    inner = seroval_obj(data_keys, data_vals, idx=1)
    outer = seroval_obj(["data"], [inner], idx=0)
    return json.dumps({"t": outer, "f": 63, "m": []})


def fetch_json(url: str, cookie: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Cookie": f"__Secure-better-auth.session_token={cookie}",
        "x-tsr-serverFn": "true",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def parse_seroval_games(data: dict) -> list[dict]:
    """Extract game records from seroval-encoded response."""
    result_node = data["p"]["v"][0]  # result field
    if result_node.get("t") == 2:  # null/undefined
        return []
    games = []
    for item in result_node.get("a", []):
        keys = item["p"]["k"]
        vals = item["p"]["v"]
        record = {}
        for k, v in zip(keys, vals):
            if v["t"] == 1:  # string
                record[k] = v["s"]
            elif v["t"] == 0:  # number
                record[k] = v["s"]
            elif v["t"] == 5:  # date
                record[k] = v["s"]
        games.append(record)
    return games


def parse_seroval_url(data: dict) -> str:
    """Extract presigned URL string from seroval response."""
    return data["p"]["v"][0]["s"]


def fetch_matches(cookie: str, limit: int = 50, cursor: str | None = None,
                  team_ids: list[str] | None = None,
                  min_rating: float | None = None) -> tuple[list[dict], str | None]:
    """Fetch a page of matches from the REST API."""
    url = f"{BASE}/api/matches?limit={limit}"
    if cursor:
        url += f"&cursor={urllib.parse.quote(cursor)}"
    req = urllib.request.Request(url, headers={
        "Cookie": f"__Secure-better-auth.session_token={cookie}",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    matches = data.get("matches", [])
    next_cursor = data.get("nextCursor")
    if min_rating is not None:
        matches = [m for m in matches
                   if m.get("teamARating", 0) >= min_rating
                   or m.get("teamBRating", 0) >= min_rating]
    if team_ids:
        tid_set = set(team_ids)
        matches = [m for m in matches
                   if m.get("teamAId") in tid_set or m.get("teamBId") in tid_set]
    return matches, next_cursor


def fetch_games(cookie: str, match_id: str) -> list[dict]:
    """Fetch game details (including replayS3Key) for a match."""
    payload = make_payload(["matchId"], [seroval_str(match_id, 2)])
    encoded = urllib.parse.quote(payload)
    url = f"{BASE}{GAMES_FN}?payload={encoded}"
    data = fetch_json(url, cookie)
    return parse_seroval_games(data)


def fetch_replay_url(cookie: str, replay_key: str) -> str:
    """Get a presigned S3 URL for a replay file."""
    payload = make_payload(["replayS3Key"], [seroval_str(replay_key, 2)])
    encoded = urllib.parse.quote(payload)
    url = f"{BASE}{REPLAY_URL_FN}?payload={encoded}"
    data = fetch_json(url, cookie)
    return parse_seroval_url(data)


def sanitize_name(name: str) -> str:
    """Sanitize a team name for use in filenames."""
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:30]


def replay_filename(match: dict, game: dict) -> str:
    """Build a descriptive replay filename: teamA_teamB_elo_date_gN.replay26"""
    team_a = sanitize_name(match.get("teamAName", "unknown"))
    team_b = sanitize_name(match.get("teamBName", "unknown"))
    avg_elo = int((match.get("teamARating", 0) + match.get("teamBRating", 0)) / 2)
    date = match.get("completedAt", match.get("createdAt", ""))[:10]  # YYYY-MM-DD
    game_num = game.get("gameNumber", 0)
    map_name = game.get("mapName", "unknown")
    return f"{team_a}_vs_{team_b}_{avg_elo}elo_{date}_g{game_num}_{map_name}.replay26"


def download_replay(cookie: str, replay_key: str, dest: Path) -> bool:
    """Download a single replay file. Returns True if downloaded, False if skipped."""
    if dest.exists():
        return False
    presigned_url = fetch_replay_url(cookie, replay_key)
    req = urllib.request.Request(presigned_url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return True


def main():
    parser = argparse.ArgumentParser(description="Download ladder replays")
    parser.add_argument("--cookie", default=None,
                        help="Value of __Secure-better-auth.session_token (default: from .env)")
    parser.add_argument("--pages", type=int, default=10,
                        help="Number of match pages to fetch (50 matches each)")
    parser.add_argument("--min-rating", type=float, default=None,
                        help="Only download matches where at least one team has this rating")
    parser.add_argument("--team-ids", nargs="*", default=None,
                        help="Only download matches involving these team IDs")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Auto-select top N teams from ladder by rating")
    parser.add_argument("--out", type=str, default=str(REPLAY_DIR),
                        help="Output directory for replays")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Delay between API requests (seconds)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List matches without downloading")
    args = parser.parse_args()

    load_dotenv()
    if not args.cookie:
        args.cookie = os.environ.get("BATTLECODE_SESSION_COOKIE")
    if not args.cookie:
        parser.error("No cookie: set BATTLECODE_SESSION_COOKIE in .env or pass --cookie")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    team_ids = args.team_ids

    # Auto-select top teams from ladder
    if args.top_n:
        print(f"Fetching ladder to select top {args.top_n} teams...")
        req = urllib.request.Request(f"{BASE}/api/ladder", headers={
            "Cookie": f"__Secure-better-auth.session_token={args.cookie}",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            ladder = json.loads(resp.read())
        ladder.sort(key=lambda t: t.get("rating", 0), reverse=True)
        team_ids = [t["teamId"] for t in ladder[:args.top_n]]
        for t in ladder[:args.top_n]:
            print(f"  {t['teamName']:30s}  rating={t['rating']:.0f}")

    # Collect matches
    all_matches = []
    cursor = None
    for page in range(args.pages):
        matches, cursor = fetch_matches(
            args.cookie, limit=50, cursor=cursor,
            team_ids=team_ids, min_rating=args.min_rating,
        )
        all_matches.extend(matches)
        print(f"Page {page+1}: {len(matches)} matches (total {len(all_matches)})")
        if not cursor:
            print("No more matches.")
            break
        time.sleep(args.delay)

    print(f"\n{len(all_matches)} matches to process")

    if args.dry_run:
        for m in all_matches[:20]:
            print(f"  {m['teamAName']:25s} vs {m['teamBName']:25s}  "
                  f"({m['scoreA']}-{m['scoreB']})")
        if len(all_matches) > 20:
            print(f"  ... and {len(all_matches) - 20} more")
        return

    # Download replays, append to JSONL index as we go
    downloaded = 0
    skipped = 0
    errors = 0
    index_path = out_dir / "index.jsonl"

    # Load existing entries to avoid duplicates
    existing_files: set[str] = set()
    if index_path.exists():
        for line in index_path.read_text().splitlines():
            if line.strip():
                existing_files.add(json.loads(line).get("file", ""))

    index_fp = open(index_path, "a")

    for i, match in enumerate(all_matches):
        match_id = match["id"]
        team_a = match.get("teamAName", "?")
        team_b = match.get("teamBName", "?")

        try:
            games = fetch_games(args.cookie, match_id)
        except Exception as e:
            print(f"  [{i+1}/{len(all_matches)}] Error fetching games for {match_id}: {e}")
            errors += 1
            time.sleep(args.delay)
            continue

        for game in games:
            replay_key = game.get("replayS3Key")
            if not replay_key:
                continue
            fname = replay_filename(match, game)
            dest = out_dir / fname

            try:
                if download_replay(args.cookie, replay_key, dest):
                    downloaded += 1
                    status = "downloaded"
                else:
                    skipped += 1
                    status = "exists"
            except Exception as e:
                errors += 1
                status = f"error: {e}"

            if status == "downloaded" or "error" in status:
                print(f"  [{i+1}/{len(all_matches)}] {team_a} vs {team_b} "
                      f"g{game.get('gameNumber', 0)} ({game.get('mapName', '?')}): {status}")

            if dest.exists() and fname not in existing_files:
                record = json.dumps({
                    "file": fname,
                    "match_id": match_id,
                    "game_number": game.get("gameNumber"),
                    "map": game.get("mapName"),
                    "team_a": match.get("teamAName"),
                    "team_b": match.get("teamBName"),
                    "team_a_rating": match.get("teamARating"),
                    "team_b_rating": match.get("teamBRating"),
                    "winner_side": game.get("winnerSide"),
                    "win_condition": game.get("winCondition"),
                    "turns": game.get("turnsPlayed"),
                }, separators=(",", ":"))
                index_fp.write(record + "\n")
                index_fp.flush()
                existing_files.add(fname)

            time.sleep(args.delay * 0.5)

        time.sleep(args.delay)

    index_fp.close()
    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {errors} errors")
    print(f"Index: {index_path} ({len(existing_files)} entries)")


if __name__ == "__main__":
    main()
