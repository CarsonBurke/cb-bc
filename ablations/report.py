#!/usr/bin/env python3
"""Generate a markdown report from ablations/history.json.

Usage:
    python ablations/report.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_DIR = PROJECT_ROOT / "ablations"
HISTORY_FILE = ABLATIONS_DIR / "history.json"
REPORT_FILE = ABLATIONS_DIR / "REPORT.md"


def load_history() -> dict:
    if not HISTORY_FILE.exists():
        print(f"No history found at {HISTORY_FILE}. Run history.py first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(HISTORY_FILE.read_text())


def section_rankings(history: dict) -> str:
    """Current rankings from the latest round."""
    latest = history["rounds"][-1]
    bots = sorted(latest["bots"].items(), key=lambda x: x[1]["elo"], reverse=True)

    lines = ["## Current Rankings", ""]
    lines.append(f"Round {latest['round_id']} ({latest['timestamp']})", )
    lines.append("")
    lines.append("| Rank | Bot | ELO | W | L | D |")
    lines.append("|-|-|-|-|-|-|")
    for i, (name, data) in enumerate(bots, 1):
        lines.append(f"| {i} | {name} | {data['elo']} | {data['wins']} | {data['losses']} | {data['draws']} |")
    return "\n".join(lines)


def section_elo_progression(history: dict) -> str:
    """ELO over time: rounds as rows, bots as columns."""
    if len(history["rounds"]) < 1:
        return ""

    # Collect all bots across all rounds
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


def section_lineage(history: dict) -> str:
    """Text tree of bot lineage."""
    lineage = history.get("lineage", {})
    if not lineage:
        return ""

    # Build children map
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for bot, info in lineage.items():
        parent = info.get("parent", "")
        if not parent or parent not in lineage:
            roots.append(bot)
        else:
            children[parent].append(bot)

    # Sort for deterministic output
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


def section_winrate_matrix(history: dict) -> str:
    """Head-to-head win rates from the latest round."""
    latest = history["rounds"][-1]
    matches = latest.get("matches", [])
    bots = sorted(latest["bots"])

    # Count wins for row bot against column bot
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


def section_economy_stats(history: dict) -> str:
    """End-game economy stats per bot from match snapshots."""
    latest = history["rounds"][-1]
    bots = sorted(latest["bots"].items(), key=lambda x: x[1]["elo"], reverse=True)

    # Check if any bot has stats
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


def section_key_stats(history: dict) -> str:
    """Best bot, biggest ELO jump, most improved."""
    latest = history["rounds"][-1]
    bots = latest["bots"]

    lines = ["## Key Stats", ""]

    # Best bot
    best = max(bots.items(), key=lambda x: x[1]["elo"])
    lines.append(f"- **Best bot**: {best[0]} (ELO {best[1]['elo']})")

    # If multiple rounds, compute ELO changes
    if len(history["rounds"]) >= 2:
        prev = history["rounds"][-2]
        jumps: list[tuple[str, int]] = []
        for name, data in bots.items():
            if name in prev["bots"]:
                delta = data["elo"] - prev["bots"][name]["elo"]
                jumps.append((name, delta))

        if jumps:
            biggest = max(jumps, key=lambda x: x[1])
            most_improved = max(jumps, key=lambda x: x[1])
            biggest_drop = min(jumps, key=lambda x: x[1])
            lines.append(f"- **Biggest ELO jump**: {biggest[0]} (+{biggest[1]})" if biggest[1] > 0 else f"- **Biggest ELO jump**: none")
            if biggest_drop[1] < 0:
                lines.append(f"- **Biggest ELO drop**: {biggest_drop[0]} ({biggest_drop[1]})")
    else:
        lines.append("- *Only one round recorded; trends available after round 2.*")

    # Total matches in latest round
    total = len(latest.get("matches", []))
    lines.append(f"- **Matches in latest round**: {total}")

    return "\n".join(lines)


def main():
    history = load_history()

    if not history["rounds"]:
        print("No rounds in history.", file=sys.stderr)
        sys.exit(1)

    sections = [
        "# Ablation Report",
        "",
        section_rankings(history),
        "",
        section_elo_progression(history),
        "",
        section_winrate_matrix(history),
        "",
        section_lineage(history),
        "",
        section_economy_stats(history),
        "",
        section_key_stats(history),
        "",
    ]

    report = "\n".join(sections) + "\n"

    # Write to file and stdout
    REPORT_FILE.write_text(report)
    print(report)
    print(f"Report written to {REPORT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
