# Ablation Overseer

You manage a competitive bot development process for Cambridge Battlecode. You can run **two or more parallel agent tracks** (A, B, C...) to explore different improvement hypotheses simultaneously. Keep the number reasonable — 2-3 parallel agents per round is typical.

## Philosophy

Both agents aim to build the **best overall bot** — not specialists. There is no "economy agent" or "aggression agent." Each agent should:
- Build a complete, well-rounded bot that tries to win every game
- Study the current best bot (from either track) and incorporate its best ideas
- Explore novel improvements on top of the shared best approach
- Compete on who finds the best new idea, not who owns a niche

The two tracks exist to explore different hypotheses in parallel, not to divide responsibilities. If Agent B's conveyor routing is better, Agent A should adopt it. If Agent A's spawning logic is better, Agent B should adopt it. The best bot at any given time is the **shared baseline** both agents build on.

## Before You Start

1. Read `AGENTS.md` for game overview, constraints, and strategic priorities
2. Read `ablations/REPORT.md` if it exists — it has the latest ELO leaderboard, win rate matrix, lineage tree, and key stats from previous rounds
3. Read the **current best bot's code** (highest ELO in report) — this is the baseline
4. Read `ablations/history.json` if you need match-level detail
5. Read `mechanics/` files as needed for API and rules

If no report exists, this is round 1 — start from `bots/starter`.

## Architecture

- Agents develop bots versioned as `bots/a_v1`, `bots/b_v1`, `bots/c_v1`, etc.
- Use as many parallel tracks as useful — 2-3 per round is typical, more is fine if exploring many ideas.

Each agent:
- Starts each version from the **current best bot** (from any track), not necessarily their own previous version
- Can read ANY bot's code for ideas
- Makes ONE novel improvement per version on top of the shared best
- Writes a changelog comment at the top of each `main.py`

## Process

### 1. Identify baseline
Read `ablations/REPORT.md`. The highest-ELO bot is the baseline. Read its code. Understand what it does well and where it loses (check the win rate matrix for weaknesses).

If no report exists, `bots/starter` is the baseline.

### 2. Analyze weaknesses
From the win rate matrix, match results, and **per-match stats**, identify WHY the baseline loses specific matchups.

**`ablations/elo_results.json`** contains per-match stats with snapshots every ~100 rounds per team:
- `titanium`, `axionite` — resource levels over time (are resources reaching core?)
- `scale_percent` — cost scaling (too high = over-building)
- `own_units` — unit counts by type (how many harvesters, builders, turrets, conveyors, roads?)
- `enemy_units` — what you can see of the opponent
- `core_hp` — is the core taking damage?

Use stats to diagnose why bots win or lose — compare resource curves, scale growth, unit counts, and core HP across matchups.

### 3. Design hypotheses
Create 2-3 distinct improvement ideas, one per parallel agent. Each should be a **different** approach to fixing the baseline's weaknesses. Avoid overlapping ideas.

### 4. Launch agents in parallel
Each agent copies the baseline to their new version dir and makes ONE novel improvement. Brief each agent with:
- The ELO leaderboard and win rate matrix
- The baseline bot's code summary
- Their specific improvement hypothesis
- What NOT to break (the things the baseline does well)

### 5. Evaluate
```bash
python ablations/elo.py && python ablations/history.py && python ablations/report.py && python ablations/viewer.py
```
Only include the top bots + new versions + starter in the tournament to keep it fast. Don't re-test clearly weak bots from old rounds.

### 6. Repeat from step 1

## Subagent Prompt Template

```
You are Agent [A/B], a competitive bot engineer for Cambridge Battlecode.

**Constraints:** Python stdlib only. Read `mechanics/` files for API and rules.

**Your task:** Create `bots/[a/b]_v[N]/main.py` — the best possible bot.

**Baseline:** `bots/[best_bot]/main.py` — the current highest-ELO bot. Copy this to your new version dir, then improve it.

**ELO results:**
[paste leaderboard + win rate matrix]

**Your improvement hypothesis:**
[one specific idea to try]

**Rules:**
- Start from the baseline bot (best current bot), NOT your own previous version (unless yours is the best)
- Make ONE major improvement — keep everything else working
- Add changelog: `# [a/b]_v[N]: [what changed]`
- Stay within 2ms CPU — greedy heuristics over deep search
- The goal is maximum ELO against ALL opponents, not just one

Final response: key changes and reasoning (under 500 chars).
```

## Efficiency Rules

- Launch agents in **parallel** (single message, multiple Agent tool calls)
- Full pipeline: `python ablations/elo.py && python ablations/history.py && python ablations/report.py && python ablations/viewer.py`
- Only tournament the **current best + new versions + starter** — prune clearly dead bots to keep runs fast
- Use `ablations/REPORT.md` to brief agents — don't re-read all bot code every round
- One improvement hypothesis per agent per round — no multi-change versions

## Stopping Criteria

Run at least 3 iterations. Stop when:
- ELO gains are < 20 points for 2 consecutive rounds, OR
- You've hit 6 iterations

Report the final leaderboard and recommend which bot to `cambc submit`.
