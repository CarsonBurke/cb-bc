# cb-bc

## Setup

```bash
python -m venv .venv
.venv/bin/pip install cambc
```

All CLI commands use `.venv/bin/cambc`. Map names require the `.map26` extension.

## CLI Commands

### Local

| Command | Description |
|-|-|
| `cambc starter` | Init project with starter bot template |
| `cambc run <a> <b> [map.map26]` | Run local match (no time limits). `--watch` to auto-open visualizer |
| `cambc watch [replay]` | Open browser visualizer. `--match <id>` for platform matches |
| `cambc map-editor` | Launch map editor. `--platform` for online version |

### Platform

| Command | Description |
|-|-|
| `cambc login` / `logout` | Authenticate with platform |
| `cambc submit <path>` | Upload bot (directory, `.py`, or `.zip`) to ladder |
| `cambc status` | Show team info, rating, rank |
| `cambc unrated <team_id>` | Challenge team to unrated match |
| `cambc test-run <a> <b> [map.map26]` | Remote match with 2ms CPU time limit (AWS Graviton3) |
| `cambc matches` | View match history. Filter by type, team, limit |
| `cambc match <id>` | Show detailed match results |
| `cambc test-matches` | Show remote test run history |
| `cambc teams search <q>` | Search teams by name |
| `cambc teams info <id>` | View team details |

## Ablations

```bash
python ablations/elo.py                    # All bots, all maps, round-robin
python ablations/elo.py bot_a bot_b        # Specific bots only
```

Results: `ablations/elo_results.json`. Replays: `replays/`.
