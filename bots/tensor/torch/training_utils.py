"""Shared training utilities: League, session management, trajectory I/O."""

from __future__ import annotations

import json
import os
import pickle
import random
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"
ELO_K = 32
ELO_DEFAULT = 1000


def _recommended_worker_count(n_games: int) -> int:
    """Bound collection workers by available CPU, leaving one core free."""
    cpu = os.cpu_count() or 1
    if cpu <= 2:
        return min(n_games, 1)
    return max(1, min(n_games, cpu - 1))


def _session_dir(name: str) -> Path:
    return SESSIONS_DIR / name


def _init_session(name: str) -> dict:
    """Create or load a session directory. Returns paths dict."""
    d = _session_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    league_dir = d / "league"
    league_dir.mkdir(exist_ok=True)
    replay_dir = d / "replays"
    replay_dir.mkdir(exist_ok=True)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump({"name": name, "created": time.time(), "mode": "self-play"}, f)
    return {
        "dir": d,
        "league_dir": league_dir,
        "league_json": league_dir / "league.json",
        "weight_path": d / "weights.pkl",
        "log_path": d / "train_log.json",
        "replay_dir": replay_dir,
    }


def _default_session_name() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"sp_{ts}"


# ---- League (no torch dependency) ----

class League:
    def __init__(self, league_dir: Path, league_json: Path, max_size: int = 20):
        self.max_size = max_size
        self.league_dir = league_dir
        self.league_json = league_json
        league_dir.mkdir(parents=True, exist_ok=True)
        if league_json.exists():
            with open(league_json) as f:
                data = json.load(f)
            self.models = data.get("models", [])
            self.match_history = data.get("matches", [])
        else:
            self.models = []
            self.match_history = []

    def save(self):
        with open(self.league_json, "w") as f:
            json.dump({"models": self.models, "matches": self.match_history}, f, indent=2)

    def add_model(self, weights: list[list[float]], gen: int) -> dict:
        path = str(self.league_dir / f"gen_{gen}.pkl")
        with open(path, "wb") as f:
            pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self._register(path, gen)

    def register_model(self, path: str, gen: int) -> dict:
        """Register an externally-saved checkpoint (e.g. torch state_dict)."""
        return self._register(path, gen)

    def _register(self, path: str, gen: int) -> dict:
        entry = {
            "id": f"gen_{gen}", "gen": gen, "elo": ELO_DEFAULT,
            "path": path, "timestamp": time.time(),
            "wins": 0, "losses": 0, "draws": 0,
        }
        self.models.append(entry)
        if len(self.models) > self.max_size:
            self.models.sort(key=lambda m: m["elo"], reverse=True)
            for removed in self.models[self.max_size:]:
                try:
                    os.remove(removed["path"])
                except OSError:
                    pass
            self.models = self.models[:self.max_size]
        self.save()
        return entry

    def sample_opponents(self, k: int) -> list[dict]:
        if not self.models:
            return []
        return random.sample(self.models, min(k, len(self.models)))

    def update_elo(self, model_a_id: str, model_b_id: str, winner: str | None):
        a = next((m for m in self.models if m["id"] == model_a_id), None)
        b = next((m for m in self.models if m["id"] == model_b_id), None)
        if not a or not b:
            return
        ea = 1.0 / (1.0 + 10 ** ((b["elo"] - a["elo"]) / 400))
        if winner == "a":
            sa = 1.0; a["wins"] += 1; b["losses"] += 1
        elif winner == "b":
            sa = 0.0; a["losses"] += 1; b["wins"] += 1
        else:
            sa = 0.5; a["draws"] += 1; b["draws"] += 1
        a["elo"] = round(a["elo"] + ELO_K * (sa - ea), 1)
        b["elo"] = round(b["elo"] + ELO_K * (1 - sa - (1 - ea)), 1)
        self.match_history.append({
            "a": model_a_id, "b": model_b_id, "winner": winner, "timestamp": time.time(),
        })

    def get_rankings(self) -> list[dict]:
        return sorted(self.models, key=lambda m: m["elo"], reverse=True)

    def latest_gen(self) -> int:
        return max((m["gen"] for m in self.models), default=-1)


# ---- Trajectory I/O ----

def _read_trajectory_file(path: str | None) -> list[list]:
    """Read all pickle chunks from a trajectory file (one per builder)."""
    if not path or not os.path.exists(path):
        return []
    trajs = []
    try:
        with open(path, "rb") as f:
            while True:
                try:
                    traj = pickle.load(f)
                    if traj:
                        trajs.append(traj)
                except EOFError:
                    break
    except Exception:
        pass
    return trajs
