"""Shared training utilities: League, session management, trajectory I/O."""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import trueskill

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"

# TrueSkill environment — matches ablation system settings
TS_ENV = trueskill.TrueSkill(draw_probability=0.02)


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
        "log_path": d / "train_log.jsonl",
        "replay_dir": replay_dir,
    }


def _default_session_name() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"sp_{ts}"


# ---- League (TrueSkill-based, no torch dependency) ----

def _ts_rating(mu: float, sigma: float) -> trueskill.Rating:
    return TS_ENV.create_rating(mu=mu, sigma=sigma)


def _ts_elo(rating: trueskill.Rating) -> float:
    """Conservative skill estimate (mu - 3*sigma), used for ranking/eviction."""
    return round(rating.mu - 3 * rating.sigma, 1)


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
        # Inherit rating from parent (previous gen) if it exists
        parent = next((m for m in self.models if m["gen"] == gen - 1), None)
        if parent is not None:
            # Inherit mu, widen sigma slightly to reflect uncertainty about the new weights
            mu = parent["mu"]
            sigma = min(parent["sigma"] + 1.0, TS_ENV.sigma)
        else:
            mu = TS_ENV.mu
            sigma = TS_ENV.sigma

        entry = {
            "id": f"gen_{gen}", "gen": gen,
            "mu": round(mu, 3), "sigma": round(sigma, 3),
            "elo": _ts_elo(_ts_rating(mu, sigma)),
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
        """Sample opponents weighted by matchmaking quality.

        Prefers opponents close in skill (small |mu_a - mu_b|) and those with
        high uncertainty (large sigma), since games against them are most
        informative.  Falls back to uniform if the league has <=2 models.
        """
        if not self.models:
            return []
        if len(self.models) <= 2:
            return random.sample(self.models, min(k, len(self.models)))

        # Current model is the latest gen
        latest = max(self.models, key=lambda m: m["gen"])
        latest_mu = latest["mu"]

        weights = []
        for m in self.models:
            if m["id"] == latest["id"]:
                # Don't play against yourself
                weights.append(0.0)
            else:
                # Prefer close skill + high uncertainty opponents
                skill_diff = abs(m["mu"] - latest_mu)
                closeness = math.exp(-skill_diff / 10.0)
                uncertainty = m["sigma"] / TS_ENV.sigma
                weights.append(closeness + 0.5 * uncertainty + 0.1)  # floor avoids starvation

        total = sum(weights)
        if total == 0:
            return random.sample(self.models, min(k, len(self.models)))

        # Weighted sampling without replacement
        pool = list(range(len(self.models)))
        selected = []
        for _ in range(min(k, len(self.models) - 1)):
            if not pool:
                break
            probs = [weights[i] for i in pool]
            ptotal = sum(probs)
            if ptotal == 0:
                break
            r = random.random() * ptotal
            cumulative = 0.0
            chosen_idx = pool[-1]
            for pi, idx in enumerate(pool):
                cumulative += probs[pi]
                if r <= cumulative:
                    chosen_idx = idx
                    break
            selected.append(self.models[chosen_idx])
            pool.remove(chosen_idx)

        return selected

    def update_elo(self, model_a_id: str, model_b_id: str, winner: str | None):
        a = next((m for m in self.models if m["id"] == model_a_id), None)
        b = next((m for m in self.models if m["id"] == model_b_id), None)
        if not a or not b:
            return

        ra = _ts_rating(a["mu"], a["sigma"])
        rb = _ts_rating(b["mu"], b["sigma"])

        if winner == "a":
            new_a, new_b = TS_ENV.rate_1vs1(ra, rb)
            a["wins"] += 1; b["losses"] += 1
        elif winner == "b":
            new_b, new_a = TS_ENV.rate_1vs1(rb, ra)
            a["losses"] += 1; b["wins"] += 1
        else:
            new_a, new_b = TS_ENV.rate_1vs1(ra, rb, drawn=True)
            a["draws"] += 1; b["draws"] += 1

        a["mu"] = round(new_a.mu, 3); a["sigma"] = round(new_a.sigma, 3)
        b["mu"] = round(new_b.mu, 3); b["sigma"] = round(new_b.sigma, 3)
        a["elo"] = _ts_elo(new_a)
        b["elo"] = _ts_elo(new_b)

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
