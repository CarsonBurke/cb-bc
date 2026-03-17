#!/usr/bin/env python3
"""Self-play training with league system.

Uses fork-based multiprocessing for game collection (no torch in workers),
then imports torch for GPU training. Pickle for trajectory I/O.

Usage:
    python bots/tensor/torch/train_selfplay.py                     # Defaults
    python bots/tensor/torch/train_selfplay.py --episodes 200 --opponents 3
    python bots/tensor/torch/train_selfplay.py --league-size 20 --games-per-opp 2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import tempfile
import time
from pathlib import Path

# No torch import here — deferred until after fork-based game collection

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"
ELO_K = 32
ELO_DEFAULT = 1000


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
        "weight_path": d / "weights.json",
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
        path = str(self.league_dir / f"gen_{gen}.json")
        with open(path, "w") as f:
            json.dump(weights, f)
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


# ---- Game runner (no torch, safe to fork) ----

def _run_game(args: tuple) -> tuple[str | None, str | None]:
    bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed = args
    os.environ["TENSOR_TRAIN_LOG"] = log_path
    try:
        from cambc.cambc_engine import run_game
        run_game(bot_a, bot_b, engine_root, map_path, replay_path, seed)
    except Exception as e:
        print(f"  Game error: {e}")
        return None, None
    finally:
        os.environ.pop("TENSOR_TRAIN_LOG", None)

    traj_path = log_path if os.path.exists(log_path) else None
    # Winner determined later from replay; return None for now
    return traj_path, None


def collect_games(
    opponent_weights_path: str,
    n_games: int,
    maps: list[str],
    n_workers: int,
    episode_num: int,
    opp_id: str,
    replay_dir: Path | None = None,
) -> tuple[list[list], list[str | None]]:
    """Run games with fork multiprocessing. Returns (trajectories, winners)."""
    bot_a = str(PROJECT_ROOT / "bots" / "tensor" / "main.py")
    bot_b = str(PROJECT_ROOT / "bots" / "tensor_opp" / "main.py")
    engine_root = str(PROJECT_ROOT)
    if replay_dir is None:
        replay_dir = PROJECT_ROOT / "replays" / "training"
    replay_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="selfplay_")
    game_args = []
    for i in range(n_games):
        map_path = str(PROJECT_ROOT / "maps" / random.choice(maps))
        log_path = os.path.join(tmpdir, f"traj_{i}.pkl")
        replay_path = str(replay_dir / f"ep{episode_num}_{opp_id}_g{i}.replay26")
        seed = random.randint(1, 2**31)
        game_args.append((bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed))

    os.environ["TENSOR_OPP_WEIGHTS"] = opponent_weights_path
    # Use fork — no torch imported yet, so no fork issues
    ctx = mp.get_context("fork")
    with ctx.Pool(min(n_workers, n_games)) as pool:
        results = pool.map(_run_game, game_args)
    os.environ.pop("TENSOR_OPP_WEIGHTS", None)

    trajectories, winners = [], []
    for traj_path, _winner in results:
        game_trajs = _read_trajectory_file(traj_path)
        trajectories.extend(game_trajs)
        # Winner: did our bot (A) place any harvesters on ore?
        if game_trajs:
            total_harv = sum(
                sum(1 for s in t if isinstance(s, (list, tuple)) and len(s) >= 10 and s[9] > 0.5)
                for t in game_trajs
            )
            winners.append("a" if total_harv > 0 else "b")
        else:
            winners.append(None)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return trajectories, winners


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


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Self-play training with league system")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponents", type=int, default=3, help="Opponents per episode")
    parser.add_argument("--games-per-opp", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0, help="0=auto")
    parser.add_argument("--league-size", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--session", type=str, default="", help="Session name (empty=new)")
    parser.add_argument("--save", type=str, default="", help="Override weight path")
    args = parser.parse_args()

    # Session setup
    session_name = args.session or _default_session_name()
    sess = _init_session(session_name)
    weight_path = args.save or str(sess["weight_path"])
    log_path = sess["log_path"]
    replay_dir = sess["replay_dir"]

    n_workers = args.workers or min(os.cpu_count() or 4, args.opponents * args.games_per_opp)
    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]
    league = League(sess["league_dir"], sess["league_json"], args.league_size)

    # Seed league with current weights if empty
    if not league.models:
        weights = None
        # Try session weights first, then global weights
        for wp in [weight_path, str(WEIGHT_PATH)]:
            if os.path.exists(wp):
                with open(wp) as f:
                    weights = json.load(f)
                if weights:
                    break
        if weights:
            league.add_model(weights, 0)
            print(f"Seeded league with gen_0 (ELO {ELO_DEFAULT})")

    print(f"Session: {session_name}")
    print(f"Self-play: {args.episodes} episodes, {args.opponents} opponents x {args.games_per_opp} games")
    print(f"League size: {args.league_size} | Workers: {n_workers}")

    # ---- Collection phase uses fork (no torch) ----
    # ---- After all collection, import torch for training ----

    t_start = time.time()
    train_log = []
    model = None  # lazy init after first collection

    for ep in range(args.episodes):
        t0 = time.time()
        gen = league.latest_gen() + 1
        current_id = f"gen_{gen}"

        opponents = league.sample_opponents(min(args.opponents, len(league.models)))
        if not opponents:
            # No league yet — need to init model first
            if model is None:
                from model import BuilderMLP, DEVICE, train_on_trajectories
                import torch
                model = BuilderMLP().to(DEVICE)
                model.load_weights(weight_path)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)
                league.add_model(model.export_weights(), 0)
                print(f"Seeded league with gen_0")
                opponents = league.sample_opponents(1)

        # Save current weights for bot A
        if model is not None:
            model.save_weights(weight_path)

        # Collect games (fork, no torch)
        all_trajs = []
        total_wins, total_losses, total_games = 0, 0, 0
        per_opp_results: list[tuple[dict, list[str | None]]] = []

        for opp in opponents:
            trajs, winners = collect_games(
                opp["path"], args.games_per_opp, maps, n_workers, ep + 1, opp["id"],
                replay_dir=replay_dir,
            )
            all_trajs.extend(trajs)
            per_opp_results.append((opp, winners))
            for w in winners:
                total_games += 1
                if w == "a": total_wins += 1
                elif w == "b": total_losses += 1

        t_collect = time.time() - t0

        # ---- Now import torch and train (only on first iteration) ----
        if model is None:
            from model import BuilderMLP, DEVICE, train_on_trajectories
            import torch
            model = BuilderMLP().to(DEVICE)
            model.load_weights(weight_path)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)
            print(f"Device: {DEVICE}")

        t1 = time.time()
        stats = train_on_trajectories(model, optimizer, all_trajs)
        scheduler.step()
        t_train = time.time() - t1

        # Checkpoint to league + per-game ELO updates
        if (ep + 1) % args.checkpoint_every == 0 or ep == 0:
            league.add_model(model.export_weights(), gen)
            for opp, winners in per_opp_results:
                for w in winners:
                    if w is not None:
                        league.update_elo(current_id, opp["id"], w)
            league.save()

        # Stats
        def _traj_reward(traj):
            r_harv, r_conv, r_dti, r_dmg = 0.0, 0.0, 0.0, 0.0
            for s in traj:
                if isinstance(s, (list, tuple)):
                    n = len(s)
                    dti = s[6]
                    harv = s[9] if n > 9 else 0.0
                    dmg = s[10] if n > 10 else 0.0
                    conv = s[12] if n > 12 else 0.0
                else:
                    dti = s.get("dti", 0)
                    harv = s.get("harv", 0)
                    dmg = s.get("dmg", 0)
                    conv = s.get("conv", 0)
                r_harv += harv * 5.0
                r_conv += conv * 2.0
                r_dti += max(dti, 0) * 0.2
                r_dmg += dmg * 1.0
            return r_harv + r_conv + r_dti + r_dmg, r_harv, r_conv, r_dti, r_dmg

        def _traj_ti(traj):
            last = traj[-1]
            return last[8] if isinstance(last, (list, tuple)) else last.get("ti", 0)

        n_trajs = max(len(all_trajs), 1)
        rewards = [_traj_reward(t) for t in all_trajs]
        mean_reward = sum(r[0] for r in rewards) / n_trajs
        mean_r_harv = sum(r[1] for r in rewards) / n_trajs
        mean_r_conv = sum(r[2] for r in rewards) / n_trajs
        mean_r_dti = sum(r[3] for r in rewards) / n_trajs
        mean_r_dmg = sum(r[4] for r in rewards) / n_trajs
        mean_ti = sum(_traj_ti(t) for t in all_trajs if t) / n_trajs
        elapsed = time.time() - t_start
        train_log.append({
            "episode": ep + 1,
            "reward": round(mean_reward, 4),
            "r_harv": round(mean_r_harv, 4),
            "r_conv": round(mean_r_conv, 4),
            "r_dti": round(mean_r_dti, 4),
            "r_dmg": round(mean_r_dmg, 4),
            "resources": round(mean_ti, 2),
            "best_resources": round(mean_ti, 2),
            "pi_loss": round(stats["pi_loss"], 5),
            "vf_loss": round(stats["vf_loss"], 5),
            "entropy": round(stats["entropy"], 4),
            "approx_kl": round(stats.get("approx_kl", 0), 6),
            "clip_frac": round(stats.get("clip_frac", 0), 4),
            "explained_var": stats.get("explained_var", 0),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "sps": stats["n_steps"],
            "elapsed": round(elapsed, 1),
            "winrate": round(total_wins / max(total_games, 1), 3),
            "league_size": len(league.models),
        })

        if (ep + 1) % 5 == 0 or ep == 0 or ep == args.episodes - 1:
            with open(str(log_path), "w") as f:
                json.dump({"episodes": train_log, "config": {
                    "envs": args.opponents * args.games_per_opp,
                    "ep_len": "self-play", "grid": "real", "lr": args.lr,
                    "total_episodes": args.episodes, "mode": "self-play",
                    "league_size": args.league_size,
                    "session": session_name,
                }}, f)

        if (ep + 1) % 5 == 0 or ep == 0:
            rankings = league.get_rankings()
            top3 = ", ".join(f"{r['id']}({r['elo']:.0f})" for r in rankings[:3])
            print(
                f"  Ep {ep + 1:3d}/{args.episodes}: "
                f"W/L={total_wins}/{total_losses}/{total_games}  "
                f"reward={mean_reward:+.2f}  ti={mean_ti:.0f}  "
                f"pi={stats['pi_loss']:+.4f}  "
                f"kl={stats.get('approx_kl', 0):.4f}  "
                f"ev={stats.get('explained_var', 0):.2f}  "
                f"clip={stats.get('clip_frac', 0):.2f}  "
                f"collect={t_collect:.1f}s train={t_train:.1f}s  [{elapsed:.0f}s]"
            )

    model.save_weights(weight_path)
    # Also copy final weights to global path for the bot to use
    shutil.copy2(weight_path, str(WEIGHT_PATH))
    league.save()

    print(f"\nDone in {time.time() - t_start:.0f}s")
    print(f"\nLeague rankings:")
    for r in league.get_rankings():
        print(f"  {r['id']:>12s}  ELO={r['elo']:6.1f}  W={r['wins']} L={r['losses']} D={r['draws']}")


if __name__ == "__main__":
    main()
