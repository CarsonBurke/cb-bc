#!/usr/bin/env python3
"""Train tensor bot using the real cambc engine with CPU-parallel games.

Runs N games in parallel against an external opponent, collects trajectories
logged by the bot, then trains on GPU with PPO.

Usage:
    python bots/tensor/torch/train_engine.py                  # Defaults
    python bots/tensor/torch/train_engine.py --games 20 --episodes 100
    python bots/tensor/torch/train_engine.py --opponent b_v6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

from model import (
    BuilderMLP, DEVICE, PROJECT_ROOT, WEIGHT_PATH,
    train_on_trajectories,
)

SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"


def _init_session(name: str) -> dict:
    import time as _time
    d = SESSIONS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "replays").mkdir(exist_ok=True)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump({"name": name, "created": _time.time(), "mode": "engine"}, f)
    return {
        "dir": d,
        "weight_path": d / "weights.json",
        "log_path": d / "train_log.json",
        "replay_dir": d / "replays",
    }


def _default_session_name() -> str:
    return f"eng_{time.strftime('%Y%m%d_%H%M%S')}"


# ---- Game runner ----

def _run_game(args: tuple) -> str | None:
    bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed = args
    os.environ["TENSOR_TRAIN_LOG"] = log_path
    try:
        from cambc.cambc_engine import run_game
        run_game(bot_a, bot_b, engine_root, map_path, replay_path, seed)
    except Exception as e:
        print(f"  Game failed (seed={seed}): {e}")
        return None
    finally:
        os.environ.pop("TENSOR_TRAIN_LOG", None)
    return log_path if os.path.exists(log_path) else None


def run_parallel_games(
    n_games: int, opponent: str, maps: list[str], n_workers: int,
    episode_num: int = 0, replay_dir: Path | None = None,
) -> list[list[dict]]:
    bot_a = str(PROJECT_ROOT / "bots" / "tensor" / "main.py")
    opp_path = PROJECT_ROOT / "bots" / opponent / "main.py"
    if not opp_path.exists():
        raise FileNotFoundError(f"Opponent not found: {opp_path}")
    bot_b = str(opp_path)
    engine_root = str(PROJECT_ROOT)
    if replay_dir is None:
        replay_dir = PROJECT_ROOT / "replays" / "training"
    replay_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="tensor_train_")

    game_args = []
    for i in range(n_games):
        map_path = str(PROJECT_ROOT / "maps" / random.choice(maps))
        log_path = os.path.join(tmpdir, f"traj_{i}.json")
        replay_path = str(replay_dir / f"ep{episode_num}_game{i}.replay26")
        seed = random.randint(1, 2**31)
        game_args.append((bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed))

    ctx = mp.get_context("spawn")
    with ctx.Pool(min(n_workers, n_games)) as pool:
        results = pool.map(_run_game, game_args)

    trajectories = []
    for log_path in results:
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    traj = json.load(f)
                if traj:
                    trajectories.append(traj)
            except Exception:
                pass
    shutil.rmtree(tmpdir, ignore_errors=True)
    return trajectories


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Train with real cambc engine")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", type=str, default="starter")
    parser.add_argument("--workers", type=int, default=0, help="0=auto")
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

    n_workers = args.workers or min(os.cpu_count() or 4, args.games)
    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]

    print(f"Session: {session_name}")
    print(f"Device: {DEVICE}")
    print(f"Real-engine training: {args.episodes} episodes x {args.games} games/ep")
    print(f"Opponent: {args.opponent} | Workers: {n_workers} | Maps: {len(maps)}")

    model = BuilderMLP().to(DEVICE)
    if model.load_weights(weight_path):
        print(f"Loaded weights from {weight_path}")

    import torch
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    t_start = time.time()
    train_log = []

    for ep in range(args.episodes):
        t0 = time.time()
        model.save_weights(weight_path)

        trajectories = run_parallel_games(args.games, args.opponent, maps, n_workers, ep + 1, replay_dir)
        t_collect = time.time() - t0

        t1 = time.time()
        stats = train_on_trajectories(model, optimizer, trajectories)
        scheduler.step()
        t_train = time.time() - t1

        mean_reward = sum(
            sum(s["dti"] * 0.1 + s["dax"] * 0.05 + 0.01 for s in t) for t in trajectories
        ) / max(len(trajectories), 1)
        mean_ti = sum(t[-1]["ti"] for t in trajectories if t) / max(len(trajectories), 1) if trajectories else 0
        elapsed = time.time() - t_start
        train_log.append({
            "episode": ep + 1,
            "reward": round(mean_reward, 4),
            "resources": round(mean_ti, 2),
"best_resources": round(mean_ti, 2),
            "pi_loss": round(stats["pi_loss"], 5),
            "vf_loss": round(stats["vf_loss"], 5),
            "entropy": round(stats["entropy"], 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "sps": stats["n_steps"],
            "elapsed": round(elapsed, 1),
        })

        if (ep + 1) % 5 == 0 or ep == 0 or ep == args.episodes - 1:
            with open(str(log_path), "w") as f:
                json.dump({"episodes": train_log, "config": {
                    "envs": args.games, "ep_len": "real-engine",
                    "grid": "real", "lr": args.lr,
                    "total_episodes": args.episodes, "opponent": args.opponent,
                    "session": session_name,
                }}, f)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Ep {ep + 1:3d}/{args.episodes}: "
                f"games={len(trajectories)} steps={stats['n_steps']:,}  "
                f"reward={mean_reward:+.2f}  ti={mean_ti:.0f}  "
                f"pi={stats['pi_loss']:+.4f} vf={stats['vf_loss']:.4f}  "
                f"collect={t_collect:.1f}s train={t_train:.1f}s  [{elapsed:.0f}s]"
            )

    model.save_weights(weight_path)
    import shutil
    shutil.copy2(weight_path, str(WEIGHT_PATH))
    print(f"\nDone in {time.time() - t_start:.0f}s. Weights saved to {weight_path}")


if __name__ == "__main__":
    main()
