#!/usr/bin/env python3
"""Benchmark a single training episode with timing breakdowns.

Default workload is exactly one episode, one opponent, one game. The benchmark
uses the real collection and PPO paths rather than a synthetic microbench.

Examples:
    python bots/tensor/torch/bench_episode.py
    python bots/tensor/torch/bench_episode.py --mode selfplay
    python bots/tensor/torch/bench_episode.py --json
"""

from __future__ import annotations

import argparse
import ast  # noqa: F401 - engine exec expects ast in sys.modules
import json
import os
import pickle
import random
import shutil
import time
from pathlib import Path


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f}s"


def _print_breakdown(title: str, rows: list[tuple[str, float]], total: float) -> None:
    print(f"\n{title}:")
    for label, seconds in rows:
        pct = 0.0 if total <= 0 else (seconds / total) * 100.0
        print(f"  {label:<24} {_fmt(seconds):>9}  {pct:5.1f}%")


def _maybe_cleanup(path: Path, keep: bool) -> None:
    if not keep:
        shutil.rmtree(path, ignore_errors=True)


def _label_from_key(key: str) -> str:
    if key.endswith("_s"):
        key = key[:-2]
    return key.replace("_", " ")


def _load_weight_state(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_model_compat(model, *paths: str | Path) -> bool:
    for path in paths:
        data = _load_weight_state(path)
        if data is None:
            continue
        try:
            if isinstance(data, dict):
                model.load_state_dict(data)
            else:
                model.import_weights(data)
            return True
        except Exception:
            continue
    return False


def _benchmark_batched(args) -> dict:
    from train_batched import (
        BuilderMLP,
        DEVICE,
        League,
        PROJECT_ROOT,
        TEACHER_HIDDEN,
        WEIGHT_PATH,
        _default_session_name,
        _init_session,
        collect_games_batched,
        torch,
        train_on_trajectories,
    )

    session_name = args.session or f"bench_batched_{_default_session_name()}"
    sess = _init_session(session_name)
    session_dir = Path(sess["dir"])
    weight_path = str(sess["weight_path"])
    replay_dir = sess["replay_dir"]
    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]

    total_start = time.perf_counter()

    setup_start = time.perf_counter()
    hidden = TEACHER_HIDDEN if args.teacher else None
    model = BuilderMLP(hidden=hidden).to(DEVICE)
    _load_model_compat(model, weight_path, WEIGHT_PATH)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start

    seed_start = time.perf_counter()
    league = League(sess["league_dir"], sess["league_json"], args.league_size)
    if not league.models:
        ckpt = str(league.league_dir / "gen_0.pkl")
        model.save_weights(ckpt)
        league.register_model(ckpt, 0)
    opponents = league.sample_opponents(1)
    if not opponents:
        raise RuntimeError("Could not seed an opponent for the benchmark.")
    seed_s = time.perf_counter() - seed_start

    save_start = time.perf_counter()
    model.save_weights(weight_path)
    save_weights_s = time.perf_counter() - save_start

    opp_paths = [opponents[0]["path"]]
    opp_ids = [opponents[0]["id"]]

    collection_profile: dict[str, float] = {}
    all_trajs, all_winners, per_opp_winners = collect_games_batched(
        model,
        opp_paths,
        opp_ids,
        maps,
        args.workers,
        1,
        replay_dir,
        opp_hidden=hidden,
        profile=collection_profile,
    )
    if not all_trajs:
        raise RuntimeError("Benchmark game produced no trajectories.")

    train_profile: dict[str, float] = {}
    train_start = time.perf_counter()
    model.train()
    stats = train_on_trajectories(model, optimizer, all_trajs, profile=train_profile)
    scheduler.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_total_s = time.perf_counter() - train_start

    checkpoint_start = time.perf_counter()
    ckpt = str(league.league_dir / "gen_1.pkl")
    model.save_weights(ckpt)
    league.register_model(ckpt, 1)
    for winner in per_opp_winners.get(opponents[0]["id"], []):
        if winner is not None:
            league.update_elo("gen_1", opponents[0]["id"], winner)
    league.save()
    checkpoint_s = time.perf_counter() - checkpoint_start

    total_s = time.perf_counter() - total_start
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "mode": "batched",
        "session": session_name,
        "device": str(DEVICE),
        "teacher": bool(args.teacher),
        "games": 1,
        "wins": sum(1 for w in all_winners if w == "a"),
        "losses": sum(1 for w in all_winners if w == "b"),
        "trajectories": len(all_trajs),
        "steps": int(stats.get("n_steps", 0)),
        "gpu_inferences": int(collection_profile.get("gpu_inferences", 0)),
        "params": int(n_params),
        "total_s": total_s,
        "setup_s": setup_s,
        "seed_s": seed_s,
        "save_weights_s": save_weights_s,
        "collection_s": collection_profile.get("collection_total_s", 0.0),
        "train_s": train_total_s,
        "checkpoint_s": checkpoint_s,
        "collection_profile": collection_profile,
        "train_profile": train_profile,
        "ppo_stats": stats,
    }

    _maybe_cleanup(session_dir, args.keep_session)
    return result


def _benchmark_selfplay(args) -> dict:
    from train_selfplay import (
        ELO_DEFAULT,
        League,
        PROJECT_ROOT,
        WEIGHT_PATH,
        _default_session_name,
        _init_session,
        collect_all_games,
    )

    session_name = args.session or f"bench_selfplay_{_default_session_name()}"
    sess = _init_session(session_name)
    session_dir = Path(sess["dir"])
    weight_path = str(sess["weight_path"])
    replay_dir = sess["replay_dir"]
    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]

    global_weight_backup = None
    global_weight_exists = WEIGHT_PATH.exists()
    if global_weight_exists:
        global_weight_backup = WEIGHT_PATH.read_bytes()

    total_start = time.perf_counter()
    seed_start = time.perf_counter()
    league = League(sess["league_dir"], sess["league_json"], args.league_size)
    seed_weights = None
    if not league.models:
        for wp in [weight_path, str(WEIGHT_PATH)]:
            state = _load_weight_state(wp)
            if isinstance(state, list) and state:
                seed_weights = state
                break
        if not seed_weights:
            raise RuntimeError("Self-play benchmark requires existing weights to seed the league.")
        league.add_model(seed_weights, 0)
    opponents = league.sample_opponents(1)
    if not opponents:
        raise RuntimeError("Could not seed an opponent for the benchmark.")
    if seed_weights is None:
        state = _load_weight_state(opponents[0]["path"])
        if isinstance(state, list) and state:
            seed_weights = state
    if seed_weights is None:
        raise RuntimeError("Could not recover JSON-compatible seed weights for self-play benchmark.")
    seed_s = time.perf_counter() - seed_start

    # Tensor self-play bots read this path as JSON during collection.
    WEIGHT_PATH.write_text(json.dumps(seed_weights, separators=(",", ":")))

    collection_profile: dict[str, float] = {}
    all_trajs, per_opp_winners = collect_all_games(
        opponents,
        1,
        maps,
        args.workers,
        1,
        replay_dir=replay_dir,
        profile=collection_profile,
    )
    if not all_trajs:
        raise RuntimeError("Benchmark game produced no trajectories.")

    from model import BuilderMLP, DEVICE, TEACHER_HIDDEN, train_on_trajectories
    import torch

    setup_start = time.perf_counter()
    hidden = TEACHER_HIDDEN if args.teacher else None
    model = BuilderMLP(hidden=hidden).to(DEVICE)
    _load_model_compat(model, weight_path, WEIGHT_PATH)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start

    save_start = time.perf_counter()
    model.save_weights(weight_path)
    model.save_weights(str(WEIGHT_PATH))
    save_weights_s = time.perf_counter() - save_start

    train_profile: dict[str, float] = {}
    train_start = time.perf_counter()
    stats = train_on_trajectories(model, optimizer, all_trajs, profile=train_profile)
    scheduler.step()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_total_s = time.perf_counter() - train_start

    checkpoint_start = time.perf_counter()
    ckpt = str(league.league_dir / "gen_1.pkl")
    model.save_weights(ckpt)
    league.register_model(ckpt, 1)
    for winner in per_opp_winners.get(opponents[0]["id"], []):
        if winner is not None:
            league.update_elo("gen_1", opponents[0]["id"], winner)
    league.save()
    checkpoint_s = time.perf_counter() - checkpoint_start

    total_s = time.perf_counter() - total_start
    winners = per_opp_winners.get(opponents[0]["id"], [])
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "mode": "selfplay",
        "session": session_name,
        "device": str(DEVICE),
        "teacher": bool(args.teacher),
        "games": 1,
        "wins": sum(1 for w in winners if w == "a"),
        "losses": sum(1 for w in winners if w == "b"),
        "trajectories": len(all_trajs),
        "steps": int(stats.get("n_steps", 0)),
        "params": int(n_params),
        "seed_elo": ELO_DEFAULT,
        "total_s": total_s,
        "setup_s": setup_s,
        "seed_s": seed_s,
        "save_weights_s": save_weights_s,
        "collection_s": collection_profile.get("collection_total_s", 0.0),
        "train_s": train_total_s,
        "checkpoint_s": checkpoint_s,
        "collection_profile": collection_profile,
        "train_profile": train_profile,
        "ppo_stats": stats,
    }

    if global_weight_exists:
        WEIGHT_PATH.write_bytes(global_weight_backup)
    else:
        WEIGHT_PATH.unlink(missing_ok=True)
    _maybe_cleanup(session_dir, args.keep_session)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one full training episode")
    parser.add_argument("--mode", choices=("batched", "selfplay"), default="batched")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--league-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--keep-session", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    result = _benchmark_batched(args) if args.mode == "batched" else _benchmark_selfplay(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    total = result["total_s"]
    print(f"Mode: {result['mode']}")
    print(f"Device: {result['device']}")
    print(f"Session: {result['session']}")
    print(
        f"Game result: {result['wins']} win / {result['losses']} loss | "
        f"{result['trajectories']} trajectories | {result['steps']} steps"
    )
    if "gpu_inferences" in result:
        print(f"GPU inferences: {result['gpu_inferences']}")
    print(f"Parameters: {result['params']:,}")
    print(f"Total wall time: {_fmt(total)}")

    _print_breakdown(
        "Episode Breakdown",
        [
            ("setup", result["setup_s"]),
            ("seed league", result["seed_s"]),
            ("save weights", result["save_weights_s"]),
            ("collection", result["collection_s"]),
            ("train", result["train_s"]),
            ("checkpoint", result["checkpoint_s"]),
        ],
        total,
    )

    collection_rows = [
        (_label_from_key(k), v)
        for k, v in sorted(result["collection_profile"].items())
        if k.endswith("_s")
    ]
    if collection_rows:
        _print_breakdown("Collection Breakdown", collection_rows, result["collection_s"])

    train_rows = [
        (_label_from_key(k), v)
        for k, v in sorted(result["train_profile"].items())
        if k.endswith("_s")
    ]
    if train_rows:
        _print_breakdown("Train Breakdown", train_rows, result["train_s"])

    stats = result["ppo_stats"]
    print(
        "\nPPO stats: "
        f"pi={stats.get('pi_loss', 0):+.4f}  "
        f"vf={stats.get('vf_loss', 0):+.4f}  "
        f"entropy={stats.get('entropy', 0):.4f}  "
        f"kl={stats.get('approx_kl', 0):.5f}  "
        f"ev={stats.get('explained_var', 0)}"
    )


if __name__ == "__main__":
    main()
