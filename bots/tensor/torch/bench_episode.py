#!/usr/bin/env python3
"""Benchmark a single training episode with timing breakdowns.

Reuses the real collection and PPO paths from train_batched so the benchmark
always matches production behavior.

Examples:
    python bots/tensor/torch/bench_episode.py
    python bots/tensor/torch/bench_episode.py --opponents 2 --games-per-opp 2
    python bots/tensor/torch/bench_episode.py --dummy-opponent --json
"""

from __future__ import annotations

import argparse
import ast  # noqa: F401 - engine exec expects ast in sys.modules
import json
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


def _label_from_key(key: str) -> str:
    if key.endswith("_s"):
        key = key[:-2]
    return key.replace("_", " ")


def _benchmark_batched(args) -> dict:
    from train_batched import (
        BuilderMLP,
        DEVICE,
        DUMMY_OPP_ID,
        DUMMY_OPP_PATH,
        League,
        PROJECT_ROOT,
        TEACHER_HIDDEN,
        TRAIN_MAX_TURNS,
        WEIGHT_PATH,
        _default_session_name,
        _init_session,
        collect_games_batched,
        load_centralized_critic_checkpoint,
        reset_centralized_critic,
        save_centralized_critic_checkpoint,
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

    # --- Setup (matches train_batched main loop) ---
    setup_start = time.perf_counter()
    hidden = TEACHER_HIDDEN if args.teacher else None
    model = BuilderMLP(hidden=hidden).to(DEVICE)
    model.load_weights(WEIGHT_PATH)
    reset_centralized_critic()
    load_centralized_critic_checkpoint(WEIGHT_PATH, lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    setup_s = time.perf_counter() - setup_start

    # --- League + opponent selection (matches train_batched main loop) ---
    seed_start = time.perf_counter()
    league = League(sess["league_dir"], args.league_size)
    if not league.models:
        ckpt = str(league.league_dir / "gen_0.pkl")
        model.save_weights(ckpt)
        league.register_model(ckpt, 0)
    if args.dummy_opponent:
        opponents = [{"id": DUMMY_OPP_ID, "path": DUMMY_OPP_PATH}]
    else:
        opponents = league.sample_opponents(max(1, min(args.opponents, len(league.models))))
        if not opponents:
            raise RuntimeError("Could not seed an opponent for the benchmark.")
    seed_s = time.perf_counter() - seed_start

    # --- Build game list (matches train_batched episode loop) ---
    save_start = time.perf_counter()
    model.save_weights(weight_path)
    save_weights_s = time.perf_counter() - save_start

    all_opp_paths = []
    all_opp_ids = []
    for opp in opponents:
        games_per_opp = args.games_per_opp * (max(args.opponents, 1) if args.dummy_opponent else 1)
        for _ in range(games_per_opp):
            all_opp_paths.append(opp["path"])
            all_opp_ids.append(opp["id"])

    # --- Collection (identical call to train_batched) ---
    collection_profile: dict[str, float] = {}
    all_trajs, all_winners, per_opp_winners, all_global_states, traj_game_map, _game_origin, _game_delivery_totals, _game_specs = collect_games_batched(
        model, all_opp_paths, all_opp_ids, maps,
        1, replay_dir,
        opp_hidden=hidden,
        coalesce_ms=args.coalesce_ms,
        max_training_turns=TRAIN_MAX_TURNS,
        profile=collection_profile,
    )
    if not all_trajs:
        raise RuntimeError("Benchmark game produced no trajectories.")

    # --- Training (identical call to train_batched) ---
    train_profile: dict[str, float] = {}
    train_start = time.perf_counter()
    model.train()
    stats = train_on_trajectories(
        model, optimizer, all_trajs,
        profile=train_profile,
        global_states_list=all_global_states,
        traj_game_map=traj_game_map if all_global_states else None,
        game_winners=all_winners,
    )
    model.save_weights(weight_path)
    save_centralized_critic_checkpoint(weight_path)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_total_s = time.perf_counter() - train_start

    checkpoint_start = time.perf_counter()
    ckpt = str(league.league_dir / "gen_1.pkl")
    model.save_weights(ckpt)
    league.register_model(ckpt, 1)
    if not args.dummy_opponent:
        for opp_id, winners in per_opp_winners.items():
            for winner in winners:
                if winner is not None:
                    league.update_elo("gen_1", opp_id, winner)
    league.save()
    checkpoint_s = time.perf_counter() - checkpoint_start

    total_s = time.perf_counter() - total_start
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "mode": "batched",
        "session": session_name,
        "device": str(DEVICE),
        "teacher": bool(args.teacher),
        "dummy_opponent": bool(args.dummy_opponent),
        "games": len(all_opp_paths),
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

    if not args.keep_session:
        shutil.rmtree(session_dir, ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one full training episode")
    parser.add_argument("--league-size", type=int, default=2)
    parser.add_argument("--opponents", type=int, default=2)
    parser.add_argument("--games-per-opp", type=int, default=2)
    parser.add_argument("--coalesce-ms", type=float, default=0.25)
    parser.add_argument("--dummy-opponent", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--keep-session", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    result = _benchmark_batched(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    total = result["total_s"]
    print(f"\nMode: {result['mode']}")
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
