#!/usr/bin/env python3
"""Batched GPU self-play training via pipe-based proxy bots.

Architecture:
- Main process: GPU model + inference server
- Game workers (spawn): Run engine with proxy bot, no torch needed
- Proxy bot: sends obs via OS pipe, blocks on pipe read for actions
- Server: collects obs from all games via select(), batches on GPU, responds

Usage:
    python bots/tensor/torch/train_batched.py --episodes 50 --games 8
"""

from __future__ import annotations

import argparse
import ast  # noqa: F401 — engine needs ast in sys.modules when forking
import json
import multiprocessing as mp
import os
import pickle
import random
import select
import shutil
import struct
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


from model import BuilderMLP, DEVICE, train_on_trajectories, TEACHER_HIDDEN, distill, _parse_trajectories, OBS_SIZE, H_DIM
from replay_parser import augment_trajectories_with_working_counts

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"

from train_selfplay import (
    League, _init_session, _default_session_name, _read_trajectory_file,
    ELO_K, ELO_DEFAULT,
)


def _run_game_proxy(args):
    """Run a single game with the proxy bot. No torch needed."""
    bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed = args
    os.environ["TENSOR_TRAIN_LOG"] = log_path
    try:
        from cambc.cambc_engine import run_game
        from replay_parser import extract_working_harvesters
        run_game(bot_a, bot_b, engine_root, map_path, replay_path, seed)
        working = extract_working_harvesters(replay_path)
    except Exception as e:
        print(f"  Game error: {e}")
        return None, None
    finally:
        os.environ.pop("TENSOR_TRAIN_LOG", None)
    return (log_path if os.path.exists(log_path) else None), working


# Struct-based binary IPC protocol constants
_OBS_FLOATS = OBS_SIZE + H_DIM  # 423 + 16 = 439
_OBS_FLOAT_BYTES = _OBS_FLOATS * 4
_LEN_PREFIX = struct.Struct('=I')


def _read_exact(fd, n):
    """Read exactly n bytes from fd."""
    data = b""
    while len(data) < n:
        chunk = os.read(fd, n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _pipe_read_obs(fd):
    """Read struct-packed observation. Returns raw float bytes or None."""
    raw = _read_exact(fd, 4)
    if raw is None:
        return None
    length = _LEN_PREFIX.unpack(raw)[0]
    payload = _read_exact(fd, length)
    return payload  # raw float bytes: OBS_SIZE + H_DIM floats


def _pipe_write_action_a(fd, bt, bd, md, h_next_bytes, old_lp):
    """Write struct-packed action for training bot (A)."""
    payload = struct.pack('=iii', bt, bd, md) + h_next_bytes + struct.pack('=f', old_lp)
    os.write(fd, _LEN_PREFIX.pack(len(payload)) + payload)


def _pipe_write_action_b(fd, bt, bd, md, h_next_bytes):
    """Write struct-packed action for opponent bot (B)."""
    payload = struct.pack('=iii', bt, bd, md) + h_next_bytes
    os.write(fd, _LEN_PREFIX.pack(len(payload)) + payload)


def _game_worker(args):
    """Run a single game with pipe FDs for both sides. Module-level for pickle."""
    *game_arg, obs_w_a, act_r_a, obs_w_b, act_r_b = args
    os.environ["TENSOR_OBS_WFD"] = str(obs_w_a)
    os.environ["TENSOR_ACT_RFD"] = str(act_r_a)
    os.environ["TENSOR_OBS_WFD_B"] = str(obs_w_b)
    os.environ["TENSOR_ACT_RFD_B"] = str(act_r_b)
    try:
        result = _run_game_proxy(tuple(game_arg))
    finally:
        for key in ("TENSOR_OBS_WFD", "TENSOR_ACT_RFD", "TENSOR_OBS_WFD_B", "TENSOR_ACT_RFD_B"):
            os.environ.pop(key, None)
        for fd in (obs_w_a, act_r_a, obs_w_b, act_r_b):
            try:
                os.close(fd)
            except OSError:
                pass
    return result


def collect_games_batched(
    model: BuilderMLP,
    opp_weight_paths: list[str],
    opp_ids: list[str],
    maps: list[str],
    n_workers: int,
    episode_num: int,
    replay_dir: Path,
    opp_hidden: list[int] | None = None,
    profile: dict[str, float] | None = None,
) -> tuple[list[list], list[str | None], dict[str, list[str | None]]]:
    """Run all games with batched GPU inference via pipes. Returns (trajs, winners, per_opp_winners)."""
    t_total = time.perf_counter() if profile is not None else 0.0
    bot_a = str(PROJECT_ROOT / "bots" / "tensor_proxy" / "main.py")
    bot_b = str(PROJECT_ROOT / "bots" / "tensor_proxy_opp" / "main.py")
    engine_root = str(PROJECT_ROOT)
    replay_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="batched_")
    n_games = len(opp_weight_paths)

    # Create pipe pairs per game: A pipes (training bot) + B pipes (opponent bot)
    pipes_a = []  # (obs_r, obs_w, act_r, act_w) for bot A
    pipes_b = []  # (obs_r, obs_w, act_r, act_w) for bot B
    for i in range(n_games):
        obs_r, obs_w = os.pipe()
        act_r, act_w = os.pipe()
        pipes_a.append((obs_r, obs_w, act_r, act_w))
        obs_r, obs_w = os.pipe()
        act_r, act_w = os.pipe()
        pipes_b.append((obs_r, obs_w, act_r, act_w))

    # Build game args
    game_args = []
    for i in range(n_games):
        map_path = str(PROJECT_ROOT / "maps" / random.choice(maps))
        log_path = os.path.join(tmpdir, f"traj_{i}.pkl")
        replay_path = str(replay_dir / f"ep{episode_num}_{opp_ids[i]}_g{i}.replay26")
        seed = random.randint(1, 2**31)
        game_args.append((bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed))

    # Worker args: game_args + A write/read FDs + B write/read FDs
    worker_args = [
        (*game_args[i], pipes_a[i][1], pipes_a[i][2], pipes_b[i][1], pipes_b[i][2])
        for i in range(n_games)
    ]

    # Load opponent models on GPU (deduplicate by weight path)
    t_phase = time.perf_counter() if profile is not None else 0.0
    opp_models = {}  # weight_path -> BuilderMLP
    game_to_opp = {}  # game_idx -> weight_path
    for i in range(n_games):
        wp = opp_weight_paths[i]
        game_to_opp[i] = wp
        if wp not in opp_models:
            opp_m = BuilderMLP(hidden=opp_hidden).to(DEVICE)
            opp_m.load_weights(wp)
            opp_m.eval()
            opp_models[wp] = opp_m
    if profile is not None and DEVICE.type == "cuda":
        torch.cuda.synchronize()
    if profile is not None:
        profile["opp_model_load_s"] = time.perf_counter() - t_phase

    # Fork workers
    t_phase = time.perf_counter() if profile is not None else 0.0
    ctx = mp.get_context("fork")
    pool = ctx.Pool(min(n_workers, n_games))
    async_result = pool.map_async(_game_worker, worker_args)
    if profile is not None:
        profile["pool_dispatch_s"] = time.perf_counter() - t_phase

    # Close child-side FDs in parent
    fd_to_info = {}  # fd -> (game_idx, side)  side='a' or 'b'
    act_fds = {}  # (game_idx, side) -> act_write_fd
    active_fds = set()

    for i in range(n_games):
        # A pipes: parent reads obs, writes acts
        os.close(pipes_a[i][1])  # close obs_w
        os.close(pipes_a[i][2])  # close act_r
        fd_to_info[pipes_a[i][0]] = (i, "a")
        act_fds[(i, "a")] = pipes_a[i][3]
        active_fds.add(pipes_a[i][0])

        # B pipes: parent reads obs, writes acts
        os.close(pipes_b[i][1])  # close obs_w
        os.close(pipes_b[i][2])  # close act_r
        fd_to_info[pipes_b[i][0]] = (i, "b")
        act_fds[(i, "b")] = pipes_b[i][3]
        active_fds.add(pipes_b[i][0])

    # Inference server loop
    model.eval()
    total_inferences = 0

    while active_fds:
        try:
            t_phase = time.perf_counter() if profile is not None else 0.0
            readable, _, _ = select.select(list(active_fds), [], [], 1.0)
            if profile is not None:
                profile["ipc_wait_s"] = profile.get("ipc_wait_s", 0.0) + (time.perf_counter() - t_phase)
        except (ValueError, OSError):
            break

        if not readable:
            if async_result.ready():
                break
            continue

        # Read all available observations, split by side
        pending_a = []  # (game_idx, float_bytes)
        pending_b = []  # (game_idx, float_bytes, opp_weight_path)
        t_phase = time.perf_counter() if profile is not None else 0.0
        for fd in readable:
            game_idx, side = fd_to_info[fd]
            float_bytes = _pipe_read_obs(fd)
            if float_bytes is None:
                active_fds.discard(fd)
                try:
                    os.close(fd)
                except OSError:
                    pass
                continue
            if side == "a":
                pending_a.append((game_idx, float_bytes))
            else:
                pending_b.append((game_idx, float_bytes, game_to_opp[game_idx]))
        if profile is not None:
            profile["ipc_read_s"] = profile.get("ipc_read_s", 0.0) + (time.perf_counter() - t_phase)

        # Batch A inference (all same model — training bot with exploration)
        if pending_a:
            n_a = len(pending_a)
            # Concatenate raw bytes → tensor (no Python float objects)
            all_bytes = bytearray()
            for _, fb in pending_a:
                all_bytes.extend(fb)

            t_phase = time.perf_counter() if profile is not None else 0.0
            with torch.no_grad():
                all_t = torch.frombuffer(all_bytes, dtype=torch.float32).reshape(n_a, _OBS_FLOATS).to(DEVICE)
                obs_t = all_t[:, :OBS_SIZE]
                h_t = all_t[:, OBS_SIZE:]
                bt_l, bd_l, md_l, values, h_next = model(obs_t, h_t)
                bt_d = Categorical(logits=bt_l)
                bd_d = Categorical(logits=bd_l)
                md_d = Categorical(logits=md_l)
                bt_act = bt_d.sample()
                bd_act = bd_d.sample()
                md_act = md_d.sample()
                # Mask build_dir log_prob when build_type is "none" (idx 0)
                bd_lp = bd_d.log_prob(bd_act)
                bd_mask = (bt_act != 0).float()
                lp = bt_d.log_prob(bt_act) + bd_lp * bd_mask + md_d.log_prob(md_act)

                # Single bulk transfer to CPU — one sync point
                # h_next is already tanh'd by model.forward()
                results_cpu = torch.stack([
                    bt_act.float(), bd_act.float(), md_act.float(), lp,
                ]).cpu()
                h_cpu = h_next.cpu()
            if profile is not None and DEVICE.type == "cuda":
                torch.cuda.synchronize()
            if profile is not None:
                profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + (time.perf_counter() - t_phase)

            _h_fmt = struct.Struct(f'={H_DIM}f')
            t_phase = time.perf_counter() if profile is not None else 0.0
            for i, (game_idx, _) in enumerate(pending_a):
                try:
                    _pipe_write_action_a(
                        act_fds[(game_idx, "a")],
                        int(results_cpu[0, i]), int(results_cpu[1, i]), int(results_cpu[2, i]),
                        _h_fmt.pack(*h_cpu[i].tolist()), float(results_cpu[3, i]),
                    )
                except OSError:
                    pass
            if profile is not None:
                profile["ipc_write_a_s"] = profile.get("ipc_write_a_s", 0.0) + (time.perf_counter() - t_phase)
            total_inferences += n_a

        # Batch B inference grouped by opponent model (low-temp sampling for diversity)
        OPP_TEMP = 0.5
        if pending_b:
            by_opp = {}
            for game_idx, fb, wp in pending_b:
                by_opp.setdefault(wp, []).append((game_idx, fb))

            for wp, items in by_opp.items():
                opp_m = opp_models[wp]
                n_b = len(items)
                all_bytes = bytearray()
                for _, fb in items:
                    all_bytes.extend(fb)

                t_phase = time.perf_counter() if profile is not None else 0.0
                with torch.no_grad():
                    all_t = torch.frombuffer(all_bytes, dtype=torch.float32).reshape(n_b, _OBS_FLOATS).to(DEVICE)
                    obs_t = all_t[:, :OBS_SIZE]
                    h_t = all_t[:, OBS_SIZE:]
                    bt_l, bd_l, md_l, _, h_next = opp_m(obs_t, h_t)
                    bt_act = Categorical(logits=bt_l / OPP_TEMP).sample()
                    bd_act = Categorical(logits=bd_l / OPP_TEMP).sample()
                    md_act = Categorical(logits=md_l / OPP_TEMP).sample()
                    # h_next is already tanh'd by model.forward()
                    acts_cpu = torch.stack([bt_act, bd_act, md_act]).cpu()
                    h_cpu = h_next.cpu()
                if profile is not None and DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                if profile is not None:
                    profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + (time.perf_counter() - t_phase)

                _h_fmt = struct.Struct(f'={H_DIM}f')
                t_phase = time.perf_counter() if profile is not None else 0.0
                for i, (game_idx, _) in enumerate(items):
                    try:
                        _pipe_write_action_b(
                            act_fds[(game_idx, "b")],
                            int(acts_cpu[0, i]), int(acts_cpu[1, i]), int(acts_cpu[2, i]),
                            _h_fmt.pack(*h_cpu[i].tolist()),
                        )
                    except OSError:
                        pass
                if profile is not None:
                    profile["ipc_write_b_s"] = profile.get("ipc_write_b_s", 0.0) + (time.perf_counter() - t_phase)
                total_inferences += n_b

        if not pending_a and not pending_b:
            if async_result.ready():
                break

    # Close remaining fds
    for fd in list(active_fds):
        try:
            os.close(fd)
        except OSError:
            pass
    for key, fd in act_fds.items():
        try:
            os.close(fd)
        except OSError:
            pass

    # Get results
    try:
        t_phase = time.perf_counter() if profile is not None else 0.0
        results = async_result.get(timeout=30)
    except Exception:
        results = [None] * n_games
    if profile is not None:
        profile["worker_join_s"] = time.perf_counter() - t_phase

    pool.close()
    pool.join()

    # Read trajectories and augment with replay ground truth
    t_phase = time.perf_counter() if profile is not None else 0.0
    trajectories, winners = [], []
    per_opp_winners = {}
    for i, result in enumerate(results):
        oid = opp_ids[i]
        if isinstance(result, tuple):
            log_path, working = result
        else:
            log_path, working = result, None
        if log_path is None:
            per_opp_winners.setdefault(oid, []).append(None)
            continue
        game_trajs = _read_trajectory_file(log_path)
        game_trajs = augment_trajectories_with_working_counts(game_trajs, working, team=0)
        trajectories.extend(game_trajs)
        if game_trajs:
            total_harv = sum(
                sum(1 for s in t if isinstance(s, (list, tuple)) and len(s) >= 10 and s[9] > 0.5)
                for t in game_trajs
            )
            w = "a" if total_harv > 0 else "b"
        else:
            w = None
        winners.append(w)
        per_opp_winners.setdefault(oid, []).append(w)
    if profile is not None:
        profile["postprocess_s"] = time.perf_counter() - t_phase
        profile["collection_total_s"] = time.perf_counter() - t_total
        profile["gpu_inferences"] = float(total_inferences)
        profile["games"] = float(n_games)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"    [{n_games} games, {total_inferences} GPU inferences]")
    return trajectories, winners, per_opp_winners


def main():
    parser = argparse.ArgumentParser(description="Batched GPU self-play training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--games-per-opp", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--league-size", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--teacher", action="store_true", help="Train large teacher, distill to student")
    parser.add_argument("--distill-epochs", type=int, default=20, help="Distillation epochs")
    parser.add_argument("--distill-lr", type=float, default=1e-3, help="Distillation learning rate")
    parser.add_argument("--session", type=str, default="")
    args = parser.parse_args()

    session_name = args.session or _default_session_name()
    sess = _init_session(session_name)
    weight_path = str(sess["weight_path"])
    log_path = sess["log_path"]
    replay_dir = sess["replay_dir"]

    n_workers = args.workers or min(os.cpu_count() or 4, args.opponents * args.games_per_opp)
    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]
    league = League(sess["league_dir"], sess["league_json"], args.league_size)

    hidden = TEACHER_HIDDEN if args.teacher else None
    model = BuilderMLP(hidden=hidden).to(DEVICE)
    model.load_weights(weight_path) or model.load_weights(str(WEIGHT_PATH))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    if not league.models:
        ckpt = str(league.league_dir / "gen_0.pkl")
        model.save_weights(ckpt)
        league.register_model(ckpt, 0)

    n_params = sum(p.numel() for p in model.parameters())
    mode = "TEACHER" if args.teacher else "BATCHED GPU"
    print(f"Session: {session_name} [{mode}]")
    print(f"Device: {DEVICE} | {n_params:,} params")
    print(f"Self-play: {args.episodes} episodes, {args.opponents}x{args.games_per_opp} games")
    print(f"Workers: {n_workers}")

    t_start = time.time()
    train_log = []

    for ep in range(args.episodes):
        t0 = time.time()
        gen = league.latest_gen() + 1
        current_id = f"gen_{gen}"
        opponents = league.sample_opponents(min(args.opponents, len(league.models)))
        if not opponents:
            league.add_model(model.export_weights(), 0)
            opponents = league.sample_opponents(1)

        model.save_weights(weight_path)

        # Build all games across all opponents for parallel execution
        all_opp_paths = []
        all_opp_ids = []
        for opp in opponents:
            for _ in range(args.games_per_opp):
                all_opp_paths.append(opp["path"])
                all_opp_ids.append(opp["id"])

        all_trajs, all_winners, per_opp_winners = collect_games_batched(
            model, all_opp_paths, all_opp_ids, maps, n_workers,
            ep + 1, replay_dir,
            opp_hidden=hidden,
        )

        total_wins, total_losses, total_games = 0, 0, 0
        per_opp_results = []
        for opp in opponents:
            opp_winners = per_opp_winners.get(opp["id"], [])
            per_opp_results.append((opp, opp_winners))
            for w in opp_winners:
                total_games += 1
                if w == "a": total_wins += 1
                elif w == "b": total_losses += 1

        t_collect = time.time() - t0

        t1 = time.time()
        model.train()
        stats = train_on_trajectories(model, optimizer, all_trajs)
        scheduler.step()
        t_train = time.time() - t1

        if (ep + 1) % args.checkpoint_every == 0 or ep == 0:
            ckpt = str(league.league_dir / f"gen_{gen}.pkl")
            model.save_weights(ckpt)
            league.register_model(ckpt, gen)
            for opp, winners in per_opp_results:
                for w in winners:
                    if w is not None:
                        league.update_elo(current_id, opp["id"], w)
            league.save()

        # Stats
        def _traj_reward(traj):
            r_harv, r_conv, r_work, r_dmg = 0.0, 0.0, 0.0, 0.0
            for s in traj:
                if isinstance(s, (list, tuple)):
                    n = len(s)
                    n_working = s[6] if n > 6 else 0
                    conv_r = s[7] if n > 7 else 0.0
                    harv_r = s[9] if n > 9 else 0.0
                    dmg = s[10] if n > 10 else 0.0
                else:
                    n_working = s.get("n_working", 0)
                    conv_r = s.get("conv_r", 0)
                    harv_r = s.get("harv_r", 0)
                    dmg = s.get("dmg", 0)
                r_harv += harv_r; r_conv += conv_r
                r_work += n_working * 0.5; r_dmg += dmg * 20.0
            total = r_harv + r_conv + r_work + r_dmg
            return total, r_harv, r_conv, r_work, r_dmg

        def _traj_ti(traj):
            last = traj[-1]
            return last[8] if isinstance(last, (list, tuple)) else last.get("ti", 0)

        n_trajs = max(len(all_trajs), 1)
        rewards = [_traj_reward(t) for t in all_trajs]
        mean_reward = sum(r[0] for r in rewards) / n_trajs
        mean_r_harv = sum(r[1] for r in rewards) / n_trajs
        mean_r_conv = sum(r[2] for r in rewards) / n_trajs
        mean_r_work = sum(r[3] for r in rewards) / n_trajs
        mean_r_dmg = sum(r[4] for r in rewards) / n_trajs
        mean_ti = sum(_traj_ti(t) for t in all_trajs if t) / n_trajs
        elapsed = time.time() - t_start

        train_log.append({
            "episode": ep + 1, "reward": round(mean_reward, 4),
            "r_harv": round(mean_r_harv, 4), "r_conv": round(mean_r_conv, 4),
            "r_work": round(mean_r_work, 4), "r_dmg": round(mean_r_dmg, 4),
            "resources": round(mean_ti, 2), "best_resources": round(mean_ti, 2),
            "pi_loss": round(stats["pi_loss"], 5), "vf_loss": round(stats["vf_loss"], 5),
            "entropy": round(stats["entropy"], 4),
            "approx_kl": round(stats.get("approx_kl", 0), 6),
            "clip_frac": round(stats.get("clip_frac", 0), 4),
            "explained_var": stats.get("explained_var", 0),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "sps": stats["n_steps"], "elapsed": round(elapsed, 1),
            "winrate": round(total_wins / max(total_games, 1), 3),
            "league_size": len(league.models),
        })

        if (ep + 1) % 5 == 0 or ep == 0 or ep == args.episodes - 1:
            with open(str(log_path), "w") as f:
                json.dump({"episodes": train_log, "config": {
                    "envs": args.opponents * args.games_per_opp,
                    "ep_len": "self-play", "grid": "real", "lr": args.lr,
                    "total_episodes": args.episodes, "mode": "batched-gpu",
                    "session": session_name,
                }}, f)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Ep {ep + 1:3d}/{args.episodes}: "
                f"W/L={total_wins}/{total_losses}/{total_games}  "
                f"reward={mean_reward:+.2f}  ti={mean_ti:.0f}  "
                f"pi={stats['pi_loss']:+.4f}  "
                f"kl={stats.get('approx_kl', 0):.4f}  "
                f"ev={stats.get('explained_var', 0):.2f}  "
                f"collect={t_collect:.1f}s train={t_train:.1f}s  [{elapsed:.0f}s]"
            )

    model.save_weights(weight_path)
    league.save()

    if not args.teacher:
        model.save_weights(str(WEIGHT_PATH))

    print(f"\nDone in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
