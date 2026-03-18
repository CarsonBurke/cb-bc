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
import ast  # noqa: F401 — engine imports can still expect ast in sys.modules
import gc
import json
import os
import pickle
import random
import selectors
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _maybe_run_worker_mode() -> None:
    if "--worker-result" not in sys.argv:
        return
    from game_worker import main as _game_worker_main

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-result", required=True)
    parser.add_argument("--worker-pipes", required=True)
    parser.add_argument("--worker-game-args", required=True)
    args, _ = parser.parse_known_args()
    _game_worker_main([
        "--result", args.worker_result,
        "--pipes", args.worker_pipes,
        "--game-args", args.worker_game_args,
    ])
    raise SystemExit(0)


_maybe_run_worker_mode()

import torch

from model import (
    BuilderMLP,
    DEVICE,
    H_DIM,
    OBS_SIZE,
    TILE_FEATURES,
    TEACHER_HIDDEN,
    compute_action_masks,
    sample_logits_gumbel,
    train_on_trajectories,
)
from replay_parser import augment_trajectories_with_working_counts

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor" / "weights.json"

from training_utils import (
    League, _init_session, _default_session_name, _read_trajectory_file,
    ELO_DEFAULT,
)

# Struct-based binary IPC protocol constants
_OBS_FLOATS = OBS_SIZE + H_DIM
_OBS_FLOAT_BYTES = _OBS_FLOATS * 4
_ACT_HEAD = struct.Struct('=iii')
_H_STRUCT = struct.Struct(f'={H_DIM}f')
_LP_STRUCT = struct.Struct('=f')
_H_BYTES = H_DIM * 4
DEFAULT_COALESCE_MS = 0.25 if DEVICE.type == "cuda" else 0.0
WORKER_LAUNCH_STAGGER_S = 0.02
DUMMY_OPP_PATH = "__dummy__"
DUMMY_OPP_ID = "dummy"
_ZERO_H_BYTES = bytes(_H_BYTES)
_DUMMY_DIR_TILES = (6, 7, 8, 11, 12, 13, 16, 17, 18)
_dummy_idx_cache = {}


def _read_exact(fd, n):
    """Read exactly n bytes from fd."""
    data = bytearray(n)
    view = memoryview(data)
    used = 0
    while used < n:
        chunk = os.read(fd, n - used)
        if not chunk:
            return None
        end = used + len(chunk)
        view[used:end] = chunk
        used = end
    return data


def _pipe_read_obs(fd):
    """Read fixed-size float32 observation packet."""
    return _read_exact(fd, _OBS_FLOAT_BYTES)


def _pipe_write_action_a(fd, bt, bd, md, h_next_bytes, old_lp):
    """Write fixed-size action packet for training bot (A)."""
    os.writev(fd, (_ACT_HEAD.pack(bt, bd, md), h_next_bytes, _LP_STRUCT.pack(old_lp)))


def _pipe_write_action_b(fd, bt, bd, md, h_next_bytes):
    """Write fixed-size action packet for opponent bot (B)."""
    os.writev(fd, (_ACT_HEAD.pack(bt, bd, md), h_next_bytes))


def _dummy_indices(device: torch.device):
    cached = _dummy_idx_cache.get(device)
    if cached is not None:
        return cached
    ore_ti = torch.tensor(
        [tile * TILE_FEATURES + 1 for tile in _DUMMY_DIR_TILES],
        dtype=torch.long,
        device=device,
    )
    ore_ax = torch.tensor(
        [tile * TILE_FEATURES + 2 for tile in _DUMMY_DIR_TILES],
        dtype=torch.long,
        device=device,
    )
    cached = (ore_ti, ore_ax)
    _dummy_idx_cache[device] = cached
    return cached


def _dummy_policy(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cheap scripted opponent for early curriculum episodes."""
    bt_mask, bd_mask = compute_action_masks(obs)
    ore_ti_idx, ore_ax_idx = _dummy_indices(obs.device)
    ore_score = obs.index_select(1, ore_ti_idx) + obs.index_select(1, ore_ax_idx)
    has_ore_dir = bd_mask & (ore_score > 0.0)
    fallback_dir = bd_mask.to(torch.int64).argmax(dim=1)
    ore_dir = torch.where(
        has_ore_dir,
        ore_score,
        torch.full_like(ore_score, -1.0),
    ).argmax(dim=1)

    build_type = torch.zeros(obs.shape[0], dtype=torch.int64, device=obs.device)
    can_harv = bt_mask[:, 1] & has_ore_dir.any(dim=1)
    can_conv = bt_mask[:, 2] & bd_mask.any(dim=1)
    can_road = bt_mask[:, 3] & bd_mask.any(dim=1)
    build_type[can_harv] = 1
    build_type[~can_harv & can_conv] = 2
    build_type[~can_harv & ~can_conv & can_road] = 3

    build_dir = torch.full_like(build_type, 4)
    use_fallback = build_type > 1
    build_dir[can_harv] = ore_dir[can_harv]
    build_dir[use_fallback] = fallback_dir[use_fallback]

    move_dir = torch.full_like(build_type, 4)
    return build_type, build_dir, move_dir

def _load_worker_result(result_path: str) -> tuple[str | None, dict | None, str | None]:
    try:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return (None, None, None)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            return result[0], result[1], None
    return (None, None, None)


def _recommended_worker_count(n_games: int) -> int:
    """Allow moderate oversubscription while leaving one CPU thread for the server."""
    cpu = os.cpu_count() or 1
    if cpu <= 1:
        return min(n_games, 1)
    return max(1, min(n_games, (cpu - 1) * 2))


def _coalesce_readable(selector, readable, coalesce_s, fd_to_info, game_to_opp, profile=None):
    """Wait briefly for more FDs that can join the same eventual inference batch."""
    if coalesce_s <= 0.0 or not readable:
        return readable

    target_groups = set()
    for fd in readable:
        game_idx, side = fd_to_info[fd]
        group = ("a", None) if side == "a" else ("b", game_to_opp[game_idx])
        target_groups.add(group)

    ready = set(readable)
    t_phase = time.perf_counter() if profile is not None else 0.0
    try:
        events = selector.select(coalesce_s)
    except OSError:
        events = []
    for key, _ in events:
        fd = key.fd
        if fd in ready or fd not in fd_to_info:
            continue
        game_idx, side = fd_to_info[fd]
        group = ("a", None) if side == "a" else ("b", game_to_opp[game_idx])
        if group in target_groups:
            ready.add(fd)
    if profile is not None:
        profile["coalesce_wait_s"] = profile.get("coalesce_wait_s", 0.0) + (time.perf_counter() - t_phase)
    return list(ready)


def collect_games_batched(
    model: BuilderMLP,
    opp_weight_paths: list[str],
    opp_ids: list[str],
    maps: list[str],
    episode_num: int,
    replay_dir: Path,
    opp_hidden: list[int] | None = None,
    coalesce_ms: float = DEFAULT_COALESCE_MS,
    profile: dict[str, float] | None = None,
) -> tuple[list[list], list[str | None], dict[str, list[str | None]]]:
    """Run all games with batched GPU inference via pipes. Returns (trajs, winners, per_opp_winners)."""
    t_total = time.perf_counter() if profile is not None else 0.0
    bot_a = str(PROJECT_ROOT / "bots" / "tensor_proxy" / "main.py")
    bot_b_proxy = str(PROJECT_ROOT / "bots" / "tensor_proxy_opp" / "main.py")
    bot_b_dummy = str(PROJECT_ROOT / "bots" / "tensor_dummy" / "main.py")
    engine_root = str(PROJECT_ROOT)
    replay_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="batched_")
    n_games = len(opp_weight_paths)

    # Build game args
    game_args = []
    for i in range(n_games):
        map_path = str(PROJECT_ROOT / "maps" / random.choice(maps))
        log_path = os.path.join(tmpdir, f"traj_{i}.pkl")
        replay_path = str(replay_dir / f"ep{episode_num}_{opp_ids[i]}_g{i}.replay26")
        seed = random.randint(1, 2**31)
        bot_b = bot_b_dummy if opp_weight_paths[i] == DUMMY_OPP_PATH else bot_b_proxy
        game_args.append((bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed))

    # Load opponent models on GPU (deduplicate by weight path)
    t_phase = time.perf_counter() if profile is not None else 0.0
    opp_models = {}  # weight_path -> BuilderMLP
    game_to_opp = {}  # game_idx -> weight_path
    for i in range(n_games):
        wp = opp_weight_paths[i]
        game_to_opp[i] = wp
        if wp == DUMMY_OPP_PATH:
            continue
        if wp not in opp_models:
            opp_m = BuilderMLP(hidden=opp_hidden).to(DEVICE)
            opp_m.load_weights(wp)
            opp_m.eval()
            opp_models[wp] = opp_m
    if profile is not None:
        profile["opp_model_load_s"] = time.perf_counter() - t_phase

    t_phase = time.perf_counter() if profile is not None else 0.0
    max_workers = _recommended_worker_count(n_games)
    if profile is not None:
        profile["pool_dispatch_s"] = time.perf_counter() - t_phase
        profile["worker_count"] = float(max_workers)

    selector = selectors.DefaultSelector()
    results = [None] * n_games
    processes = {}
    launch_order = sorted(range(n_games), key=lambda i: (opp_weight_paths[i], i))
    pending_indices = launch_order[:]

    fd_to_info = {}  # fd -> (game_idx, side)  side='a' or 'b'
    act_fds = {}  # (game_idx, side) -> act_write_fd
    active_fds = set()

    def _close_read_fd(fd: int) -> None:
        active_fds.discard(fd)
        fd_to_info.pop(fd, None)
        try:
            selector.unregister(fd)
        except Exception:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def _launch_game(i: int) -> None:
        obs_r_a, obs_w_a = os.pipe()
        act_r_a, act_w_a = os.pipe()
        use_proxy_b = opp_weight_paths[i] != DUMMY_OPP_PATH
        obs_r_b = obs_w_b = act_r_b = act_w_b = -1
        if use_proxy_b:
            obs_r_b, obs_w_b = os.pipe()
            act_r_b, act_w_b = os.pipe()
        result_path = os.path.join(tmpdir, f"result_{i}.pkl")
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = ""
        worker_fds = tuple(fd for fd in (obs_w_a, act_r_a, obs_w_b, act_r_b) if fd >= 0)
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-result", result_path,
                "--worker-pipes", f"{obs_w_a},{act_r_a},{obs_w_b},{act_r_b}",
                "--worker-game-args", json.dumps(game_args[i]),
            ],
            close_fds=True,
            cwd=str(PROJECT_ROOT),
            env=worker_env,
            pass_fds=worker_fds,
        )
        processes[i] = {"proc": proc, "result_path": result_path}
        if WORKER_LAUNCH_STAGGER_S > 0.0:
            time.sleep(WORKER_LAUNCH_STAGGER_S)

        os.close(obs_w_a)
        os.close(act_r_a)
        if use_proxy_b:
            os.close(obs_w_b)
            os.close(act_r_b)

        fd_to_info[obs_r_a] = (i, "a")
        act_fds[(i, "a")] = act_w_a
        active_fds.add(obs_r_a)
        selector.register(obs_r_a, selectors.EVENT_READ)
        if use_proxy_b:
            fd_to_info[obs_r_b] = (i, "b")
            act_fds[(i, "b")] = act_w_b
            active_fds.add(obs_r_b)
            selector.register(obs_r_b, selectors.EVENT_READ)

    def _reap_processes() -> None:
        for game_idx, proc_info in list(processes.items()):
            proc = proc_info["proc"]
            if proc.poll() is None:
                continue
            results[game_idx] = _load_worker_result(proc_info["result_path"])
            processes.pop(game_idx, None)

    while pending_indices and len(processes) < max_workers:
        _launch_game(pending_indices.pop(0))

    # Inference server loop — disable cyclic GC to avoid pauses from
    # the collector scanning the many live pipe buffers and tensor temporaries.
    gc_was_enabled = gc.isenabled()
    gc.disable()

    model.eval()
    total_inferences = 0
    coalesce_s = max(coalesce_ms, 0.0) / 1000.0
    batch_a_calls = batch_a_items = batch_a_max = 0
    batch_b_calls = batch_b_items = batch_b_max = 0

    while active_fds or pending_indices or processes:
        _reap_processes()
        while pending_indices and len(processes) < max_workers:
            _launch_game(pending_indices.pop(0))

        if not active_fds:
            if not processes and not pending_indices:
                break
            time.sleep(0.001)
            continue

        try:
            t_phase = time.perf_counter() if profile is not None else 0.0
            events = selector.select(1.0)
            readable = [key.fd for key, _ in events]
            if profile is not None:
                profile["ipc_wait_s"] = profile.get("ipc_wait_s", 0.0) + (time.perf_counter() - t_phase)
        except OSError:
            break

        if not readable:
            if not processes and not pending_indices:
                break
            continue

        readable = _coalesce_readable(
            selector, readable, coalesce_s, fd_to_info, game_to_opp, profile=profile,
        )

        # Read all available observations, split by side
        pending_a = []  # (game_idx, float_bytes)
        pending_b = []  # (game_idx, float_bytes, opp_weight_path)
        t_phase = time.perf_counter() if profile is not None else 0.0
        for fd in readable:
            game_idx, side = fd_to_info[fd]
            float_bytes = _pipe_read_obs(fd)
            if float_bytes is None:
                _close_read_fd(fd)
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
            batch_a_calls += 1
            batch_a_items += n_a
            batch_a_max = max(batch_a_max, n_a)
            # Concatenate raw bytes → tensor (no Python float objects)
            t_phase = time.perf_counter() if profile is not None else 0.0
            all_bytes = bytearray().join(fb for _, fb in pending_a)
            if profile is not None:
                profile["batch_pack_a_s"] = profile.get("batch_pack_a_s", 0.0) + (time.perf_counter() - t_phase)

            with torch.no_grad():
                t_phase = time.perf_counter() if profile is not None else 0.0
                all_t = torch.frombuffer(all_bytes, dtype=torch.float32).reshape(n_a, _OBS_FLOATS).to(DEVICE)
                obs_t = all_t[:, :OBS_SIZE]
                if profile is not None:
                    dt = time.perf_counter() - t_phase
                    profile["h2d_a_s"] = profile.get("h2d_a_s", 0.0) + dt
                    profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + dt

                t_phase = time.perf_counter() if profile is not None else 0.0
                bt_l, bd_l, md_l, h_next = model.infer_packed(all_t)
                if profile is not None:
                    dt = time.perf_counter() - t_phase
                    profile["forward_a_s"] = profile.get("forward_a_s", 0.0) + dt
                    profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + dt

                t_phase = time.perf_counter() if profile is not None else 0.0
                # Apply action masks before sampling
                bt_m, bd_m = compute_action_masks(obs_t)
                bt_l.masked_fill_(~bt_m, -1e9)
                bd_l.masked_fill_(~bd_m, -1e9)
                bt_act, bt_lp = sample_logits_gumbel(bt_l)
                bd_act, bd_lp = sample_logits_gumbel(bd_l)
                md_act, md_lp = sample_logits_gumbel(md_l)
                # Mask build_dir log_prob when build_type is "none" (idx 0)
                bd_causal = (bt_act != 0).float()
                lp = bt_lp + bd_lp * bd_causal + md_lp
                if profile is not None:
                    dt = time.perf_counter() - t_phase
                    profile["sample_a_s"] = profile.get("sample_a_s", 0.0) + dt
                    profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + dt

                t_phase = time.perf_counter() if profile is not None else 0.0
                # Single bulk transfer to CPU — one sync point
                # h_next is already tanh'd by model.forward()
                results_cpu = torch.stack([
                    bt_act.float(), bd_act.float(), md_act.float(), lp,
                ]).cpu()
                h_cpu = h_next.cpu().contiguous()
                if profile is not None:
                    dt = time.perf_counter() - t_phase
                    profile["d2h_a_s"] = profile.get("d2h_a_s", 0.0) + dt
                    profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + dt

            t_phase = time.perf_counter() if profile is not None else 0.0
            h_blob = bytes(h_cpu.untyped_storage())
            h_view = memoryview(h_blob)
            for i, (game_idx, _) in enumerate(pending_a):
                try:
                    h_start = i * _H_BYTES
                    _pipe_write_action_a(
                        act_fds[(game_idx, "a")],
                        int(results_cpu[0, i]), int(results_cpu[1, i]), int(results_cpu[2, i]),
                        h_view[h_start:h_start + _H_BYTES], float(results_cpu[3, i]),
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
                n_b = len(items)
                batch_b_calls += 1
                batch_b_items += n_b
                batch_b_max = max(batch_b_max, n_b)
                t_phase = time.perf_counter() if profile is not None else 0.0
                all_bytes = bytearray().join(fb for _, fb in items)
                if profile is not None:
                    profile["batch_pack_b_s"] = profile.get("batch_pack_b_s", 0.0) + (time.perf_counter() - t_phase)

                if wp == DUMMY_OPP_PATH:
                    t_phase = time.perf_counter() if profile is not None else 0.0
                    obs_t = torch.frombuffer(all_bytes, dtype=torch.float32).reshape(n_b, _OBS_FLOATS)[:, :OBS_SIZE]
                    bt_act, bd_act, md_act = _dummy_policy(obs_t)
                    if profile is not None:
                        profile["dummy_b_s"] = profile.get("dummy_b_s", 0.0) + (time.perf_counter() - t_phase)

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    for i, (game_idx, _) in enumerate(items):
                        try:
                            _pipe_write_action_b(
                                act_fds[(game_idx, "b")],
                                int(bt_act[i]), int(bd_act[i]), int(md_act[i]),
                                _ZERO_H_BYTES,
                            )
                        except OSError:
                            pass
                    if profile is not None:
                        profile["ipc_write_b_s"] = profile.get("ipc_write_b_s", 0.0) + (time.perf_counter() - t_phase)
                    continue

                opp_m = opp_models[wp]
                with torch.no_grad():
                    t_phase = time.perf_counter() if profile is not None else 0.0
                    all_t = torch.frombuffer(all_bytes, dtype=torch.float32).reshape(n_b, _OBS_FLOATS).to(DEVICE)
                    obs_t = all_t[:, :OBS_SIZE]
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["h2d_b_s"] = profile.get("h2d_b_s", 0.0) + dt
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    bt_l, bd_l, md_l, h_next = opp_m.infer_packed(all_t)
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["forward_b_s"] = profile.get("forward_b_s", 0.0) + dt
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    bt_m, bd_m = compute_action_masks(obs_t)
                    bt_l.masked_fill_(~bt_m, -1e9)
                    bd_l.masked_fill_(~bd_m, -1e9)
                    bt_act, _ = sample_logits_gumbel(bt_l, OPP_TEMP, return_log_prob=False)
                    bd_act, _ = sample_logits_gumbel(bd_l, OPP_TEMP, return_log_prob=False)
                    md_act, _ = sample_logits_gumbel(md_l, OPP_TEMP, return_log_prob=False)
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["sample_b_s"] = profile.get("sample_b_s", 0.0) + dt
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    acts_cpu = torch.stack([bt_act, bd_act, md_act]).cpu()
                    h_cpu = h_next.cpu().contiguous()
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["d2h_b_s"] = profile.get("d2h_b_s", 0.0) + dt
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                t_phase = time.perf_counter() if profile is not None else 0.0
                h_blob = bytes(h_cpu.untyped_storage())
                h_view = memoryview(h_blob)
                for i, (game_idx, _) in enumerate(items):
                    try:
                        h_start = i * _H_BYTES
                        _pipe_write_action_b(
                            act_fds[(game_idx, "b")],
                            int(acts_cpu[0, i]), int(acts_cpu[1, i]), int(acts_cpu[2, i]),
                            h_view[h_start:h_start + _H_BYTES],
                        )
                    except OSError:
                        pass
                if profile is not None:
                    profile["ipc_write_b_s"] = profile.get("ipc_write_b_s", 0.0) + (time.perf_counter() - t_phase)
                total_inferences += n_b

    t_phase = time.perf_counter() if profile is not None else 0.0
    deadline = time.perf_counter() + 30.0
    while processes and time.perf_counter() < deadline:
        _reap_processes()
        if processes:
            time.sleep(0.001)
    _reap_processes()
    for proc_info in processes.values():
        proc = proc_info["proc"]
        if proc.poll() is None:
            proc.terminate()
    kill_deadline = time.perf_counter() + 5.0
    while processes and time.perf_counter() < kill_deadline:
        _reap_processes()
        if processes:
            time.sleep(0.001)
    for proc_info in processes.values():
        proc = proc_info["proc"]
        if proc.poll() is None:
            proc.kill()
    _reap_processes()
    for game_idx in list(processes):
        results[game_idx] = (None, None)
        processes.pop(game_idx, None)
    if profile is not None:
        profile["worker_join_s"] = time.perf_counter() - t_phase

    selector.close()

    for fd in list(active_fds):
        _close_read_fd(fd)
    for fd in act_fds.values():
        try:
            os.close(fd)
        except OSError:
            pass

    if gc_was_enabled:
        gc.enable()
        gc.collect()

    # Read trajectories and augment with replay ground truth
    t_phase = time.perf_counter() if profile is not None else 0.0
    trajectories, winners = [], []
    per_opp_winners = {}
    for i, result in enumerate(results):
        oid = opp_ids[i]
        if isinstance(result, tuple) and len(result) == 3:
            log_path, working, w = result
        elif isinstance(result, tuple) and len(result) == 2:
            log_path, working = result
            w = None
        else:
            log_path, working, w = None, None, None
        if log_path is None:
            per_opp_winners.setdefault(oid, []).append(None)
            continue
        game_trajs = _read_trajectory_file(log_path)
        game_trajs = augment_trajectories_with_working_counts(game_trajs, working, team=0)
        trajectories.extend(game_trajs)
        winners.append(w)
        per_opp_winners.setdefault(oid, []).append(w)
    if profile is not None:
        profile["postprocess_s"] = time.perf_counter() - t_phase
        profile["collection_total_s"] = time.perf_counter() - t_total
        profile["gpu_inferences"] = float(total_inferences)
        profile["games"] = float(n_games)
        profile["batch_a_mean"] = batch_a_items / max(batch_a_calls, 1)
        profile["batch_a_max"] = float(batch_a_max)
        profile["batch_b_mean"] = batch_b_items / max(batch_b_calls, 1)
        profile["batch_b_max"] = float(batch_b_max)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"    [{n_games} games, {total_inferences} GPU inferences]")
    return trajectories, winners, per_opp_winners


def main():
    parser = argparse.ArgumentParser(description="Batched GPU self-play training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--games-per-opp", type=int, default=2)
    parser.add_argument("--dummy-episodes", type=int, default=0, help="Use only scripted dummy opponents for the first N episodes")
    parser.add_argument("--league-size", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--teacher", action="store_true", help="Train large teacher, distill to student")
    parser.add_argument("--distill-epochs", type=int, default=20, help="Distillation epochs")
    parser.add_argument("--distill-lr", type=float, default=1e-3, help="Distillation learning rate")
    parser.add_argument("--coalesce-ms", type=float, default=DEFAULT_COALESCE_MS)
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--worker-result", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pipes", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-game-args", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    session_name = args.session or _default_session_name()
    sess = _init_session(session_name)
    weight_path = str(sess["weight_path"])
    log_path = sess["log_path"]
    replay_dir = sess["replay_dir"]

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
    if args.dummy_episodes > 0:
        print(f"Dummy curriculum: first {args.dummy_episodes} episodes use scripted side-B")
    print(f"Workers: auto")

    t_start = time.time()
    train_log = []

    for ep in range(args.episodes):
        t0 = time.time()
        current_gen = league.latest_gen()
        current_id = f"gen_{current_gen}" if current_gen >= 0 else None
        dummy_phase = ep < args.dummy_episodes
        games_per_opp = args.games_per_opp
        if dummy_phase:
            opponents = [{"id": DUMMY_OPP_ID, "path": DUMMY_OPP_PATH}]
            games_per_opp *= max(args.opponents, 1)
        else:
            opponents = league.sample_opponents(min(args.opponents, len(league.models)))
            if not opponents:
                league.add_model(model.export_weights(), 0)
                opponents = league.sample_opponents(1)

        model.save_weights(weight_path)

        # Build all games across all opponents for parallel execution
        all_opp_paths = []
        all_opp_ids = []
        for opp in opponents:
            for _ in range(games_per_opp):
                all_opp_paths.append(opp["path"])
                all_opp_ids.append(opp["id"])

        all_trajs, all_winners, per_opp_winners = collect_games_batched(
            model, all_opp_paths, all_opp_ids, maps,
            ep + 1, replay_dir,
            opp_hidden=hidden,
            coalesce_ms=args.coalesce_ms,
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
            gen = league.latest_gen() + 1
            ckpt = str(league.league_dir / f"gen_{gen}.pkl")
            model.save_weights(ckpt)
            league.register_model(ckpt, gen)
            current_id = f"gen_{gen}"

        # Update ELO every episode using the latest registered model
        if current_id is not None:
            for opp, winners in per_opp_results:
                if opp["path"] == DUMMY_OPP_PATH:
                    continue
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
                r_work += n_working * 2.5; r_dmg += dmg * 20.0
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
            "elo": round(next((m["elo"] for m in league.models if m["id"] == current_id), ELO_DEFAULT), 1),
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
