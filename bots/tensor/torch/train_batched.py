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
from collections import OrderedDict
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REPLAY_PARSER_PYTHON = PROJECT_ROOT / "packages" / "replay_parser" / "python"
if str(REPLAY_PARSER_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPLAY_PARSER_PYTHON))


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
import torch.nn.functional as F


from model import (
    BuilderMLP,
    CONVEYOR_BUILD_TYPES,
    DEVICE,
    HARVESTER_BUILD_TYPE,
    H_DIM,
    OBS_SIZE,
    REWARD_INCOME_SCALE,
    TILE_FEATURES,
    TEACHER_HIDDEN,
    get_centralized_critic,
    _parse_trajectories,
    conveyor_dir_mask,
    compute_action_masks,
    compute_conditional_build_type_mask,
    configure_centralized_critic_runtime,
    load_centralized_critic_checkpoint,
    reset_centralized_critic,
    save_centralized_critic_checkpoint,
    distill,
    sample_logits_gumbel,
    train_on_trajectories,
    GraphedInference,
    TERMINAL_WIN_REWARD,
    TERMINAL_LOSS_REWARD,
    REWARD_MODE,
    REWARD_FLOW_SCALE,
    RewardMode,
    BRIDGE_BUILD_TYPE,
    N_BRIDGE_TARGETS,
    compute_harvester_build_dir_mask,
    _unimix_logits,
)


SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor_inference" / "weights.json"
WORKER_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

from training_utils import (
    League, _init_session, _default_session_name, _read_trajectory_file,
)

# Struct-based binary IPC protocol constants
_OBS_FLOATS = OBS_SIZE + H_DIM
_OBS_FLOAT_BYTES = _OBS_FLOATS * 4
_ACT_HEAD = struct.Struct('=iiiii')
_H_STRUCT = struct.Struct(f'={H_DIM}f')
_LP_STRUCT = struct.Struct('=f')
_H_BYTES = H_DIM * 4
DEFAULT_COALESCE_MS = 0.25 if DEVICE.type == "cuda" else 0.0

# KL-adaptive actor learning rate
KL_TARGET = 0.01
KL_EMA_ALPHA = 0.1
ACTOR_LR_MIN = 3e-5   # lr / 10
ACTOR_LR_MAX = 3e-3   # lr * 10
import ctypes as _ctypes
WORKER_LAUNCH_STAGGER_S = 0.005
DUMMY_OPP_PATH = "__dummy__"
DUMMY_OPP_ID = "dummy"
_ZERO_H_BYTES = bytes(_H_BYTES)
_DUMMY_DIR_TILES = (6, 7, 8, 11, 12, 13, 16, 17, 18)
_dummy_idx_cache = {}
_OPP_GRAPH_TEMP = 0.5
_OPP_GRAPH_BATCH_BUCKETS = (8, 16, 32, 64, 128, 256)
_OPP_MODEL_CACHE_MAX = 32
_OPP_GRAPH_CACHE_MAX = 64
_opp_model_cache: OrderedDict[tuple[object, ...], BuilderMLP] = OrderedDict()
_opp_graph_cache: OrderedDict[tuple[object, ...], GraphedInference] = OrderedDict()
_opp_graph_failures: set[tuple[object, ...]] = set()


def _cache_put(cache: OrderedDict, key, value, max_items: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _hidden_key(hidden: list[int] | None) -> tuple[int, ...]:
    return tuple(hidden) if hidden is not None else ()


def _weight_sig(weight_path: str) -> tuple[int, int]:
    st = os.stat(weight_path)
    return st.st_mtime_ns, st.st_size


def _graph_batch_bucket(batch_size: int) -> int | None:
    for bucket in _OPP_GRAPH_BATCH_BUCKETS:
        if batch_size <= bucket:
            return bucket
    return None


def _opponent_model_cache_key(weight_path: str, hidden: list[int] | None) -> tuple[object, ...]:
    return (weight_path, _hidden_key(hidden), *_weight_sig(weight_path), DEVICE.type)


def _get_cached_opponent_model(weight_path: str, hidden: list[int] | None) -> BuilderMLP:
    key = _opponent_model_cache_key(weight_path, hidden)
    cached = _opp_model_cache.get(key)
    if cached is not None:
        _opp_model_cache.move_to_end(key)
        cached.eval()
        return cached

    model = BuilderMLP(hidden=hidden).to(DEVICE)
    model.load_weights(weight_path)
    model.eval()
    _cache_put(_opp_model_cache, key, model, _OPP_MODEL_CACHE_MAX)
    return model


def _get_cached_opponent_graph(
    model_key: tuple[object, ...],
    model: BuilderMLP,
    batch_size: int,
    temperature: float,
) -> GraphedInference | None:
    if DEVICE.type != "cuda":
        return None
    bucket = _graph_batch_bucket(batch_size)
    if bucket is None:
        return None

    dtype_key = str(next(model.parameters()).dtype)
    key = (
        *model_key,
        bucket,
        temperature,
        False,
        dtype_key,
    )
    if key in _opp_graph_failures:
        return None
    cached = _opp_graph_cache.get(key)
    if cached is not None:
        _opp_graph_cache.move_to_end(key)
        return cached

    try:
        graph = GraphedInference(
            model,
            max_batch=bucket,
            temperature=temperature,
            return_log_prob=False,
        )
        graph.capture()
    except Exception:
        _opp_graph_failures.add(key)
        return None
    _cache_put(_opp_graph_cache, key, graph, _OPP_GRAPH_CACHE_MAX)
    return graph


def _session_state_path(sess: dict) -> Path:
    return Path(sess["dir"]) / "train_state.pt"


def _atomic_torch_save(payload: object, path: str | Path) -> None:
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _read_last_logged_state(log_path: str | Path) -> tuple[int, float]:
    path = Path(log_path)
    if not path.exists():
        return 0, 0.0
    last_ep = 0
    last_elapsed = 0.0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict) or "episode" not in entry:
                    continue
                ep = entry.get("episode", 0)
                if isinstance(ep, int) and ep > last_ep:
                    last_ep = ep
                    elapsed = entry.get("elapsed", 0.0)
                    if isinstance(elapsed, (int, float)):
                        last_elapsed = float(elapsed)
    except OSError:
        return 0, 0.0
    return last_ep, last_elapsed


def _load_resume_state(
    state_path: str | Path,
    *,
    optimizer: torch.optim.Optimizer,
) -> dict | None:
    path = Path(state_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location=DEVICE)
    if not isinstance(payload, dict):
        return None
    opt_state = payload.get("optimizer")
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    return payload


def _load_critic_init_weights(weight_path: str | Path, *, lr: float) -> bool:
    """Load PPO critic init weights from a plain state_dict or sidecar checkpoint."""
    path = Path(weight_path)
    if not path.exists():
        return False

    try:
        payload = torch.load(path, map_location=DEVICE)
        if isinstance(payload, dict) and "critic" in payload:
            critic_state = payload["critic"]
            opt_state = payload.get("optimizer")
        else:
            critic_state = payload
            opt_state = None

        if not isinstance(critic_state, dict):
            raise ValueError("critic weights must be a state_dict-compatible mapping")

        reset_centralized_critic()
        critic, critic_opt = get_centralized_critic(lr=lr)
        critic.load_state_dict(critic_state)
        if opt_state is not None:
            critic_opt.load_state_dict(opt_state)
        return True
    except Exception as e:
        print(f"Could not load critic init weights from {path}: {e}")
        reset_centralized_critic()
        return False


def _load_resume_critic_checkpoint(weight_path: str | Path, *, lr: float) -> bool:
    """Load the session-sidecar critic checkpoint for PPO resume."""
    return load_centralized_critic_checkpoint(weight_path, lr=lr)


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


def _pipe_write_action_a(fd, bt, bd, cd, brt, md, h_next_bytes, old_lp):
    """Write fixed-size action packet for training bot (A)."""
    os.writev(fd, (_ACT_HEAD.pack(bt, bd, cd, brt, md), h_next_bytes, _LP_STRUCT.pack(old_lp)))


def _pipe_write_action_b(fd, bt, bd, cd, brt, md, h_next_bytes):
    """Write fixed-size action packet for opponent bot (B)."""
    os.writev(fd, (_ACT_HEAD.pack(bt, bd, cd, brt, md), h_next_bytes))


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


def _dummy_policy(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    build_dir = torch.full((obs.shape[0],), 4, dtype=torch.int64, device=obs.device)
    has_ore = has_ore_dir.any(dim=1)
    build_dir[has_ore] = ore_dir[has_ore]
    build_dir[~has_ore] = fallback_dir[~has_ore]

    bt_mask = compute_conditional_build_type_mask(obs, build_dir, bt_mask)
    build_type = torch.zeros(obs.shape[0], dtype=torch.int64, device=obs.device)
    can_harv = bt_mask[:, 1] & has_ore
    can_conv = bt_mask[:, 2]
    can_road = bt_mask[:, 3]
    build_type[can_harv] = 1
    build_type[~can_harv & can_conv] = 2
    build_type[~can_harv & ~can_conv & can_road] = 3

    conv_dir = torch.zeros_like(build_type)  # 0 = N (first cardinal)
    move_dir = torch.full_like(build_type, 4)
    return build_type, build_dir, conv_dir, move_dir

def _load_worker_result(
    result_path: str,
) -> tuple[str | None, dict | None, str | None, dict | None, dict | None, dict | None]:
    try:
        with open(result_path, "rb") as f:
            result = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError):
        return (None, None, None, None, None, None)
    if isinstance(result, tuple):
        if len(result) == 6:
            return result
        if len(result) == 5:
            return result[0], result[1], result[2], result[3], result[4], None
        if len(result) == 4:
            return result[0], result[1], result[2], result[3], None, None
        if len(result) == 3:
            return result[0], result[1], result[2], None, None, None
        if len(result) == 2:
            return result[0], result[1], None, None, None, None
    return (None, None, None, None, None, None)


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
) -> tuple[list[list], list[str | None], dict[str, list[str | None]], list[dict | None], list[int]]:
    """Run all games with batched GPU inference via pipes. Returns (trajs, winners, per_opp_winners, global_states, traj_game_map)."""
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
    opp_model_keys = {}  # weight_path -> cache key
    game_to_opp = {}  # game_idx -> weight_path
    for i in range(n_games):
        wp = opp_weight_paths[i]
        game_to_opp[i] = wp
        if wp == DUMMY_OPP_PATH:
            continue
        if wp not in opp_models:
            opp_model_keys[wp] = _opponent_model_cache_key(wp, opp_hidden)
            opp_models[wp] = _get_cached_opponent_model(wp, opp_hidden)
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
        worker_python = WORKER_PYTHON if os.path.exists(WORKER_PYTHON) else sys.executable
        proc = subprocess.Popen(
            [
                worker_python,
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

    # Inference server loop — disable cyclic GC to avoid pauses from
    # the collector scanning the many live pipe buffers and tensor temporaries.
    gc_was_enabled = gc.isenabled()
    gc.disable()

    model.eval()
    _max_a_batch = n_games * 4
    graphed_a = GraphedInference(model, max_batch=_max_a_batch)
    graphed_a.capture()
    total_inferences = 0

    while pending_indices and len(processes) < max_workers:
        _launch_game(pending_indices.pop(0))
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
                if profile is not None:
                    dt = time.perf_counter() - t_phase
                    profile["h2d_a_s"] = profile.get("h2d_a_s", 0.0) + dt
                    profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + dt

                t_phase = time.perf_counter() if profile is not None else 0.0
                bt_act, bd_act, cd_act, brt_act, md_act, lp, h_next = graphed_a(all_t)
                results_cpu = torch.stack([
                    bt_act.float(), bd_act.float(), cd_act.float(), brt_act.float(), md_act.float(), lp,
                ]).cpu()
                h_cpu = h_next.cpu().contiguous()
                if profile is not None:
                    dt = time.perf_counter() - t_phase
                    profile["d2h_a_s"] = profile.get("d2h_a_s", 0.0) + dt
                    profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + dt

            t_phase = time.perf_counter() if profile is not None else 0.0
            h_view = memoryview((_ctypes.c_char * (n_a * _H_BYTES)).from_address(h_cpu.data_ptr()))
            for i, (game_idx, _) in enumerate(pending_a):
                try:
                    h_start = i * _H_BYTES
                    _pipe_write_action_a(
                        act_fds[(game_idx, "a")],
                        int(results_cpu[0, i]), int(results_cpu[1, i]), int(results_cpu[2, i]),
                        int(results_cpu[3, i]), int(results_cpu[4, i]), h_view[h_start:h_start + _H_BYTES], float(results_cpu[5, i]),
                    )
                except OSError:
                    pass
            if profile is not None:
                profile["ipc_write_a_s"] = profile.get("ipc_write_a_s", 0.0) + (time.perf_counter() - t_phase)
            total_inferences += n_a

        # Batch B inference grouped by opponent model (low-temp sampling for diversity)
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
                    bt_act, bd_act, cd_act, md_act = _dummy_policy(obs_t)
                    brt_act = torch.zeros_like(bt_act)
                    if profile is not None:
                        profile["dummy_b_s"] = profile.get("dummy_b_s", 0.0) + (time.perf_counter() - t_phase)

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    for i, (game_idx, _) in enumerate(items):
                        try:
                            _pipe_write_action_b(
                                act_fds[(game_idx, "b")],
                                int(bt_act[i]), int(bd_act[i]), int(cd_act[i]), int(brt_act[i]), int(md_act[i]),
                                _ZERO_H_BYTES,
                            )
                        except OSError:
                            pass
                    if profile is not None:
                        profile["ipc_write_b_s"] = profile.get("ipc_write_b_s", 0.0) + (time.perf_counter() - t_phase)
                    continue

                opp_m = opp_models[wp]
                opp_g = _get_cached_opponent_graph(opp_model_keys[wp], opp_m, n_b, _OPP_GRAPH_TEMP)
                with torch.no_grad():
                    t_phase = time.perf_counter() if profile is not None else 0.0
                    all_t = torch.frombuffer(all_bytes, dtype=torch.float32).reshape(n_b, _OBS_FLOATS).to(DEVICE)
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["h2d_b_s"] = profile.get("h2d_b_s", 0.0) + dt
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    if opp_g is not None and n_b <= opp_g.max_batch:
                        # CUDA-graphed path: forward + mask + sample in one graph replay
                        bt_act, bd_act, cd_act, brt_act, md_act, _, h_next = opp_g(all_t)
                        acts_cpu = torch.stack([bt_act, bd_act, cd_act, brt_act, md_act]).cpu()
                        h_cpu = h_next.cpu().contiguous()
                    else:
                        # Fallback: ungraphed inference
                        obs_t = all_t[:, :OBS_SIZE]
                        bt_l, bd_l, cd_l, _brt_l, md_l, _eps, h_next = opp_m.infer_packed(all_t)
                        if profile is not None:
                            dt = time.perf_counter() - t_phase
                            profile["forward_b_s"] = profile.get("forward_b_s", 0.0) + dt
                            profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                        t_phase = time.perf_counter() if profile is not None else 0.0
                        bt_m, bd_m = compute_action_masks(obs_t)
                        cd_m = conveyor_dir_mask(obs_t.device, obs_t.shape[0])
                        bt_l.masked_fill_(~bt_m, -1e9)
                        cd_l.masked_fill_(~cd_m, -1e9)
                        bt_act, _ = sample_logits_gumbel(bt_l, _OPP_GRAPH_TEMP, mask=bt_m, return_log_prob=False)
                        bd_l.masked_fill_(~bd_m, -1e9)
                        bd_act, _ = sample_logits_gumbel(bd_l, _OPP_GRAPH_TEMP, mask=bd_m, return_log_prob=False)
                        cd_act, _ = sample_logits_gumbel(cd_l, _OPP_GRAPH_TEMP, mask=cd_m, return_log_prob=False)
                        brt_act, _ = sample_logits_gumbel(_brt_l, _OPP_GRAPH_TEMP, return_log_prob=False)
                        md_act, _ = sample_logits_gumbel(md_l, _OPP_GRAPH_TEMP, return_log_prob=False)
                        acts_cpu = torch.stack([bt_act, bd_act, cd_act, brt_act, md_act]).cpu()
                        h_cpu = h_next.cpu().contiguous()
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                t_phase = time.perf_counter() if profile is not None else 0.0
                h_blob = bytes(h_cpu.untyped_storage())
                h_view = memoryview(h_blob)
                for i, (game_idx, _) in enumerate(items):
                    try:
                        h_start = i * _H_BYTES
                        _pipe_write_action_b(
                            act_fds[(game_idx, "b")],
                            int(acts_cpu[0, i]), int(acts_cpu[1, i]), int(acts_cpu[2, i]), int(acts_cpu[3, i]), int(acts_cpu[4, i]),
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
        results[game_idx] = (None, None, None, None)
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
    all_global_states = []
    traj_game_map = []  # maps each trajectory index -> game index (for global states)
    game_counter = 0
    game_origin = []  # game_counter -> original game index i
    per_opp_winners = {}
    for i, result in enumerate(results):
        oid = opp_ids[i]
        if isinstance(result, tuple) and len(result) == 6:
            log_path, working, w, gstates = result[:4]
        elif isinstance(result, tuple) and len(result) >= 4:
            log_path, working, w, gstates = result[:4]
        elif isinstance(result, tuple) and len(result) == 3:
            log_path, working, w = result
            gstates = None
        elif isinstance(result, tuple) and len(result) == 2:
            log_path, working = result
            w, gstates = None, None
        else:
            log_path, working, w, gstates = None, None, None, None
        if log_path is None:
            per_opp_winners.setdefault(oid, []).append(None)
            continue
        game_trajs = _read_trajectory_file(log_path)
        # Inject replay-derived resource reward as per-step reward (training only)
        if gstates and REWARD_MODE in (RewardMode.INCOME, RewardMode.FLOW):
            from flow_reward import compute_dense_delivery_rewards, compute_dense_flow_rewards, augment_trajectories_with_flow_rewards
            if REWARD_MODE == RewardMode.INCOME:
                rewards = compute_dense_delivery_rewards(gstates)
            else:
                rewards = compute_dense_flow_rewards(gstates)
            game_trajs = augment_trajectories_with_flow_rewards(game_trajs, rewards)
        n_before = len(trajectories)
        trajectories.extend(game_trajs)
        # Map each new trajectory to this game's global state index
        for _ in range(len(game_trajs)):
            traj_game_map.append(game_counter)
        all_global_states.append(gstates)
        game_origin.append(i)
        game_counter += 1
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

    # Read path-based buffers into memory before deleting tmpdir
    for gs in all_global_states:
        if gs is None:
            continue
        for key, buf_key in [("states_path", "states_buf"), ("scalars_path", "scalars_buf"),
                             ("transfer_events_path", "transfer_events_buf"), ("ti_delivered_path", "ti_delivered_buf"),
                             ("ax_delivered_path", "ax_delivered_buf")]:
            if gs.get(buf_key) is None and gs.get(key):
                try:
                    gs[buf_key] = Path(gs[key]).read_bytes()
                except OSError:
                    pass

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"    [{n_games} games, {total_inferences} GPU inferences]")
    return trajectories, winners, per_opp_winners, all_global_states, traj_game_map, game_origin


def _step_round_index(step) -> int | None:
    if not isinstance(step, (list, tuple)) or len(step) < 1:
        return None
    obs = step[0]
    if isinstance(obs, (bytes, bytearray, memoryview)):
        import struct as _struct
        _OBS_ROUND_BYTE_OFFSET = 475 * 4
        if len(obs) <= _OBS_ROUND_BYTE_OFFSET + 3:
            return None
        rnd_val = _struct.unpack_from('<f', obs, _OBS_ROUND_BYTE_OFFSET)[0]
    elif isinstance(obs, (list, tuple)):
        if len(obs) <= 475:
            return None
        rnd_val = obs[475]
    else:
        return None
    return max(0, min(int(round(rnd_val * 2000.0)), 1999))


def _step_build_reward(step) -> float:
    _credit = REWARD_FLOW_SCALE if REWARD_MODE == RewardMode.FLOW else REWARD_INCOME_SCALE
    if isinstance(step, (list, tuple)):
        n = len(step)
        if n > 12:
            return float(step[7] if n > 7 else 0.0) * _credit
        return float(step[6] if n > 6 else 0.0) * _credit
    return float(step.get("n_working", 0.0)) * _credit


def _shared_flow_reward_total(game_trajs: list[list]) -> float:
    round_rewards: dict[int, float] = {}
    for traj in game_trajs:
        for step in traj:
            rnd = _step_round_index(step)
            if rnd is None:
                continue
            reward = _step_build_reward(step)
            if reward > round_rewards.get(rnd, 0.0):
                round_rewards[rnd] = reward
    return sum(round_rewards.values())


def main():
    parser = argparse.ArgumentParser(description="Batched GPU self-play training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponents", type=int, default=3)
    parser.add_argument("--games-per-opp", type=int, default=2)
    parser.add_argument("--dummy-episodes", type=int, default=0, help="Use only scripted dummy opponents for the first N episodes")
    parser.add_argument("--league-size", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=0, help="Critic learning rate (default: same as --lr)")
    parser.add_argument(
        "--critic-minibatch-size",
        type=int,
        default=0,
        help="Critic PPO minibatch size (default: auto, larger than actor minibatches on CUDA)",
    )
    parser.add_argument("--teacher", action="store_true", help="Train large teacher, distill to student")
    parser.add_argument("--distill-epochs", type=int, default=20, help="Distillation epochs")
    parser.add_argument("--distill-lr", type=float, default=1e-3, help="Distillation learning rate")
    parser.add_argument("--coalesce-ms", type=float, default=DEFAULT_COALESCE_MS)
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--init-weights", type=str, default="", help="Initialize actor from a checkpoint")
    parser.add_argument(
        "--critic-init-weights",
        type=str,
        default="",
        help="Initialize critic from a plain state_dict or sidecar checkpoint",
    )
    parser.add_argument("--critic-no-compile", action="store_true", help="Disable torch.compile for PPO critic updates")
    parser.add_argument("--freeze-critic", action="store_true", help="Skip critic updates during PPO (actor-only training)")
    parser.add_argument("--resume", action="store_true", help="Resume a prior session from saved trainer state")
    parser.add_argument("--worker-result", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pipes", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-game-args", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    session_name = args.session or _default_session_name()
    session_dir = SESSIONS_DIR / session_name
    if session_dir.exists() and not args.resume:
        raise SystemExit(
            f"Session '{session_name}' already exists. Use --resume or choose a new --session."
        )
    sess = _init_session(session_name)
    weight_path = str(sess["weight_path"])
    log_path = sess["log_path"]
    replay_dir = sess["replay_dir"]
    state_path = _session_state_path(sess)
    game_log_path = Path(str(log_path).replace("train_log.jsonl", "game_log.jsonl"))

    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]
    league = League(sess["league_dir"], args.league_size)

    hidden = TEACHER_HIDDEN if args.teacher else None
    model = BuilderMLP(hidden=hidden).to(DEVICE)
    configure_centralized_critic_runtime(enable_compile=not args.critic_no_compile)
    reset_centralized_critic()
    _critic_lr = args.critic_lr if args.critic_lr > 0 else args.lr
    if args.resume:
        model.load_weights(weight_path)
        _load_resume_critic_checkpoint(weight_path, lr=_critic_lr)
    elif args.init_weights:
        if not model.load_weights(args.init_weights):
            raise SystemExit(f"Could not load actor weights from {args.init_weights}")
        print(f"Initialized actor from {args.init_weights}")
    _deferred_critic_path = args.critic_init_weights or None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _kl_ema = KL_TARGET

    start_episode = 0
    elapsed_offset = 0.0
    resumed_exact = False
    if args.resume:
        resume_state = _load_resume_state(
            state_path,
            optimizer=optimizer,
        )
        if resume_state is not None:
            start_episode = int(resume_state.get("episode", 0))
            elapsed_offset = float(resume_state.get("elapsed_offset_s", 0.0))
            _kl_ema = float(resume_state.get("kl_ema", KL_TARGET))
            resumed_exact = True
        else:
            start_episode, elapsed_offset = _read_last_logged_state(log_path)

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
    if args.resume:
        if resumed_exact:
            print(f"Resume: continuing from episode {start_episode + 1}")
        elif start_episode > 0:
            print(f"Resume: continuing from episode {start_episode + 1} (optimizer reset)")
        else:
            print("Resume: no prior trainer state found, starting fresh")

    if start_episode >= args.episodes:
        print(f"Nothing to do: session already reached episode {start_episode}")
        return

    run_started_at = time.time()
    train_log = []

    _log_config = {
        "envs": args.opponents * args.games_per_opp,
        "ep_len": "self-play", "grid": "real", "lr": args.lr,
        "total_episodes": args.episodes, "mode": "batched-gpu",
        "session": session_name,
    }
    if not args.resume:
        # A non-resume run reusing a session name should start with a clean
        # per-run dashboard state instead of inheriting stale game rows/replays.
        for old_replay in replay_dir.glob("*.replay26"):
            try:
                old_replay.unlink()
            except OSError:
                pass
        try:
            game_log_path.unlink()
        except FileNotFoundError:
            pass

    if not args.resume or not Path(log_path).exists():
        with open(str(log_path), "w") as f:
            f.write(json.dumps({"config": _log_config}) + "\n")

    for ep in range(start_episode, args.episodes):
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

        all_trajs, all_winners, per_opp_winners, all_global_states, traj_game_map, game_origin = collect_games_batched(
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
        if _deferred_critic_path is not None:
            if not _load_critic_init_weights(_deferred_critic_path, lr=_critic_lr):
                raise SystemExit(f"Could not load critic from {_deferred_critic_path}")
            print(f"Loaded critic init weights from {_deferred_critic_path}")
            _deferred_critic_path = None
        model.train()
        stats = train_on_trajectories(
            model, optimizer, all_trajs,
            global_states_list=all_global_states,
            traj_game_map=traj_game_map if all_global_states else None,
            game_winners=all_winners,
            target_kl=0.03,
            critic_minibatch_size=args.critic_minibatch_size or None,
            freeze_critic=args.freeze_critic,
        )
        model.save_weights(weight_path)
        save_centralized_critic_checkpoint(weight_path)
        # KL-adaptive actor LR
        ep_kl = stats.get("approx_kl", KL_TARGET)
        _kl_ema = KL_EMA_ALPHA * ep_kl + (1 - KL_EMA_ALPHA) * _kl_ema
        if _kl_ema > 0:
            lr_scale = KL_TARGET / _kl_ema
            new_lr = max(ACTOR_LR_MIN, min(ACTOR_LR_MAX, optimizer.param_groups[0]["lr"] * lr_scale))
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
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
            r_work, r_dmg, r_own_dmg = 0.0, 0.0, 0.0
            harv_attempts = harv_failures = harv_prox_gate = harv_prox_paid = 0
            conv_attempts = conv_failures = conv_prox_gate = conv_prox_paid = 0
            for s in traj:
                if isinstance(s, (list, tuple)):
                    n = len(s)
                    if n > 12:
                        n_working = s[7] if n > 7 else 0
                        dmg = s[11] if n > 11 else 0.0
                        harv_attempt = s[13] if n > 13 else 0
                        harv_fail = s[14] if n > 14 else 0
                        harv_gate = s[15] if n > 15 else 0
                        harv_paid = s[16] if n > 16 else 0
                        conv_attempt = s[17] if n > 17 else 0
                        conv_fail = s[18] if n > 18 else 0
                        conv_gate = s[19] if n > 19 else 0
                        conv_paid = s[20] if n > 20 else 0
                        own_dmg = s[21] if n > 21 else 0.0
                    else:
                        n_working = s[6] if n > 6 else 0
                        dmg = s[10] if n > 10 else 0.0
                        harv_attempt = harv_fail = harv_gate = harv_paid = 0
                        conv_attempt = conv_fail = conv_gate = conv_paid = 0
                        own_dmg = 0.0
                else:
                    n_working = s.get("n_working", 0)
                    dmg = s.get("dmg", 0)
                    own_dmg = s.get("own_dmg", 0)
                    harv_attempt = s.get("harv_attempt", 0)
                    harv_fail = s.get("harv_fail", 0)
                    harv_gate = s.get("harv_gate", 0)
                    harv_paid = s.get("harv_paid", 0)
                    conv_attempt = s.get("conv_attempt", 0)
                    conv_fail = s.get("conv_fail", 0)
                    conv_gate = s.get("conv_gate", 0)
                    conv_paid = s.get("conv_paid", 0)
                _credit = REWARD_FLOW_SCALE if REWARD_MODE == RewardMode.FLOW else REWARD_INCOME_SCALE
                r_work += n_working * _credit
                harv_attempts += harv_attempt
                harv_failures += harv_fail
                harv_prox_gate += harv_gate
                harv_prox_paid += harv_paid
                conv_attempts += conv_attempt
                conv_failures += conv_fail
                conv_prox_gate += conv_gate
                conv_prox_paid += conv_paid
            total = r_work
            return (
                total, r_work, 0.0, 0.0,
                harv_attempts, harv_failures, harv_prox_gate, harv_prox_paid,
                conv_attempts, conv_failures, conv_prox_gate, conv_prox_paid,
            )

        def _traj_ti(traj):
            last = traj[-1]
            if isinstance(last, (list, tuple)):
                return last[9] if len(last) > 12 else last[8]
            return last.get("ti", 0)

        n_trajs = max(len(all_trajs), 1)
        rewards = []
        for traj_i, traj in enumerate(all_trajs):
            rewards.append(_traj_reward(traj))
        game_trajs: dict[int, list[list]] = {}
        for traj_i, traj in enumerate(all_trajs):
            gi = traj_game_map[traj_i] if traj_i < len(traj_game_map) else 0
            game_trajs.setdefault(gi, []).append(traj)
        game_shared_work = {
            gi: _shared_flow_reward_total(trajs)
            for gi, trajs in game_trajs.items()
        }
        n_games_logged = max(len(game_shared_work), 1)
        mean_r_work = sum(game_shared_work.values()) / n_games_logged
        mean_r_dmg = sum(r[2] for r in rewards) / n_trajs
        mean_r_own_dmg = sum(r[3] for r in rewards) / n_trajs
        mean_harv_attempts = sum(r[4] for r in rewards) / n_trajs
        mean_harv_failures = sum(r[5] for r in rewards) / n_trajs
        mean_harv_prox_gate = sum(r[6] for r in rewards) / n_trajs
        mean_harv_prox_paid = sum(r[7] for r in rewards) / n_trajs
        mean_conv_attempts = sum(r[8] for r in rewards) / n_trajs
        mean_conv_failures = sum(r[9] for r in rewards) / n_trajs
        mean_conv_prox_gate = sum(r[10] for r in rewards) / n_trajs
        mean_conv_prox_paid = sum(r[11] for r in rewards) / n_trajs
        r_outcome = (total_wins * TERMINAL_WIN_REWARD + total_losses * TERMINAL_LOSS_REWARD) / max(total_games, 1) if REWARD_MODE == RewardMode.OUTCOME else 0.0
        mean_reward = (mean_r_work + r_outcome)
        mean_ti = sum(_traj_ti(t) for t in all_trajs if t) / n_trajs
        elapsed = elapsed_offset + (time.time() - run_started_at)

        ep_entry = {
            "episode": ep + 1, "reward": round(mean_reward, 4),
            "r_work": round(mean_r_work, 4), "r_dmg": round(mean_r_dmg, 4),
            "r_own_dmg": round(mean_r_own_dmg, 4),
            "harv_attempts": round(mean_harv_attempts, 3),
            "harv_failures": round(mean_harv_failures, 3),
            "harv_prox_gate": round(mean_harv_prox_gate, 3),
            "harv_prox_paid": round(mean_harv_prox_paid, 3),
            "conv_attempts": round(mean_conv_attempts, 3),
            "conv_failures": round(mean_conv_failures, 3),
            "conv_prox_gate": round(mean_conv_prox_gate, 3),
            "conv_prox_paid": round(mean_conv_prox_paid, 3),
            "resources": round(mean_ti, 2),
            "pi_loss": round(stats["pi_loss"], 5), "vf_loss": round(stats["vf_loss"], 5),
            "entropy": round(stats["entropy"], 4),
            "approx_kl": round(stats.get("approx_kl", 0), 6),
            "actor_lr": optimizer.param_groups[0]["lr"],
            "clip_frac": round(stats.get("clip_frac", 0), 4),
            "explained_var": stats.get("explained_var", 0),
            "actor_grad_norm": round(stats.get("actor_grad_norm", 0), 4),
            "critic_grad_norm": round(stats.get("critic_grad_norm", 0), 4),
            "critic_value_delta": round(stats.get("critic_value_delta", 0), 4),
            "ret_min": stats.get("ret_min", 0),
            "ret_max": stats.get("ret_max", 0),
            "ret_mean": stats.get("ret_mean", 0),
            "ret_oob_frac": stats.get("ret_oob_frac", 0),
            "ls_bt_mean": round(stats.get("ls_bt_mean", 0), 4),
            "ls_bd_mean": round(stats.get("ls_bd_mean", 0), 4),
            "ls_cd_mean": round(stats.get("ls_cd_mean", 0), 4),
            "ls_md_mean": round(stats.get("ls_md_mean", 0), 4),
            "ls_bt_std": round(stats.get("ls_bt_std", 0), 4),
            "ls_bd_std": round(stats.get("ls_bd_std", 0), 4),
            "ls_cd_std": round(stats.get("ls_cd_std", 0), 4),
            "ls_md_std": round(stats.get("ls_md_std", 0), 4),
            "reward_raw_std": round(stats.get("reward_raw_std", 0), 4),
            "sps": stats["n_steps"], "elapsed": round(elapsed, 1),
            "winrate": round(total_wins / max(total_games, 1), 3),
        }
        train_log.append(ep_entry)

        # Append single line — O(1) per episode instead of O(n) rewrite
        with open(str(log_path), "a") as f:
            f.write(json.dumps(ep_entry) + "\n")

        # Per-game log: aggregate trajectories by game index
        _N_REWARD_FIELDS = 12  # matches _traj_reward tuple length
        game_rewards = {}  # game_idx -> [fields..., n_units]
        for traj_i, r in enumerate(rewards):
            gi = traj_game_map[traj_i] if traj_i < len(traj_game_map) else 0
            if gi not in game_rewards:
                game_rewards[gi] = [0.0] * _N_REWARD_FIELDS
                game_rewards[gi].append(0)  # n_units
            for k in range(_N_REWARD_FIELDS):
                game_rewards[gi][k] += r[k]
            game_rewards[gi][_N_REWARD_FIELDS] += 1
        with open(game_log_path, "a") as gf:
            for gi in sorted(game_rewards):
                gr = game_rewards[gi]
                w = all_winners[gi] if gi < len(all_winners) else None
                orig_i = game_origin[gi] if gi < len(game_origin) else gi
                oid = all_opp_ids[orig_i] if orig_i < len(all_opp_ids) else "?"
                ti = 0
                # Sum resources from all trajs in this game
                for traj_i, traj in enumerate(all_trajs):
                    if (traj_game_map[traj_i] if traj_i < len(traj_game_map) else 0) == gi and traj:
                        ti += _traj_ti(traj)
                nu = max(gr[_N_REWARD_FIELDS], 1)
                gf.write(json.dumps({
                    "episode": ep + 1, "game": orig_i,
                    "opp": oid, "winner": w,
                    "reward": round(game_shared_work.get(gi, 0.0), 3), "units": gr[_N_REWARD_FIELDS],
                    "harv_attempts": round(gr[4] / nu, 3),
                    "harv_failures": round(gr[5] / nu, 3),
                    "harv_prox_gate": round(gr[6] / nu, 3),
                    "harv_prox_paid": round(gr[7] / nu, 3),
                    "conv_attempts": round(gr[8] / nu, 3),
                    "conv_failures": round(gr[9] / nu, 3),
                    "conv_prox_gate": round(gr[10] / nu, 3),
                    "conv_prox_paid": round(gr[11] / nu, 3),
                    "resources": round(ti, 1),
                    "replay": f"ep{ep + 1}_{oid}_g{orig_i}.replay26",
                }) + "\n")

        _atomic_torch_save({
            "episode": ep + 1,
            "elapsed_offset_s": elapsed,
            "optimizer": optimizer.state_dict(),
            "kl_ema": _kl_ema,
        }, state_path)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Ep {ep + 1:3d}/{args.episodes}: "
                f"W/L={total_wins}/{total_losses}/{total_games}  "
                f"reward/unit={mean_reward:+.2f}  flow={mean_r_work:+.2f}  ti={mean_ti:.0f}  "
                f"harv a/f/g/p={mean_harv_attempts:.2f}/{mean_harv_failures:.2f}/{mean_harv_prox_gate:.2f}/{mean_harv_prox_paid:.2f}  "
                f"pi={stats['pi_loss']:+.4f}  "
                f"kl={stats.get('approx_kl', 0):.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.1e}  "
                f"ev={stats.get('explained_var', 0):.2f}  "
                f"collect={t_collect:.1f}s train={t_train:.1f}s  [{elapsed:.0f}s]"
            )

    model.save_weights(weight_path)
    save_centralized_critic_checkpoint(weight_path)
    league.save()

    if not args.teacher:
        model.save_weights(str(WEIGHT_PATH))

    print(f"\nDone in {elapsed_offset + (time.time() - run_started_at):.0f}s")


if __name__ == "__main__":
    main()
