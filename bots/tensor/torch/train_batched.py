#!/usr/bin/env python3
"""Batched GPU self-play training via shared-memory ring-buffer collectors.

Architecture:
- Main process: GPU model + inference server
- Game workers (spawn): Run engine with proxy bot, no torch needed
- Proxy bot: enqueues obs into shared-memory rings and waits on tiny notify pipes
- Server: drains ready rings via select(), batches on GPU, responds

Usage:
    python bots/tensor/torch/train_batched.py --episodes 50 --games 8
"""

from __future__ import annotations

import argparse
import ast  # noqa: F401 — engine imports can still expect ast in sys.modules
from collections import OrderedDict
from dataclasses import asdict, dataclass
import gc
import json
import math
import mmap
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPLAY_PARSER_PYTHON = PROJECT_ROOT / "packages" / "replay_parser" / "python"
if str(REPLAY_PARSER_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPLAY_PARSER_PYTHON))


def _maybe_run_worker_mode() -> None:
    if "--worker-result" not in sys.argv:
        return
    from game_worker import main as _game_worker_main

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-result", required=True)
    parser.add_argument("--worker-collector-fds", required=True)
    parser.add_argument("--worker-game-args", required=True)
    args, _ = parser.parse_known_args()
    _game_worker_main([
        "--result", args.worker_result,
        "--collector-fds", args.worker_collector_fds,
        "--game-args", args.worker_game_args,
    ])
    raise SystemExit(0)


_maybe_run_worker_mode()

import torch
import torch.nn.functional as F

try:
    from normuon import make_normuon_optimizer
    from hidden_state_bank import HiddenStateBank
    from replay_buffers import (
        AX_DELIVERED_BUFFER,
        SCALAR_BUFFER,
        STATE_BUFFER,
        TI_DELIVERED_BUFFER,
        TRANSFER_EVENTS_BUFFER,
        load_replay_buffer,
        preload_replay_buffers,
    )
    from shared_collector import (
        COLLECTOR_RING_SLOTS_DEFAULT,
        CollectorServerSide,
        OBS_BYTES,
        collector_bytes,
        initialize_collector,
    )
    from training_profile import (
        EpisodeProfilerConfig,
        parse_profile_episode_spec,
        trace_scope,
    )
    from trajectory_schema import (
        TrajectoryStep,
        WorkerGameResult,
        make_trajectory_step,
        validate_worker_game_result,
    )
except ImportError:
    from bots.tensor.torch.normuon import make_normuon_optimizer
    from bots.tensor.torch.hidden_state_bank import HiddenStateBank
    from bots.tensor.torch.replay_buffers import (
        AX_DELIVERED_BUFFER,
        SCALAR_BUFFER,
        STATE_BUFFER,
        TI_DELIVERED_BUFFER,
        TRANSFER_EVENTS_BUFFER,
        load_replay_buffer,
        preload_replay_buffers,
    )
    from bots.tensor.shared_collector import (
        COLLECTOR_RING_SLOTS_DEFAULT,
        CollectorServerSide,
        OBS_BYTES,
        collector_bytes,
        initialize_collector,
    )
    from bots.tensor.torch.training_profile import (
        EpisodeProfilerConfig,
        parse_profile_episode_spec,
        trace_scope,
    )
    from bots.tensor.trajectory_schema import (
        TrajectoryStep,
        WorkerGameResult,
        make_trajectory_step,
        validate_worker_game_result,
    )


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
    N_BUILD,
    N_CONV_DIR,
    N_BRIDGE_TARGETS,
    N_DIR,
    N_ACTION_TEMPERATURES,
    bd_has_ore,
    _sample_actions_from_policy_heads,
    _OBS_GLOBAL_START,
)


SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor_inference" / "weights.json"
WORKER_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")

from training_utils import League, _init_session, _default_session_name, jsonl_append, jsonl_init

# Shared collector protocol constants
_OBS_FLOATS = OBS_SIZE + H_DIM
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
TRAIN_MAX_TURNS = 300
_DUMMY_DIR_TILES = (6, 7, 8, 11, 12, 13, 16, 17, 18)
_dummy_idx_cache = {}
_OPP_GRAPH_TEMP = 0.5
_OPP_GRAPH_BATCH_BUCKETS = (8, 16, 32, 64, 128, 256)
_OPP_MODEL_CACHE_MAX = 32
_OPP_GRAPH_CACHE_MAX = 64
_opp_model_cache: OrderedDict[tuple[object, ...], BuilderMLP] = OrderedDict()
_opp_graph_cache: OrderedDict[tuple[object, ...], GraphedInference] = OrderedDict()
_opp_graph_failures: set[tuple[object, ...]] = set()


@dataclass(frozen=True, slots=True)
class GameLaunchSpec:
    game_idx: int
    opponent_id: str
    opponent_weight_path: str
    map_path: str
    replay_path: str
    seed: int
    bot_a: str
    bot_b: str
    engine_root: str
    swapped: bool = False  # True when training bot plays as team B

    @property
    def actor_team(self) -> int:
        return 1 if self.swapped else 0

    def worker_args(self) -> tuple[str, str, str, str, str, int, int]:
        return (self.bot_a, self.bot_b, self.map_path, self.engine_root, self.replay_path, self.seed, self.actor_team)

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload["map_name"] = Path(self.map_path).name
        payload["bot_b_kind"] = "dummy" if self.opponent_weight_path == DUMMY_OPP_PATH else "proxy"
        return payload


@dataclass(slots=True)
class ActiveCollector:
    collector_fd: int
    collector_mm: mmap.mmap
    sides: dict[str, CollectorServerSide]
    req_read_fds: dict[str, int]
    resp_write_fds: dict[str, int]
    obs_tensors: dict[tuple[str, int], torch.Tensor]



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


def _resolve_benchmark_maps(maps: list[str], spec: str) -> list[str]:
    """Resolve a comma-separated benchmark map selection against the available map names."""
    if not spec.strip():
        return maps
    requested: list[str] = []
    seen: set[str] = set()
    for chunk in spec.split(","):
        map_name = chunk.strip()
        if not map_name or map_name in seen:
            continue
        seen.add(map_name)
        requested.append(map_name)
    if not requested:
        raise ValueError("benchmark map list is empty")
    available = set(maps)
    missing = [map_name for map_name in requested if map_name not in available]
    if missing:
        raise ValueError(f"unknown benchmark map(s): {', '.join(missing)}")
    return requested


def _build_game_launch_specs(
    *,
    opp_weight_paths: list[str],
    opp_ids: list[str],
    maps: list[str],
    episode_num: int,
    replay_dir: Path,
    bot_a: str,
    bot_b_proxy: str,
    bot_b_dummy: str,
    engine_root: str,
    benchmark_seed: int | None = None,
    benchmark_maps: list[str] | None = None,
) -> list[GameLaunchSpec]:
    """Build per-game launch specs, optionally from a deterministic benchmark workload."""
    if benchmark_maps is None:
        candidate_maps = sorted(maps) if benchmark_seed is not None else list(maps)
    else:
        candidate_maps = list(benchmark_maps)
    if not candidate_maps:
        raise ValueError("at least one map is required to launch games")

    rng = random if benchmark_seed is None else random.Random(benchmark_seed + episode_num * 1_000_003)
    game_specs: list[GameLaunchSpec] = []
    for i, (opp_id, opp_weight_path) in enumerate(zip(opp_ids, opp_weight_paths, strict=True)):
        map_name = rng.choice(candidate_maps)
        replay_path = str(replay_dir / f"ep{episode_num}_{opp_id}_g{i}.replay26")
        seed = rng.randint(1, 2**31)
        bot_b_bot = bot_b_dummy if opp_weight_path == DUMMY_OPP_PATH else bot_b_proxy
        # Randomly swap sides so the training bot learns to play as both team A and B.
        swapped = bool(rng.getrandbits(1))
        if swapped:
            spec_bot_a, spec_bot_b = bot_b_bot, bot_a
        else:
            spec_bot_a, spec_bot_b = bot_a, bot_b_bot
        game_specs.append(
            GameLaunchSpec(
                game_idx=i,
                opponent_id=opp_id,
                opponent_weight_path=opp_weight_path,
                map_path=str(PROJECT_ROOT / "maps" / map_name),
                replay_path=replay_path,
                seed=seed,
                bot_a=spec_bot_a,
                bot_b=spec_bot_b,
                engine_root=engine_root,
                swapped=swapped,
            )
        )
    return game_specs


def _get_cached_opponent_model(weight_path: str, hidden: list[int] | None) -> BuilderMLP:
    key = _opponent_model_cache_key(weight_path, hidden)
    cached = _opp_model_cache.get(key)
    if cached is not None:
        _opp_model_cache.move_to_end(key)
        cached.eval()
        return cached

    model = BuilderMLP(hidden=hidden).to(DEVICE)
    if not model.load_weights(weight_path):
        raise RuntimeError(f"Could not load opponent weights from {weight_path}")
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


def _drain_notify(fd: int) -> int | None:
    """Drain notify bytes from a pipe. None means EOF/closed."""
    token = os.read(fd, 4096)
    if not token:
        return None
    return len(token)


def _tensor_byte_view(tensor: torch.Tensor, *, n_bytes: int | None = None) -> memoryview:
    """Return a zero-copy byte view over a contiguous CPU tensor."""
    if tensor.device.type != "cpu":
        raise ValueError("tensor byte views require CPU tensors")
    if not tensor.is_contiguous():
        raise ValueError("tensor byte views require contiguous tensors")
    size = tensor.numel() * tensor.element_size() if n_bytes is None else int(n_bytes)
    return memoryview((_ctypes.c_char * size).from_address(tensor.data_ptr()))


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

def _load_worker_result(result_path: str) -> WorkerGameResult:
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    return validate_worker_game_result(result)


def _create_collector_fd(tmpdir: str, game_idx: int, slot_count: int) -> int:
    """Create an anonymous mmap backing file, preferring /dev/shm."""
    backing_dir = "/dev/shm"
    if not os.path.isdir(backing_dir) or not os.access(backing_dir, os.W_OK):
        backing_dir = tmpdir
    fd, path = tempfile.mkstemp(prefix=f"tensor_collector_{game_idx}_", dir=backing_dir)
    try:
        os.unlink(path)
    except OSError:
        pass
    os.ftruncate(fd, collector_bytes(slot_count))
    return fd


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
    benchmark_seed: int | None = None,
    benchmark_maps: list[str] | None = None,
    collector_ring_slots: int = COLLECTOR_RING_SLOTS_DEFAULT,
    max_training_turns: int | None = None,
    profile: dict[str, float] | None = None,
    trace=None,
) -> tuple[list[list[TrajectoryStep]], list[str | None], dict[str, list[str | None]], list[dict | None], list[int], list[int], dict[int, float], list[GameLaunchSpec]]:
    """Run all games with batched GPU inference via shared-memory rings."""
    if max_training_turns is not None and max_training_turns < 2000 and REWARD_MODE == RewardMode.OUTCOME:
        raise ValueError("max_training_turns requires dense per-turn rewards; outcome-only reward mode is unsupported")
    t_total = time.perf_counter() if profile is not None else 0.0
    bot_a = str(PROJECT_ROOT / "bots" / "tensor_proxy" / "main.py")
    bot_b_proxy = str(PROJECT_ROOT / "bots" / "tensor_proxy_opp" / "main.py")
    bot_b_dummy = str(PROJECT_ROOT / "bots" / "tensor_dummy" / "main.py")
    engine_root = str(PROJECT_ROOT)
    replay_dir.mkdir(parents=True, exist_ok=True)
    temp_root = PROJECT_ROOT / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="batched_", dir=str(temp_root))
    if collector_ring_slots <= 0:
        raise ValueError(f"collector_ring_slots must be > 0, got {collector_ring_slots}")
    n_games = len(opp_weight_paths)

    game_specs = _build_game_launch_specs(
        opp_weight_paths=opp_weight_paths,
        opp_ids=opp_ids,
        maps=maps,
        episode_num=episode_num,
        replay_dir=replay_dir,
        bot_a=bot_a,
        bot_b_proxy=bot_b_proxy,
        bot_b_dummy=bot_b_dummy,
        engine_root=engine_root,
        benchmark_seed=benchmark_seed,
        benchmark_maps=benchmark_maps,
    )

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
    results: list[WorkerGameResult | None] = [None] * n_games
    actor_trajectories: dict[int, dict[int, list[TrajectoryStep]]] = {}
    processes = {}
    launch_order = sorted(range(n_games), key=lambda i: (opp_weight_paths[i], i))
    pending_indices = launch_order[:]

    fd_to_info = {}  # fd -> (game_idx, side)  side='a' or 'b'
    active_collectors: dict[int, ActiveCollector] = {}
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

    def _close_game_collector(game_idx: int) -> None:
        collector = active_collectors.pop(game_idx, None)
        if collector is None:
            return
        for fd in collector.req_read_fds.values():
            _close_read_fd(fd)
        for fd in collector.resp_write_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        collector.obs_tensors.clear()
        for side_view in collector.sides.values():
            side_view.close()
        collector.collector_mm.close()
        try:
            os.close(collector.collector_fd)
        except OSError:
            pass
        actor_hidden_bank.release_game(game_idx)
        opponent_hidden_bank.release_game(game_idx)

    def _launch_game(i: int) -> None:
        collector_fd = _create_collector_fd(tmpdir, i, collector_ring_slots)
        collector_mm = mmap.mmap(collector_fd, 0, access=mmap.ACCESS_WRITE)
        initialize_collector(collector_mm, collector_ring_slots)
        req_r_a, req_w_a = os.pipe()
        resp_r_a, resp_w_a = os.pipe()
        use_proxy_b = opp_weight_paths[i] != DUMMY_OPP_PATH
        req_r_b = req_w_b = resp_r_b = resp_w_b = -1
        if use_proxy_b:
            req_r_b, req_w_b = os.pipe()
            resp_r_b, resp_w_b = os.pipe()
        result_path = os.path.join(tmpdir, f"result_{i}.pkl")
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = ""
        if max_training_turns is None:
            worker_env.pop("TENSOR_TRAIN_MAX_TURNS", None)
        else:
            worker_env["TENSOR_TRAIN_MAX_TURNS"] = str(int(max_training_turns))
        worker_fds = tuple(fd for fd in (collector_fd, req_w_a, resp_r_a, req_w_b, resp_r_b) if fd >= 0)
        worker_python = WORKER_PYTHON if os.path.exists(WORKER_PYTHON) else sys.executable
        proc = subprocess.Popen(
            [
                worker_python,
                str(Path(__file__).resolve()),
                "--worker-result", result_path,
                "--worker-collector-fds", f"{collector_fd},{req_w_a},{resp_r_a},{req_w_b},{resp_r_b}",
                "--worker-game-args", json.dumps(game_specs[i].worker_args()),
            ],
            close_fds=True,
            cwd=str(PROJECT_ROOT),
            env=worker_env,
            pass_fds=worker_fds,
        )
        processes[i] = {"proc": proc, "result_path": result_path}
        if WORKER_LAUNCH_STAGGER_S > 0.0:
            time.sleep(WORKER_LAUNCH_STAGGER_S)

        os.close(req_w_a)
        os.close(resp_r_a)
        if use_proxy_b:
            os.close(req_w_b)
            os.close(resp_r_b)

        sides = {"a": CollectorServerSide(collector_mm, "a")}
        obs_tensors = {
            ("a", slot_index): torch.frombuffer(
                sides["a"].request_obs_buffer(slot_index),
                dtype=torch.float32,
            ).reshape(OBS_SIZE)
            for slot_index in range(sides["a"].slot_count)
        }
        fd_to_info[req_r_a] = (i, "a")
        active_fds.add(req_r_a)
        selector.register(req_r_a, selectors.EVENT_READ)
        if use_proxy_b:
            sides["b"] = CollectorServerSide(collector_mm, "b")
            for slot_index in range(sides["b"].slot_count):
                obs_tensors[("b", slot_index)] = torch.frombuffer(
                    sides["b"].request_obs_buffer(slot_index),
                    dtype=torch.float32,
                ).reshape(OBS_SIZE)
            fd_to_info[req_r_b] = (i, "b")
            active_fds.add(req_r_b)
            selector.register(req_r_b, selectors.EVENT_READ)
        active_collectors[i] = ActiveCollector(
            collector_fd=collector_fd,
            collector_mm=collector_mm,
            sides=sides,
            req_read_fds={k: v for k, v in (("a", req_r_a), ("b", req_r_b)) if v >= 0},
            resp_write_fds={k: v for k, v in (("a", resp_w_a), ("b", resp_w_b)) if v >= 0},
            obs_tensors=obs_tensors,
        )

    def _reap_processes() -> None:
        for game_idx, proc_info in list(processes.items()):
            proc = proc_info["proc"]
            if proc.poll() is None:
                continue
            if proc.returncode != 0:
                raise RuntimeError(
                    f"game worker {game_idx} exited with code {proc.returncode}"
                )
            results[game_idx] = _load_worker_result(proc_info["result_path"])
            processes.pop(game_idx, None)
            _close_game_collector(game_idx)

    # Inference server loop — disable cyclic GC to avoid pauses from
    # the collector scanning the many live pipe buffers and tensor temporaries.
    gc_was_enabled = gc.isenabled()
    gc.disable()

    model.eval()
    _max_a_batch = n_games * 4
    # Side-A rollout depends on exact behavior log-probs for PPO, so only the
    # actor forward pass is graphed; masks/sampling/log-prob stay eager.
    graphed_a = GraphedInference(model, max_batch=_max_a_batch, graph_sampling=False)
    total_inferences = 0
    actor_graph_disabled = False
    _use_pinned = DEVICE.type == "cuda"
    # Pre-allocate pinned staging buffers for H2D/D2H to enable non_blocking transfers.
    # Sized for max batch; reused each iteration to avoid repeated allocation.
    _hidden_bank_capacity = max(4096, n_games * 512)
    actor_hidden_bank = HiddenStateBank(
        hidden_dim=H_DIM,
        device=DEVICE,
        initial_capacity=_hidden_bank_capacity,
        growth_capacity=4096,
    )
    opponent_hidden_bank = HiddenStateBank(
        hidden_dim=H_DIM,
        device=DEVICE,
        initial_capacity=_hidden_bank_capacity,
        growth_capacity=4096,
    )
    if _use_pinned:
        _pin_obs = torch.empty(_max_a_batch, OBS_SIZE, dtype=torch.float32, pin_memory=True)
        _pin_results = torch.empty(5, _max_a_batch, dtype=torch.float32, pin_memory=True)
        _pin_h_prev = torch.empty(_max_a_batch, H_DIM, dtype=torch.float32, pin_memory=True)
        _pin_actor_bt_l = torch.empty(_max_a_batch, N_BUILD, dtype=torch.float32, pin_memory=True)
        _pin_actor_bd_l = torch.empty(_max_a_batch, N_DIR, dtype=torch.float32, pin_memory=True)
        _pin_actor_cd_l = torch.empty(_max_a_batch, N_CONV_DIR, dtype=torch.float32, pin_memory=True)
        _pin_actor_brt_l = torch.empty(_max_a_batch, N_BRIDGE_TARGETS, dtype=torch.float32, pin_memory=True)
        _pin_actor_md_l = torch.empty(_max_a_batch, N_DIR, dtype=torch.float32, pin_memory=True)
        _pin_actor_temperature = torch.empty(_max_a_batch, N_ACTION_TEMPERATURES, dtype=torch.float32, pin_memory=True)
    else:
        _pin_obs = torch.empty(_max_a_batch, OBS_SIZE, dtype=torch.float32)
        _pin_results = torch.empty(5, _max_a_batch, dtype=torch.float32)
        _pin_h_prev = torch.empty(_max_a_batch, H_DIM, dtype=torch.float32)
    _actor_trunk = torch.empty(_max_a_batch, OBS_SIZE + H_DIM, dtype=torch.float32, device=DEVICE)
    _opponent_trunk = torch.empty(_max_a_batch, OBS_SIZE + H_DIM, dtype=torch.float32, device=DEVICE)

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

        # Drain all available requests from shared rings, split by side.
        pending_a = []  # (game_idx, request)
        pending_b = []  # (game_idx, request, opp_weight_path)
        t_phase = time.perf_counter() if profile is not None else 0.0
        for fd in readable:
            game_idx, side = fd_to_info[fd]
            notify_count = _drain_notify(fd)
            if notify_count is None:
                _close_read_fd(fd)
                continue
            collector = active_collectors.get(game_idx)
            if collector is None:
                continue
            requests = collector.sides[side].pop_requests()
            if len(requests) < notify_count:
                raise RuntimeError(
                    f"collector underflow for game {game_idx} side {side}: "
                    f"{len(requests)} requests for {notify_count} notify bytes"
                )
            if side == "a":
                for request in requests:
                    pending_a.append((game_idx, request))
            else:
                for request in requests:
                    pending_b.append((game_idx, request, game_to_opp[game_idx]))
        if profile is not None:
            profile["ipc_read_s"] = profile.get("ipc_read_s", 0.0) + (time.perf_counter() - t_phase)

        # Batch A inference (all same model — training bot with exploration)
        if pending_a:
            n_a = len(pending_a)
            batch_a_calls += 1
            batch_a_items += n_a
            batch_a_max = max(batch_a_max, n_a)
            t_phase = time.perf_counter() if profile is not None else 0.0
            batch_obs = _pin_obs[:n_a]
            for row, (game_idx, request) in enumerate(pending_a):
                collector = active_collectors[game_idx]
                batch_obs[row].copy_(collector.obs_tensors[("a", request.slot_index)])
            actor_keys = [(game_idx, request.unit_id) for game_idx, request in pending_a]
            if profile is not None:
                profile["batch_pack_a_s"] = profile.get("batch_pack_a_s", 0.0) + (time.perf_counter() - t_phase)

            with trace_scope(trace, "collect.actor_batch"), torch.no_grad():
                actor_all_t = _actor_trunk[:n_a]
                t_phase = time.perf_counter() if profile is not None else 0.0
                with trace_scope(trace, "collect.actor_h2d"):
                    actor_all_t[:, :OBS_SIZE].copy_(batch_obs, non_blocking=_use_pinned)
                    actor_row_index = actor_hidden_bank.gather_into(
                        actor_keys,
                        actor_all_t[:, OBS_SIZE:],
                    )
                if profile is not None:
                    dt = time.perf_counter() - t_phase
                    profile["h2d_a_s"] = profile.get("h2d_a_s", 0.0) + dt

                obs_cpu = batch_obs
                t_infer = time.perf_counter() if profile is not None else 0.0
                while True:
                    bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_next = graphed_a.policy_heads(
                        actor_all_t,
                        trace=trace,
                        scope_prefix="collect.actor",
                    )
                    if profile is not None:
                        profile["infer_a_s"] = profile.get("infer_a_s", 0.0) + (time.perf_counter() - t_infer)

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    with trace_scope(trace, "collect.actor_d2h"):
                        if _use_pinned:
                            _pin_actor_bt_l[:n_a].copy_(bt_l, non_blocking=True)
                            _pin_actor_bd_l[:n_a].copy_(bd_l, non_blocking=True)
                            _pin_actor_cd_l[:n_a].copy_(cd_l, non_blocking=True)
                            _pin_actor_brt_l[:n_a].copy_(brt_l, non_blocking=True)
                            _pin_actor_md_l[:n_a].copy_(md_l, non_blocking=True)
                            _pin_actor_temperature[:n_a].copy_(head_temperature, non_blocking=True)
                            _pin_h_prev[:n_a].copy_(actor_all_t[:, OBS_SIZE:], non_blocking=True)
                            torch.cuda.synchronize()
                            bt_l_cpu = _pin_actor_bt_l[:n_a]
                            bd_l_cpu = _pin_actor_bd_l[:n_a]
                            cd_l_cpu = _pin_actor_cd_l[:n_a]
                            brt_l_cpu = _pin_actor_brt_l[:n_a]
                            md_l_cpu = _pin_actor_md_l[:n_a]
                            temperature_cpu = _pin_actor_temperature[:n_a]
                            h_prev_cpu = _pin_h_prev[:n_a]
                        else:
                            bt_l_cpu = bt_l.float().cpu()
                            bd_l_cpu = bd_l.float().cpu()
                            cd_l_cpu = cd_l.float().cpu()
                            brt_l_cpu = brt_l.float().cpu()
                            md_l_cpu = md_l.float().cpu()
                            temperature_cpu = head_temperature.float().cpu()
                            h_prev_cpu = actor_all_t[:, OBS_SIZE:].cpu().contiguous()
                    if profile is not None:
                        profile["d2h_a_s"] = profile.get("d2h_a_s", 0.0) + (time.perf_counter() - t_phase)

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    with trace_scope(trace, "collect.actor.sample"):
                        bt_act, bd_act, cd_act, brt_act, md_act, lp, _ = _sample_actions_from_policy_heads(
                            obs_cpu,
                            bt_l_cpu,
                            bd_l_cpu,
                            cd_l_cpu,
                            brt_l_cpu,
                            md_l_cpu,
                            temperature_cpu,
                            h_prev_cpu,
                            temperature=1.0,
                            return_log_prob=True,
                        )
                    if profile is not None:
                        profile["sample_a_s"] = profile.get("sample_a_s", 0.0) + (time.perf_counter() - t_phase)
                    if lp is not None and torch.isfinite(lp).all():
                        break
                    if actor_graph_disabled:
                        bad_idx = int((~torch.isfinite(lp)).nonzero(as_tuple=False)[0, 0])
                        bad_game_idx, bad_request = pending_a[bad_idx]
                        raise RuntimeError(
                            "non-finite behavior log-prob after actor graph fallback "
                            f"for game {bad_game_idx}, unit {bad_request.unit_id}, round {bad_request.round_idx}: {float(lp[bad_idx])!r}"
                        )
                    print("    actor graph disabled after non-finite behavior log-prob; falling back to eager actor inference")
                    graphed_a.disable_graph()
                    actor_graph_disabled = True
                    t_infer = time.perf_counter() if profile is not None else 0.0
                actor_hidden_bank.update(actor_row_index, h_next)

            t_phase = time.perf_counter() if profile is not None else 0.0
            with trace_scope(trace, "collect.actor_fanout"):
                h_prev_view = _tensor_byte_view(h_prev_cpu)
                for i, (game_idx, request) in enumerate(pending_a):
                    unit_id = request.unit_id
                    round_idx = request.round_idx
                    old_lp = float(lp[i])
                    if not math.isfinite(old_lp):
                        raise RuntimeError(
                            f"non-finite behavior log-prob for game {game_idx}, unit {unit_id}, round {round_idx}: {old_lp!r}"
                        )
                    trajectories_by_unit = actor_trajectories.setdefault(game_idx, {})
                    unit_traj = trajectories_by_unit.setdefault(unit_id, [])
                    # dbg_flags from the request reports the previous action's
                    # debug counters — retroactively update the last step.
                    _df = request.dbg_flags
                    if _df and unit_traj:
                        unit_traj[-1] = unit_traj[-1]._replace(
                            harv_attempt=(_df) & 1,
                            harv_fail=(_df >> 1) & 1,
                            harv_prox_gate=(_df >> 2) & 1,
                            harv_prox_paid=(_df >> 3) & 1,
                            conv_attempt=(_df >> 4) & 1,
                            conv_fail=(_df >> 5) & 1,
                            conv_prox_gate=(_df >> 6) & 1,
                            conv_prox_paid=(_df >> 7) & 1,
                        )
                    unit_traj.append(
                        make_trajectory_step(
                            obs=request.obs_buffer,
                            h_prev=h_prev_view[i * _H_BYTES:(i + 1) * _H_BYTES],
                            obs_bytes_len=OBS_BYTES,
                            h_bytes_len=H_DIM * 4,
                            round_idx=round_idx,
                            bt=int(bt_act[i]),
                            bd=int(bd_act[i]),
                            cd=int(cd_act[i]),
                            brt=int(brt_act[i]),
                            md=int(md_act[i]),
                            reward=0.0,
                            old_log_prob=old_lp,
                            enemy_core_dmg=0.0,
                            own_core_dmg=0.0,
                            harv_attempt=0,
                            harv_fail=0,
                            harv_prox_gate=0,
                            harv_prox_paid=0,
                            conv_attempt=0,
                            conv_fail=0,
                            conv_prox_gate=0,
                            conv_prox_paid=0,
                        )
                    )
                    collector = active_collectors.get(game_idx)
                    if collector is None:
                        continue
                    try:
                        collector.sides["a"].write_response(
                            int(bt_act[i]),
                            int(bd_act[i]),
                            int(cd_act[i]),
                            int(brt_act[i]),
                            int(md_act[i]),
                        )
                        os.write(collector.resp_write_fds["a"], b"\x01")
                    except OSError:
                        pass
            if profile is not None:
                profile["ipc_write_a_s"] = profile.get("ipc_write_a_s", 0.0) + (time.perf_counter() - t_phase)
            total_inferences += n_a

        # Batch B inference grouped by opponent model (low-temp sampling for diversity)
        if pending_b:
            by_opp = {}
            for game_idx, request, wp in pending_b:
                by_opp.setdefault(wp, []).append((game_idx, request))

            for wp, items in by_opp.items():
                n_b = len(items)
                batch_b_calls += 1
                batch_b_items += n_b
                batch_b_max = max(batch_b_max, n_b)
                t_phase = time.perf_counter() if profile is not None else 0.0
                batch_obs = _pin_obs[:n_b]
                for row, (game_idx, request) in enumerate(items):
                    collector = active_collectors[game_idx]
                    batch_obs[row].copy_(collector.obs_tensors[("b", request.slot_index)])
                if profile is not None:
                    profile["batch_pack_b_s"] = profile.get("batch_pack_b_s", 0.0) + (time.perf_counter() - t_phase)

                if wp == DUMMY_OPP_PATH:
                    t_phase = time.perf_counter() if profile is not None else 0.0
                    obs_t = batch_obs
                    bt_act, bd_act, cd_act, md_act = _dummy_policy(obs_t)
                    brt_act = torch.zeros_like(bt_act)
                    if profile is not None:
                        profile["dummy_b_s"] = profile.get("dummy_b_s", 0.0) + (time.perf_counter() - t_phase)

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    for i, (game_idx, _request) in enumerate(items):
                        collector = active_collectors.get(game_idx)
                        if collector is None:
                            continue
                        try:
                            collector.sides["b"].write_response(
                                int(bt_act[i]),
                                int(bd_act[i]),
                                int(cd_act[i]),
                                int(brt_act[i]),
                                int(md_act[i]),
                            )
                            os.write(collector.resp_write_fds["b"], b"\x01")
                        except OSError:
                            pass
                    if profile is not None:
                        profile["ipc_write_b_s"] = profile.get("ipc_write_b_s", 0.0) + (time.perf_counter() - t_phase)
                    continue

                opp_m = opp_models[wp]
                opp_keys = [(game_idx, request.unit_id) for game_idx, request in items]
                opp_g = None
                with trace_scope(trace, "collect.opponent_batch"), torch.no_grad():
                    opp_all_t = _opponent_trunk[:n_b]
                    t_phase = time.perf_counter() if profile is not None else 0.0
                    opp_all_t[:, :OBS_SIZE].copy_(batch_obs, non_blocking=_use_pinned and n_b <= _max_a_batch)
                    opp_row_index = opponent_hidden_bank.gather_into(
                        opp_keys,
                        opp_all_t[:, OBS_SIZE:],
                    )
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["h2d_b_s"] = profile.get("h2d_b_s", 0.0) + dt
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                    t_phase = time.perf_counter() if profile is not None else 0.0
                    if opp_g is not None and n_b <= opp_g.max_batch:
                        # CUDA-graphed path: forward + mask + sample in one graph replay
                        bt_act, bd_act, cd_act, brt_act, md_act, _, h_next = opp_g(opp_all_t)
                        if _use_pinned and n_b <= _max_a_batch:
                            _pin_results[:5, :n_b].copy_(torch.stack([bt_act, bd_act, cd_act, brt_act, md_act]), non_blocking=True)
                            torch.cuda.synchronize()
                            acts_cpu = _pin_results[:5, :n_b]
                        else:
                            acts_cpu = torch.stack([bt_act, bd_act, cd_act, brt_act, md_act]).cpu()
                    else:
                        # Fallback: ungraphed inference
                        bt_l, bd_l, cd_l, _brt_l, md_l, head_temperature, h_next = opp_m.infer_packed(opp_all_t)
                        if profile is not None:
                            dt = time.perf_counter() - t_phase
                            profile["forward_b_s"] = profile.get("forward_b_s", 0.0) + dt
                            profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt

                        t_phase = time.perf_counter() if profile is not None else 0.0
                        obs_t_cpu = batch_obs
                        bt_l_cpu = bt_l.float().cpu()
                        bd_l_cpu = bd_l.float().cpu()
                        cd_l_cpu = cd_l.float().cpu()
                        brt_l_cpu = _brt_l.float().cpu()
                        md_l_cpu = md_l.float().cpu()
                        head_temperature_cpu = head_temperature.float().cpu()
                        bt_m, bd_m = compute_action_masks(obs_t_cpu)
                        cd_m = conveyor_dir_mask(obs_t_cpu.device, obs_t_cpu.shape[0])
                        bt_l_cpu.masked_fill_(~bt_m, -1e9)
                        cd_l_cpu.masked_fill_(~cd_m, -1e9)
                        bt_act, _ = sample_logits_gumbel(
                            bt_l_cpu,
                            temperature=head_temperature_cpu[:, 0] * _OPP_GRAPH_TEMP,
                            mask=bt_m,
                            return_log_prob=False,
                        )
                        bd_l_cpu.masked_fill_(~bd_m, -1e9)
                        bd_act, _ = sample_logits_gumbel(
                            bd_l_cpu,
                            temperature=head_temperature_cpu[:, 1] * _OPP_GRAPH_TEMP,
                            mask=bd_m,
                            return_log_prob=False,
                        )
                        cd_act, _ = sample_logits_gumbel(
                            cd_l_cpu,
                            temperature=head_temperature_cpu[:, 2] * _OPP_GRAPH_TEMP,
                            mask=cd_m,
                            return_log_prob=False,
                        )
                        brt_act, _ = sample_logits_gumbel(
                            brt_l_cpu,
                            temperature=head_temperature_cpu[:, 3] * _OPP_GRAPH_TEMP,
                            return_log_prob=False,
                        )
                        md_act, _ = sample_logits_gumbel(
                            md_l_cpu,
                            temperature=head_temperature_cpu[:, 4] * _OPP_GRAPH_TEMP,
                            return_log_prob=False,
                        )
                        acts_cpu = torch.stack([bt_act, bd_act, cd_act, brt_act, md_act])
                    if profile is not None:
                        dt = time.perf_counter() - t_phase
                        profile["infer_b_s"] = profile.get("infer_b_s", 0.0) + dt
                opponent_hidden_bank.update(opp_row_index, h_next)

                t_phase = time.perf_counter() if profile is not None else 0.0
                for i, (game_idx, request) in enumerate(items):
                    collector = active_collectors.get(game_idx)
                    if collector is None:
                        continue
                    try:
                        collector.sides["b"].write_response(
                            int(acts_cpu[0, i]),
                            int(acts_cpu[1, i]),
                            int(acts_cpu[2, i]),
                            int(acts_cpu[3, i]),
                            int(acts_cpu[4, i]),
                        )
                        os.write(collector.resp_write_fds["b"], b"\x01")
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
        results[game_idx] = None
        processes.pop(game_idx, None)
        _close_game_collector(game_idx)
    if profile is not None:
        profile["worker_join_s"] = time.perf_counter() - t_phase

    selector.close()

    for fd in list(active_fds):
        _close_read_fd(fd)
    for game_idx in list(active_collectors):
        _close_game_collector(game_idx)

    if gc_was_enabled:
        gc.enable()
        gc.collect()

    # Read trajectories and augment with replay ground truth
    t_phase = time.perf_counter() if profile is not None else 0.0
    trajectories: list[list[TrajectoryStep]] = []
    winners = []
    all_global_states = []
    traj_game_map = []  # maps each trajectory index -> game index (for global states)
    game_counter = 0
    game_origin = []  # game_counter -> original game index i
    game_delivery_totals: dict[int, float] = {}  # game_counter -> total delivery reward
    per_opp_winners = {}
    for i, result in enumerate(results):
        oid = opp_ids[i]
        if result is None:
            raise RuntimeError(f"missing worker result for game {i}")
        trajectories_by_unit = actor_trajectories.get(i)
        if not trajectories_by_unit:
            raise RuntimeError(f"missing collected actor trajectories for game {i}")
        game_trajs = [trajectories_by_unit[unit_id] for unit_id in sorted(trajectories_by_unit)]
        # Normalize winner to training bot's perspective: "a" = our bot won.
        # When swapped (training bot played as team B), flip the raw winner.
        w = result.winner
        if game_specs[i].swapped and w in ("a", "b"):
            w = "b" if w == "a" else "a"
        gstates = result.global_states
        actor_team = game_specs[i].actor_team
        # Inject replay-derived resource reward as per-step reward (training only)
        if gstates and REWARD_MODE in (RewardMode.INCOME, RewardMode.FLOW):
            from flow_reward import compute_dense_delivery_rewards, compute_dense_flow_rewards, augment_trajectories_with_flow_rewards
            if REWARD_MODE == RewardMode.INCOME:
                rewards = compute_dense_delivery_rewards(gstates)
            else:
                rewards = compute_dense_flow_rewards(gstates, team=actor_team)
            game_trajs = augment_trajectories_with_flow_rewards(game_trajs, rewards)
            # Track delivery totals separately for logging
            delivery_rewards = compute_dense_delivery_rewards(gstates)
            if max_training_turns is not None:
                delivery_rewards = _truncate_reward_dict(delivery_rewards, max_training_turns)
            game_delivery_totals[game_counter] = sum(delivery_rewards.values())
        if max_training_turns is not None:
            game_trajs = _truncate_trajectories_to_turn_limit(game_trajs, max_training_turns)
            gstates = _truncate_global_states_to_turn_limit(gstates, max_training_turns)
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
        profile["actor_graph_disabled"] = float(actor_graph_disabled)

    # Hydrate path-backed buffers before the temporary replay directory goes away.
    preload_replay_buffers(all_global_states)

    shutil.rmtree(tmpdir, ignore_errors=True)
    n_swapped = sum(1 for spec in game_specs if spec.swapped)
    print(f"    [{n_games} games ({n_swapped} swapped), {total_inferences} GPU inferences]")
    return trajectories, winners, per_opp_winners, all_global_states, traj_game_map, game_origin, game_delivery_totals, game_specs


def _step_round_index(step: TrajectoryStep) -> int:
    if not isinstance(step, TrajectoryStep):
        raise TypeError(f"Expected TrajectoryStep, got {type(step).__name__}")
    return step.round_idx


def _step_build_reward(step: TrajectoryStep) -> float:
    if not isinstance(step, TrajectoryStep):
        raise TypeError(f"Expected TrajectoryStep, got {type(step).__name__}")
    return float(step.reward)


def _shared_flow_reward_total(game_trajs: list[list[TrajectoryStep]]) -> float:
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


def _truncate_trajectories_to_turn_limit(
    trajectories: list[list[TrajectoryStep]],
    max_turns: int,
) -> list[list[TrajectoryStep]]:
    """Keep only non-empty trajectories with steps strictly before ``max_turns``."""
    if max_turns <= 0:
        return []
    truncated = [
        [step for step in traj if _step_round_index(step) < max_turns]
        for traj in trajectories
    ]
    return [traj for traj in truncated if traj]


def _truncate_reward_dict(rewards: dict[int, float], max_turns: int) -> dict[int, float]:
    if max_turns <= 0:
        return {}
    return {rnd: reward for rnd, reward in rewards.items() if rnd < max_turns}


def _truncate_global_states_to_turn_limit(
    global_states: dict | None,
    max_turns: int,
) -> dict | None:
    """Slice replay-backed per-turn buffers so training only sees early turns."""
    if global_states is None:
        return None
    n_turns = int(global_states.get("n_turns", 0))
    if n_turns <= max_turns:
        return global_states

    capped_turns = max(0, int(max_turns))
    state_stride = int(global_states.get("state_stride", 0))
    n_scalars = int(global_states.get("n_scalars", 0))
    bytes_per_turn = {
        STATE_BUFFER: state_stride * 4,
        SCALAR_BUFFER: n_scalars * 4,
        TRANSFER_EVENTS_BUFFER: 8,
        TI_DELIVERED_BUFFER: 8,
        AX_DELIVERED_BUFFER: 8,
    }

    for spec, per_turn in bytes_per_turn.items():
        limit = capped_turns * per_turn
        required = spec in (STATE_BUFFER, SCALAR_BUFFER)
        if per_turn <= 0:
            global_states[spec.buf_key] = None
            global_states[spec.path_key] = None
            continue
        path = global_states.get(spec.path_key)
        if path:
            try:
                with Path(path).open("rb") as f:
                    global_states[spec.buf_key] = bytearray(f.read(limit))
                global_states[spec.path_key] = None
                continue
            except OSError:
                pass
        buf = global_states.get(spec.buf_key)
        if buf is None:
            if required:
                raise RuntimeError(
                    f"missing required replay buffer during truncation: {spec.path_key}"
                )
            global_states[spec.buf_key] = None
            global_states[spec.path_key] = None
            continue
        global_states[spec.buf_key] = bytearray(memoryview(buf)[:limit])
        global_states[spec.path_key] = None

    global_states["n_turns"] = capped_turns
    return global_states


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
        "--actor-minibatch-size",
        type=int,
        default=0,
        help="Actor PPO minibatch size (default: max(N//16, 8192) for GPU saturation)",
    )
    parser.add_argument("--teacher", action="store_true", help="Train large teacher, distill to student")
    parser.add_argument("--distill-epochs", type=int, default=20, help="Distillation epochs")
    parser.add_argument("--distill-lr", type=float, default=1e-3, help="Distillation learning rate")
    parser.add_argument("--coalesce-ms", type=float, default=DEFAULT_COALESCE_MS)
    parser.add_argument(
        "--collector-ring-slots",
        type=int,
        default=COLLECTOR_RING_SLOTS_DEFAULT,
        help="Per-side shared-memory request/response ring depth",
    )
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
    parser.add_argument(
        "--use-pmpo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PMPO sign-weighted policy updates and disable advantage normalization",
    )
    parser.add_argument("--profile-episodes", type=str, default="", help="1-based episode list/ranges to profile end-to-end, e.g. 1,5-7")
    parser.add_argument("--profile-dir", type=str, default="", help="Output dir for profiler traces (default: <session>/profiles)")
    parser.add_argument("--profile-ops", action="store_true", help="Also capture torch op-level profiling data (much heavier than scope tracing)")
    parser.add_argument("--profile-shapes", action="store_true", help="Record tensor shapes in end-to-end profiler output")
    parser.add_argument("--profile-memory", action="store_true", help="Record memory events in end-to-end profiler output")
    parser.add_argument("--profile-stack", action="store_true", help="Record Python stacks in end-to-end profiler output")
    parser.add_argument("--benchmark-seed", type=int, default=None, help="Use a deterministic local RNG for map and game-seed assignment")
    parser.add_argument("--benchmark-maps", type=str, default="", help="Optional comma-separated map filenames for deterministic benchmark runs")
    parser.add_argument("--resume", action="store_true", help="Resume a prior session from saved trainer state")
    parser.add_argument("--worker-result", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-collector-fds", type=str, default="", help=argparse.SUPPRESS)
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
    game_log_path = sess["game_log_path"]
    profile_episodes = parse_profile_episode_spec(args.profile_episodes)
    profile_cfg = None
    if profile_episodes:
        profile_output_dir = Path(args.profile_dir) if args.profile_dir else session_dir / "profiles"
        profile_cfg = EpisodeProfilerConfig(
            episodes=profile_episodes,
            output_dir=profile_output_dir,
            profile_ops=args.profile_ops,
            record_shapes=args.profile_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.profile_stack,
        )

    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]
    benchmark_maps = _resolve_benchmark_maps(maps, args.benchmark_maps)
    league = League(sess["league_dir"], args.league_size)

    hidden = TEACHER_HIDDEN if args.teacher else None
    model = BuilderMLP(hidden=hidden).to(DEVICE)
    configure_centralized_critic_runtime(enable_compile=not args.critic_no_compile)
    reset_centralized_critic()
    _critic_lr = args.critic_lr if args.critic_lr > 0 else args.lr
    if args.resume:
        if not model.load_weights(weight_path):
            raise SystemExit(f"Could not load actor weights from {weight_path}")
        _load_resume_critic_checkpoint(weight_path, lr=_critic_lr)
    elif args.init_weights:
        if not model.load_weights(args.init_weights):
            raise SystemExit(f"Could not load actor weights from {args.init_weights}")
        print(f"Initialized actor from {args.init_weights}")
    _deferred_critic_path = args.critic_init_weights or None
    optimizer = make_normuon_optimizer(model, lr=args.lr, weight_decay=0.0)
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
    print(f"Collector ring slots: {args.collector_ring_slots}")
    print(f"Training turn cap: first {TRAIN_MAX_TURNS} turns per match")
    if args.benchmark_seed is not None or args.benchmark_maps:
        print(
            "Benchmark workload:"
            f" seed={args.benchmark_seed} maps={','.join(benchmark_maps)}"
        )
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
        jsonl_init(log_path, config=_log_config)

    for ep in range(start_episode, args.episodes):
        t0 = time.time()
        episode_trace = profile_cfg.create(ep + 1) if profile_cfg is not None else None
        episode_profile = {} if episode_trace is not None else None
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

        with trace_scope(episode_trace, "episode.save_actor_weights"):
            model.save_weights(weight_path)

        # Build all games across all opponents for parallel execution
        all_opp_paths = []
        all_opp_ids = []
        for opp in opponents:
            for _ in range(games_per_opp):
                all_opp_paths.append(opp["path"])
                all_opp_ids.append(opp["id"])

        with trace_scope(episode_trace, "episode.collect"):
            all_trajs, all_winners, per_opp_winners, all_global_states, traj_game_map, game_origin, game_delivery_totals, game_specs = collect_games_batched(
                model, all_opp_paths, all_opp_ids, maps,
                ep + 1, replay_dir,
                opp_hidden=hidden,
                coalesce_ms=args.coalesce_ms,
                benchmark_seed=args.benchmark_seed,
                benchmark_maps=benchmark_maps if (args.benchmark_seed is not None or args.benchmark_maps) else None,
                collector_ring_slots=args.collector_ring_slots,
                max_training_turns=TRAIN_MAX_TURNS,
                profile=episode_profile,
                trace=episode_trace,
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
            with trace_scope(episode_trace, "episode.load_critic_init"):
                if not _load_critic_init_weights(_deferred_critic_path, lr=_critic_lr):
                    raise SystemExit(f"Could not load critic from {_deferred_critic_path}")
                print(f"Loaded critic init weights from {_deferred_critic_path}")
                _deferred_critic_path = None
        model.train()
        with trace_scope(episode_trace, "episode.train"):
            stats = train_on_trajectories(
                model, optimizer, all_trajs,
                global_states_list=all_global_states,
                traj_game_map=traj_game_map if all_global_states else None,
                game_winners=all_winners,
                target_kl=0.03,
                actor_minibatch_size=args.actor_minibatch_size or None,
                freeze_critic=args.freeze_critic,
                use_pmpo=args.use_pmpo,
                profile=episode_profile,
                trace=episode_trace,
            )
        with trace_scope(episode_trace, "episode.save_training_state"):
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
            with trace_scope(episode_trace, "episode.checkpoint"):
                gen = league.latest_gen() + 1
                ckpt = str(league.league_dir / f"gen_{gen}.pkl")
                model.save_weights(ckpt)
                league.register_model(ckpt, gen)
                current_id = f"gen_{gen}"

        # Update ELO every episode using the latest registered model
        if current_id is not None:
            with trace_scope(episode_trace, "episode.elo_update"):
                for opp, winners in per_opp_results:
                    if opp["path"] == DUMMY_OPP_PATH:
                        continue
                    for w in winners:
                        if w is not None:
                            league.update_elo(current_id, opp["id"], w)
                league.save()

        # Stats
        def _traj_reward(traj):
            r_flow, r_dmg, r_own_dmg = 0.0, 0.0, 0.0
            harv_attempts = harv_failures = harv_prox_gate = harv_prox_paid = 0
            conv_attempts = conv_failures = conv_prox_gate = conv_prox_paid = 0
            for s in traj:
                if not isinstance(s, TrajectoryStep):
                    raise TypeError(f"Expected TrajectoryStep, got {type(s).__name__}")
                n_working = s.reward
                dmg = s.enemy_core_dmg
                own_dmg = s.own_core_dmg
                harv_attempt = s.harv_attempt
                harv_fail = s.harv_fail
                harv_gate = s.harv_prox_gate
                harv_paid = s.harv_prox_paid
                conv_attempt = s.conv_attempt
                conv_fail = s.conv_fail
                conv_gate = s.conv_prox_gate
                conv_paid = s.conv_prox_paid
                r_flow += n_working
                harv_attempts += harv_attempt
                harv_failures += harv_fail
                harv_prox_gate += harv_gate
                harv_prox_paid += harv_paid
                conv_attempts += conv_attempt
                conv_failures += conv_fail
                conv_prox_gate += conv_gate
                conv_prox_paid += conv_paid
            total = r_flow
            return (
                total, r_flow, 0.0, 0.0,
                harv_attempts, harv_failures, harv_prox_gate, harv_prox_paid,
                conv_attempts, conv_failures, conv_prox_gate, conv_prox_paid,
            )

        n_trajs = max(len(all_trajs), 1)
        rewards = []
        for traj_i, traj in enumerate(all_trajs):
            rewards.append(_traj_reward(traj))
        game_trajs: dict[int, list[list]] = {}
        for traj_i, traj in enumerate(all_trajs):
            gi = traj_game_map[traj_i] if traj_i < len(traj_game_map) else 0
            game_trajs.setdefault(gi, []).append(traj)
        game_shared_flow = {
            gi: _shared_flow_reward_total(trajs)
            for gi, trajs in game_trajs.items()
        }
        n_games_logged = max(len(game_shared_flow), 1)
        mean_r_flow = sum(game_shared_flow.values()) / n_games_logged
        mean_r_delivery = sum(game_delivery_totals.values()) * REWARD_INCOME_SCALE / n_games_logged
        mean_r_flow -= mean_r_delivery
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
        mean_reward = (mean_r_flow + mean_r_delivery + r_outcome)
        mean_ti = sum(game_delivery_totals.values()) / max(len(game_delivery_totals), 1)
        elapsed = elapsed_offset + (time.time() - run_started_at)

        ep_entry = {
            "episode": ep + 1, "reward": round(mean_reward, 4),
            "r_flow": round(mean_r_flow, 4), "r_delivery": round(mean_r_delivery, 4),
            "r_dmg": round(mean_r_dmg, 4),
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
            "temp_bt_mean": round(stats.get("temp_bt_mean", 0), 4),
            "temp_bd_mean": round(stats.get("temp_bd_mean", 0), 4),
            "temp_cd_mean": round(stats.get("temp_cd_mean", 0), 4),
            "temp_brt_mean": round(stats.get("temp_brt_mean", 0), 4),
            "temp_md_mean": round(stats.get("temp_md_mean", 0), 4),
            "temp_bt_std": round(stats.get("temp_bt_std", 0), 4),
            "temp_bd_std": round(stats.get("temp_bd_std", 0), 4),
            "temp_cd_std": round(stats.get("temp_cd_std", 0), 4),
            "temp_brt_std": round(stats.get("temp_brt_std", 0), 4),
            "temp_md_std": round(stats.get("temp_md_std", 0), 4),
            "reward_raw_std": round(stats.get("reward_raw_std", 0), 4),
            "sps": stats["n_steps"], "elapsed": round(elapsed, 1),
            "winrate": round(total_wins / max(total_games, 1), 3),
        }
        if episode_trace is not None:
            ep_entry["profile_summary"] = str(episode_trace.summary_path)
            ep_entry["profile_trace"] = str(episode_trace.trace_path)
        train_log.append(ep_entry)

        # Append single line — O(1) per episode instead of O(n) rewrite
        with trace_scope(episode_trace, "episode.logging"):
            jsonl_append(log_path, ep_entry)

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
        for gi in sorted(game_rewards):
            gr = game_rewards[gi]
            w = all_winners[gi] if gi < len(all_winners) else None
            orig_i = game_origin[gi] if gi < len(game_origin) else gi
            oid = all_opp_ids[orig_i] if orig_i < len(all_opp_ids) else "?"
            ti = game_delivery_totals.get(gi, 0.0)
            nu = max(gr[_N_REWARD_FIELDS], 1)
            jsonl_append(game_log_path, {
                "episode": ep + 1, "game": orig_i,
                "opp": oid, "winner": w,
                "reward": round(game_shared_flow.get(gi, 0.0), 3), "units": gr[_N_REWARD_FIELDS],
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
            })

        with trace_scope(episode_trace, "episode.resume_state"):
            _atomic_torch_save({
                "episode": ep + 1,
                "elapsed_offset_s": elapsed,
                "optimizer": optimizer.state_dict(),
                "kl_ema": _kl_ema,
            }, state_path)

        if episode_trace is not None:
            trace_metadata = {
                "episode": ep + 1,
                "collect_s": t_collect,
                "train_s": t_train,
                "wins": total_wins,
                "losses": total_losses,
                "games": total_games,
                "n_steps": stats.get("n_steps", 0),
            }
            if args.benchmark_seed is not None or args.benchmark_maps:
                trace_metadata["benchmark_seed"] = args.benchmark_seed
                trace_metadata["benchmark_maps"] = list(benchmark_maps)
                trace_metadata["benchmark_workload"] = [spec.metadata() for spec in game_specs]
            episode_trace.finish(
                coarse_metrics=episode_profile,
                metadata=trace_metadata,
            )

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Ep {ep + 1:3d}/{args.episodes}: "
                f"W/L={total_wins}/{total_losses}/{total_games}  "
                f"reward/unit={mean_reward:+.2f}  flow={mean_r_flow:+.2f}  ti={mean_ti:.0f}  "
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
