"""Replay dataset: loads BC samples from ladder replays via the sequential parser.

Handles both actor (per-unit obs+action) and critic (global state+value label)
data from the same replay files.
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
from threading import Lock

import torch

from replay_parser import (
    parse_replay_full,
    parse_replay_sequential,
    parse_replay_working,
    parse_replays_working_batch,
)

_CACHE_DIR_NAME = ".pretrain_cache"
_ACTOR_CACHE_VERSION = 8  # v8: obs_size 498→548 (TILE_FEATURES 19→21)
_CRITIC_CACHE_VERSION = 14  # v14: income mode (no terminal reward)
from model import (
    GAMMA,
    REWARD_MODE,
    RewardMode,
    REWARD_INCOME_SCALE,
    TERMINAL_WIN_REWARD,
    TERMINAL_LOSS_REWARD,
    HARVESTER_BUILD_TYPE,
    N_BUILD,
    N_DIR,
    UNIT_CONTEXT_DIM,
    compute_action_masks,
    compute_harvester_build_dir_mask,
    _OBS_X_NORM,
    _OBS_Y_NORM,
    _OBS_CORE_DX,
    _OBS_CORE_DY,
    _OBS_ECORE_DX,
    _OBS_ECORE_DY,
    _OBS_GLOBAL_START,
    _OBS_SPAWN_ROUND,
)


def _file_signature(path: str | Path) -> tuple[int, int]:
    st = Path(path).stat()
    return st.st_size, st.st_mtime_ns


def _cache_dir(replay_dir: str | Path) -> Path:
    d = Path(replay_dir) / _CACHE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(
    cache_dir: Path,
    kind: str,
    source_path: str | Path,
    team: int,
) -> Path:
    path = Path(source_path)
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]
    return cache_dir / f"{path.stem}.{kind}.team{team}.{digest}.pt"


def _load_torch_cache(
    cache_path: Path,
    *,
    version: int,
    file_sig: tuple[int, int],
    team: int,
):
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != version:
        return None
    if tuple(payload.get("file_sig", ())) != file_sig:
        return None
    if payload.get("team") != team:
        return None
    return payload


def _atomic_torch_save(payload: dict, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _raise_stale_turn_metadata(
    path: str | Path,
    *,
    expected_n_turns: int,
    actual_n_turns: int,
    source: str,
) -> None:
    raise RuntimeError(
        f"Stale replay metadata for {path}: index reports n_turns={expected_n_turns}, "
        f"but {source} parsed n_turns={actual_n_turns}"
    )


def _winner_from_index_entry(entry: dict) -> int | None:
    ws = entry.get("winner_side")
    if ws in ("A", "a", 0):
        return 0
    if ws in ("B", "b", 1):
        return 1
    return None


def _turns_from_index_entry(entry: dict) -> int:
    turns = entry.get("turns", 0)
    if isinstance(turns, int):
        return max(turns, 0)
    return 0


def _critic_winloss_targets(
    n_turns: int,
    *,
    base_value: float,
    gamma: float = GAMMA,
    ti_delivered: torch.Tensor | None = None,
    ax_delivered: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discounted returns for critic pretraining: resource income + terminal.

    Per-round reward = (ti_delivered[t] + ax_delivered[t] * 4) * REWARD_BUILD_CREDIT,
    plus terminal reward at the last step. Returns are computed via reverse-scan:
      R_t = r_t + γ * R_{t+1}
    """
    AX_CONVERSION_RATE = 4  # GameConstants.AXIONITE_CONVERSION_TITANIUM_RATE

    if n_turns <= 0:
        return torch.zeros(0, dtype=torch.float32)

    rewards = torch.zeros(n_turns, dtype=torch.float32)
    if REWARD_MODE == RewardMode.INCOME and ti_delivered is not None and ti_delivered.shape[0] >= n_turns:
        rewards = ti_delivered[:n_turns].float() * REWARD_INCOME_SCALE
        if ax_delivered is not None and ax_delivered.shape[0] >= n_turns:
            rewards += ax_delivered[:n_turns].float() * AX_CONVERSION_RATE * REWARD_INCOME_SCALE
    if REWARD_MODE == RewardMode.OUTCOME:
        rewards[-1] += float(base_value)

    # Reverse-scan for discounted returns
    returns = torch.zeros(n_turns, dtype=torch.float32)
    R = 0.0
    for t in range(n_turns - 1, -1, -1):
        R = rewards[t].item() + gamma * R
        returns[t] = R
    return returns


def _decode_sequential_replay(
    path: str,
    *,
    team: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode sequential replay buffers into raw tensors on CPU."""
    data = parse_replay_sequential(path, team=team)
    n = data["n_samples"]
    obs_size = data.get("obs_size", 548)
    if n == 0:
        return (
            torch.zeros(0, obs_size, dtype=torch.float32),
            torch.zeros(0, 5, dtype=torch.long),
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
        )

    raw_obs = torch.frombuffer(
        bytearray(data["obs"]), dtype=torch.float32,
    ).reshape(n, obs_size).clone()
    act_raw = bytearray(data["actions"])
    act_stride = len(act_raw) // n
    raw_actions = torch.tensor(
        [act_raw[i * act_stride: i * act_stride + act_stride] for i in range(n)],
        dtype=torch.long,
    )
    if act_stride == 4:
        raw_actions = torch.cat([
            raw_actions[:, :3],
            torch.zeros(n, 1, dtype=torch.long),
            raw_actions[:, 3:],
        ], dim=1)

    bid_raw = bytearray(data["bot_ids"])
    raw_bot_ids = torch.tensor(
        [struct.unpack_from("<i", bid_raw, i * 4)[0] for i in range(n)],
        dtype=torch.int32,
    )
    turn_raw = bytearray(data["turns"])
    raw_turns = torch.tensor(
        [struct.unpack_from("<I", turn_raw, i * 4)[0] for i in range(n)],
        dtype=torch.int32,
    )
    return raw_obs, raw_actions, raw_bot_ids, raw_turns


def _load_working_batch(paths: list[Path]) -> dict[str, dict]:
    """Load winner/turn metadata for many replays with batch fallback."""
    if not paths:
        return {}
    try:
        metas = parse_replays_working_batch([str(path) for path in paths])
        if len(metas) != len(paths):
            raise ValueError("metadata batch size mismatch")
    except Exception:
        metas = [parse_replay_working(str(path)) for path in paths]
    return {
        str(path): (meta if isinstance(meta, dict) else {})
        for path, meta in zip(paths, metas)
    }


def _critic_step_features_from_obs(
    obs: torch.Tensor,
    turns: torch.Tensor,
    *,
    map_w: int,
    map_h: int,
    n_turns: int,
    core_x: int = 0,
    core_y: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-bot centralized critic inputs from sequential replay samples."""
    if obs.shape[0] == 0:
        return (
            torch.zeros(0, dtype=torch.int16),
            torch.zeros(0, 2, dtype=torch.int16),
            torch.zeros(0, UNIT_CONTEXT_DIM, dtype=torch.float16),
            torch.zeros(n_turns + 1, dtype=torch.long),
        )

    step_rounds = turns.to(torch.long).clamp_(0, max(n_turns - 1, 0))
    order = step_rounds.argsort(stable=True)
    step_rounds = step_rounds[order]
    obs = obs[order]

    x_norm = obs[:, _OBS_X_NORM]
    y_norm = obs[:, _OBS_Y_NORM]
    unit_xy = torch.empty(obs.shape[0], 2, dtype=torch.int16)
    unit_xy[:, 0] = torch.clamp((x_norm * map_w + core_x).round().to(torch.long), 0, map_w - 1).to(torch.int16)
    unit_xy[:, 1] = torch.clamp((y_norm * map_h + core_y).round().to(torch.long), 0, map_h - 1).to(torch.int16)

    unit_ctx = torch.empty(obs.shape[0], UNIT_CONTEXT_DIM, dtype=torch.float16)
    unit_ctx[:, 0] = x_norm.to(torch.float16)
    unit_ctx[:, 1] = y_norm.to(torch.float16)
    unit_ctx[:, 2] = obs[:, _OBS_CORE_DX].to(torch.float16)
    unit_ctx[:, 3] = obs[:, _OBS_CORE_DY].to(torch.float16)
    unit_ctx[:, 4] = obs[:, _OBS_ECORE_DX].to(torch.float16)
    unit_ctx[:, 5] = obs[:, _OBS_ECORE_DY].to(torch.float16)
    unit_ctx[:, 6] = (obs[:, _OBS_GLOBAL_START] - obs[:, _OBS_SPAWN_ROUND]).to(torch.float16)

    counts = torch.bincount(step_rounds, minlength=n_turns)
    round_offsets = torch.zeros(n_turns + 1, dtype=torch.long)
    round_offsets[1:] = counts.cumsum(0)
    return step_rounds.to(torch.int16), unit_xy, unit_ctx, round_offsets


def _extract_friendly_core_xy(state0: torch.Tensor) -> tuple[int, int]:
    """Extract the friendly core tile from round-0 replay state channels.

    Channel 11 is core presence (both teams); channel 12 is the team sign
    (+1 friendly, -1 enemy).  Intersect both to find the friendly core.
    """
    if state0.numel() == 0:
        return 0, 0
    friendly_core = (state0[11] > 0.5) & (state0[12] > 0.5)
    hits = torch.nonzero(friendly_core, as_tuple=False)
    if hits.numel() == 0:
        # Fallback: any core
        hits = torch.nonzero(state0[11] > 0.5, as_tuple=False)
        if hits.numel() == 0:
            return 0, 0
    cy, cx = hits[0].tolist()
    return int(cx), int(cy)


def _sort_by_bot_and_turn(
    obs: torch.Tensor,
    bot_ids: torch.Tensor,
    turns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort replay steps into recurrent actor order: (bot_id, turn)."""
    if obs.shape[0] == 0:
        return obs, bot_ids, turns
    sort_keys = bot_ids.long() * 100000 + turns.long()
    order = sort_keys.argsort()
    return obs[order].contiguous(), bot_ids[order].contiguous(), turns[order].contiguous()


class ActorBCDataset:
    """Lazy-loading actor BC dataset from ladder replays.

    Stores lightweight metadata per replay. Loads and parses replays on demand
    via load_game_batch(), using an LRU cache to avoid redundant parsing.
    """

    def __init__(
        self,
        replay_dir: str | Path,
        index_path: str | Path | None = None,
        min_rating: float = 0.0,
        winner_only: bool = True,
        builds_only: bool = False,
        cache_size: int = 4,
    ):
        self.replay_dir = Path(replay_dir)
        self.cache_dir = _cache_dir(self.replay_dir)
        self.winner_only = winner_only
        self.builds_only = builds_only

        if index_path is None:
            index_path = self.replay_dir / "index.jsonl"
        entries = _load_index(index_path, min_rating)

        # Lightweight metadata per (file, team)
        self.games: list[dict] = []
        self.total_samples = 0
        self._cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._round_cache: dict[str, dict[str, torch.Tensor | int]] = {}
        self._cache_order: collections.deque[str] = collections.deque()
        self._round_cache_order: collections.deque[str] = collections.deque()
        self._cache_size = cache_size
        self._cache_lock = Lock()
        self._scan_metadata(entries)

    def _scan_metadata(self, entries: list[dict]):
        seen: set[str] = set()
        pending_meta: list[Path] = []
        scanned: list[tuple[Path, int | None, int]] = []
        for entry in entries:
            fname = entry["file"]
            if fname in seen:
                continue
            seen.add(fname)
            path = self.replay_dir / fname
            if not path.exists():
                continue

            winner = _winner_from_index_entry(entry)
            n_turns = _turns_from_index_entry(entry)
            scanned.append((path, winner, n_turns))
            if n_turns <= 0 or winner is None:
                pending_meta.append(path)

        meta_by_path = _load_working_batch(pending_meta)

        for path, winner, n_turns in scanned:
            if n_turns <= 0 or winner is None:
                meta = meta_by_path.get(str(path), {})
                if winner is None:
                    winner = meta.get("winner")
                if n_turns <= 0:
                    n_turns = meta.get("n_turns", 0)
            if n_turns == 0:
                continue

            if self.winner_only:
                if winner is None:
                    continue  # skip draws
                teams = [winner]
            else:
                teams = [0, 1]

            for team in teams:
                self.games.append({
                    "path": str(path),
                    "team": team,
                    "n_turns": n_turns,
                    "winner": "a" if winner == team else "b" if winner is not None else None,
                })

        # Estimate total samples (actual count determined at parse time)
        self.total_samples = sum(g["n_turns"] for g in self.games)

    def _get_parsed(
        self, path: str, team: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load or retrieve cached (obs, actions, bot_ids, turns) for a replay.

        Returns are already sorted by (bot_id, turn) for BPTT.
        """
        key = f"{path}:{team}"
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached

        path_obj = Path(path)
        file_sig = _file_signature(path_obj)
        cache_path = _cache_path(self.cache_dir, "actor", path_obj, team)
        payload = _load_torch_cache(
            cache_path,
            version=_ACTOR_CACHE_VERSION,
            file_sig=file_sig,
            team=team,
        )
        if payload is None:
            raw_obs, raw_actions, raw_bot_ids, raw_turns = _decode_sequential_replay(
                path, team=team,
            )
            if raw_obs.shape[0] > 0:
                # Sort by (bot_id, turn) for BPTT
                sort_keys = raw_bot_ids.long() * 100000 + raw_turns.long()
                order = sort_keys.argsort()
                raw_obs = raw_obs[order].contiguous()
                raw_actions = raw_actions[order].contiguous()
                raw_bot_ids = raw_bot_ids[order].contiguous()
                raw_turns = raw_turns[order].contiguous()

            payload = {
                "version": _ACTOR_CACHE_VERSION,
                "file_sig": list(file_sig),
                "team": team,
                "obs": raw_obs,
                "actions": raw_actions,
                "bot_ids": raw_bot_ids,
                "turns": raw_turns,
            }
            try:
                _atomic_torch_save(payload, cache_path)
            except OSError:
                pass
        else:
            raw_obs = payload["obs"]
            raw_actions = payload["actions"]
            raw_bot_ids = payload["bot_ids"]
            raw_turns = payload["turns"]

        result = (raw_obs, raw_actions, raw_bot_ids, raw_turns)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            self._cache[key] = result
            self._cache_order.append(key)
            while self._cache_size > 0 and len(self._cache_order) > self._cache_size:
                evict = self._cache_order.popleft()
                self._cache.pop(evict, None)
        return result

    def _get_round_major(
        self,
        path: str,
        team: int,
        *,
        n_rounds_hint: int | None = None,
    ) -> dict[str, torch.Tensor | int]:
        """Load or derive the round-major actor layout used by TBPTT staging."""
        key = f"{path}:{team}"
        with self._cache_lock:
            cached = self._round_cache.get(key)
        if cached is not None:
            return cached

        path_obj = Path(path)
        file_sig = _file_signature(path_obj)
        cache_path = _cache_path(self.cache_dir, "actor", path_obj, team)
        payload = _load_torch_cache(
            cache_path,
            version=_ACTOR_CACHE_VERSION,
            file_sig=file_sig,
            team=team,
        )

        required = {
            "round_obs",
            "round_actions",
            "round_bot_slots",
            "round_offsets",
            "round_n_slots",
            "round_bt_mask",
            "round_bd_mask",
            "round_harv_bd_mask",
        }
        if payload is not None and required.issubset(payload):
            round_obs = payload["round_obs"]
            round_actions = payload["round_actions"]
            round_bot_slots = payload["round_bot_slots"]
            round_offsets = payload["round_offsets"]
            round_n_slots = int(payload["round_n_slots"])
            round_bt_mask = payload["round_bt_mask"]
            round_bd_mask = payload["round_bd_mask"]
            round_harv_bd_mask = payload["round_harv_bd_mask"]
        else:
            obs, actions, bot_ids, turns = self._get_parsed(path, team)
            if obs.shape[0] > 0:
                order = turns.argsort(stable=True)
                round_obs = obs[order].to(torch.float16).contiguous()
                round_actions = actions[order].to(torch.int16).contiguous()
                round_bot_ids = bot_ids[order].contiguous()
                round_turns = turns[order].long().contiguous()
                observed_rounds = int(round_turns[-1].item()) + 1
                n_rounds = max(observed_rounds, n_rounds_hint or 0)
                counts = torch.bincount(round_turns, minlength=n_rounds)
                round_offsets = torch.zeros(n_rounds + 1, dtype=torch.long)
                round_offsets[1:] = counts.cumsum(0)
                unique_bots, round_bot_slots = round_bot_ids.unique(sorted=True, return_inverse=True)
                round_n_slots = int(unique_bots.shape[0])
                round_bot_slots = round_bot_slots.to(torch.long).contiguous()
                obs_f = round_obs.float()
                round_bt_mask, round_bd_mask = compute_action_masks(obs_f)
                round_harv_bd_mask = compute_harvester_build_dir_mask(obs_f, round_bd_mask)
                round_bt_mask = round_bt_mask.clone()
                round_bt_mask[:, HARVESTER_BUILD_TYPE] &= round_harv_bd_mask.any(dim=1)
            else:
                n_rounds = max(n_rounds_hint or 0, 0)
                round_obs = torch.zeros(0, 548, dtype=torch.float16)
                round_actions = torch.zeros(0, 5, dtype=torch.int16)
                round_bot_slots = torch.zeros(0, dtype=torch.long)
                round_offsets = torch.zeros(n_rounds + 1, dtype=torch.long)
                round_n_slots = 0
                round_bt_mask = torch.zeros(0, N_BUILD, dtype=torch.bool)
                round_bd_mask = torch.zeros(0, N_DIR, dtype=torch.bool)
                round_harv_bd_mask = torch.zeros(0, N_DIR, dtype=torch.bool)

            if payload is None:
                payload = {
                    "version": _ACTOR_CACHE_VERSION,
                    "file_sig": list(file_sig),
                    "team": team,
                    "obs": obs,
                    "actions": actions,
                    "bot_ids": bot_ids,
                    "turns": turns,
                }
            payload.update({
                "round_obs": round_obs,
                "round_actions": round_actions,
                "round_bot_slots": round_bot_slots,
                "round_offsets": round_offsets,
                "round_n_slots": round_n_slots,
                "round_bt_mask": round_bt_mask,
                "round_bd_mask": round_bd_mask,
                "round_harv_bd_mask": round_harv_bd_mask,
            })
            try:
                _atomic_torch_save(payload, cache_path)
            except OSError:
                pass

        result = {
            "obs": round_obs,
            "actions": round_actions,
            "bot_slots": round_bot_slots,
            "round_offsets": round_offsets,
            "n_slots": round_n_slots,
            "bt_mask": round_bt_mask,
            "bd_mask": round_bd_mask,
            "harv_bd_mask": round_harv_bd_mask,
        }
        with self._cache_lock:
            cached = self._round_cache.get(key)
            if cached is not None:
                return cached
            self._round_cache[key] = result
            self._round_cache_order.append(key)
            while self._cache_size > 0 and len(self._round_cache_order) > self._cache_size:
                evict = self._round_cache_order.popleft()
                self._round_cache.pop(evict, None)
        return result

    @property
    def n_games(self) -> int:
        return len(self.games)

    def load_game_batch(
        self,
        n_games: int,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load a flat actor batch from the first n_games dataset entries."""
        obs_batches: list[torch.Tensor] = []
        action_batches: list[torch.Tensor] = []
        for game in self.games[:n_games]:
            obs, actions, _bot_ids, _turns = self._get_parsed(
                game["path"], game["team"],
            )
            if obs.shape[0] == 0:
                continue
            obs_batches.append(obs)
            action_batches.append(actions)

        if not obs_batches:
            dev = torch.device(device)
            return (
                torch.zeros(0, 548, dtype=torch.float32, device=dev),
                torch.zeros(0, 5, dtype=torch.long, device=dev),
            )

        return (
            torch.cat(obs_batches, dim=0).to(device=device),
            torch.cat(action_batches, dim=0).to(device=device),
        )

    def load_all_trajectories(self) -> list[dict]:
        """Load all games and return per-bot trajectories sorted by turn.

        Each trajectory is a dict with:
            obs:     (T, 498) on CPU
            actions: (T, 5) long on CPU
        where T is the number of turns this bot acted.
        Kept on CPU to avoid OOM; training loop moves to GPU per trajectory.
        """
        trajectories: list[dict] = []
        for game in self.games:
            obs, actions, bot_ids, turns = self._get_parsed(
                game["path"], game["team"],
            )
            if obs.shape[0] == 0:
                continue
            # Data is already sorted by (bot_id, turn) from _get_parsed.
            # Split into per-bot trajectories.
            unique_bots = bot_ids.unique()
            for bid in unique_bots:
                mask = bot_ids == bid
                t_obs = obs[mask].half()  # float16 to halve RAM
                t_act = actions[mask].to(torch.int16)  # int16 saves 6x vs long
                if t_obs.shape[0] > 0:
                    trajectories.append({"obs": t_obs, "actions": t_act})
        return trajectories


class CriticBCDataset:
    """Lazy-loading critic dataset from ladder replays.

    Stores only file paths and metadata. Loads replay bytes on demand
    via sample_batch(), using an LRU cache to avoid redundant parsing.
    """

    def __init__(
        self,
        replay_dir: str | Path,
        index_path: str | Path | None = None,
        min_rating: float = 0.0,
        cache_size: int = 256,
    ):
        self.replay_dir = Path(replay_dir)
        self.cache_dir = _cache_dir(self.replay_dir)

        if index_path is None:
            index_path = self.replay_dir / "index.jsonl"
        entries = _load_index(index_path, min_rating)

        # Lightweight metadata per (file, team) — no raw bytes stored
        self.games: list[dict] = []
        self.total_turns = 0
        self._cache: dict[str, dict] = {}  # key -> parsed data
        self._cache_order: collections.deque[str] = collections.deque()
        self._cache_size = cache_size
        self._cache_lock = Lock()
        self._game_lookup: dict[str, dict] = {}
        self._scan_metadata(entries)

    def _scan_metadata(self, entries: list[dict]):
        seen: set[str] = set()
        pending_meta: list[Path] = []
        scanned: list[tuple[Path, int | None, int]] = []
        for entry in entries:
            fname = entry["file"]
            if fname in seen:
                continue
            seen.add(fname)
            path = self.replay_dir / fname
            if not path.exists():
                continue

            winner = _winner_from_index_entry(entry)
            n_turns = _turns_from_index_entry(entry)
            scanned.append((path, winner, n_turns))
            if n_turns <= 0 or winner is None:
                pending_meta.append(path)

        meta_by_path = _load_working_batch(pending_meta)

        for path, winner, n_turns in scanned:
            if n_turns <= 0 or winner is None:
                meta = meta_by_path.get(str(path), {})
                if winner is None:
                    winner = meta.get("winner")
                if n_turns <= 0:
                    n_turns = meta.get("n_turns", 0)
            if n_turns == 0:
                continue

            for team in (0, 1):
                if winner == team:
                    base_value = 1.0
                elif winner is not None:
                    base_value = -1.0
                else:
                    base_value = 0.0

                game = {
                    "path": str(path),
                    "team": team,
                    "n_turns": n_turns,
                    "base_value": base_value,
                    "winner": "a" if winner == team else "b" if winner is not None else None,
                }
                self.games.append(game)
                self._game_lookup[f"{path}:{team}"] = game
                self.total_turns += n_turns

    def _get_buffers(self, path: str, team: int) -> dict:
        """Load or retrieve cached replay data (buffers + map dims)."""
        key = f"{path}:{team}"
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        path_obj = Path(path)
        file_sig = _file_signature(path_obj)
        cache_path = _cache_path(self.cache_dir, "critic", path_obj, team)
        payload = _load_torch_cache(
            cache_path,
            version=_CRITIC_CACHE_VERSION,
            file_sig=file_sig,
            team=team,
        )
        game_meta = self._game_lookup.get(key)
        expected_n_turns = None if game_meta is None else int(game_meta["n_turns"])
        base_value = 0.0 if game_meta is None else float(game_meta["base_value"])
        if payload is None:
            data = parse_replay_full(path, team=team)
            full_turns = data["n_turns"]
            if expected_n_turns is not None and full_turns != expected_n_turns:
                _raise_stale_turn_metadata(
                    path,
                    expected_n_turns=expected_n_turns,
                    actual_n_turns=full_turns,
                    source="parse_replay_full",
                )
            map_w = data["map_w"]
            map_h = data["map_h"]
            n_channels = data.get("n_channels", 19)
            n_scalars = data.get("n_scalars", 17)
            states = torch.frombuffer(
                bytearray(data["states"]), dtype=torch.float32,
            ).reshape(full_turns, n_channels, map_h, map_w).to(torch.float16).clone()
            scalars = torch.frombuffer(
                bytearray(data["scalars"]), dtype=torch.float32,
            ).reshape(full_turns, n_scalars).to(torch.float16).clone()
            core_x, core_y = _extract_friendly_core_xy(states[0])
            seq_obs, _seq_actions, seq_bot_ids, seq_turns = _decode_sequential_replay(
                path, team=team,
            )
            actor_obs, actor_bot_ids, actor_turns = _sort_by_bot_and_turn(
                seq_obs, seq_bot_ids, seq_turns,
            )
            actor_step_order = actor_turns.to(torch.long).clamp_(0, max(full_turns - 1, 0)).argsort(stable=True)
            step_rounds, step_unit_xy, step_unit_ctx, round_offsets = _critic_step_features_from_obs(
                actor_obs[actor_step_order],
                actor_turns[actor_step_order],
                map_w=map_w,
                map_h=map_h,
                n_turns=full_turns,
                core_x=core_x,
                core_y=core_y,
            )
            # Extract per-round titanium/axionite delivered for this team
            ti_buf = data.get("titanium_delivered", b"")
            ti_delivered = None
            if ti_buf and len(ti_buf) >= full_turns * 8:
                ti_raw = torch.frombuffer(bytearray(ti_buf), dtype=torch.int32).reshape(-1, 2)
                ti_delivered = ti_raw[:full_turns, team].clone()

            ax_buf = data.get("axionite_delivered", b"")
            ax_delivered = None
            if ax_buf and len(ax_buf) >= full_turns * 8:
                ax_raw = torch.frombuffer(bytearray(ax_buf), dtype=torch.int32).reshape(-1, 2)
                ax_delivered = ax_raw[:full_turns, team].clone()

            round_targets = _critic_winloss_targets(
                full_turns,
                base_value=base_value,
                ti_delivered=ti_delivered,
                ax_delivered=ax_delivered,
            )
            step_targets = round_targets.index_select(0, step_rounds.to(torch.long))
            entry = {
                "version": _CRITIC_CACHE_VERSION,
                "file_sig": list(file_sig),
                "team": team,
                "n_turns": full_turns,
                "states": states,
                "scalars": scalars,
                "targets": round_targets,
                "step_rounds": step_rounds,
                "step_unit_xy": step_unit_xy,
                "step_unit_ctx": step_unit_ctx,
                "step_targets": step_targets,
                "round_offsets": round_offsets,
                "actor_obs": actor_obs.to(torch.float16),
                "actor_bot_ids": actor_bot_ids,
                "actor_step_order": actor_step_order,
                "map_w": map_w,
                "map_h": map_h,
                "core_x": core_x,
                "core_y": core_y,
                "n_channels": n_channels,
                "n_scalars": n_scalars,
            }
            try:
                _atomic_torch_save(entry, cache_path)
            except OSError:
                pass
        else:
            cached_n_turns = int(payload.get("n_turns", payload["states"].shape[0]))
            if expected_n_turns is not None and cached_n_turns != expected_n_turns:
                _raise_stale_turn_metadata(
                    path,
                    expected_n_turns=expected_n_turns,
                    actual_n_turns=cached_n_turns,
                    source="cached critic replay",
                )
            entry = {
                "n_turns": cached_n_turns,
                "states": payload["states"],
                "scalars": payload["scalars"],
                "targets": payload["targets"],
                "step_rounds": payload["step_rounds"],
                "step_unit_xy": payload["step_unit_xy"],
                "step_unit_ctx": payload["step_unit_ctx"],
                "step_targets": payload["step_targets"],
                "round_offsets": payload["round_offsets"],
                "map_w": payload["map_w"],
                "map_h": payload["map_h"],
                "core_x": payload.get("core_x", 0),
                "core_y": payload.get("core_y", 0),
                "n_channels": payload["n_channels"],
                "n_scalars": payload["n_scalars"],
            }
            # Prefer baked-in actor latents; fall back to raw staging tensors
            if "step_actor_latent" in payload:
                entry["step_actor_latent"] = payload["step_actor_latent"]
            else:
                entry["actor_obs"] = payload["actor_obs"]
                entry["actor_bot_ids"] = payload["actor_bot_ids"]
                entry["actor_step_order"] = payload["actor_step_order"]
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            self._cache[key] = entry
            self._cache_order.append(key)
            while self._cache_size > 0 and len(self._cache_order) > self._cache_size:
                evict = self._cache_order.popleft()
                self._cache.pop(evict, None)
        return entry

    @property
    def n_games(self) -> int:
        return len(self.games)

    def load_game_batch(
        self,
        n_games: int = 4,
        max_turns_per_game: int = 0,
        max_map_w: int = 50,
        max_map_h: int = 50,
        n_channels: int = 19,
        n_scalars: int = 17,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load full game states for a random subset of games.

        Bulk-loads all turns from a few games, pads on CPU, then optionally
        performs one bulk transfer to the target device.

        Returns:
            states:  (N, C, max_H, max_W)
            masks:   (N, 1, max_H, max_W)
            scalars: (N, n_scalars)
            values:  (N,)
        """
        import random

        selected = random.sample(self.games, min(n_games, len(self.games)))
        return self.load_games(
            selected,
            max_turns_per_game=max_turns_per_game,
            max_map_w=max_map_w,
            max_map_h=max_map_h,
            n_channels=n_channels,
            n_scalars=n_scalars,
            device=device,
            pin_memory=pin_memory,
        )

    def load_games(
        self,
        selected: list[dict],
        *,
        max_turns_per_game: int = 0,
        max_map_w: int = 50,
        max_map_h: int = 50,
        n_channels: int = 19,
        n_scalars: int = 17,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a specific list of replay/team entries into one padded batch."""

        # Pre-compute total turns for one contiguous allocation
        game_infos = []
        total_turns = 0
        for game in selected:
            bufs = self._get_buffers(game["path"], game["team"])
            nt = game["n_turns"]
            if max_turns_per_game > 0 and nt > max_turns_per_game:
                nt = max_turns_per_game
            game_infos.append((game, bufs, nt))
            total_turns += nt

        use_pinned_staging = pin_memory and torch.cuda.is_available()

        # Build one CPU batch first, then issue a single bulk transfer per tensor.
        states = torch.zeros(
            total_turns, n_channels, max_map_h, max_map_w,
            dtype=torch.float32, pin_memory=use_pinned_staging,
        )
        masks = torch.zeros(
            total_turns, 1, max_map_h, max_map_w,
            dtype=torch.float32, pin_memory=use_pinned_staging,
        )
        scalars = torch.zeros(
            total_turns, n_scalars,
            dtype=torch.float32, pin_memory=use_pinned_staging,
        )
        values = torch.zeros(
            total_turns, dtype=torch.float32, pin_memory=use_pinned_staging,
        )

        offset = 0
        for game, bufs, nt in game_infos:
            mw, mh = bufs["map_w"], bufs["map_h"]
            nc = bufs["n_channels"]
            ns = bufs["n_scalars"]
            full_turns = game["n_turns"]

            # Bulk decode
            game_states = bufs["states"]
            game_scalars = bufs["scalars"]

            if nt < full_turns:
                idx = torch.randperm(full_turns)[:nt].sort().values
                game_states = game_states[idx]
                game_scalars = game_scalars[idx]

            # Direct-write into the pre-allocated tensor.
            states[offset:offset + nt, :nc, :mh, :mw] = game_states
            masks[offset:offset + nt, :, :mh, :mw] = 1.0
            scalars[offset:offset + nt, :ns] = game_scalars

            game_targets = bufs["targets"]
            if nt < full_turns:
                game_targets = game_targets[idx]
            values[offset:offset + nt] = game_targets

            offset += nt

        if device != "cpu":
            states = states.to(device=device, non_blocking=use_pinned_staging)
            masks = masks.to(device=device, non_blocking=use_pinned_staging)
            scalars = scalars.to(device=device, non_blocking=use_pinned_staging)
            values = values.to(device=device, non_blocking=use_pinned_staging)

        return states, masks, scalars, values

    def load_staged_games(
        self,
        selected: list[dict],
        *,
        pin_memory: bool = False,
    ) -> list[dict]:
        """Load game metadata plus tensor references for critic staging.

        We intentionally avoid pinning or cloning whole-game tensors here.
        The critic trainer materializes pinned minibatch buffers later, so
        duplicating full staged games on the host only wastes RAM.
        """
        _ = pin_memory

        def _stage_game(game: dict) -> dict:
            bufs = self._get_buffers(game["path"], game["team"])
            return {
                "path": game["path"],
                "team": game["team"],
                "n_turns": int(game["n_turns"]),
                "states": bufs["states"],
                "scalars": bufs["scalars"],
                "targets": bufs["targets"],
                "step_rounds": bufs["step_rounds"],
                "step_unit_xy": bufs["step_unit_xy"],
                "step_unit_ctx": bufs["step_unit_ctx"],
                "step_targets": bufs["step_targets"],
                "round_offsets": bufs["round_offsets"],
                "map_w": int(bufs["map_w"]),
                "map_h": int(bufs["map_h"]),
                "n_channels": int(bufs["n_channels"]),
                "n_scalars": int(bufs["n_scalars"]),
            }

        if len(selected) <= 1:
            return [_stage_game(game) for game in selected]

        workers = min(8, max(1, len(selected)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_stage_game, selected))

    def load_game_window(
        self,
        ordered_games: list[dict],
        *,
        max_staging_turns: int,
        max_map_w: int,
        max_map_h: int,
        n_channels: int,
        n_scalars: int,
        device: str = "cpu",
        pin_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Load consecutive games until a turn budget is reached."""
        if max_staging_turns <= 0:
            raise ValueError("max_staging_turns must be positive")
        if not ordered_games:
            empty_states = torch.zeros(0, n_channels, max_map_h, max_map_w, dtype=torch.float32)
            empty_masks = torch.zeros(0, 1, max_map_h, max_map_w, dtype=torch.float32)
            empty_scalars = torch.zeros(0, n_scalars, dtype=torch.float32)
            empty_values = torch.zeros(0, dtype=torch.float32)
            return empty_states, empty_masks, empty_scalars, empty_values, 0

        selected: list[dict] = []
        total_turns = 0
        for game in ordered_games:
            nt = int(game["n_turns"])
            if nt <= 0:
                continue
            if selected and total_turns + nt > max_staging_turns:
                break
            selected.append(game)
            total_turns += nt
            if total_turns >= max_staging_turns:
                break

        if not selected:
            selected = [ordered_games[0]]

        states, masks, scalars, values = self.load_games(
            selected,
            max_turns_per_game=0,
            max_map_w=max_map_w,
            max_map_h=max_map_h,
            n_channels=n_channels,
            n_scalars=n_scalars,
            device=device,
            pin_memory=pin_memory,
        )
        return states, masks, scalars, values, len(selected)


def _load_index(path: str | Path, min_rating: float = 0.0) -> list[dict]:
    """Load JSONL index, filtering by minimum average rating."""
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        avg_rating = (
            (entry.get("team_a_rating") or 0) + (entry.get("team_b_rating") or 0)
        ) / 2.0
        if avg_rating >= min_rating:
            entries.append(entry)
    return entries
