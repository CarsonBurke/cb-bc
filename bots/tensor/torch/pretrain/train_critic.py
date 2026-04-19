#!/usr/bin/env python3
"""Pretrain the centralized critic from ladder replays.

Uses discounted terminal win/loss returns so critic pretraining matches the
post-training reward objective.

Usage:
    python bots/tensor/torch/pretrain/train_critic.py --replay-dir replay_data --epochs 30
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_TORCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TORCH_DIR))

from model import (
    ACTOR_LATENT_DIM,
    BuilderMLP,
    CentralizedCritic,
    DEVICE, GLOBAL_MAP_CHANNELS, GLOBAL_MAP_MAX_W, GLOBAL_MAP_MAX_H,
    GLOBAL_SCALAR_DIM, H_DIM, OBS_PREFIX_SIZE, UNIT_CONTEXT_DIM,
    soft_cross_entropy_from_logits,
    value_target_probs,
    _make_bucket_centers, _normalize_actor_latent_dim,
    win_probability,
)
from normuon import make_normuon_optimizer

from training_utils import jsonl_line, jsonl_init
from dataset import CriticBCDataset, _cache_path, _file_signature, _atomic_torch_save, _CRITIC_CACHE_VERSION
from pretrain.gpu import (
    autocast_context,
    configure_cuda_runtime,
    maybe_compile,
    prepare_spatial_batch,
    resolve_cuda_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "replay_data"
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
DEFAULT_PRETRAIN_CHUNK_SIZE = 4096


def _atomic_torch_save(payload: dict, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


class CriticPretrainStep(nn.Module):
    """Compiled replay-critic training step body."""

    def __init__(self, critic: CentralizedCritic):
        super().__init__()
        self.critic = critic

    def forward(
        self,
        states: torch.Tensor,
        masks: torch.Tensor,
        scalars: torch.Tensor,
        round_idx: torch.Tensor,
        unit_xy: torch.Tensor,
        unit_ctx: torch.Tensor,
        actor_latent: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.critic.encode_map(states, masks)
        values, logits = self.critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent, actions,
        )
        return values, logits


@dataclass
class _CriticProfile:
    prefetch_wait_ms: float = 0.0
    transfer_submit_ms: float = 0.0
    transfer_gpu_ms: float = 0.0
    forward_gpu_ms: float = 0.0
    backward_gpu_ms: float = 0.0
    optimizer_gpu_ms: float = 0.0
    post_step_ms: float = 0.0
    batch_rounds_sum: int = 0
    batch_steps_sum: int = 0
    batch_rounds_min: int = 1 << 30
    batch_rounds_max: int = 0
    batch_steps_min: int = 1 << 30
    batch_steps_max: int = 0
    steps: int = 0

    def record_step(
        self,
        *,
        prefetch_wait_ms: float,
        transfer_submit_ms: float,
        transfer_gpu_ms: float,
        forward_gpu_ms: float,
        backward_gpu_ms: float,
        optimizer_gpu_ms: float,
        post_step_ms: float,
        batch_rounds: int,
        batch_steps: int,
    ) -> None:
        self.prefetch_wait_ms += prefetch_wait_ms
        self.transfer_submit_ms += transfer_submit_ms
        self.transfer_gpu_ms += transfer_gpu_ms
        self.forward_gpu_ms += forward_gpu_ms
        self.backward_gpu_ms += backward_gpu_ms
        self.optimizer_gpu_ms += optimizer_gpu_ms
        self.post_step_ms += post_step_ms
        self.batch_rounds_sum += batch_rounds
        self.batch_steps_sum += batch_steps
        self.batch_rounds_min = min(self.batch_rounds_min, batch_rounds)
        self.batch_rounds_max = max(self.batch_rounds_max, batch_rounds)
        self.batch_steps_min = min(self.batch_steps_min, batch_steps)
        self.batch_steps_max = max(self.batch_steps_max, batch_steps)
        self.steps += 1

    def summary(self) -> dict[str, float | int]:
        if self.steps <= 0:
            return {"steps": 0}
        steps = float(self.steps)
        return {
            "steps": self.steps,
            "prefetch_wait_ms": round(self.prefetch_wait_ms / steps, 3),
            "transfer_submit_ms": round(self.transfer_submit_ms / steps, 3),
            "transfer_gpu_ms": round(self.transfer_gpu_ms / steps, 3),
            "forward_gpu_ms": round(self.forward_gpu_ms / steps, 3),
            "backward_gpu_ms": round(self.backward_gpu_ms / steps, 3),
            "optimizer_gpu_ms": round(self.optimizer_gpu_ms / steps, 3),
            "post_step_ms": round(self.post_step_ms / steps, 3),
            "batch_rounds_avg": round(self.batch_rounds_sum / steps, 1),
            "batch_rounds_min": self.batch_rounds_min,
            "batch_rounds_max": self.batch_rounds_max,
            "batch_steps_avg": round(self.batch_steps_sum / steps, 1),
            "batch_steps_min": self.batch_steps_min,
            "batch_steps_max": self.batch_steps_max,
        }


def _aggregate_win_confidence(
    win_prob: torch.Tensor,
    game_idx_cpu: torch.Tensor,
    round_num_cpu: torch.Tensor,
) -> dict[int, dict[str, float | int]]:
    """Aggregate per-step win probabilities into one per-game dashboard point."""
    if win_prob.numel() == 0:
        return {}
    game_idx = game_idx_cpu.to(torch.long)
    round_num = round_num_cpu.to(torch.long)
    out: dict[int, dict[str, float | int]] = {}
    for gi in game_idx.unique(sorted=True).tolist():
        mask = game_idx == gi
        if not bool(mask.any()):
            continue
        game_rounds = round_num[mask]
        latest_round = int(game_rounds.max().item())
        latest_mask = mask & (round_num == latest_round)
        out[int(gi)] = {
            "win_confidence": float(win_prob[latest_mask].mean().item()),
            "round_offset": latest_round,
        }
    return out


def _explained_var_from_sums(
    sum_y: torch.Tensor,
    sum_y2: torch.Tensor,
    sum_err: torch.Tensor,
    sum_err2: torch.Tensor,
    count: int,
) -> float:
    """Match PPO's explained-variance statistic from running sums."""
    if count <= 0:
        return 0.0
    n = float(count)
    var_y = (sum_y2 / n - (sum_y / n) ** 2).item()
    if var_y <= 0.0:
        return 0.0
    var_err = (sum_err2 / n - (sum_err / n) ** 2).item()
    return 1.0 - (var_err / var_y)


def _empty_staging_tensor(shape: tuple[int, ...], dtype: torch.dtype, *, pin_memory: bool) -> torch.Tensor:
    if pin_memory and DEVICE.type == "cuda":
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.empty(shape, dtype=dtype)


@dataclass
class _CriticBatchBuffers:
    round_states: torch.Tensor | None = None
    round_masks: torch.Tensor | None = None
    round_scalars: torch.Tensor | None = None
    step_round_idx: torch.Tensor | None = None
    step_unit_xy: torch.Tensor | None = None
    step_unit_ctx: torch.Tensor | None = None
    step_actor_latent: torch.Tensor | None = None
    step_targets: torch.Tensor | None = None
    step_game_idx: torch.Tensor | None = None
    step_round_num: torch.Tensor | None = None

    def ensure(
        self,
        *,
        total_rounds: int,
        total_steps: int,
        pin_memory: bool,
        need_game_log: bool,
    ) -> None:
        if self.round_states is None or self.round_states.shape[0] < total_rounds:
            self.round_states = _empty_staging_tensor(
                (total_rounds, GLOBAL_MAP_CHANNELS, GLOBAL_MAP_MAX_H, GLOBAL_MAP_MAX_W),
                torch.float32,
                pin_memory=pin_memory,
            )
            self.round_masks = _empty_staging_tensor(
                (total_rounds, 1, GLOBAL_MAP_MAX_H, GLOBAL_MAP_MAX_W),
                torch.float32,
                pin_memory=pin_memory,
            )
            self.round_scalars = _empty_staging_tensor(
                (total_rounds, GLOBAL_SCALAR_DIM),
                torch.float32,
                pin_memory=pin_memory,
            )
        if self.step_round_idx is None or self.step_round_idx.shape[0] < total_steps:
            self.step_round_idx = _empty_staging_tensor((total_steps,), torch.long, pin_memory=pin_memory)
            self.step_unit_xy = _empty_staging_tensor((total_steps, 2), torch.int16, pin_memory=pin_memory)
            self.step_unit_ctx = _empty_staging_tensor((total_steps, UNIT_CONTEXT_DIM), torch.float16, pin_memory=pin_memory)
            self.step_actor_latent = _empty_staging_tensor((total_steps, ACTOR_LATENT_DIM), torch.float16, pin_memory=pin_memory)
            self.step_targets = _empty_staging_tensor((total_steps,), torch.float32, pin_memory=pin_memory)
        if need_game_log and (self.step_game_idx is None or self.step_game_idx.shape[0] < total_steps):
            self.step_game_idx = _empty_staging_tensor((total_steps,), torch.int32, pin_memory=pin_memory)
            self.step_round_num = _empty_staging_tensor((total_steps,), torch.int32, pin_memory=pin_memory)


def _compute_actor_latents_for_replay(
    actor: BuilderMLP,
    obs_pre_bt: torch.Tensor,
    bot_ids_bt: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Replay actor latents in bot-major recurrent order. Runs on `device`."""
    if obs_pre_bt.shape[0] == 0:
        return torch.zeros(0, ACTOR_LATENT_DIM, dtype=torch.float16)
    if device is None:
        device = DEVICE

    with torch.no_grad():
        obs_pre = obs_pre_bt.to(device=device, dtype=torch.float32)
        bids = bot_ids_bt.to(device=device, dtype=torch.int32)

        new_segment = torch.ones(obs_pre.shape[0], dtype=torch.bool, device=device)
        new_segment[1:] = bids[1:] != bids[:-1]
        seg_starts = new_segment.nonzero(as_tuple=False).squeeze(1)
        seg_ends = torch.cat([seg_starts[1:], bids.new_tensor([obs_pre.shape[0]], dtype=torch.long)])
        seg_lens = seg_ends - seg_starts
        max_len = int(seg_lens.max().item())

        h_live = torch.zeros(seg_starts.shape[0], H_DIM, device=device)
        actor_latent = torch.empty(obs_pre.shape[0], ACTOR_LATENT_DIM, device=device)

        for t in range(max_len):
            active = seg_lens > t
            active_ids = active.nonzero(as_tuple=False).squeeze(1)
            step_idx = seg_starts[active_ids] + t
            h_prev = h_live[active_ids]
            raw_latent = actor._latent_from_obs_pre(obs_pre[step_idx], h_prev)
            actor_latent[step_idx] = _normalize_actor_latent_dim(raw_latent)
            h_gate, h_cand = actor._project_recurrent_heads(raw_latent, h_prev)
            z = torch.sigmoid(h_gate)
            h_live.index_copy_(0, active_ids, ((1.0 - z) * h_prev + z * h_cand).float())

    return actor_latent.to(dtype=torch.float16).cpu()


def _compute_actor_obs_pre(
    actor: BuilderMLP,
    obs_bt_cpu: torch.Tensor,
    device: torch.device | None = None,
    chunk_size: int = DEFAULT_PRETRAIN_CHUNK_SIZE,
) -> torch.Tensor:
    """Project replay observations through the actor's obs-only encoder in chunks."""
    if obs_bt_cpu.shape[0] == 0:
        enc_dim = OBS_PREFIX_SIZE
        return torch.zeros(0, enc_dim, dtype=torch.float16)
    if device is None:
        device = DEVICE

    enc_dim = OBS_PREFIX_SIZE
    obs_pre_cpu = torch.empty(obs_bt_cpu.shape[0], enc_dim, dtype=torch.float16)

    with torch.no_grad():
        for start in range(0, obs_bt_cpu.shape[0], chunk_size):
            end = min(start + chunk_size, obs_bt_cpu.shape[0])
            obs = obs_bt_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            obs_pre = actor._obs_encoder_pre(obs)
            obs_pre_cpu[start:end] = obs_pre.to(dtype=torch.float16).cpu()

    return obs_pre_cpu


def _compute_actor_obs_pre_for_replay(
    actor: BuilderMLP,
    obs_bt_cpu: torch.Tensor,
    chunk_size: int = DEFAULT_PRETRAIN_CHUNK_SIZE,
) -> torch.Tensor:
    """Compatibility wrapper for replay staging tests and one-off materialization."""
    return _compute_actor_obs_pre(actor, obs_bt_cpu, chunk_size=chunk_size)


def _ensure_step_actor_latents(
    bufs: dict,
    actor: BuilderMLP,
) -> torch.Tensor:
    """Materialize per-step actor latents from replay staging tensors if needed."""
    step_actor_latent = bufs.get("step_actor_latent")
    if step_actor_latent is not None:
        return step_actor_latent

    actor_obs_pre = bufs.get("actor_obs_pre")
    if actor_obs_pre is None:
        actor_obs = bufs.get("actor_obs")
        if actor_obs is None:
            raise KeyError("missing actor_obs staging tensor")
        actor_obs_pre = _compute_actor_obs_pre_for_replay(actor, actor_obs)
        bufs["actor_obs_pre"] = actor_obs_pre

    actor_bot_ids = bufs.get("actor_bot_ids")
    if actor_bot_ids is None:
        raise KeyError("missing actor_bot_ids staging tensor")
    actor_step_order = bufs.get("actor_step_order")
    if actor_step_order is None:
        raise KeyError("missing actor_step_order staging tensor")

    actor_latent_bt = _compute_actor_latents_for_replay(actor, actor_obs_pre, actor_bot_ids)
    step_actor_latent = actor_latent_bt.index_select(0, actor_step_order.to(torch.long))
    bufs["step_actor_latent"] = step_actor_latent
    bufs.pop("actor_obs_pre", None)
    bufs.pop("actor_obs", None)
    bufs.pop("actor_bot_ids", None)
    bufs.pop("actor_step_order", None)
    return step_actor_latent


def precompute_all_actor_latents(
    dataset: "CriticBCDataset",
    actor: BuilderMLP,
    device: torch.device,
    *,
    target_samples_per_batch: int = 65536,
) -> None:
    """Batch-compute actor latents for ALL games on GPU, saving back to disk cache.

    After this, every disk-cached game has 'step_actor_latent' baked in and the
    raw actor staging tensors (actor_obs, actor_bot_ids, actor_step_order) are
    removed. Subsequent loads skip actor inference entirely.
    """
    actor = actor.to(device).eval()
    games = dataset.games
    cache_dir = dataset.cache_dir
    n = len(games)
    n_computed = 0
    t0 = time.time()
    sample_budget = max(int(target_samples_per_batch), 1)

    # Temporarily disable in-memory cache eviction during precompute
    old_cache_size = dataset._cache_size
    dataset._cache_size = 0  # unlimited

    pending: list[tuple[int, dict, dict]] = []
    pending_samples = 0

    def _flush_pending() -> int:
        nonlocal pending, pending_samples
        if not pending:
            return 0

        combined_obs = torch.cat([bufs["actor_obs"] for _idx, _game, bufs in pending], dim=0)
        combined_bot_ids = torch.empty(combined_obs.shape[0], dtype=torch.int32)
        split_ranges: list[tuple[int, int]] = []
        cursor = 0
        next_bot_base = 0
        for _idx, _game, bufs in pending:
            bot_ids = bufs["actor_bot_ids"].to(torch.long)
            n_steps = int(bot_ids.shape[0])
            if n_steps > 0:
                bot_min = int(bot_ids.min().item())
                bot_max = int(bot_ids.max().item())
                shifted = bot_ids - bot_min + next_bot_base
                combined_bot_ids[cursor:cursor + n_steps].copy_(shifted.to(torch.int32))
                next_bot_base += bot_max - bot_min + 1
            split_ranges.append((cursor, cursor + n_steps))
            cursor += n_steps

        actor_obs_pre = _compute_actor_obs_pre(actor, combined_obs, device)
        combined_latent_bt = _compute_actor_latents_for_replay(
            actor, actor_obs_pre, combined_bot_ids, device,
        )

        flushed = 0
        for (game_idx, game, bufs), (start, end) in zip(pending, split_ranges):
            actor_latent_bt = combined_latent_bt[start:end]
            step_actor_latent = actor_latent_bt.index_select(0, bufs["actor_step_order"].to(torch.long))
            bufs["step_actor_latent"] = step_actor_latent

            # Skip disk caching — states tensor is ~100MB/game, infeasible at scale.
            # In-memory LRU in the dataset handles within-epoch reuse.

            bufs.pop("actor_obs_pre", None)
            bufs.pop("actor_obs", None)
            bufs.pop("actor_bot_ids", None)
            bufs.pop("actor_step_order", None)
            flushed += 1

            if (game_idx + 1) % 500 == 0:
                print(f"  Actor latents: {game_idx+1}/{n} games, {n_computed + flushed} computed ({time.time()-t0:.1f}s)")

        pending = []
        pending_samples = 0
        return flushed

    for i, game in enumerate(games):
        path_str = game["path"]
        team = game["team"]
        bufs = dataset._get_buffers(path_str, team)

        if bufs.get("step_actor_latent") is not None:
            continue  # already has latents (from disk cache)

        actor_obs = bufs.get("actor_obs")
        if actor_obs is None:
            continue  # no actor obs available
        pending.append((i, game, bufs))
        pending_samples += int(actor_obs.shape[0])
        if pending_samples >= sample_budget:
            n_computed += _flush_pending()

    n_computed += _flush_pending()

    # Restore cache eviction and flush to fit limit
    dataset._cache_size = old_cache_size
    with dataset._cache_lock:
        while dataset._cache_size > 0 and len(dataset._cache_order) > dataset._cache_size:
            evict = dataset._cache_order.popleft()
            dataset._cache.pop(evict, None)

    print(f"  Actor latents: done, {n_computed} computed / {n - n_computed} cached ({time.time()-t0:.1f}s)")


def _build_round_groups(
    games: list[dict],
    *,
    rounds_per_group: int,
) -> list[dict]:
    groups: list[dict] = []
    for game_idx, game in enumerate(games):
        n_turns = int(game["n_turns"])
        for round_start in range(0, n_turns, rounds_per_group):
            round_end = min(round_start + rounds_per_group, n_turns)
            groups.append({
                "game_idx": game_idx,
                "round_start": round_start,
                "round_end": round_end,
            })
    return groups


def _shuffle_round_groups(
    round_groups: list[dict],
    *,
    games_per_batch: int,
    chunk_groups_per_batch: int,
) -> list[dict]:
    if not round_groups:
        return []
    if games_per_batch <= 0:
        games_per_batch = 1

    groups_by_block: dict[int, list[dict]] = {}
    for group in round_groups:
        block_id = group["game_idx"] // games_per_batch
        groups_by_block.setdefault(block_id, []).append(group)

    block_ids = list(groups_by_block)
    random.shuffle(block_ids)
    for block_id in block_ids:
        random.shuffle(groups_by_block[block_id])

    cursors = {block_id: 0 for block_id in block_ids}
    shuffled: list[dict] = []
    active = block_ids[:]
    stride = max(1, chunk_groups_per_batch)
    while active:
        next_active: list[int] = []
        for block_id in active:
            block = groups_by_block[block_id]
            start = cursors[block_id]
            end = min(start + stride, len(block))
            shuffled.extend(block[start:end])
            cursors[block_id] = end
            if end < len(block):
                next_active.append(block_id)
        active = next_active
    return shuffled


def _materialize_critic_batch(
    batch_groups: list[dict],
    games: list[dict],
    dataset: CriticBCDataset,
    buffers: _CriticBatchBuffers,
    *,
    pin_memory: bool,
    need_game_log: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    prepared: list[tuple[dict, dict, int, int, int, int, int]] = []
    loaded_buffers: dict[int, dict] = {}
    total_rounds = 0
    total_steps = 0

    for group in sorted(batch_groups, key=lambda g: (g["game_idx"], g["round_start"])):
        game_idx = group["game_idx"]
        game = games[game_idx]
        bufs = loaded_buffers.get(game_idx)
        if bufs is None:
            bufs = dataset._get_buffers(game["path"], game["team"])
            loaded_buffers[game_idx] = bufs
        rs = group["round_start"]
        re = group["round_end"]
        round_offsets = bufs["round_offsets"]
        ss = int(round_offsets[rs].item())
        se = int(round_offsets[re].item())
        if se <= ss:
            continue
        prepared.append((game, bufs, rs, re, ss, se, game_idx))
        total_rounds += re - rs
        total_steps += se - ss

    if not prepared:
        return (
            _empty_staging_tensor((0, GLOBAL_MAP_CHANNELS, GLOBAL_MAP_MAX_H, GLOBAL_MAP_MAX_W), torch.float32, pin_memory=pin_memory),
            _empty_staging_tensor((0, 1, GLOBAL_MAP_MAX_H, GLOBAL_MAP_MAX_W), torch.float32, pin_memory=pin_memory),
            _empty_staging_tensor((0, GLOBAL_SCALAR_DIM), torch.float32, pin_memory=pin_memory),
            _empty_staging_tensor((0,), torch.long, pin_memory=pin_memory),
            _empty_staging_tensor((0, 2), torch.int16, pin_memory=pin_memory),
            _empty_staging_tensor((0, UNIT_CONTEXT_DIM), torch.float16, pin_memory=pin_memory),
            _empty_staging_tensor((0, ACTOR_LATENT_DIM), torch.float16, pin_memory=pin_memory),
            _empty_staging_tensor((0,), torch.float32, pin_memory=pin_memory),
            _empty_staging_tensor((0,), torch.int32, pin_memory=pin_memory) if need_game_log else None,
            _empty_staging_tensor((0,), torch.int32, pin_memory=pin_memory) if need_game_log else None,
        )

    buffers.ensure(
        total_rounds=total_rounds,
        total_steps=total_steps,
        pin_memory=pin_memory,
        need_game_log=need_game_log,
    )
    assert buffers.round_states is not None
    assert buffers.round_masks is not None
    assert buffers.round_scalars is not None
    assert buffers.step_round_idx is not None
    assert buffers.step_unit_xy is not None
    assert buffers.step_unit_ctx is not None
    assert buffers.step_actor_latent is not None
    assert buffers.step_targets is not None

    round_states = buffers.round_states[:total_rounds]
    round_masks = buffers.round_masks[:total_rounds]
    round_scalars = buffers.round_scalars[:total_rounds]
    step_round_idx = buffers.step_round_idx[:total_steps]
    step_unit_xy = buffers.step_unit_xy[:total_steps]
    step_unit_ctx = buffers.step_unit_ctx[:total_steps]
    step_actor_latent = buffers.step_actor_latent[:total_steps]
    step_targets = buffers.step_targets[:total_steps]
    step_game_idx = buffers.step_game_idx[:total_steps] if need_game_log and buffers.step_game_idx is not None else None
    step_round_num = buffers.step_round_num[:total_steps] if need_game_log and buffers.step_round_num is not None else None

    round_masks.zero_()
    round_scalars.zero_()

    round_cursor = 0
    step_cursor = 0
    for game, bufs, rs, re, ss, se, gi in prepared:
        n_rounds = re - rs
        n_steps = se - ss
        mh = int(bufs["map_h"])
        mw = int(bufs["map_w"])
        nc = int(bufs["n_channels"])
        ns = int(bufs["n_scalars"])

        round_states[round_cursor:round_cursor + n_rounds, :nc, :mh, :mw].copy_(bufs["states"][rs:re])
        round_masks[round_cursor:round_cursor + n_rounds, :, :mh, :mw] = 1.0
        round_scalars[round_cursor:round_cursor + n_rounds, :ns].copy_(bufs["scalars"][rs:re])
        actor_latent = bufs["step_actor_latent"]
        step_rounds_slice = bufs["step_rounds"][ss:se]
        step_round_idx[step_cursor:step_cursor + n_steps] = step_rounds_slice.to(torch.long) - rs + round_cursor
        step_unit_xy[step_cursor:step_cursor + n_steps] = bufs["step_unit_xy"][ss:se]
        step_unit_ctx[step_cursor:step_cursor + n_steps] = bufs["step_unit_ctx"][ss:se]
        step_actor_latent[step_cursor:step_cursor + n_steps] = actor_latent[ss:se]
        step_targets[step_cursor:step_cursor + n_steps] = bufs["step_targets"][ss:se]
        if step_game_idx is not None and step_round_num is not None:
            step_game_idx[step_cursor:step_cursor + n_steps].fill_(gi)
            step_round_num[step_cursor:step_cursor + n_steps] = step_rounds_slice.to(torch.int32)

        round_cursor += n_rounds
        step_cursor += n_steps

    return round_states, round_masks, round_scalars, step_round_idx, step_unit_xy, step_unit_ctx, step_actor_latent, step_targets, step_game_idx, step_round_num


def train_critic(
    critic: CentralizedCritic,
    dataset: CriticBCDataset,
    epochs: int = 30,
    lr: float = 3e-4,
    games_per_batch: int = 160,
    rounds_per_group: int = 32,
    chunk_groups_per_batch: int = 8,
    grad_clip: float = 1.0,
    log_every: int = 5,
    step_log_every: int = 4,
    log_path: Path | None = None,
    game_log_path: Path | None = None,
    weights_path: Path | None = None,
    stop_after_optimizer_steps: int | None = None,
    enable_compile: bool = True,
    profile_optimizer_steps: int = 0,
    profile_skip_optimizer_steps: int = 0,
) -> dict:
    """Pretrain critic on real per-bot replay samples over staged game batches."""
    n_games = dataset.n_games
    if n_games == 0:
        print("No data for critic pretraining")
        return {}
    if games_per_batch <= 0:
        raise ValueError("games_per_batch must be positive")
    if rounds_per_group <= 0:
        raise ValueError("rounds_per_group must be positive")
    if chunk_groups_per_batch <= 0:
        raise ValueError("chunk_groups_per_batch must be positive")
    if step_log_every <= 0:
        raise ValueError("step_log_every must be positive")

    cuda_cfg = resolve_cuda_config(DEVICE, spatial_inputs=True, enable_compile=enable_compile)
    if cuda_cfg.enabled:
        configure_cuda_runtime(DEVICE)
    if cuda_cfg.channels_last:
        critic = critic.to(memory_format=torch.channels_last)

    bucket_centers = _make_bucket_centers().to(DEVICE)

    optimizer = make_normuon_optimizer(critic, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    critic.train()
    train_step = maybe_compile(CriticPretrainStep(critic), cuda_cfg, label="critic BC")
    train_step.train()

    stats: dict = {"epoch_losses": [], "optimizer_steps": 0}
    t0 = time.time()
    global_optimizer_step = 0
    log_episode = 0
    profile = _CriticProfile() if profile_optimizer_steps > 0 else None
    print(
        f"  {n_games} game-sides per epoch, globally shuffled "
        f"({rounds_per_group}-round groups, {chunk_groups_per_batch} groups/update)"
    )

    log_file = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else None
    game_log_file = open(game_log_path, "a", encoding="utf-8", buffering=1) if game_log_path else None
    weights_dirty = False

    def _flush_weights() -> None:
        nonlocal weights_dirty
        if weights_path is None or not weights_dirty:
            return
        _atomic_torch_save(critic.state_dict(), weights_path)
        weights_dirty = False

    def _flush_win_confidence(ep_num, batch_num, wp, game_idx_cpu, round_num_cpu, game_list):
        if not game_log_file or wp.numel() == 0:
            return
        wc = _aggregate_win_confidence(wp, game_idx_cpu, round_num_cpu.to(torch.long))
        for gi in sorted(wc):
            game = game_list[gi] if gi < len(game_list) else {}
            replay_name = Path(game.get("path", "")).stem if game else f"game_{gi}"
            team = game.get("team", 0)
            wc_data = wc[gi]
            entry = {
                "episode": ep_num,
                "game": gi,
                "opp": replay_name,
                "winner": game.get("winner"),
                "replay": f"ep{ep_num}_b{batch_num}_{replay_name}_t{team}_g{gi}.replay26",
                "win_confidence": wc_data["win_confidence"],
            }
            if wc_data["round_offset"] > 0:
                entry["round_offset"] = wc_data["round_offset"]
            game_log_file.write(jsonl_line(entry))

    try:
        stop_requested = False
        for epoch in range(epochs):
            epoch_loss = torch.zeros((), device=DEVICE)
            epoch_sum_y = torch.zeros((), device=DEVICE)
            epoch_sum_y2 = torch.zeros((), device=DEVICE)
            epoch_sum_err = torch.zeros((), device=DEVICE)
            epoch_sum_err2 = torch.zeros((), device=DEVICE)
            epoch_count = 0
            n_batches = 0
            n_optimizer_steps = 0
            total_samples = 0
            next_batch_idx = 0
            game_order = list(range(n_games))
            random.shuffle(game_order)
            games = [dataset.games[idx] for idx in game_order]
            round_groups = _build_round_groups(games, rounds_per_group=rounds_per_group)
            if not round_groups:
                continue
            round_groups = _shuffle_round_groups(
                round_groups,
                games_per_batch=games_per_batch,
                chunk_groups_per_batch=chunk_groups_per_batch,
            )

            batch_starts = list(range(0, len(round_groups), chunk_groups_per_batch))
            if not batch_starts:
                continue
            buffer_slots = [_CriticBatchBuffers(), _CriticBatchBuffers()]
            transfer_stream = torch.cuda.Stream(device=DEVICE) if cuda_cfg.enabled else None

            def _submit_batch(prefetch_pool: ThreadPoolExecutor, start_idx: int, slot_idx: int) -> Future:
                batch_groups = round_groups[start_idx:start_idx + chunk_groups_per_batch]
                return prefetch_pool.submit(
                    _materialize_critic_batch,
                    batch_groups,
                    games,
                    dataset,
                    buffer_slots[slot_idx],
                    pin_memory=cuda_cfg.enabled,
                    need_game_log=game_log_file is not None,
                )

            def _transfer_to_gpu(cpu_tensors):
                """Stage pinned CPU tensors to GPU on the transfer stream."""
                (states_cpu, masks_cpu, scalars_cpu, round_idx_cpu,
                 unit_xy_cpu, unit_ctx_cpu, actor_latent_cpu, targets_cpu,
                 game_idx_cpu, round_num_cpu) = cpu_tensors
                ctx = torch.cuda.stream(transfer_stream) if transfer_stream else contextlib.nullcontext()
                transfer_start = None
                transfer_end = None
                with ctx:
                    if profile is not None and cuda_cfg.enabled:
                        transfer_start = torch.cuda.Event(enable_timing=True)
                        transfer_end = torch.cuda.Event(enable_timing=True)
                        transfer_start.record(transfer_stream)
                    states = prepare_spatial_batch(states_cpu, DEVICE, non_blocking=True, channels_last=cuda_cfg.channels_last)
                    masks = prepare_spatial_batch(masks_cpu, DEVICE, non_blocking=True, channels_last=cuda_cfg.channels_last)
                    scalars = scalars_cpu.to(device=DEVICE, non_blocking=True)
                    round_idx = round_idx_cpu.to(device=DEVICE, non_blocking=True)
                    unit_xy = unit_xy_cpu.to(device=DEVICE, dtype=torch.long, non_blocking=True)
                    unit_ctx = unit_ctx_cpu.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
                    actor_latent = actor_latent_cpu.to(device=DEVICE, dtype=torch.float32, non_blocking=True)
                    values = targets_cpu.to(device=DEVICE, non_blocking=True)
                    targets = value_target_probs(values, bucket_centers)
                    if transfer_end is not None:
                        transfer_end.record(transfer_stream)
                    if transfer_stream:
                        event = transfer_stream.record_event()
                    else:
                        event = None
                return (
                    states, masks, scalars, round_idx, unit_xy, unit_ctx, actor_latent, values, targets,
                    game_idx_cpu, round_num_cpu, event, transfer_start, transfer_end,
                )

            with ThreadPoolExecutor(max_workers=2) as prefetch_pool:
                if next_batch_idx >= len(batch_starts):
                    future = None
                else:
                    future = _submit_batch(prefetch_pool, batch_starts[next_batch_idx], 0)

                # Pipeline: prefetch N+1 on CPU, transfer N to GPU, train on N-1
                staged_gpu = None  # transferred batch ready for training

                for local_batch_idx, group_start in enumerate(batch_starts[next_batch_idx:]):
                    batch_idx = next_batch_idx + local_batch_idx
                    if future is None and staged_gpu is None:
                        break

                    # Wait for CPU prefetch, start transfer
                    new_staged = None
                    if future is not None:
                        prefetch_wait_t0 = time.perf_counter()
                        cpu_tensors = future.result()
                        prefetch_wait_ms = (time.perf_counter() - prefetch_wait_t0) * 1000.0
                        next_start = batch_starts[batch_idx + 1] if batch_idx + 1 < len(batch_starts) else None
                        future = (
                            _submit_batch(prefetch_pool, next_start, (batch_idx + 1) % 2)
                            if next_start is not None
                            else None
                        )
                        if cpu_tensors[7].numel() > 0:  # targets_cpu
                            transfer_submit_t0 = time.perf_counter()
                            new_staged = _transfer_to_gpu(cpu_tensors)
                            transfer_submit_ms = (time.perf_counter() - transfer_submit_t0) * 1000.0
                        else:
                            transfer_submit_ms = 0.0
                    else:
                        prefetch_wait_ms = 0.0
                        transfer_submit_ms = 0.0

                    # Train on previously staged batch (overlaps with current transfer)
                    if staged_gpu is not None:
                        (states, masks, scalars, round_idx, unit_xy, unit_ctx,
                         actor_latent, values, targets, sg_game_idx_cpu, sg_round_num_cpu, event,
                         transfer_start, transfer_end) = staged_gpu
                        if event is not None:
                            torch.cuda.current_stream(DEVICE).wait_event(event)
                        transfer_gpu_ms = 0.0
                        if profile is not None and transfer_start is not None and transfer_end is not None:
                            transfer_end.synchronize()
                            transfer_gpu_ms = transfer_start.elapsed_time(transfer_end)

                        fwd_start = fwd_end = bwd_end = opt_end = None
                        if profile is not None and cuda_cfg.enabled:
                            fwd_start = torch.cuda.Event(enable_timing=True)
                            fwd_end = torch.cuda.Event(enable_timing=True)
                            bwd_end = torch.cuda.Event(enable_timing=True)
                            opt_end = torch.cuda.Event(enable_timing=True)
                            fwd_start.record(torch.cuda.current_stream(DEVICE))
                        with autocast_context(cuda_cfg):
                            zero_actions = torch.zeros(
                                actor_latent.shape[0], 5, dtype=torch.long, device=actor_latent.device,
                            )
                            pred_values, logits = train_step(
                                states, masks, scalars, round_idx,
                                unit_xy, unit_ctx, actor_latent, zero_actions,
                            )
                            loss = soft_cross_entropy_from_logits(logits, targets).mean()
                        if fwd_end is not None:
                            fwd_end.record(torch.cuda.current_stream(DEVICE))
                        with torch.no_grad():
                            err = values - pred_values
                            batch_count = int(values.shape[0])
                            epoch_sum_y = epoch_sum_y + values.sum()
                            epoch_sum_y2 = epoch_sum_y2 + values.square().sum()
                            epoch_sum_err = epoch_sum_err + err.sum()
                            epoch_sum_err2 = epoch_sum_err2 + err.square().sum()

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if bwd_end is not None:
                            bwd_end.record(torch.cuda.current_stream(DEVICE))
                        grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                        optimizer.step()
                        if opt_end is not None:
                            opt_end.record(torch.cuda.current_stream(DEVICE))
                        weights_dirty = True
                        should_profile_step = (
                            profile is not None
                            and global_optimizer_step + 1 > profile_skip_optimizer_steps
                            and profile.steps < profile_optimizer_steps
                        )
                        post_step_t0 = time.perf_counter() if should_profile_step else None

                        epoch_loss = epoch_loss + loss.detach()
                        n_batches += 1
                        n_optimizer_steps += 1
                        global_optimizer_step += 1
                        total_samples += batch_count
                        epoch_count += batch_count

                        if global_optimizer_step % step_log_every == 0 or global_optimizer_step == 1:
                            log_episode += 1

                        if game_log_file is not None and global_optimizer_step % step_log_every == 0:
                            wp = win_probability(logits, bucket_centers).cpu()
                            _flush_win_confidence(
                                log_episode, batch_idx,
                                wp, sg_game_idx_cpu, sg_round_num_cpu, games,
                            )

                        if log_file is not None and (
                            global_optimizer_step % step_log_every == 0 or global_optimizer_step == 1
                        ):
                            value_loss = round(loss.detach().item(), 6)
                            _ev = _explained_var_from_sums(
                                epoch_sum_y,
                                epoch_sum_y2,
                                epoch_sum_err,
                                epoch_sum_err2,
                                epoch_count,
                            )
                            entry = {
                                "episode": log_episode,
                                "optimizer_step": global_optimizer_step,
                                "loss": value_loss,
                                "value_loss": value_loss,
                                "explained_var": round(_ev, 4),
                                "grad_norm": round(grad_norm.item(), 4),
                                "elapsed": round(time.time() - t0, 1),
                                "n_samples": total_samples,
                            }
                            log_file.write(jsonl_line(entry))


                        if stop_after_optimizer_steps is not None and global_optimizer_step >= stop_after_optimizer_steps:
                            _flush_weights()
                            stop_requested = True

                        if should_profile_step:
                            if opt_end is not None:
                                opt_end.synchronize()
                                forward_gpu_ms = fwd_start.elapsed_time(fwd_end) if fwd_start is not None and fwd_end is not None else 0.0
                                backward_gpu_ms = fwd_end.elapsed_time(bwd_end) if fwd_end is not None and bwd_end is not None else 0.0
                                optimizer_gpu_ms = bwd_end.elapsed_time(opt_end) if bwd_end is not None else 0.0
                            else:
                                forward_gpu_ms = 0.0
                                backward_gpu_ms = 0.0
                                optimizer_gpu_ms = 0.0
                            profile.record_step(
                                prefetch_wait_ms=prefetch_wait_ms,
                                transfer_submit_ms=transfer_submit_ms,
                                transfer_gpu_ms=transfer_gpu_ms,
                                forward_gpu_ms=forward_gpu_ms,
                                backward_gpu_ms=backward_gpu_ms,
                                optimizer_gpu_ms=optimizer_gpu_ms,
                                post_step_ms=((time.perf_counter() - post_step_t0) * 1000.0) if post_step_t0 is not None else 0.0,
                                batch_rounds=int(states.shape[0]),
                                batch_steps=int(values.shape[0]),
                            )

                    staged_gpu = new_staged
                    if stop_requested:
                        break

                # Drain: train the last staged batch
                if staged_gpu is not None and not stop_requested:
                    (states, masks, scalars, round_idx, unit_xy, unit_ctx,
                     actor_latent, values, targets, sg_game_idx_cpu, sg_round_num_cpu, event,
                     transfer_start, transfer_end) = staged_gpu
                    if event is not None:
                        torch.cuda.current_stream(DEVICE).wait_event(event)
                    transfer_gpu_ms = 0.0
                    if profile is not None and transfer_start is not None and transfer_end is not None:
                        transfer_end.synchronize()
                        transfer_gpu_ms = transfer_start.elapsed_time(transfer_end)
                    fwd_start = fwd_end = bwd_end = opt_end = None
                    if profile is not None and cuda_cfg.enabled:
                        fwd_start = torch.cuda.Event(enable_timing=True)
                        fwd_end = torch.cuda.Event(enable_timing=True)
                        bwd_end = torch.cuda.Event(enable_timing=True)
                        opt_end = torch.cuda.Event(enable_timing=True)
                        fwd_start.record(torch.cuda.current_stream(DEVICE))
                    with autocast_context(cuda_cfg):
                        zero_actions = torch.zeros(
                            actor_latent.shape[0], 5, dtype=torch.long, device=actor_latent.device,
                        )
                        pred_values, logits = train_step(
                            states, masks, scalars, round_idx,
                            unit_xy, unit_ctx, actor_latent, zero_actions,
                        )
                        loss = soft_cross_entropy_from_logits(logits, targets).mean()
                    if fwd_end is not None:
                        fwd_end.record(torch.cuda.current_stream(DEVICE))
                    with torch.no_grad():
                        err = values - pred_values
                        batch_count = int(values.shape[0])
                        epoch_sum_y = epoch_sum_y + values.sum()
                        epoch_sum_y2 = epoch_sum_y2 + values.square().sum()
                        epoch_sum_err = epoch_sum_err + err.sum()
                        epoch_sum_err2 = epoch_sum_err2 + err.square().sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if bwd_end is not None:
                        bwd_end.record(torch.cuda.current_stream(DEVICE))
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    optimizer.step()
                    if opt_end is not None:
                        opt_end.record(torch.cuda.current_stream(DEVICE))
                    weights_dirty = True
                    should_profile_step = (
                        profile is not None
                        and global_optimizer_step + 1 > profile_skip_optimizer_steps
                        and profile.steps < profile_optimizer_steps
                    )
                    post_step_t0 = time.perf_counter() if should_profile_step else None
                    epoch_loss = epoch_loss + loss.detach()
                    n_batches += 1
                    n_optimizer_steps += 1
                    global_optimizer_step += 1
                    total_samples += batch_count
                    epoch_count += batch_count
                    if should_profile_step:
                        if opt_end is not None:
                            opt_end.synchronize()
                            forward_gpu_ms = fwd_start.elapsed_time(fwd_end) if fwd_start is not None and fwd_end is not None else 0.0
                            backward_gpu_ms = fwd_end.elapsed_time(bwd_end) if fwd_end is not None and bwd_end is not None else 0.0
                            optimizer_gpu_ms = bwd_end.elapsed_time(opt_end) if bwd_end is not None else 0.0
                        else:
                            forward_gpu_ms = 0.0
                            backward_gpu_ms = 0.0
                            optimizer_gpu_ms = 0.0
                        profile.record_step(
                            prefetch_wait_ms=0.0,
                            transfer_submit_ms=0.0,
                            transfer_gpu_ms=transfer_gpu_ms,
                            forward_gpu_ms=forward_gpu_ms,
                            backward_gpu_ms=backward_gpu_ms,
                            optimizer_gpu_ms=optimizer_gpu_ms,
                            post_step_ms=((time.perf_counter() - post_step_t0) * 1000.0) if post_step_t0 is not None else 0.0,
                            batch_rounds=int(states.shape[0]),
                            batch_steps=int(values.shape[0]),
                        )

            if stop_requested:
                _flush_weights()
                break

            scheduler.step()
            _flush_weights()
            avg = (epoch_loss / max(n_batches, 1)).item()
            epoch_ev = _explained_var_from_sums(
                epoch_sum_y,
                epoch_sum_y2,
                epoch_sum_err,
                epoch_sum_err2,
                epoch_count,
            )
            stats["epoch_losses"].append(avg)
            stats["optimizer_steps"] = global_optimizer_step

            if (epoch + 1) % log_every == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  "
                    f"ev={epoch_ev:.3f}  "
                    f"({n_batches} batches, {n_optimizer_steps} opt steps)  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.1f}s]"
                )

    finally:
        _flush_weights()
        if log_file is not None:
            log_file.close()
        if game_log_file is not None:
            game_log_file.close()

    stats["optimizer_steps"] = global_optimizer_step
    stats["final_loss"] = stats["epoch_losses"][-1] if stats["epoch_losses"] else 0.0
    stats["n_games"] = n_games
    if profile is not None:
        stats["profile"] = profile.summary()
        print(f"  profile: {json.dumps(stats['profile'], sort_keys=True)}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Pretrain critic from replays")
    parser.add_argument("--replay-dir", type=str, default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--index-path", type=str, default="")
    parser.add_argument("--min-rating", type=float, default=1500.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--games-per-batch", type=int, default=240)
    parser.add_argument("--rounds-per-group", type=int, default=32)
    parser.add_argument(
        "--chunk-groups-per-batch",
        type=int,
        default=96 if DEVICE.type == "cuda" else 8,
    )
    parser.add_argument("--step-log-every", type=int, default=4)
    parser.add_argument("--rounds-per-batch", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--staging-turns", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--actor-weights", type=str, default="", help="Frozen actor weights used to build replay actor latents")
    parser.add_argument(
        "--allow-random-actor-latents",
        action="store_true",
        help="Allow critic pretraining to use fresh actor-init latents when no actor checkpoint is provided",
    )
    parser.add_argument("--init-weights", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile even on CUDA")
    parser.add_argument("--session", type=str, default="",
                        help="Session name for dashboard logging")
    parser.add_argument("--stop-after-optimizer-steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--profile-optimizer-steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--profile-skip-optimizer-steps", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)

    session_name = args.session or f"critic_pretrain_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir = SESSIONS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "train_log.jsonl"
    game_log_path = session_dir / "game_log.jsonl"
    weights_path = session_dir / "critic_weights.pt"
    config = {
        "mode": "critic_pretrain", "lr": args.lr, "epochs": args.epochs,
        "min_rating": args.min_rating,
        "games_per_batch": args.games_per_batch,
        "rounds_per_group": args.rounds_per_group,
        "chunk_groups_per_batch": args.chunk_groups_per_batch,
        "step_log_every": args.step_log_every,
        "session": session_name,
        "actor_weights": args.actor_weights,
    }
    jsonl_init(log_path, config=config)
    jsonl_init(game_log_path)

    print(f"Replay dir: {replay_dir}")
    print(f"Session: {session_name}")
    print(f"Device: {DEVICE}")
    cuda_cfg = resolve_cuda_config(DEVICE, spatial_inputs=True, enable_compile=not args.no_compile)
    if cuda_cfg.enabled:
        print(
            f"CUDA pretrain: amp={cuda_cfg.amp_dtype}, "
            f"channels_last={'on' if cuda_cfg.channels_last else 'off'}, "
            f"compile={'on' if cuda_cfg.compile_model else 'off'}"
        )

    t0 = time.time()
    index_path = args.index_path or None
    dataset = CriticBCDataset(replay_dir, index_path=index_path, min_rating=args.min_rating)
    print(f"Loaded {dataset.total_turns} turns from {dataset.n_games} games in {time.time()-t0:.1f}s")

    critic = CentralizedCritic().to(DEVICE)
    actor = BuilderMLP().to(DEVICE).eval()
    actor_weight_path = args.actor_weights
    if not actor_weight_path and not args.allow_random_actor_latents:
        raise SystemExit(
            "Critic pretraining requires --actor-weights. "
            "Run actor BC first, then point critic pretrain at that checkpoint.",
        )
    if actor_weight_path:
        if not actor.load_weights(actor_weight_path):
            raise SystemExit(f"Could not load actor weights from {actor_weight_path}")
        print(f"Loaded frozen actor latents from {actor_weight_path}")
    else:
        print("Using fresh actor init for replay latents (--allow-random-actor-latents)")
    for p in actor.parameters():
        p.requires_grad_(False)

    if args.init_weights:
        state = torch.load(args.init_weights, map_location=DEVICE, weights_only=True)
        critic.load_state_dict(state)
        print(f"Loaded initial critic weights from {args.init_weights}")
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"Critic: {n_params:,} params")

    precompute_all_actor_latents(dataset, actor, DEVICE)

    stats = train_critic(
        critic, dataset, epochs=args.epochs, lr=args.lr,
        games_per_batch=args.games_per_batch,
        rounds_per_group=args.rounds_per_group,
        chunk_groups_per_batch=args.chunk_groups_per_batch,
        step_log_every=args.step_log_every,
        log_path=log_path,
        game_log_path=game_log_path,
        weights_path=weights_path,
        stop_after_optimizer_steps=args.stop_after_optimizer_steps or None,
        enable_compile=not args.no_compile,
        profile_optimizer_steps=args.profile_optimizer_steps,
        profile_skip_optimizer_steps=args.profile_skip_optimizer_steps,
    )

    out_path = args.output or str(PROJECT_ROOT / "bots" / "tensor" / "sessions" / "pretrained_critic.pt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(critic.state_dict(), out_path)
    print(f"Critic weights saved to {out_path}")
    print(f"Final loss: {stats.get('final_loss', 0):.4f}")


if __name__ == "__main__":
    main()
