#!/usr/bin/env python3
"""Pretrain the actor via behavioral cloning from ladder replays.

Uses round-major truncated BPTT over contiguous K-round chunks sampled across
games. Within each round step, all active bots from the sampled game chunks
are batched together on GPU while carrying recurrent hidden state forward.
Simple cross-entropy on all action heads — no weighting, oversampling, or
focal loss.

Usage:
    python bots/tensor/torch/pretrain/train_bc.py --epochs 20
    python bots/tensor/torch/pretrain/train_bc.py --both-teams --epochs 10
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import json
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
    BPTT_K,
    BRIDGE_BUILD_TYPE,
    BuilderMLP,
    DEVICE,
    HARVESTER_BUILD_TYPE,
    H_DIM,
    N_BRIDGE_TARGETS,
    N_BUILD,
    N_CONV_DIR,
    N_DIR,
    OBS_SIZE,
    compute_action_masks,
    compute_build_dir_mask_for_type,
    compute_harvester_build_dir_mask,
    conveyor_dir_mask,
)

from dataset import ActorBCDataset
from pretrain.gpu import (
    autocast_context,
    configure_cuda_runtime,
    maybe_compile,
    resolve_cuda_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "replay_data"
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"


class _DiffScatter(torch.autograd.Function):
    """Differentiable scatter: write vals into h at idx, gradient flows both ways."""

    @staticmethod
    def forward(ctx, h, idx, vals):
        ctx.save_for_backward(idx)
        out = h.detach().clone()
        out[idx] = vals
        return out

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        g_h = g.clone()
        g_h[idx] = 0
        return g_h, None, g[idx]


def _stage_cpu_tensor(t: torch.Tensor) -> torch.Tensor:
    """Keep resident training data on CPU, pinned when CUDA transfers matter."""
    t = t.contiguous()
    if DEVICE.type == "cuda":
        return t.pin_memory()
    return t


def _empty_staging_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Allocate a transient staging buffer on CPU, pinned when useful."""
    if DEVICE.type == "cuda":
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    return torch.empty(shape, dtype=dtype)


def _stage_device_tensor_to_cpu(t: torch.Tensor) -> torch.Tensor:
    """Copy a tensor to CPU staging without an extra pageable->pinned hop."""
    if t.device.type == "cpu":
        return _stage_cpu_tensor(t)
    out = torch.empty_like(t, device="cpu", pin_memory=DEVICE.type == "cuda")
    out.copy_(t, non_blocking=False)
    return out


def _materialize_group_prep_tensors(group: dict, games: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build transient CPU tensors for one group's prep scan."""
    if "obs" in group and "prep_slots" in group:
        return group["obs"], group["prep_slots"]

    n_samples = int(group["n_samples"])
    if n_samples == 0:
        return (
            _empty_staging_tensor((0, OBS_SIZE), torch.float16),
            _empty_staging_tensor((0,), torch.long),
        )

    obs = _empty_staging_tensor((n_samples, OBS_SIZE), torch.float16)
    prep_slots = _empty_staging_tensor((n_samples,), torch.long)
    cursor = 0
    for seg_idx, local_game_idx in enumerate(group["segment_local_game_indices"]):
        game = games[group["game_indices"][local_game_idx]]
        rs = group["segment_starts"][seg_idx]
        re = group["segment_ends"][seg_idx]
        n = re - rs
        obs[cursor:cursor + n].copy_(game["obs"][rs:re])
        prep_slots[cursor:cursor + n].copy_(game["global_slots"][rs:re])
        cursor += n
    return obs, prep_slots


def _materialize_group_training_tensors(
    group: dict,
    games: list[dict],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build and cache one group's reusable CPU training tensors."""
    cached = (
        group.get("obs"),
        group.get("act"),
        group.get("prep_slots"),
        group.get("train_slots"),
        group.get("bt_mask"),
        group.get("bd_mask"),
        group.get("harv_bd_mask"),
    )
    if all(t is not None for t in cached):
        return cached  # type: ignore[return-value]

    n_samples = int(group["n_samples"])
    if n_samples == 0:
        empty_obs = _empty_staging_tensor((0, OBS_SIZE), torch.float16)
        empty_act = _empty_staging_tensor((0, 5), torch.int16)
        empty_prep_slots = _empty_staging_tensor((0,), torch.long)
        empty_slots = _empty_staging_tensor((0,), torch.long)
        empty_bt_mask = _empty_staging_tensor((0, N_BUILD), torch.bool)
        empty_bd_mask = _empty_staging_tensor((0, N_DIR), torch.bool)
        empty_harv_bd_mask = _empty_staging_tensor((0, N_DIR), torch.bool)
        group["obs"] = empty_obs
        group["act"] = empty_act
        group["prep_slots"] = empty_prep_slots
        group["train_slots"] = empty_slots
        group["bt_mask"] = empty_bt_mask
        group["bd_mask"] = empty_bd_mask
        group["harv_bd_mask"] = empty_harv_bd_mask
        return empty_obs, empty_act, empty_prep_slots, empty_slots, empty_bt_mask, empty_bd_mask, empty_harv_bd_mask

    obs = _empty_staging_tensor((n_samples, OBS_SIZE), torch.float16)
    act = _empty_staging_tensor((n_samples, 5), torch.int16)
    prep_slots = _empty_staging_tensor((n_samples,), torch.long)
    train_slots = _empty_staging_tensor((n_samples,), torch.long)
    bt_mask = _empty_staging_tensor((n_samples, N_BUILD), torch.bool)
    bd_mask = _empty_staging_tensor((n_samples, N_DIR), torch.bool)
    harv_bd_mask = _empty_staging_tensor((n_samples, N_DIR), torch.bool)
    cursor = 0

    for seg_idx, local_game_idx in enumerate(group["segment_local_game_indices"]):
        game = games[group["game_indices"][local_game_idx]]
        rs = group["segment_starts"][seg_idx]
        re = group["segment_ends"][seg_idx]
        n = re - rs
        slot_base = group["slot_offsets"][local_game_idx]
        game_bt_mask = game.get("bt_mask")
        game_bd_mask = game.get("bd_mask")
        game_harv_bd_mask = game.get("harv_bd_mask")
        if game_bt_mask is None or game_bd_mask is None or game_harv_bd_mask is None:
            game_obs_f = game["obs"].float()
            game_bt_mask, game_bd_mask = compute_action_masks(game_obs_f)
            game_harv_bd_mask = compute_harvester_build_dir_mask(game_obs_f, game_bd_mask)
            game_bt_mask = game_bt_mask.clone()
            game_bt_mask[:, HARVESTER_BUILD_TYPE] &= game_harv_bd_mask.any(dim=1)

        obs[cursor:cursor + n].copy_(game["obs"][rs:re])
        act[cursor:cursor + n].copy_(game["act"][rs:re])
        prep_slots[cursor:cursor + n].copy_(game["global_slots"][rs:re])
        train_slots[cursor:cursor + n].copy_(game["bot_slots"][rs:re] + slot_base)
        bt_mask[cursor:cursor + n].copy_(game_bt_mask[rs:re])
        bd_mask[cursor:cursor + n].copy_(game_bd_mask[rs:re])
        harv_bd_mask[cursor:cursor + n].copy_(game_harv_bd_mask[rs:re])
        cursor += n

    group["obs"] = obs
    group["act"] = act
    group["prep_slots"] = prep_slots
    group["train_slots"] = train_slots
    group["bt_mask"] = bt_mask
    group["bd_mask"] = bd_mask
    group["harv_bd_mask"] = harv_bd_mask
    return obs, act, prep_slots, train_slots, bt_mask, bd_mask, harv_bd_mask


def _release_group_training_tensors(groups: list[dict]) -> None:
    """Drop one-shot CPU staging tensors after their optimizer step finishes."""
    for group in groups:
        group.pop("obs", None)
        group.pop("act", None)
        group.pop("prep_slots", None)
        group.pop("train_slots", None)
        group.pop("bt_mask", None)
        group.pop("bd_mask", None)
        group.pop("harv_bd_mask", None)
        group.pop("batch_h_base", None)


def _materialize_training_batch(
    batch_groups: list[dict],
    games: list[dict],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build one transient pinned CPU batch in step-major order.

    The returned obs/actions/slot ids are laid out contiguously by local TBPTT
    timestep across all groups, so the training loop can slice each step
    directly without per-step concatenation.
    """
    if batch_groups and "obs" in batch_groups[0]:
        h_base = 0
        max_steps = max(len(group["step_sizes"]) for group in batch_groups)
        step_sizes = [0] * max_steps
        for group in batch_groups:
            for dt, step_size in enumerate(group["step_sizes"]):
                step_sizes[dt] += int(step_size)
        step_offsets: list[int] = []
        obs_cursor = 0
        for step_size in step_sizes:
            step_offsets.append(obs_cursor)
            obs_cursor += step_size

        for group in batch_groups:
            group["batch_h_base"] = h_base
            h_base += int(group["h0"].shape[0])
        h_live = torch.cat([group["h0"] for group in batch_groups], dim=0)

        total_obs = obs_cursor
        batch_obs = _empty_staging_tensor((total_obs, OBS_SIZE), torch.float16)
        batch_act = _empty_staging_tensor((total_obs, 5), torch.int16)
        batch_train_slots = _empty_staging_tensor((total_obs,), torch.long)
        batch_bt_mask = _empty_staging_tensor((total_obs, N_BUILD), torch.bool)
        batch_bd_mask = _empty_staging_tensor((total_obs, N_DIR), torch.bool)
        batch_harv_bd_mask = _empty_staging_tensor((total_obs, N_DIR), torch.bool)
        write_cursor = step_offsets.copy()

        for group in batch_groups:
            base = int(group["batch_h_base"])
            group_bt_mask = group.get("bt_mask")
            group_bd_mask = group.get("bd_mask")
            group_harv_bd_mask = group.get("harv_bd_mask")
            if group_bt_mask is None or group_bd_mask is None or group_harv_bd_mask is None:
                group_obs_f = group["obs"].float()
                group_bt_mask, group_bd_mask = compute_action_masks(group_obs_f)
                group_harv_bd_mask = compute_harvester_build_dir_mask(group_obs_f, group_bd_mask)
                group_bt_mask = group_bt_mask.clone()
                group_bt_mask[:, HARVESTER_BUILD_TYPE] &= group_harv_bd_mask.any(dim=1)
            for dt, step_size in enumerate(group["step_sizes"]):
                step_size = int(step_size)
                if step_size == 0:
                    continue
                src = int(group["step_offsets"][dt])
                dst = write_cursor[dt]
                batch_obs[dst:dst + step_size].copy_(group["obs"][src:src + step_size])
                batch_act[dst:dst + step_size].copy_(group["act"][src:src + step_size])
                batch_train_slots[dst:dst + step_size].copy_(group["train_slots"][src:src + step_size] + base)
                batch_bt_mask[dst:dst + step_size].copy_(group_bt_mask[src:src + step_size])
                batch_bd_mask[dst:dst + step_size].copy_(group_bd_mask[src:src + step_size])
                batch_harv_bd_mask[dst:dst + step_size].copy_(group_harv_bd_mask[src:src + step_size])
                write_cursor[dt] += step_size

        return (
            h_live,
            batch_obs,
            batch_act,
            step_offsets,
            step_sizes,
            batch_train_slots,
            batch_bt_mask,
            batch_bd_mask,
            batch_harv_bd_mask,
        )

    total_h = sum(int(group["h0"].shape[0]) for group in batch_groups)
    h_base = 0
    h_live = _empty_staging_tensor((total_h, H_DIM), torch.float32)

    max_steps = max(len(group["step_sizes"]) for group in batch_groups)
    step_sizes = [0] * max_steps
    for group in batch_groups:
        for dt, step_size in enumerate(group["step_sizes"]):
            step_sizes[dt] += int(step_size)
    step_offsets: list[int] = []
    total_obs = 0
    for step_size in step_sizes:
        step_offsets.append(total_obs)
        total_obs += step_size

    batch_obs = _empty_staging_tensor((total_obs, OBS_SIZE), torch.float16)
    batch_act = _empty_staging_tensor((total_obs, 5), torch.int16)
    batch_train_slots = _empty_staging_tensor((total_obs,), torch.long)
    batch_bt_mask = _empty_staging_tensor((total_obs, N_BUILD), torch.bool)
    batch_bd_mask = _empty_staging_tensor((total_obs, N_DIR), torch.bool)
    batch_harv_bd_mask = _empty_staging_tensor((total_obs, N_DIR), torch.bool)
    write_cursor = step_offsets.copy()

    h_cursor = 0
    for group in batch_groups:
        group["batch_h_base"] = h_base
        h_base += int(group["h0"].shape[0])
        h_live[h_cursor:h_cursor + int(group["h0"].shape[0])].copy_(group["h0"])
        h_cursor += int(group["h0"].shape[0])

    for group in batch_groups:
        base = int(group["batch_h_base"])
        slot_offsets = group["slot_offsets"]
        seg_cursor = 0
        for dt, step_size in enumerate(group["step_sizes"]):
            if step_size == 0:
                continue
            dst = write_cursor[dt]
            copied = 0
            while copied < step_size:
                local_game_idx = group["segment_local_game_indices"][seg_cursor]
                game = games[group["game_indices"][local_game_idx]]
                rs = group["segment_starts"][seg_cursor]
                re = group["segment_ends"][seg_cursor]
                n = re - rs
                game_bt_mask = game.get("bt_mask")
                game_bd_mask = game.get("bd_mask")
                game_harv_bd_mask = game.get("harv_bd_mask")
                if game_bt_mask is None or game_bd_mask is None or game_harv_bd_mask is None:
                    game_obs_f = game["obs"].float()
                    game_bt_mask, game_bd_mask = compute_action_masks(game_obs_f)
                    game_harv_bd_mask = compute_harvester_build_dir_mask(game_obs_f, game_bd_mask)
                    game_bt_mask = game_bt_mask.clone()
                    game_bt_mask[:, HARVESTER_BUILD_TYPE] &= game_harv_bd_mask.any(dim=1)
                batch_obs[dst + copied:dst + copied + n].copy_(game["obs"][rs:re])
                batch_act[dst + copied:dst + copied + n].copy_(game["act"][rs:re])
                batch_train_slots[dst + copied:dst + copied + n].copy_(
                    game["bot_slots"][rs:re] + slot_offsets[local_game_idx] + base,
                )
                batch_bt_mask[dst + copied:dst + copied + n].copy_(game_bt_mask[rs:re])
                batch_bd_mask[dst + copied:dst + copied + n].copy_(game_bd_mask[rs:re])
                batch_harv_bd_mask[dst + copied:dst + copied + n].copy_(game_harv_bd_mask[rs:re])
                copied += n
                seg_cursor += 1
            write_cursor[dt] += copied

    return (
        h_live,
        batch_obs,
        batch_act,
        step_offsets,
        step_sizes,
        batch_train_slots,
        batch_bt_mask,
        batch_bd_mask,
        batch_harv_bd_mask,
    )


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over the causally valid subset, or zero when empty."""
    if mask is None:
        return F.cross_entropy(logits, targets)
    if not mask.any():
        return logits.new_zeros(())
    return F.cross_entropy(logits[mask], targets[mask])


def _masked_action_log_prob(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chosen-action log-prob under a legality mask."""
    if mask is None:
        return F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)

    target_mask = F.one_hot(targets, logits.shape[-1]).to(torch.bool)
    effective_mask = mask | target_mask
    safe_logits = logits.masked_fill(~effective_mask, -1e9)
    safe_logits = torch.where(
        effective_mask.any(dim=1, keepdim=True),
        safe_logits,
        torch.zeros_like(safe_logits),
    )
    return F.log_softmax(safe_logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)


def _bc_step_losses(
    obs: torch.Tensor,
    actions: torch.Tensor,
    bt_l: torch.Tensor,
    bd_l: torch.Tensor,
    cd_l: torch.Tensor,
    brt_l: torch.Tensor,
    md_l: torch.Tensor,
    *,
    bt_mask: torch.Tensor | None = None,
    bd_mask: torch.Tensor | None = None,
    harv_bd_mask: torch.Tensor | None = None,
    cd_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Per-sample joint BC loss with inference-consistent causal masking.

    Returns NLL *sums* (not means) plus sample count so callers can compute
    a correctly-weighted per-sample average across variable-size time steps.
    """
    bt_act = actions[:, 0]
    bd_act = actions[:, 1]
    cd_act = actions[:, 2]
    brt_act = actions[:, 3]
    md_act = actions[:, 4]

    if bt_mask is None or bd_mask is None or harv_bd_mask is None:
        obs_f = obs.float()
        bt_mask, bd_mask = compute_action_masks(obs_f)
        harv_bd_mask = compute_harvester_build_dir_mask(obs_f, bd_mask)
        bt_mask = bt_mask.clone()
        bt_mask[:, HARVESTER_BUILD_TYPE] &= harv_bd_mask.any(dim=1)
        bd_mask = compute_build_dir_mask_for_type(obs_f, bt_act, bd_mask, harv_bd_mask)
    else:
        bd_mask = compute_build_dir_mask_for_type(obs, bt_act, bd_mask, harv_bd_mask)
    if cd_mask is None:
        cd_mask = conveyor_dir_mask(obs.device, obs.shape[0])

    build_causal = (bt_act != 0).float()
    conv_causal = ((bt_act == 2) | (bt_act == 7) | (bt_act == 8)).float()
    bridge_causal = (bt_act == BRIDGE_BUILD_TYPE).float()

    bt_nll = -_masked_action_log_prob(bt_l, bt_act, bt_mask)
    bd_nll = -_masked_action_log_prob(bd_l, bd_act, bd_mask) * build_causal
    cd_nll = -_masked_action_log_prob(cd_l, cd_act, cd_mask) * conv_causal
    brt_nll = -_masked_action_log_prob(brt_l, brt_act) * bridge_causal
    md_nll = -_masked_action_log_prob(md_l, md_act)

    total_nll = bt_nll + bd_nll + cd_nll + brt_nll + md_nll
    n = obs.shape[0]
    return (
        total_nll.sum(),
        bt_nll.sum(),
        bd_nll.sum(),
        cd_nll.sum(),
        brt_nll.sum(),
        md_nll.sum(),
        n,
    )


def _fused_actor_head_params(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fused actor projection once for repeated recurrent steps."""
    combined_w = torch.cat([
        model.head_build_type.weight,
        model.head_build_dir.weight,
        model.head_conv_dir.weight,
        model.head_bridge_target.weight,
        model.head_move_dir.weight,
    ], dim=0)
    combined_b = torch.cat([
        model.head_build_type.bias,
        model.head_build_dir.bias,
        model.head_conv_dir.bias,
        model.head_bridge_target.bias,
        model.head_move_dir.bias,
    ])
    return combined_w, combined_b


def _fused_recurrent_head_params(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the recurrent-state projection used for no-grad chunk-boundary scans."""
    combined_w = torch.cat([
        model.head_h_gate.weight,
        model.head_h_cand.weight,
    ], dim=0)
    combined_b = torch.cat([
        model.head_h_gate.bias,
        model.head_h_cand.bias,
    ])
    return combined_w, combined_b


def _forward_from_obs_pre_step_fused(
    model,
    obs_pre: torch.Tensor,
    h_prev: torch.Tensor,
    combined_w: torch.Tensor,
    combined_b: torch.Tensor,
):
    """Recurrent actor step with a cached fused head projection."""
    h = model._latent_from_obs_pre(obs_pre, h_prev)
    combined = F.linear(h, combined_w, combined_b)
    bt_l, bd_l, cd_l, brt_l, md_l = combined.split(
        [N_BUILD, N_DIR, N_CONV_DIR, N_BRIDGE_TARGETS, N_DIR],
        dim=-1,
    )
    recurrent_in = torch.cat([h, h_prev], dim=-1)
    h_gate = model.head_h_gate(recurrent_in)
    h_cand = torch.tanh(model.head_h_cand(recurrent_in))
    z = torch.sigmoid(h_gate)
    h_next = (1.0 - z) * h_prev + z * h_cand
    return bt_l, bd_l, cd_l, brt_l, md_l, h_next


def _hidden_update_from_obs_pre_step_fused(
    model,
    obs_pre: torch.Tensor,
    h_prev: torch.Tensor,
    combined_w: torch.Tensor,
    combined_b: torch.Tensor,
) -> torch.Tensor:
    """Recurrent hidden-state update without projecting policy heads."""
    h = model._latent_from_obs_pre(obs_pre, h_prev)
    recurrent_in = torch.cat([h, h_prev], dim=-1)
    h_gate, h_cand = F.linear(recurrent_in, combined_w, combined_b).split([H_DIM, H_DIM], dim=-1)
    z = torch.sigmoid(h_gate)
    h_cand = torch.tanh(h_cand)
    return (1.0 - z) * h_prev + z * h_cand


def _actor_bc_optimizer(
    model: BuilderMLP,
    *,
    lr: float,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Actor-BC optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _run_eval_games(
    weight_path: str,
    replay_dir: Path,
    n_games: int = 2,
):
    """Run a few games with the BC-pretrained weights and print a replay summary."""
    import shutil
    import subprocess

    replay_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = PROJECT_ROOT / "maps"
    maps = [f.name for f in maps_dir.glob("*.map26")]
    if not maps:
        print("No maps found, skipping eval games")
        return

    cambc_bin = PROJECT_ROOT / ".venv" / "bin" / "cambc"
    if not cambc_bin.exists():
        print("cambc not found in .venv, skipping eval games")
        return

    bot_a = str(PROJECT_ROOT / "bots" / "tensor_inference")
    bot_b = str(PROJECT_ROOT / "bots" / "tensor_dummy")

    inference_weights = PROJECT_ROOT / "bots" / "tensor_inference" / "weights.json"
    backup = None
    if inference_weights.exists():
        backup = inference_weights.with_suffix(".json.bak")
        shutil.copy2(inference_weights, backup)
    shutil.copy2(weight_path, inference_weights)

    print(f"\nRunning {n_games} eval games...")
    eval_results: list[dict[str, object]] = []
    try:
        for i in range(n_games):
            map_name = random.choice(maps)
            replay_path = replay_dir / f"bc_eval_{i}_{map_name.replace('.map26', '')}.replay26"
            map_path = str(maps_dir / map_name)
            try:
                result = subprocess.run(
                    [str(cambc_bin), "run", bot_a, bot_b, map_path,
                     "--replay", str(replay_path)],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True, text=True, timeout=60,
                )
                if replay_path.exists():
                    from replay_parser import ReplayData
                    rd = ReplayData.from_file(str(replay_path))
                    eval_results.append({
                        "game": i + 1,
                        "map": map_name,
                        "status": "ok",
                        "winner": rd.winner_str,
                        "turns": rd.n_turns,
                        "replay": str(replay_path),
                    })
                    print(f"  game {i+1}: {map_name} → winner={rd.winner_str}, turns={rd.n_turns}")
                else:
                    eval_results.append({
                        "game": i + 1,
                        "map": map_name,
                        "status": "missing_replay",
                        "replay": str(replay_path),
                    })
                    print(f"  game {i+1}: {map_name} — no replay generated")
                    if result.stderr:
                        print(f"    stderr: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                eval_results.append({
                    "game": i + 1,
                    "map": map_name,
                    "status": "timeout",
                    "replay": str(replay_path),
                })
                print(f"  game {i+1}: {map_name} — timeout")
            except Exception as e:
                eval_results.append({
                    "game": i + 1,
                    "map": map_name,
                    "status": "error",
                    "error": str(e),
                    "replay": str(replay_path),
                })
                print(f"  game {i+1}: error — {e}")
    finally:
        if backup and backup.exists():
            shutil.move(str(backup), str(inference_weights))
        elif backup is None:
            inference_weights.unlink(missing_ok=True)

    summary_path = replay_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\nEval summary:")
    for result in eval_results:
        line = (
            f"  game {result['game']}: status={result['status']} map={result['map']} "
            f"replay={result['replay']}"
        )
        if result.get("winner") is not None:
            line += f" winner={result['winner']} turns={result['turns']}"
        if result.get("error"):
            line += f" error={result['error']}"
        print(line)
    print(f"  summary={summary_path}")


def _load_single_game(dataset, game: dict) -> dict | None:
    """Parse one game into a staged dict, or None if empty."""
    n_rounds = int(game["n_turns"])
    if hasattr(dataset, "_get_round_major"):
        round_major = dataset._get_round_major(
            game["path"],
            game["team"],
            n_rounds_hint=n_rounds,
        )
        if isinstance(round_major, dict):
            obs = round_major["obs"]
            actions = round_major["actions"]
            bot_slots = round_major["bot_slots"]
            round_offsets_t = round_major["round_offsets"]
            n_slots = int(round_major["n_slots"])
            bt_mask = round_major["bt_mask"]
            bd_mask = round_major["bd_mask"]
            harv_bd_mask = round_major["harv_bd_mask"]
        else:
            obs, actions, bot_slots, round_offsets_t, n_slots = round_major
            obs_f = obs.float()
            bt_mask, bd_mask = compute_action_masks(obs_f)
            harv_bd_mask = compute_harvester_build_dir_mask(obs_f, bd_mask)
            bt_mask = bt_mask.clone()
            bt_mask[:, HARVESTER_BUILD_TYPE] &= harv_bd_mask.any(dim=1)
        if obs.shape[0] == 0:
            return None
        observed_n_rounds = max(int(round_offsets_t.shape[0]) - 1, 0)
        if observed_n_rounds < n_rounds:
            pad = torch.full(
                (n_rounds - observed_n_rounds,),
                int(round_offsets_t[-1].item()),
                dtype=torch.long,
            )
            round_offsets_t = torch.cat([round_offsets_t, pad], dim=0)
        round_offsets = round_offsets_t.tolist()
    else:
        obs, actions, bot_ids, turns = dataset._get_parsed(game["path"], game["team"])
        if obs.shape[0] == 0:
            return None

        order = turns.argsort(stable=True)
        obs = obs[order].half()
        actions = actions[order].to(torch.int16)
        bot_ids = bot_ids[order]
        turns = turns[order].long()

        observed_n_rounds = int(turns[-1].item()) + 1
        counts = torch.bincount(turns, minlength=max(n_rounds, observed_n_rounds))
        round_offsets_t = torch.zeros(max(n_rounds, observed_n_rounds) + 1, dtype=torch.long)
        round_offsets_t[1:] = counts.cumsum(0)
        round_offsets = round_offsets_t.tolist()
        unique_bots, bot_slots = bot_ids.unique(sorted=True, return_inverse=True)
        n_slots = int(unique_bots.shape[0])
        bot_slots = bot_slots.to(torch.long)
        obs_f = obs.float()
        bt_mask, bd_mask = compute_action_masks(obs_f)
        harv_bd_mask = compute_harvester_build_dir_mask(obs_f, bd_mask)
        bt_mask = bt_mask.clone()
        bt_mask[:, HARVESTER_BUILD_TYPE] &= harv_bd_mask.any(dim=1)

    if observed_n_rounds > n_rounds:
        raise RuntimeError(
            f"Stale replay metadata for {game['path']}: index reports n_turns={n_rounds}, "
            f"but parsed steps reach round {observed_n_rounds - 1}"
        )

    return {
        "obs": _stage_cpu_tensor(obs),
        "act": _stage_cpu_tensor(actions),
        "bot_slots": _stage_cpu_tensor(bot_slots),
        "bt_mask": _stage_cpu_tensor(bt_mask),
        "bd_mask": _stage_cpu_tensor(bd_mask),
        "harv_bd_mask": _stage_cpu_tensor(harv_bd_mask),
        "round_offsets": round_offsets,
        "n_rounds": n_rounds,
        "n_slots": n_slots,
    }


_LOAD_WORKERS = 8


def _load_game_batch(
    dataset,
    game_list,
    *,
    max_staging_games: int | None = None,
):
    """Load one staging window into round-major tensors using parallel workers.

    Returns (games, total_slots, max_rounds, used_games).
    """
    to_load = game_list[:max_staging_games] if max_staging_games else game_list

    with ThreadPoolExecutor(max_workers=_LOAD_WORKERS) as pool:
        results = list(pool.map(lambda g: _load_single_game(dataset, g), to_load))

    games = []
    total_slots = 0
    max_rounds = 0
    used_games = len(results)

    for game_data in results:
        if game_data is None:
            continue
        game_data["slot_base"] = total_slots
        game_data["global_slots"] = _stage_cpu_tensor(game_data["bot_slots"] + total_slots)
        games.append(game_data)
        total_slots += game_data["n_slots"]
        max_rounds = max(max_rounds, game_data["n_rounds"])

    return games, total_slots, max_rounds, used_games


def _prepare_tbptt_groups(
    train_model,
    games: list[dict],
    total_slots: int,
    max_rounds: int,
    tbptt_len: int,
    cuda_cfg,
) -> list[dict]:
    """Cache hidden state at each K-round boundary, then build chunk groups."""
    if total_slots == 0 or max_rounds == 0:
        return []

    groups: list[dict] = []
    h_scan = torch.zeros(total_slots, H_DIM, device=DEVICE)

    with torch.inference_mode():
        recurrent_w, recurrent_b = _fused_recurrent_head_params(train_model)
        for chunk_start in range(0, max_rounds, tbptt_len):
            active_game_indices = [idx for idx, game in enumerate(games) if game["n_rounds"] > chunk_start]
            if not active_game_indices:
                continue
            active_games = [games[idx] for idx in active_game_indices]

            chunk_lens = [min(tbptt_len, g["n_rounds"] - chunk_start) for g in active_games]
            slot_offsets = []
            h0_parts = []
            local_slot_base = 0
            for game in active_games:
                slot_offsets.append(local_slot_base)
                s = game["slot_base"]
                e = s + game["n_slots"]
                h0_parts.append(h_scan[s:e])
                local_slot_base += game["n_slots"]

            chunk_end = min(chunk_start + tbptt_len, max_rounds)
            step_sizes: list[int] = []
            step_offsets: list[int] = []
            segment_local_game_indices: list[int] = []
            segment_starts: list[int] = []
            segment_ends: list[int] = []
            flat_count = 0

            for dt, round_idx in enumerate(range(chunk_start, chunk_end)):
                step_offsets.append(flat_count)
                step_size = 0
                for game_idx, game in enumerate(active_games):
                    if dt >= chunk_lens[game_idx]:
                        continue
                    rs = game["round_offsets"][round_idx]
                    re = game["round_offsets"][round_idx + 1]
                    if re <= rs:
                        continue
                    segment_local_game_indices.append(game_idx)
                    segment_starts.append(rs)
                    segment_ends.append(re)
                    step_size += re - rs
                step_sizes.append(step_size)
                flat_count += step_size

            group = {
                "chunk_start": chunk_start,
                "game_indices": active_game_indices,
                "chunk_lens": chunk_lens,
                "slot_offsets": slot_offsets,
                "step_sizes": step_sizes,
                "step_offsets": step_offsets,
                "segment_local_game_indices": segment_local_game_indices,
                "segment_starts": segment_starts,
                "segment_ends": segment_ends,
                "n_samples": flat_count,
                "h0": _stage_device_tensor_to_cpu(torch.cat(h0_parts, dim=0)),
            }
            groups.append(group)

            if flat_count == 0:
                continue

            group_obs_cpu, group_prep_slots_cpu = _materialize_group_prep_tensors(group, games)
            group_obs = group_obs_cpu.to(DEVICE, non_blocking=True)
            group_prep_slots = group_prep_slots_cpu.to(DEVICE, non_blocking=True)
            with autocast_context(cuda_cfg):
                obs_pre_all = train_model._obs_encoder_pre(group_obs)
                offset = 0
                for step_size in step_sizes:
                    if step_size == 0:
                        continue
                    obs_pre = obs_pre_all[offset:offset + step_size]
                    slot_ids = group_prep_slots[offset:offset + step_size]
                    b_h = h_scan[slot_ids]
                    h_new = _hidden_update_from_obs_pre_step_fused(
                        train_model, obs_pre, b_h, recurrent_w, recurrent_b,
                    )
                    h_scan.index_copy_(0, slot_ids, h_new.float())
                    offset += step_size

    return groups


def _stage_training_batch_to_device(
    cpu_batch: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[int],
        list[int],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    cuda_cfg,
    transfer_stream: torch.cuda.Stream | None,
):
    """Move one actor BC batch from CPU staging to the training device."""
    (
        h_live_cpu,
        batch_obs_cpu,
        batch_act_cpu,
        step_offsets,
        step_sizes,
        batch_train_slots_cpu,
        batch_bt_mask_cpu,
        batch_bd_mask_cpu,
        batch_harv_bd_mask_cpu,
    ) = cpu_batch
    ctx = torch.cuda.stream(transfer_stream) if transfer_stream is not None else contextlib.nullcontext()
    with ctx:
        h_live = h_live_cpu.to(DEVICE, non_blocking=True)
        batch_obs = batch_obs_cpu.to(DEVICE, non_blocking=True)
        batch_act = batch_act_cpu.to(DEVICE, non_blocking=True)
        batch_train_slots = batch_train_slots_cpu.to(DEVICE, non_blocking=True)
        batch_bt_mask = batch_bt_mask_cpu.to(DEVICE, non_blocking=True)
        batch_bd_mask = batch_bd_mask_cpu.to(DEVICE, non_blocking=True)
        batch_harv_bd_mask = batch_harv_bd_mask_cpu.to(DEVICE, non_blocking=True)
        event = transfer_stream.record_event() if transfer_stream is not None else None
    return (
        h_live,
        batch_obs,
        batch_act,
        step_offsets,
        step_sizes,
        batch_train_slots,
        batch_bt_mask,
        batch_bd_mask,
        batch_harv_bd_mask,
        event,
    )


def train_actor_bc(
    model: BuilderMLP,
    dataset: ActorBCDataset,
    epochs: int = 20,
    lr: float = 3e-4,
    games_per_batch: int = 800,
    tbptt_len: int = BPTT_K,
    chunk_groups_per_batch: int = 1,
    grad_clip: float = 1.0,
    log_path: Path | None = None,
    step_log_path: Path | None = None,
    step_log_every: int = 0,
    enable_compile: bool = True,
    session_dir: Path | None = None,
) -> dict:
    """Round-chunk BPTT behavioral cloning across many games.

    Streams through game batches. For each optimizer step, sample several
    contiguous K-round chunk groups, then batch all active bots from those
    round steps together on GPU while carrying recurrent state across the
    chunk.
    """
    if dataset.n_games == 0:
        print("No data for actor BC")
        return {}
    if tbptt_len <= 0:
        raise ValueError("tbptt_len must be positive")
    if chunk_groups_per_batch <= 0:
        raise ValueError("chunk_groups_per_batch must be positive")
    if games_per_batch <= 0:
        raise ValueError("games_per_batch must be positive")
    if step_log_every < 0:
        raise ValueError("step_log_every must be non-negative")
    if log_path is not None and step_log_path is not None and log_path != step_log_path:
        raise ValueError("log_path and step_log_path must match when both are provided")
    if step_log_path is not None:
        log_path = step_log_path

    cuda_cfg = resolve_cuda_config(DEVICE, spatial_inputs=False, enable_compile=enable_compile)
    if cuda_cfg.enabled:
        configure_cuda_runtime(DEVICE)

    optimizer = _actor_bc_optimizer(model, lr=lr, weight_decay=1e-4)
    train_model = maybe_compile(model, cuda_cfg, label="actor BC")
    train_model.train()

    n_games = dataset.n_games
    print(f"  {n_games} games per epoch, staging {games_per_batch} games in parallel")

    stats = {"epoch_losses": [], "optimizer_steps": 0}
    total_samples = 0
    t0 = time.time()
    global_optimizer_step = 0
    log_file = open(log_path, "a", encoding="utf-8", buffering=1) if log_path and step_log_every > 0 else None

    try:
        for epoch in range(epochs):
            epoch_nll_sum = torch.zeros((), device=DEVICE)
            epoch_nll_bt = torch.zeros((), device=DEVICE)
            epoch_nll_md = torch.zeros((), device=DEVICE)
            epoch_nll_bd = torch.zeros((), device=DEVICE)
            epoch_nll_cd = torch.zeros((), device=DEVICE)
            epoch_nll_brt = torch.zeros((), device=DEVICE)
            epoch_sample_count = 0
            n_optimizer_steps = 0

            # Shuffle game order each epoch
            game_order = list(range(n_games))
            random.shuffle(game_order)

            game_cursor = 0
            while game_cursor < n_games:
                remaining_games = [dataset.games[game_order[j]] for j in range(game_cursor, n_games)]
                games, total_slots, max_rounds, used_games = _load_game_batch(
                    dataset,
                    remaining_games,
                    max_staging_games=games_per_batch,
                )
                if used_games <= 0:
                    break
                game_cursor += used_games
                if not games:
                    continue

                groups = _prepare_tbptt_groups(
                    train_model,
                    games,
                    total_slots,
                    max_rounds,
                    tbptt_len,
                    cuda_cfg,
                )
                if not groups:
                    continue

                group_ids = list(range(len(groups)))
                random.shuffle(group_ids)
                batch_group_sets = []
                for group_batch_start in range(0, len(group_ids), chunk_groups_per_batch):
                    batch_groups = [groups[group_id] for group_id in group_ids[group_batch_start:group_batch_start + chunk_groups_per_batch]]
                    batch_groups = [
                        group
                        for group in batch_groups
                        if group["h0"].numel() > 0
                        and (group["n_samples"] > 0 if "n_samples" in group else group["obs"].shape[0] > 0)
                    ]
                    if batch_groups:
                        batch_group_sets.append(batch_groups)
                if not batch_group_sets:
                    continue

                combined_w, combined_b = _fused_actor_head_params(train_model)
                transfer_stream = torch.cuda.Stream(device=DEVICE) if cuda_cfg.enabled else None

                def _submit_batch(prefetch_pool: ThreadPoolExecutor, batch_idx: int):
                    return prefetch_pool.submit(
                        _materialize_training_batch,
                        batch_group_sets[batch_idx],
                        games,
                    )

                def _run_training_batch(staged_gpu) -> None:
                    nonlocal combined_w, combined_b
                    nonlocal epoch_nll_sum, epoch_nll_bt, epoch_nll_md, epoch_nll_bd, epoch_nll_cd, epoch_nll_brt
                    nonlocal epoch_sample_count, total_samples, n_optimizer_steps, global_optimizer_step

                    (
                        h_live,
                        batch_obs,
                        batch_act,
                        step_offsets,
                        step_sizes,
                        batch_train_slots,
                        batch_bt_mask,
                        batch_bd_mask,
                        batch_harv_bd_mask,
                        event,
                    ) = staged_gpu
                    if event is not None:
                        torch.cuda.current_stream(DEVICE).wait_event(event)

                    optimizer.zero_grad(set_to_none=True)
                    chunk_nll = torch.zeros((), device=DEVICE)
                    chunk_nll_bt = torch.zeros((), device=DEVICE)
                    chunk_nll_bd = torch.zeros((), device=DEVICE)
                    chunk_nll_cd = torch.zeros((), device=DEVICE)
                    chunk_nll_brt = torch.zeros((), device=DEVICE)
                    chunk_nll_md = torch.zeros((), device=DEVICE)
                    chunk_samples = 0

                    with autocast_context(cuda_cfg):
                        obs_pre_all = train_model._obs_encoder_pre(batch_obs)
                    batch_act_long = batch_act.long()
                    batch_cd_mask = conveyor_dir_mask(batch_obs.device, batch_obs.shape[0])

                    for step_offset, step_size in zip(step_offsets, step_sizes):
                        if step_size == 0:
                            continue

                        obs_step = batch_obs[step_offset:step_offset + step_size]
                        obs_pre = obs_pre_all[step_offset:step_offset + step_size]
                        b_act = batch_act_long[step_offset:step_offset + step_size]
                        slot_ids = batch_train_slots[step_offset:step_offset + step_size]
                        step_bt_mask = batch_bt_mask[step_offset:step_offset + step_size]
                        step_bd_mask = batch_bd_mask[step_offset:step_offset + step_size]
                        step_harv_bd_mask = batch_harv_bd_mask[step_offset:step_offset + step_size]
                        step_cd_mask = batch_cd_mask[step_offset:step_offset + step_size]
                        b_h = h_live[slot_ids]

                        bt_l, bd_l, cd_l, brt_l, md_l, h_new = _forward_from_obs_pre_step_fused(
                            train_model, obs_pre, b_h, combined_w, combined_b,
                        )
                        step_nll, bt_nll, bd_nll, cd_nll, brt_nll, md_nll, step_n = _bc_step_losses(
                            obs_step,
                            b_act,
                            bt_l,
                            bd_l,
                            cd_l,
                            brt_l,
                            md_l,
                            bt_mask=step_bt_mask,
                            bd_mask=step_bd_mask,
                            harv_bd_mask=step_harv_bd_mask,
                            cd_mask=step_cd_mask,
                        )

                        chunk_nll = chunk_nll + step_nll
                        chunk_nll_bt = chunk_nll_bt + bt_nll
                        chunk_nll_bd = chunk_nll_bd + bd_nll
                        chunk_nll_cd = chunk_nll_cd + cd_nll
                        chunk_nll_brt = chunk_nll_brt + brt_nll
                        chunk_nll_md = chunk_nll_md + md_nll
                        chunk_samples += step_n

                        # Differentiable scatter: create a new h_live so that
                        # gradient flows from future steps back through h_new.
                        h_live = _DiffScatter.apply(h_live, slot_ids, h_new.float())

                        epoch_nll_sum += step_nll.detach()
                        epoch_nll_bt += bt_nll.detach()
                        epoch_nll_md += md_nll.detach()
                        epoch_nll_bd += bd_nll.detach()
                        epoch_nll_cd += cd_nll.detach()
                        epoch_nll_brt += brt_nll.detach()
                        epoch_sample_count += step_n
                        total_samples += step_n

                    if chunk_samples == 0:
                        return

                    (chunk_nll / chunk_samples).backward()
                    _bc_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    train_model.mu_center_policy_heads()
                    n_optimizer_steps += 1
                    global_optimizer_step += 1

                    # Rebuild fused params after weight update
                    combined_w, combined_b = _fused_actor_head_params(train_model)

                    if (
                        log_file is not None
                        and (global_optimizer_step % step_log_every == 0 or global_optimizer_step == 1)
                    ):
                        denom = float(chunk_samples)
                        entry = {
                            "episode": global_optimizer_step,
                            "optimizer_step": global_optimizer_step,
                            "epoch_optimizer_step": n_optimizer_steps,
                            "loss": round((chunk_nll.detach() / denom).item(), 6),
                            "loss_bt": round((chunk_nll_bt.detach() / denom).item(), 6),
                            "loss_bd": round((chunk_nll_bd.detach() / denom).item(), 6),
                            "loss_cd": round((chunk_nll_cd.detach() / denom).item(), 6),
                            "loss_brt": round((chunk_nll_brt.detach() / denom).item(), 6),
                            "loss_md": round((chunk_nll_md.detach() / denom).item(), 6),
                            "grad_norm": round(_bc_grad_norm.item(), 4),
                            "elapsed": round(time.time() - t0, 1),
                            "n_samples": total_samples,
                        }
                        log_file.write(json.dumps(entry) + "\n")

                with ThreadPoolExecutor(max_workers=2) as prefetch_pool:
                    future = _submit_batch(prefetch_pool, 0)
                    staged_gpu = None
                    staged_groups = None

                    for batch_idx in range(len(batch_group_sets)):
                        cpu_batch = future.result()
                        current_groups = batch_group_sets[batch_idx]
                        next_idx = batch_idx + 1
                        future = _submit_batch(prefetch_pool, next_idx) if next_idx < len(batch_group_sets) else None
                        new_staged = None
                        if cpu_batch[1].shape[0] > 0:
                            new_staged = _stage_training_batch_to_device(
                                cpu_batch,
                                cuda_cfg=cuda_cfg,
                                transfer_stream=transfer_stream,
                            )
                        if staged_gpu is not None:
                            _run_training_batch(staged_gpu)
                            _release_group_training_tensors(staged_groups)
                        staged_gpu = new_staged
                        staged_groups = current_groups

                    if staged_gpu is not None:
                        _run_training_batch(staged_gpu)
                        _release_group_training_tensors(staged_groups)
            denom = max(epoch_sample_count, 1)
            avg = (epoch_nll_sum / denom).item()
            avg_bt = (epoch_nll_bt / denom).item()
            avg_md = (epoch_nll_md / denom).item()
            avg_bd = (epoch_nll_bd / denom).item()
            avg_cd = (epoch_nll_cd / denom).item()
            avg_brt = (epoch_nll_brt / denom).item()
            stats["epoch_losses"].append(avg)
            stats["optimizer_steps"] += n_optimizer_steps

            elapsed = time.time() - t0
            print(
                f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}  "
                f"bt={avg_bt:.3f} md={avg_md:.3f} bd={avg_bd:.3f} cd={avg_cd:.4f} brt={avg_brt:.4f}  "
                f"({n_optimizer_steps} opt steps, {epoch_sample_count} samples)  "
                f"[{elapsed:.1f}s]"
            )

            if session_dir is not None:
                ckpt_path = session_dir / f"weights_epoch{epoch+1}.json"
                model.save_weights(str(ckpt_path))
                print(f"  saved checkpoint: {ckpt_path.name}")
    finally:
        if log_file is not None:
            log_file.close()

    stats["final_loss"] = stats["epoch_losses"][-1] if stats["epoch_losses"] else 0.0
    stats["total_samples"] = total_samples
    stats["n_games"] = dataset.n_games
    return stats


def main():
    parser = argparse.ArgumentParser(description="Actor BC pretraining from replays")
    parser.add_argument("--replay-dir", type=str, default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--index-path", type=str, default="")
    parser.add_argument("--min-rating", type=float, default=1500.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--games-per-batch", type=int, default=800)
    parser.add_argument("--tbptt-len", type=int, default=BPTT_K)
    parser.add_argument("--chunk-groups-per-batch", type=int, default=4)
    parser.add_argument("--init-weights", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--step-log-every", type=int, default=1)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Disable disk caching of parsed replays")
    parser.add_argument("--session", type=str, default="")
    parser.add_argument("--winner-only", dest="winner_only", action="store_true")
    parser.add_argument("--both-teams", dest="winner_only", action="store_false", default=False)
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)

    session_name = args.session or f"bc_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir = SESSIONS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "train_log.jsonl"
    config = {
        "mode": "actor_bc", "lr": args.lr, "epochs": args.epochs,
        "min_rating": args.min_rating,
        "session": session_name,
        "step_log_every": args.step_log_every,
    }
    with open(log_path, "w") as f:
        f.write(json.dumps({"config": config}) + "\n")

    print(f"Replay dir: {replay_dir}")
    print(f"Session: {session_name}")
    print(f"Device: {DEVICE}")
    cuda_cfg = resolve_cuda_config(DEVICE, spatial_inputs=False, enable_compile=not args.no_compile)
    if cuda_cfg.enabled:
        print(f"CUDA pretrain: amp={cuda_cfg.amp_dtype}, compile={'on' if cuda_cfg.compile_model else 'off'}")

    t0 = time.time()
    ds = ActorBCDataset(
        replay_dir,
        index_path=args.index_path or None,
        min_rating=args.min_rating,
        winner_only=args.winner_only,
        no_cache=args.no_cache,
    )
    print(f"Indexed {ds.n_games} games (~{ds.total_samples} est. samples) in {time.time()-t0:.1f}s")

    model = BuilderMLP().to(DEVICE)
    if args.init_weights:
        if not model.load_weights(args.init_weights):
            raise SystemExit(f"Could not load actor weights from {args.init_weights}")
        print(f"Loaded initial weights from {args.init_weights}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Actor: {n_params:,} params")

    stats = train_actor_bc(
        model, ds, epochs=args.epochs, lr=args.lr,
        games_per_batch=args.games_per_batch,
        tbptt_len=args.tbptt_len,
        chunk_groups_per_batch=args.chunk_groups_per_batch,
        log_path=log_path,
        step_log_every=args.step_log_every,
        enable_compile=not args.no_compile,
        session_dir=session_dir,
    )

    out_path = args.output or str(PROJECT_ROOT / "bots" / "tensor_inference" / "weights_bc.json")
    model.save_weights(out_path)
    print(f"Weights saved to {out_path}")
    print(f"Stats: {stats.get('total_samples', 0)} samples, "
          f"{stats.get('n_games', 0)} games, "
          f"final_loss={stats.get('final_loss', 0):.4f}")

    _run_eval_games(out_path, session_dir / "replays", n_games=2)


if __name__ == "__main__":
    main()
