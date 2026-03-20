#!/usr/bin/env python3
"""Microbenchmark for CentralizedCritic forward + backward passes.

Measures wall-clock time for encode_map (CNN) and unit_values (MLP) under
various optimization configurations. Run with no arguments for a full
comparison, or pass specific flags to test individual optimizations.

Usage:
    python bots/tensor/torch/bench_critic.py
    python bots/tensor/torch/bench_critic.py --only baseline channels_last
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# ---- Ensure project importable ----
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import bots.tensor.torch.model as _model_mod
from bots.tensor.torch.model import (
    CentralizedCritic,
    DEVICE,
    GLOBAL_MAP_CHANNELS,
    GLOBAL_MAP_MAX_H,
    GLOBAL_MAP_MAX_W,
    GLOBAL_SCALAR_DIM,
    UNIT_CONTEXT_DIM,
    ACTOR_LATENT_DIM,
)

# ---- Benchmark parameters ----
WARMUP_ITERS = 5
BENCH_ITERS = 30
BATCH_ROUNDS = 64        # number of map rounds per forward pass
MAP_H, MAP_W = 40, 40    # typical map size
UNITS_PER_ROUND = 20     # avg units referencing each round
N_UNITS = BATCH_ROUNDS * UNITS_PER_ROUND


@dataclass
class BenchResult:
    name: str
    fwd_ms: float = 0.0
    bwd_ms: float = 0.0
    total_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


def make_synthetic_data(
    device: torch.device,
    channels_last: bool = False,
) -> dict:
    """Generate random tensors matching critic input shapes."""
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format

    map_tensor = torch.randn(
        BATCH_ROUNDS, GLOBAL_MAP_CHANNELS, MAP_H, MAP_W,
        device=device, dtype=torch.float32,
    ).to(memory_format=mem_fmt)

    valid_mask = torch.ones(
        BATCH_ROUNDS, 1, MAP_H, MAP_W,
        device=device, dtype=torch.float32,
    ).to(memory_format=mem_fmt)

    round_scalars = torch.randn(
        BATCH_ROUNDS, GLOBAL_SCALAR_DIM, device=device,
    )

    round_idx = torch.arange(BATCH_ROUNDS, device=device).repeat_interleave(UNITS_PER_ROUND)
    unit_xy = torch.randint(0, min(MAP_H, MAP_W), (N_UNITS, 2), device=device)
    unit_ctx = torch.randn(N_UNITS, UNIT_CONTEXT_DIM, device=device)
    actor_latent = torch.randn(N_UNITS, ACTOR_LATENT_DIM, device=device)

    return dict(
        map_tensor=map_tensor,
        valid_mask=valid_mask,
        round_scalars=round_scalars,
        round_idx=round_idx,
        unit_xy=unit_xy,
        unit_ctx=unit_ctx,
        actor_latent=actor_latent,
    )


def bench_forward_backward(
    critic: nn.Module,
    data: dict,
    *,
    use_amp: bool = False,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> tuple[float, float]:
    """Returns (fwd_ms, bwd_ms) averaged over iters."""

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextmanager(lambda: (yield))()

    # Warmup
    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextmanager(lambda: (yield))():
            feat = critic.encode_map(data["map_tensor"], data["valid_mask"])
            vals = critic.unit_values(
                feat, data["round_scalars"], data["round_idx"],
                data["unit_xy"], data["unit_ctx"], data["actor_latent"],
            )
            loss = vals.sum()
        loss.backward()
        critic.zero_grad(set_to_none=True)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []

    for _ in range(iters):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextmanager(lambda: (yield))():
            feat = critic.encode_map(data["map_tensor"], data["valid_mask"])
            vals = critic.unit_values(
                feat, data["round_scalars"], data["round_idx"],
                data["unit_xy"], data["unit_ctx"], data["actor_latent"],
            )
            loss = vals.sum()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        critic.zero_grad(set_to_none=True)

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)

    fwd_ms = sum(fwd_times) / len(fwd_times)
    bwd_ms = sum(bwd_times) / len(bwd_times)
    return fwd_ms, bwd_ms


def fresh_critic(**kwargs) -> CentralizedCritic:
    """Create a fresh critic on DEVICE."""
    return CentralizedCritic(**kwargs).to(DEVICE)


def run_baseline() -> BenchResult:
    """Baseline: no optimizations (disable built-in bf16)."""
    saved = _model_mod._USE_BF16
    _model_mod._USE_BF16 = False
    critic = fresh_critic()
    critic.train()
    data = make_synthetic_data(DEVICE)
    fwd, bwd = bench_forward_backward(critic, data)
    _model_mod._USE_BF16 = saved
    return BenchResult("baseline", fwd, bwd, fwd + bwd)


def _set_bf16(val: bool) -> bool:
    """Set model-level bf16 flag, return old value."""
    saved = _model_mod._USE_BF16
    _model_mod._USE_BF16 = val
    return saved


def run_cudnn_benchmark() -> BenchResult:
    """Enable cudnn.benchmark for conv autotuning."""
    saved_bf16 = _set_bf16(False)
    old = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    critic = fresh_critic()
    critic.train()
    data = make_synthetic_data(DEVICE)
    fwd, bwd = bench_forward_backward(critic, data)
    torch.backends.cudnn.benchmark = old
    _set_bf16(saved_bf16)
    return BenchResult("cudnn_benchmark", fwd, bwd, fwd + bwd)


def run_channels_last() -> BenchResult:
    """Channels-last memory format for conv layers."""
    saved_bf16 = _set_bf16(False)
    critic = fresh_critic()
    critic = critic.to(memory_format=torch.channels_last)
    critic.train()
    data = make_synthetic_data(DEVICE, channels_last=True)
    fwd, bwd = bench_forward_backward(critic, data)
    _set_bf16(saved_bf16)
    return BenchResult("channels_last", fwd, bwd, fwd + bwd)


def run_amp_bf16() -> BenchResult:
    """bfloat16 native conv weights via _USE_BF16."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return BenchResult("amp_bf16", notes=["bf16 not supported on this GPU"])
    saved_bf16 = _set_bf16(True)
    critic = fresh_critic()
    critic.train()
    data = make_synthetic_data(DEVICE)
    fwd, bwd = bench_forward_backward(critic, data)
    _set_bf16(saved_bf16)
    return BenchResult("amp_bf16", fwd, bwd, fwd + bwd)


def run_compile() -> BenchResult:
    """torch.compile the entire critic."""
    saved_bf16 = _set_bf16(False)
    critic = fresh_critic()
    critic.train()
    critic = torch.compile(critic)
    data = make_synthetic_data(DEVICE)
    # Extra warmup for compilation
    fwd, bwd = bench_forward_backward(critic, data, warmup=10)
    _set_bf16(saved_bf16)
    return BenchResult("torch_compile", fwd, bwd, fwd + bwd)


def run_channels_last_amp() -> BenchResult:
    """Channels-last + bf16 autocast combined."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return BenchResult("channels_last+amp", notes=["bf16 not supported"])
    saved_bf16 = _set_bf16(True)
    critic = fresh_critic()
    critic = critic.to(memory_format=torch.channels_last)
    critic.train()
    data = make_synthetic_data(DEVICE, channels_last=True)
    fwd, bwd = bench_forward_backward(critic, data)
    _set_bf16(saved_bf16)
    return BenchResult("channels_last+amp", fwd, bwd, fwd + bwd)


def run_compile_amp() -> BenchResult:
    """torch.compile + bf16 autocast."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return BenchResult("compile+amp", notes=["bf16 not supported"])
    saved_bf16 = _set_bf16(True)
    critic = fresh_critic()
    critic.train()
    critic = torch.compile(critic)
    data = make_synthetic_data(DEVICE)
    fwd, bwd = bench_forward_backward(critic, data, warmup=10)
    _set_bf16(saved_bf16)
    return BenchResult("compile+amp", fwd, bwd, fwd + bwd)


def run_all_combined() -> BenchResult:
    """All optimizations: channels_last + bf16 + compile."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return BenchResult("all_combined", notes=["bf16 not supported"])
    saved_bf16 = _set_bf16(True)
    critic = fresh_critic()
    critic = critic.to(memory_format=torch.channels_last)
    critic.train()
    critic = torch.compile(critic)
    data = make_synthetic_data(DEVICE, channels_last=True)
    fwd, bwd = bench_forward_backward(critic, data, warmup=10)
    _set_bf16(saved_bf16)
    return BenchResult("all_combined", fwd, bwd, fwd + bwd)


ALL_BENCHMARKS = {
    "baseline": run_baseline,
    "cudnn_benchmark": run_cudnn_benchmark,
    "channels_last": run_channels_last,
    "amp_bf16": run_amp_bf16,
    "torch_compile": run_compile,
    "channels_last+amp": run_channels_last_amp,
    "compile+amp": run_compile_amp,
    "all_combined": run_all_combined,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark CentralizedCritic optimizations")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Run only specific benchmarks (space-separated names)",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
    print(f"Benchmark: {BATCH_ROUNDS} rounds x {MAP_H}x{MAP_W} maps, "
          f"{N_UNITS} unit-steps, {BENCH_ITERS} iters")
    print()

    to_run = args.only if args.only else list(ALL_BENCHMARKS.keys())
    results: list[BenchResult] = []

    for name in to_run:
        if name not in ALL_BENCHMARKS:
            print(f"Unknown benchmark: {name}")
            continue
        print(f"Running {name}...", end=" ", flush=True)
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            r = ALL_BENCHMARKS[name]()
            results.append(r)
            if r.notes:
                print(f"SKIP ({', '.join(r.notes)})")
            else:
                print(f"fwd={r.fwd_ms:.1f}ms  bwd={r.bwd_ms:.1f}ms  total={r.total_ms:.1f}ms")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append(BenchResult(name, notes=[str(e)]))

    # Summary table
    baseline = next((r for r in results if r.name == "baseline" and r.total_ms > 0), None)

    print()
    print(f"{'Config':<22} {'Fwd ms':>8} {'Bwd ms':>8} {'Total ms':>9} {'Speedup':>8}")
    print("-" * 60)
    for r in results:
        if r.notes:
            print(f"{r.name:<22} {'—':>8} {'—':>8} {'—':>9} {', '.join(r.notes)}")
            continue
        speedup = ""
        if baseline and baseline.total_ms > 0 and r.total_ms > 0:
            s = baseline.total_ms / r.total_ms
            speedup = f"{s:.2f}x"
        print(f"{r.name:<22} {r.fwd_ms:>8.1f} {r.bwd_ms:>8.1f} {r.total_ms:>9.1f} {speedup:>8}")


if __name__ == "__main__":
    main()
