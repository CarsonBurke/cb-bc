"""Profile BPTT obs_encoder_pre batching vs sequential per-step residual."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
TENSOR_DIR = PROJECT_ROOT / "bots" / "tensor"
PRETRAIN_DIR = TORCH_DIR / "pretrain"

for d in (TORCH_DIR, TENSOR_DIR, PRETRAIN_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import model as model_mod
from model_constants import OBS_SIZE, H_DIM, OBS_PREFIX_SIZE

BuilderMLP = model_mod.BuilderMLP
BPTT_K = model_mod.BPTT_K
DEVICE = model_mod.DEVICE

B = 64
K = BPTT_K  # 32
WARMUP = 20
ITERS = 100


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _bench(fn, warmup=WARMUP, iters=ITERS) -> float:
    """Return median time in ms over `iters` runs."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


@torch.no_grad()
def test_bptt_profile():
    model = BuilderMLP().to(DEVICE).eval()

    obs_flat = torch.randn(B * K, OBS_SIZE, device=DEVICE)
    h_prev = torch.randn(B, H_DIM, device=DEVICE)

    # Pre-compute for per-step benchmarks
    obs_pre_all = model._obs_encoder_pre(obs_flat)  # (B*K, enc_dim)
    enc_dim = obs_pre_all.shape[-1]
    obs_pre_seq = obs_pre_all.view(B, K, enc_dim)  # (B, K, enc_dim)
    single_obs_pre = obs_pre_seq[:, 0]  # (B, enc_dim)

    # --- (a) Batched obs_encoder_pre for B*K ---
    def batched_enc():
        return model._obs_encoder_pre(obs_flat)

    t_enc = _bench(batched_enc)

    # --- (b) Per-step residual: h-dependent linear + remaining encoder + trunk + heads + GRU ---
    enc_linear = model.encoder[0]

    def single_step_residual():
        h_pre = F.linear(h_prev, enc_linear.weight[:, OBS_PREFIX_SIZE:], None)
        enc = model.encoder[1](single_obs_pre + h_pre)
        enc = model.encoder[2](enc)
        lat = model.actor_trunk(enc)
        bt_l, bd_l, cd_l, brt_l, md_l, eps, h_gate, h_cand = model._project_actor_heads(lat)
        z = torch.sigmoid(h_gate)
        h_new = (1.0 - z) * h_prev + z * h_cand
        return h_new

    t_residual_1 = _bench(single_step_residual)

    # --- Per-step residual x32 sequential (simulating loop with h dependency) ---
    def residual_loop_32():
        h_t = h_prev
        for t in range(K):
            h_pre = F.linear(h_t, enc_linear.weight[:, OBS_PREFIX_SIZE:], None)
            enc = model.encoder[1](obs_pre_seq[:, t] + h_pre)
            enc = model.encoder[2](enc)
            lat = model.actor_trunk(enc)
            bt_l, bd_l, cd_l, brt_l, md_l, eps, h_gate, h_cand = model._project_actor_heads(lat)
            z = torch.sigmoid(h_gate)
            h_t = (1.0 - z) * h_t + z * h_cand
        return h_t

    t_residual_32 = _bench(residual_loop_32)

    # --- (c) Current full sequential loop (encoder + everything, K steps) ---
    obs_seq = obs_flat.view(B, K, OBS_SIZE)

    def full_sequential_32():
        h_t = h_prev
        for t in range(K):
            obs_pre_t = model._obs_encoder_pre(obs_seq[:, t])
            h_pre = F.linear(h_t, enc_linear.weight[:, OBS_PREFIX_SIZE:], None)
            enc = model.encoder[1](obs_pre_t + h_pre)
            enc = model.encoder[2](enc)
            lat = model.actor_trunk(enc)
            bt_l, bd_l, cd_l, brt_l, md_l, eps, h_gate, h_cand = model._project_actor_heads(lat)
            z = torch.sigmoid(h_gate)
            h_t = (1.0 - z) * h_t + z * h_cand
        return h_t

    t_full_seq = _bench(full_sequential_32)

    # --- Summary ---
    t_projected = t_enc + t_residual_32

    print()
    print(f"Device: {DEVICE}")
    print(f"B={B}, K={K}, OBS_SIZE={OBS_SIZE}, H_DIM={H_DIM}")
    print(f"{'=' * 60}")
    print(f"Batched obs_encoder_pre (B*K):          {t_enc:7.2f} ms")
    print(f"Per-step residual (single step):        {t_residual_1:7.2f} ms")
    print(f"Per-step residual (32 steps seq):       {t_residual_32:7.2f} ms")
    print(f"Current full sequential loop (32 steps):{t_full_seq:7.2f} ms")
    print(f"{'=' * 60}")
    print(f"Projected optimized loop: batched_enc + 32*residual = {t_projected:.2f} ms")
    speedup = t_full_seq / t_projected if t_projected > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_bptt_profile()
