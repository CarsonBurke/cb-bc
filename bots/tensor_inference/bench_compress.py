"""Microbenchmark: current Linear(200,80) compress vs low-rank factored alternatives.

Isolates the compress step (lines 578-597 of nn.py infer()) to measure
wall-clock impact of replacing the single matmul with a low-rank bottleneck.
"""

from __future__ import annotations

import argparse
import math
import random
import time

_DOT = math.sumprod
_exp = math.exp


def _make_fake_encoded(n: int = 200) -> list[float]:
    """Simulate post-conv tile embeddings (25 tiles × 8 channels = 200)."""
    return [random.gauss(0.0, 0.5) for _ in range(n)]


def _make_weights(rows: int, cols: int) -> list[list[float]]:
    """Pre-transposed weight rows for sumprod: w_rows[j] has length rows."""
    return [[random.gauss(0.0, 0.1) for _ in range(rows)] for _ in range(cols)]


def _make_bias(n: int) -> list[float]:
    return [random.gauss(0.0, 0.01) for _ in range(n)]


def _make_gamma(n: int) -> list[float]:
    return [1.0] * n


# ---------- Current: Linear(200, 80) + RMSNorm + SiLU ----------

def compress_current(encoded: list[float], w_rows: list[list[float]],
                     bias: list[float], N: int,
                     gamma: list[float], eps: float) -> list[float]:
    compressed = [0.0] * N
    mean_sq = 0.0
    for j in range(N):
        x = _DOT(encoded, w_rows[j])
        x += bias[j]
        compressed[j] = x
        mean_sq += x * x
    inv_rms = 1.0 / (mean_sq / N + eps) ** 0.5
    for j in range(N):
        x = compressed[j] * inv_rms * gamma[j]
        if x >= 0:
            compressed[j] = x / (1.0 + _exp(-x))
        else:
            e = _exp(x)
            compressed[j] = x * e / (1.0 + e)
    return compressed


# ---------- Low-rank: Linear(200, r) + SiLU + Linear(r, 80) + RMSNorm + SiLU ----------

def compress_lowrank(encoded: list[float],
                     w1_rows: list[list[float]], b1: list[float], r: int,
                     w2_rows: list[list[float]], b2: list[float], N: int,
                     gamma: list[float], eps: float) -> list[float]:
    # Stage 1: Linear(200, r) + SiLU
    mid = [0.0] * r
    for j in range(r):
        x = _DOT(encoded, w1_rows[j]) + b1[j]
        if x >= 0:
            mid[j] = x / (1.0 + _exp(-x))
        else:
            e = _exp(x)
            mid[j] = x * e / (1.0 + e)

    # Stage 2: Linear(r, 80) + RMSNorm + SiLU
    compressed = [0.0] * N
    mean_sq = 0.0
    for j in range(N):
        x = _DOT(mid, w2_rows[j]) + b2[j]
        compressed[j] = x
        mean_sq += x * x
    inv_rms = 1.0 / (mean_sq / N + eps) ** 0.5
    for j in range(N):
        x = compressed[j] * inv_rms * gamma[j]
        if x >= 0:
            compressed[j] = x / (1.0 + _exp(-x))
        else:
            e = _exp(x)
            compressed[j] = x * e / (1.0 + e)
    return compressed


# ---------- Low-rank direct (no intermediate projection back to 80) ----------
# Output r dims directly, skip the second linear. Trunk input shrinks.

def compress_lowrank_direct(encoded: list[float],
                            w1_rows: list[list[float]], b1: list[float], r: int,
                            gamma: list[float], eps: float) -> list[float]:
    # Linear(200, r) + RMSNorm + SiLU
    compressed = [0.0] * r
    mean_sq = 0.0
    for j in range(r):
        x = _DOT(encoded, w1_rows[j]) + b1[j]
        compressed[j] = x
        mean_sq += x * x
    inv_rms = 1.0 / (mean_sq / r + eps) ** 0.5
    for j in range(r):
        x = compressed[j] * inv_rms * gamma[j]
        if x >= 0:
            compressed[j] = x / (1.0 + _exp(-x))
        else:
            e = _exp(x)
            compressed[j] = x * e / (1.0 + e)
    return compressed


def _bench(fn, args, iterations: int, warmup: int = 500) -> float:
    for _ in range(warmup):
        fn(*args)
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    elapsed = time.perf_counter() - start
    return elapsed * 1e6 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    args = parser.parse_args()

    random.seed(42)
    eps = 1e-6
    encoded = _make_fake_encoded(200)
    N = 80  # current compress output dim

    # Current: Linear(200, 80)
    w_curr = _make_weights(200, N)
    b_curr = _make_bias(N)
    g_curr = _make_gamma(N)

    us_current = _bench(compress_current, (encoded, w_curr, b_curr, N, g_curr, eps), args.iters)

    print(f"{'Variant':<35} {'µs':>8} {'vs current':>12} {'sumprod calls':>15} {'params':>8}")
    print("-" * 82)
    print(f"{'current Linear(200,80)':<35} {us_current:>8.2f} {'baseline':>12} {80:>15} {200*80+80:>8}")

    for r in (8, 12, 16, 24):
        # Low-rank: Linear(200, r) + SiLU + Linear(r, 80)
        w1 = _make_weights(200, r)
        b1 = _make_bias(r)
        w2 = _make_weights(r, N)
        b2 = _make_bias(N)
        g = _make_gamma(N)

        us = _bench(compress_lowrank, (encoded, w1, b1, r, w2, b2, N, g, eps), args.iters)
        params = 200 * r + r + r * N + N
        speedup = (us_current - us) / us_current * 100
        print(f"{'lowrank r=' + str(r) + ' → 80':<35} {us:>8.2f} {speedup:>+11.1f}% {r + N:>15} {params:>8}")

    for r in (16, 24, 32, 48):
        # Direct: Linear(200, r) only, no second projection
        w1 = _make_weights(200, r)
        b1 = _make_bias(r)
        g = _make_gamma(r)

        us = _bench(compress_lowrank_direct, (encoded, w1, b1, r, g, eps), args.iters)
        params = 200 * r + r
        speedup = (us_current - us) / us_current * 100
        print(f"{'direct r=' + str(r):<35} {us:>8.2f} {speedup:>+11.1f}% {r:>15} {params:>8}")


if __name__ == "__main__":
    main()
