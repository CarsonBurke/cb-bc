"""Benchmark: current tile SA vs RBF attention (softmax and unnormalized).

Measures the attention-only hot path: QKV projection through weighted sum.
Excludes tile encoding and residual/norm/SiLU (identical across variants).
"""

import math
import time

_DOT = math.sumprod
_EXP = math.exp
_SQRT = math.sqrt

TILE_EMBED_DIM = 8
TILE_GRID_N = 25
HALF = TILE_EMBED_DIM // 2


def _random_matrix(rows, cols, seed=42):
    """Deterministic pseudo-random matrix for benchmarking."""
    out = []
    x = seed
    for i in range(rows):
        row = []
        for j in range(cols):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            row.append((x / 0x7FFFFFFF - 0.5) * 2.0)
        out.append(row)
    return out


def _random_vec(n, seed=99):
    out = []
    x = seed
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((x / 0x7FFFFFFF - 0.5) * 2.0)
    return out


def setup():
    """Create shared test data."""
    tokens = _random_matrix(TILE_GRID_N, TILE_EMBED_DIM, seed=1)
    q_rows = _random_matrix(TILE_EMBED_DIM, TILE_EMBED_DIM, seed=2)
    q_bias = _random_vec(TILE_EMBED_DIM, seed=3)
    k_rows = _random_matrix(TILE_EMBED_DIM, TILE_EMBED_DIM, seed=4)
    k_bias = _random_vec(TILE_EMBED_DIM, seed=5)
    v_rows = _random_matrix(TILE_EMBED_DIM, TILE_EMBED_DIM, seed=6)
    v_bias = _random_vec(TILE_EMBED_DIM, seed=7)
    # RoPE cos/sin tables
    rope_cos = _random_matrix(TILE_GRID_N, HALF, seed=8)
    rope_sin = _random_matrix(TILE_GRID_N, HALF, seed=9)
    tau = 0.3
    gamma = 1.0
    return tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, rope_cos, rope_sin, tau, gamma


def bench_current(tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias,
                  rope_cos, rope_sin, tau):
    """Current: QK-norm + RoPE + scaled dot-product + softmax."""
    tile_sa_scale = _EXP(tau)

    # QKV projection + norms
    q_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    k_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
    scaled_k = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    q_scales = [0.0] * TILE_GRID_N

    for t in range(TILE_GRID_N):
        token = tokens[t]
        q_nsq = 0.0
        k_nsq = 0.0
        for j in range(TILE_EMBED_DIM):
            q_val = _DOT(token, q_rows[j]) + q_bias[j]
            k_val = _DOT(token, k_rows[j]) + k_bias[j]
            v_val = _DOT(token, v_rows[j]) + v_bias[j]
            q_vecs[t][j] = q_val
            k_vecs[t][j] = k_val
            v_cols[j][t] = v_val
            q_nsq += q_val * q_val
            k_nsq += k_val * k_val
        q_scale = 1.0 / max(_SQRT(q_nsq), 1e-6)
        k_scale = 1.0 / max(_SQRT(k_nsq), 1e-6)
        q_scales[t] = q_scale
        # RoPE + pre-scale keys
        cos_t = rope_cos[t]
        sin_t = rope_sin[t]
        s = k_scale * tile_sa_scale
        for ri in range(HALF):
            x1 = q_vecs[t][ri]
            x2 = q_vecs[t][HALF + ri]
            q_vecs[t][ri] = x1 * cos_t[ri] - x2 * sin_t[ri]
            q_vecs[t][HALF + ri] = x1 * sin_t[ri] + x2 * cos_t[ri]
            x1 = k_vecs[t][ri]
            x2 = k_vecs[t][HALF + ri]
            rk1 = x1 * cos_t[ri] - x2 * sin_t[ri]
            rk2 = x1 * sin_t[ri] + x2 * cos_t[ri]
            scaled_k[t][ri] = rk1 * s
            scaled_k[t][HALF + ri] = rk2 * s

    # Attention
    scores = [0.0] * TILE_GRID_N
    out = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    scaled_qi = [0.0] * TILE_EMBED_DIM

    for i in range(TILE_GRID_N):
        qs = q_scales[i]
        for j in range(TILE_EMBED_DIM):
            scaled_qi[j] = q_vecs[i][j] * qs
        best = -1e30
        for k in range(TILE_GRID_N):
            score = _DOT(scaled_qi, scaled_k[k])
            scores[k] = score
            if score > best:
                best = score
        z = 0.0
        for k in range(TILE_GRID_N):
            w = _EXP(scores[k] - best)
            scores[k] = w
            z += w
        inv_z = 1.0 / z
        for j in range(TILE_EMBED_DIM):
            out[i][j] = _DOT(scores, v_cols[j]) * inv_z
    return out


def bench_rbf_softmax(tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias,
                      gamma):
    """RBF with softmax: no QK-norm, no RoPE. softmax(2γ(QK^T) - γ||K||²)."""
    # QKV projection (no norms, no RoPE)
    q_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    k_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
    k_norm_sq = [0.0] * TILE_GRID_N

    for t in range(TILE_GRID_N):
        token = tokens[t]
        knsq = 0.0
        for j in range(TILE_EMBED_DIM):
            q_val = _DOT(token, q_rows[j]) + q_bias[j]
            k_val = _DOT(token, k_rows[j]) + k_bias[j]
            v_val = _DOT(token, v_rows[j]) + v_bias[j]
            q_vecs[t][j] = q_val
            k_vecs[t][j] = k_val
            v_cols[j][t] = v_val
            knsq += k_val * k_val
        k_norm_sq[t] = knsq

    # Attention
    scores = [0.0] * TILE_GRID_N
    out = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    g2 = 2.0 * gamma

    for i in range(TILE_GRID_N):
        best = -1e30
        for k in range(TILE_GRID_N):
            score = g2 * _DOT(q_vecs[i], k_vecs[k]) - gamma * k_norm_sq[k]
            scores[k] = score
            if score > best:
                best = score
        z = 0.0
        for k in range(TILE_GRID_N):
            w = _EXP(scores[k] - best)
            scores[k] = w
            z += w
        inv_z = 1.0 / z
        for j in range(TILE_EMBED_DIM):
            out[i][j] = _DOT(scores, v_cols[j]) * inv_z
    return out


def bench_rbf_unnorm(tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias,
                     gamma):
    """Unnormalized RBF: no QK-norm, no RoPE, no softmax. exp(-γ||q-k||²) @ V."""
    # QKV projection
    q_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    k_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
    k_norm_sq = [0.0] * TILE_GRID_N
    q_norm_sq = [0.0] * TILE_GRID_N

    for t in range(TILE_GRID_N):
        token = tokens[t]
        knsq = 0.0
        qnsq = 0.0
        for j in range(TILE_EMBED_DIM):
            q_val = _DOT(token, q_rows[j]) + q_bias[j]
            k_val = _DOT(token, k_rows[j]) + k_bias[j]
            v_val = _DOT(token, v_rows[j]) + v_bias[j]
            q_vecs[t][j] = q_val
            k_vecs[t][j] = k_val
            v_cols[j][t] = v_val
            knsq += k_val * k_val
            qnsq += q_val * q_val
        k_norm_sq[t] = knsq
        q_norm_sq[t] = qnsq

    # Attention — no max, no normalization
    weights = [0.0] * TILE_GRID_N
    out = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]

    for i in range(TILE_GRID_N):
        qnsq = q_norm_sq[i]
        for k in range(TILE_GRID_N):
            dist_sq = qnsq - 2.0 * _DOT(q_vecs[i], k_vecs[k]) + k_norm_sq[k]
            weights[k] = _EXP(-gamma * dist_sq)
        for j in range(TILE_EMBED_DIM):
            out[i][j] = _DOT(weights, v_cols[j])
    return out


def bench_rbf_unnorm_efficient(tokens, q_rows, q_bias, k_rows, k_bias,
                               v_rows, v_bias, gamma):
    """Unnormalized RBF using the efficient form: exp(-γ||q||²) * exp(2γ(q·k) - γ||k||²).

    Since exp(-γ||q||²) is constant per query, factor it out and apply once
    after the weighted sum. This avoids computing full distances."""
    # QKV projection
    q_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    k_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
    k_norm_sq = [0.0] * TILE_GRID_N
    q_norm_sq = [0.0] * TILE_GRID_N

    for t in range(TILE_GRID_N):
        token = tokens[t]
        knsq = 0.0
        qnsq = 0.0
        for j in range(TILE_EMBED_DIM):
            q_val = _DOT(token, q_rows[j]) + q_bias[j]
            k_val = _DOT(token, k_rows[j]) + k_bias[j]
            v_val = _DOT(token, v_rows[j]) + v_bias[j]
            q_vecs[t][j] = q_val
            k_vecs[t][j] = k_val
            v_cols[j][t] = v_val
            knsq += k_val * k_val
            qnsq += q_val * q_val
        k_norm_sq[t] = knsq
        q_norm_sq[t] = qnsq

    # Attention — factor out q norm
    weights = [0.0] * TILE_GRID_N
    out = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    g2 = 2.0 * gamma

    for i in range(TILE_GRID_N):
        q_factor = _EXP(-gamma * q_norm_sq[i])
        for k in range(TILE_GRID_N):
            # exp(2γ(q·k) - γ||k||²) — same sumprod as softmax RBF
            weights[k] = _EXP(g2 * _DOT(q_vecs[i], k_vecs[k]) - gamma * k_norm_sq[k])
        for j in range(TILE_EMBED_DIM):
            out[i][j] = _DOT(weights, v_cols[j]) * q_factor
    return out


def bench_rbf_softmax_fused(tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias,
                            gamma):
    """Softmax RBF with -γ folded into k_norm_sq and √(2γ) folded into q,k.

    Pre-folding saves one scalar multiply per (i, k) pair; pre-scaling q,k
    avoids the g2 multiply inside the score loop. Inner loop becomes a pure
    dot plus add."""
    sqrt_g2 = (2.0 * gamma) ** 0.5
    q_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    k_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
    neg_g_knsq = [0.0] * TILE_GRID_N

    for t in range(TILE_GRID_N):
        token = tokens[t]
        knsq = 0.0
        for j in range(TILE_EMBED_DIM):
            q_val = (_DOT(token, q_rows[j]) + q_bias[j]) * sqrt_g2
            k_val_raw = _DOT(token, k_rows[j]) + k_bias[j]
            k_val = k_val_raw * sqrt_g2
            v_val = _DOT(token, v_rows[j]) + v_bias[j]
            q_vecs[t][j] = q_val
            k_vecs[t][j] = k_val
            v_cols[j][t] = v_val
            knsq += k_val_raw * k_val_raw
        neg_g_knsq[t] = -gamma * knsq

    scores = [0.0] * TILE_GRID_N
    out = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]

    for i in range(TILE_GRID_N):
        qi = q_vecs[i]
        best = -1e30
        for k in range(TILE_GRID_N):
            score = _DOT(qi, k_vecs[k]) + neg_g_knsq[k]
            scores[k] = score
            if score > best:
                best = score
        z = 0.0
        for k in range(TILE_GRID_N):
            w = _EXP(scores[k] - best)
            scores[k] = w
            z += w
        inv_z = 1.0 / z
        for j in range(TILE_EMBED_DIM):
            out[i][j] = _DOT(scores, v_cols[j]) * inv_z
    return out


def bench_rbf_unnorm_fused(tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias,
                           gamma):
    """Non-softmax RBF + fusion: pre-fold γ into k_norm_sq and q_norm_sq,
    pre-scale q,k by √(2γ). Inner loop: exp(dot + neg_g_knsq[k]) * q_factor."""
    sqrt_g2 = (2.0 * gamma) ** 0.5
    q_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    k_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    v_cols = [[0.0] * TILE_GRID_N for _ in range(TILE_EMBED_DIM)]
    neg_g_knsq = [0.0] * TILE_GRID_N
    neg_g_qnsq = [0.0] * TILE_GRID_N

    for t in range(TILE_GRID_N):
        token = tokens[t]
        knsq = 0.0
        qnsq = 0.0
        for j in range(TILE_EMBED_DIM):
            q_val_raw = _DOT(token, q_rows[j]) + q_bias[j]
            k_val_raw = _DOT(token, k_rows[j]) + k_bias[j]
            v_val = _DOT(token, v_rows[j]) + v_bias[j]
            q_vecs[t][j] = q_val_raw * sqrt_g2
            k_vecs[t][j] = k_val_raw * sqrt_g2
            v_cols[j][t] = v_val
            qnsq += q_val_raw * q_val_raw
            knsq += k_val_raw * k_val_raw
        neg_g_knsq[t] = -gamma * knsq
        neg_g_qnsq[t] = -gamma * qnsq

    weights = [0.0] * TILE_GRID_N
    out = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]

    for i in range(TILE_GRID_N):
        qi = q_vecs[i]
        q_factor = _EXP(neg_g_qnsq[i])
        for k in range(TILE_GRID_N):
            weights[k] = _EXP(_DOT(qi, k_vecs[k]) + neg_g_knsq[k])
        for j in range(TILE_EMBED_DIM):
            out[i][j] = _DOT(weights, v_cols[j]) * q_factor
    return out


def _susie_tables(n_tokens, dim, seed=11):
    """Precomputed additive sinusoidal table * learnable pos_weight (pretend
    pos_weight=0.5, matching the ref's init). Shape: (n_tokens, dim)."""
    # Reference constructs unrotated sinusoids [sin(t/θ^{2i/d}), cos(t/θ^{2i/d})]
    # then multiplies by learnable pos_weight (head-wise, tied across sin/cos).
    # For bench purposes we use a fixed scale of 0.5.
    half = dim // 2
    theta = 10000.0
    inv_freq = [1.0 / (theta ** (2 * i / dim)) for i in range(half)]
    table = []
    for t in range(n_tokens):
        row = [0.0] * dim
        for i in range(half):
            row[i] = math.sin(t * inv_freq[i]) * 0.5
            row[half + i] = math.cos(t * inv_freq[i]) * 0.5
        table.append(row)
    return table


def bench_target(tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, gamma):
    """Target: SuSIE additive position + non-softmax RBF + 1 look-away token.

    - SuSIE: q,k gain an additive sinusoidal vector scaled by pos_weight.
    - Register: one zero-input key/value slot at index 0 (projects to bias).
    - Non-softmax: weights = exp(-γ||q-k||²), no denominator.
    - Folded: γ pre-multiplied into k_norm_sq and q_norm_sq.
    """
    N_REG = 1
    N_TOT = TILE_GRID_N + N_REG
    susie = _susie_tables(TILE_GRID_N, TILE_EMBED_DIM)
    sqrt_g2 = (2.0 * gamma) ** 0.5

    q_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]
    k_vecs = [[0.0] * TILE_EMBED_DIM for _ in range(N_TOT)]
    v_cols = [[0.0] * N_TOT for _ in range(TILE_EMBED_DIM)]
    neg_g_knsq = [0.0] * N_TOT
    neg_g_qnsq = [0.0] * TILE_GRID_N

    # Register (look-away) token at slot 0: zero input → k,v = biases.
    knsq0 = 0.0
    for j in range(TILE_EMBED_DIM):
        k_val = k_bias[j]
        v_val = v_bias[j]
        k_vecs[0][j] = k_val * sqrt_g2
        v_cols[j][0] = v_val
        knsq0 += k_val * k_val
    neg_g_knsq[0] = -gamma * knsq0

    # Real tokens
    for t in range(TILE_GRID_N):
        token = tokens[t]
        pos = susie[t]
        knsq = 0.0
        qnsq = 0.0
        for j in range(TILE_EMBED_DIM):
            q_val_raw = _DOT(token, q_rows[j]) + q_bias[j] + pos[j]
            k_val_raw = _DOT(token, k_rows[j]) + k_bias[j] + pos[j]
            v_val = _DOT(token, v_rows[j]) + v_bias[j]
            q_vecs[t][j] = q_val_raw * sqrt_g2
            k_vecs[N_REG + t][j] = k_val_raw * sqrt_g2
            v_cols[j][N_REG + t] = v_val
            qnsq += q_val_raw * q_val_raw
            knsq += k_val_raw * k_val_raw
        neg_g_knsq[N_REG + t] = -gamma * knsq
        neg_g_qnsq[t] = -gamma * qnsq

    weights = [0.0] * N_TOT
    out = [[0.0] * TILE_EMBED_DIM for _ in range(TILE_GRID_N)]

    for i in range(TILE_GRID_N):
        qi = q_vecs[i]
        q_factor = _EXP(neg_g_qnsq[i])
        for k in range(N_TOT):
            weights[k] = _EXP(_DOT(qi, k_vecs[k]) + neg_g_knsq[k])
        for j in range(TILE_EMBED_DIM):
            out[i][j] = _DOT(weights, v_cols[j]) * q_factor
    return out


def main():
    data = setup()
    tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, rope_cos, rope_sin, tau, gamma = data

    variants = [
        ("Current (QK-norm+RoPE+softmax)", lambda: bench_current(
            tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias,
            rope_cos, rope_sin, tau)),
        ("RBF softmax (no norm/RoPE)", lambda: bench_rbf_softmax(
            tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, gamma)),
        ("RBF softmax + fused γ,√2γ", lambda: bench_rbf_softmax_fused(
            tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, gamma)),
        ("RBF unnorm (full dist)", lambda: bench_rbf_unnorm(
            tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, gamma)),
        ("RBF unnorm (efficient)", lambda: bench_rbf_unnorm_efficient(
            tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, gamma)),
        ("RBF unnorm + fused γ,√2γ", lambda: bench_rbf_unnorm_fused(
            tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, gamma)),
        ("Target: unnorm+SuSIE+look-away", lambda: bench_target(
            tokens, q_rows, q_bias, k_rows, k_bias, v_rows, v_bias, gamma)),
    ]

    N_WARMUP = 500
    N_ITERS = 5000

    print(f"Benchmarking tile SA attention ({TILE_GRID_N} tiles, dim={TILE_EMBED_DIM})")
    print(f"  {N_WARMUP} warmup + {N_ITERS} iterations\n")

    results = []
    for name, fn in variants:
        # Warmup
        for _ in range(N_WARMUP):
            fn()
        # Bench
        t0 = time.perf_counter_ns()
        for _ in range(N_ITERS):
            fn()
        elapsed_ns = time.perf_counter_ns() - t0
        us_per_call = elapsed_ns / N_ITERS / 1000
        results.append((name, us_per_call))

    baseline = results[0][1]
    print(f"{'Variant':<38} {'µs/call':>8} {'vs current':>12}")
    print(f"{'-'*38} {'-'*8} {'-'*12}")
    for name, us in results:
        speedup = baseline / us
        print(f"{name:<38} {us:>8.1f} {speedup:>11.2f}x")


if __name__ == "__main__":
    main()
