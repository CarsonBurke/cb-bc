"""Microbenchmark: conv variants for cross-channel spatial mixing.

Key insight: a full conv can be decomposed as sum-of-sumprod(8) over neighbors,
avoiding the expensive gather into a 72-dim patch buffer.
"""

from __future__ import annotations

import argparse
import math
import random
import time

_DOT = math.sumprod
_exp = math.exp

N_TILES = 25
D_TILE = 8
GRID_W = 5


def _make_encoded():
    return [random.gauss(0.0, 0.5) for _ in range(N_TILES * D_TILE)]

def _make_weights(fi, fo):
    return [[random.gauss(0.0, 0.1) for _ in range(fi)] for _ in range(fo)]

def _make_bias(n):
    return [random.gauss(0.0, 0.01) for _ in range(n)]

def _make_gamma(n):
    return [1.0] * n


def _build_neighbors():
    neighbors = []
    for t in range(N_TILES):
        r, c = t // GRID_W, t % GRID_W
        nbrs = []
        for kr in range(-1, 2):
            for kc in range(-1, 2):
                nr, nc = r + kr, c + kc
                if 0 <= nr < GRID_W and 0 <= nc < GRID_W:
                    nbrs.append((nr * GRID_W + nc, (kr + 1) * 3 + (kc + 1)))
        neighbors.append(tuple(nbrs))
    return neighbors


# ========== Current: depthwise 3×3 + pointwise ==========

def conv_depthwise_pw(encoded_orig, conv_w, conv_b, pw_w_rows, pw_b,
                      post_gamma, eps, neighbors):
    encoded = list(encoded_orig)
    te_out = D_TILE; n_tiles = N_TILES; total = n_tiles * te_out

    conv_out = [0.0] * total
    for t in range(n_tiles):
        nbr = neighbors[t]; t_base = t * te_out
        for c in range(te_out):
            cw = conv_w[c]; s = conv_b[c]
            for nt, k in nbr: s += encoded[nt * te_out + c] * cw[k]
            conv_out[t_base + c] = s

    projected = [0.0] * total
    for t in range(n_tiles):
        src = conv_out[t * te_out:(t + 1) * te_out]; base = t * te_out
        for j in range(te_out):
            x = _DOT(src, pw_w_rows[j])
            if pw_b is not None: x += pw_b[j]
            projected[base + j] = x

    mean_sq = [0.0] * n_tiles
    for i in range(total):
        t = i // te_out; x = encoded[i] + projected[i]; encoded[i] = x; mean_sq[t] += x * x
    inv_rms = [1.0 / (ms / te_out + eps) ** 0.5 for ms in mean_sq]
    for i in range(total):
        c = i % te_out; x = encoded[i] * inv_rms[i // te_out] * post_gamma[c]
        if x >= 0: encoded[i] = x / (1.0 + _exp(-x))
        else: e = _exp(x); encoded[i] = x * e / (1.0 + e)
    return encoded


# ========== Full conv via sum-of-sumprod(8) per neighbor ==========
# W is reshaped as w_blocks[kpos][out_ch] = 8-dim weight vector
# For each tile: out[j] = sum over neighbors of sumprod(neighbor_embedding, w_block[kpos][j])

def conv_full_block(encoded_orig, w_blocks, bias, post_gamma, eps, neighbors):
    """Full 3×3 conv using per-kernel-position weight blocks.
    w_blocks[kpos][j] = list[float] of length D_TILE (the weight for output channel j,
    kernel position kpos, across all input channels).
    """
    encoded = list(encoded_orig)
    te_out = D_TILE; n_tiles = N_TILES; total = n_tiles * te_out

    projected = [0.0] * total
    for t in range(n_tiles):
        nbr = neighbors[t]; base = t * te_out
        # Start with bias
        for j in range(te_out): projected[base + j] = bias[j]
        # Accumulate from each neighbor
        for nt, kpos in nbr:
            src = encoded[nt * te_out:(nt + 1) * te_out]
            wb = w_blocks[kpos]
            for j in range(te_out):
                projected[base + j] += _DOT(src, wb[j])

    # Residual + RMSNorm + SiLU
    mean_sq = [0.0] * n_tiles
    for i in range(total):
        t = i // te_out; x = encoded[i] + projected[i]; encoded[i] = x; mean_sq[t] += x * x
    inv_rms = [1.0 / (ms / te_out + eps) ** 0.5 for ms in mean_sq]
    for i in range(total):
        c = i % te_out; x = encoded[i] * inv_rms[i // te_out] * post_gamma[c]
        if x >= 0: encoded[i] = x / (1.0 + _exp(-x))
        else: e = _exp(x); encoded[i] = x * e / (1.0 + e)
    return encoded


# ========== Full conv (block) × 2 layers ==========

def conv_full_block_x2(encoded_orig, wb1, b1, g1, wb2, b2, g2, eps, neighbors):
    encoded = list(encoded_orig)
    te_out = D_TILE; n_tiles = N_TILES; total = n_tiles * te_out

    for w_blocks, bias, gamma in ((wb1, b1, g1), (wb2, b2, g2)):
        projected = [0.0] * total
        for t in range(n_tiles):
            nbr = neighbors[t]; base = t * te_out
            for j in range(te_out): projected[base + j] = bias[j]
            for nt, kpos in nbr:
                src = encoded[nt * te_out:(nt + 1) * te_out]
                wb = w_blocks[kpos]
                for j in range(te_out):
                    projected[base + j] += _DOT(src, wb[j])

        mean_sq = [0.0] * n_tiles
        for i in range(total):
            t = i // te_out; x = encoded[i] + projected[i]; encoded[i] = x; mean_sq[t] += x * x
        inv_rms = [1.0 / (ms / te_out + eps) ** 0.5 for ms in mean_sq]
        for i in range(total):
            c = i % te_out; x = encoded[i] * inv_rms[i // te_out] * gamma[c]
            if x >= 0: encoded[i] = x / (1.0 + _exp(-x))
            else: e = _exp(x); encoded[i] = x * e / (1.0 + e)
    return encoded


# ========== Depthwise → Pointwise → Depthwise (cross-channel then spatial again) ==========

def conv_dw_pw_dw(encoded_orig, conv_w1, conv_b1, pw_w, pw_b, g_mid,
                  conv_w2, conv_b2, post_gamma, eps, neighbors):
    encoded = list(encoded_orig)
    te_out = D_TILE; n_tiles = N_TILES; total = n_tiles * te_out

    # First depthwise
    conv_out = [0.0] * total
    for t in range(n_tiles):
        nbr = neighbors[t]; t_base = t * te_out
        for c in range(te_out):
            cw = conv_w1[c]; s = conv_b1[c]
            for nt, k in nbr: s += encoded[nt * te_out + c] * cw[k]
            conv_out[t_base + c] = s

    # Pointwise (channel mixing) + RMSNorm + SiLU
    mixed = [0.0] * total
    mean_sq = [0.0] * n_tiles
    for t in range(n_tiles):
        src = conv_out[t * te_out:(t + 1) * te_out]; base = t * te_out
        for j in range(te_out):
            x = _DOT(src, pw_w[j])
            if pw_b is not None: x += pw_b[j]
            x = encoded[base + j] + x  # residual
            mixed[base + j] = x
            mean_sq[t] += x * x
    inv_rms = [1.0 / (ms / te_out + eps) ** 0.5 for ms in mean_sq]
    for i in range(total):
        c = i % te_out; x = mixed[i] * inv_rms[i // te_out] * g_mid[c]
        if x >= 0: mixed[i] = x / (1.0 + _exp(-x))
        else: e = _exp(x); mixed[i] = x * e / (1.0 + e)

    # Second depthwise on channel-mixed features
    projected = [0.0] * total
    for t in range(n_tiles):
        nbr = neighbors[t]; t_base = t * te_out
        for c in range(te_out):
            cw = conv_w2[c]; s = conv_b2[c]
            for nt, k in nbr: s += mixed[nt * te_out + c] * cw[k]
            projected[t_base + c] = s

    # Final residual + RMSNorm + SiLU
    mean_sq2 = [0.0] * n_tiles
    for i in range(total):
        t = i // te_out; x = mixed[i] + projected[i]; mixed[i] = x; mean_sq2[t] += x * x
    inv_rms2 = [1.0 / (ms / te_out + eps) ** 0.5 for ms in mean_sq2]
    for i in range(total):
        c = i % te_out; x = mixed[i] * inv_rms2[i // te_out] * post_gamma[c]
        if x >= 0: mixed[i] = x / (1.0 + _exp(-x))
        else: e = _exp(x); mixed[i] = x * e / (1.0 + e)
    return mixed


# ========== Full pipelines with lowrank compress ==========

def _lowrank_compress(tokens, lr_w1, lr_b1, r, lr_w2, lr_b2, out_dim, gamma, eps):
    mid = [0.0] * r
    for j in range(r):
        x = _DOT(tokens, lr_w1[j]) + lr_b1[j]
        if x >= 0: mid[j] = x / (1.0 + _exp(-x))
        else: e = _exp(x); mid[j] = x * e / (1.0 + e)
    out = [0.0] * out_dim; ms = 0.0
    for j in range(out_dim):
        x = _DOT(mid, lr_w2[j]) + lr_b2[j]; out[j] = x; ms += x * x
    inv_rms = 1.0 / (ms / out_dim + eps) ** 0.5
    for j in range(out_dim):
        x = out[j] * inv_rms * gamma[j]
        if x >= 0: out[j] = x / (1.0 + _exp(-x))
        else: e = _exp(x); out[j] = x * e / (1.0 + e)
    return out


def pipe_current(encoded, conv_w, conv_b, pw_w, pw_b, post_gamma, eps, neighbors,
                 lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80):
    tokens = conv_depthwise_pw(encoded, conv_w, conv_b, pw_w, pw_b, post_gamma, eps, neighbors)
    return _lowrank_compress(tokens, lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80, eps)

def pipe_fullblock(encoded, wb, b, post_gamma, eps, neighbors,
                   lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80):
    tokens = conv_full_block(encoded, wb, b, post_gamma, eps, neighbors)
    return _lowrank_compress(tokens, lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80, eps)

def pipe_fullblock_x2(encoded, wb1, b1, g1, wb2, b2, g2, eps, neighbors,
                      lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80):
    tokens = conv_full_block_x2(encoded, wb1, b1, g1, wb2, b2, g2, eps, neighbors)
    return _lowrank_compress(tokens, lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80, eps)

def pipe_dw_pw_dw(encoded, conv_w1, conv_b1, pw_w, pw_b, g_mid,
                  conv_w2, conv_b2, post_gamma, eps, neighbors,
                  lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80):
    tokens = conv_dw_pw_dw(encoded, conv_w1, conv_b1, pw_w, pw_b, g_mid,
                           conv_w2, conv_b2, post_gamma, eps, neighbors)
    return _lowrank_compress(tokens, lr_w1, lr_b1, r, lr_w2, lr_b2, N, g80, eps)


def _bench(fn, args, iterations, warmup=500):
    for _ in range(warmup): fn(*args)
    start = time.perf_counter()
    for _ in range(iterations): fn(*args)
    elapsed = time.perf_counter() - start
    return elapsed * 1e6 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10000)
    args = parser.parse_args()

    random.seed(42)
    eps = 1e-6
    encoded = _make_encoded()
    N = 80
    neighbors = _build_neighbors()

    # Current params
    conv_w = [[random.gauss(0, 0.1) for _ in range(9)] for _ in range(D_TILE)]
    conv_b = _make_bias(D_TILE)
    pw_w = _make_weights(D_TILE, D_TILE); pw_b = _make_bias(D_TILE)
    g = _make_gamma(D_TILE)

    # Full conv block weights: w_blocks[kpos][out_ch] = list[float] of D_TILE
    def make_w_blocks():
        return [[
            [random.gauss(0, 0.1) for _ in range(D_TILE)]
            for _ in range(D_TILE)
        ] for _ in range(9)]

    wb1 = make_w_blocks(); b1 = _make_bias(D_TILE)
    wb2 = make_w_blocks(); b2 = _make_bias(D_TILE)
    g2 = _make_gamma(D_TILE)

    # Second depthwise for DW-PW-DW
    conv_w2 = [[random.gauss(0, 0.1) for _ in range(9)] for _ in range(D_TILE)]
    conv_b2 = _make_bias(D_TILE)
    g_mid = _make_gamma(D_TILE)

    # Lowrank compress
    lr_w1 = _make_weights(N_TILES * D_TILE, 12); lr_b1 = _make_bias(12)
    lr_w2 = _make_weights(12, N); lr_b2 = _make_bias(N)
    g80 = _make_gamma(N)

    iters = args.iters
    fc_params = 9 * D_TILE * D_TILE + D_TILE  # 576 + 8 = 584 per layer
    dw_params = D_TILE * 9 + D_TILE  # 80 per depthwise
    pw_params = D_TILE * D_TILE + D_TILE  # 72
    lr_params = 200 * 12 + 12 + 12 * N + N  # 3452

    print(f"{'Variant':<55} {'µs':>8} {'vs curr':>9} {'cross-ch':>9} {'params':>8}")
    print("-" * 93)

    # === Conv block only ===
    print("=== Conv block only ===")

    us_dw = _bench(conv_depthwise_pw,
                   (encoded, conv_w, conv_b, pw_w, pw_b, g, eps, neighbors), iters)
    print(f"{'depthwise + pointwise (current)':<55} {us_dw:>8.1f} {'---':>9} {'no':>9} {dw_params+pw_params:>8}")

    us_fb = _bench(conv_full_block,
                   (encoded, wb1, b1, g, eps, neighbors), iters)
    print(f"{'full conv block (8→8)':<55} {us_fb:>8.1f} {(us_dw-us_fb)/us_dw*100:>+8.0f}% {'yes':>9} {fc_params:>8}")

    us_fb2 = _bench(conv_full_block_x2,
                    (encoded, wb1, b1, g, wb2, b2, g2, eps, neighbors), iters)
    print(f"{'full conv block × 2 (8→8→8)':<55} {us_fb2:>8.1f} {(us_dw-us_fb2)/us_dw*100:>+8.0f}% {'yes':>9} {2*fc_params:>8}")

    us_dpd = _bench(conv_dw_pw_dw,
                    (encoded, conv_w, conv_b, pw_w, pw_b, g_mid,
                     conv_w2, conv_b2, g, eps, neighbors), iters)
    print(f"{'depthwise → pointwise → depthwise':<55} {us_dpd:>8.1f} {(us_dw-us_dpd)/us_dw*100:>+8.0f}% {'yes':>9} {2*dw_params+pw_params:>8}")

    print()
    print("=== Full pipeline (conv + lowrank r=12 → 80) ===")

    us_curr = _bench(pipe_current,
                     (encoded, conv_w, conv_b, pw_w, pw_b, g, eps, neighbors,
                      lr_w1, lr_b1, 12, lr_w2, lr_b2, N, g80), iters)
    print(f"{'depthwise+pw + lowrank (current best)':<55} {us_curr:>8.1f} {'---':>9} {'no':>9} {dw_params+pw_params+lr_params:>8}")

    us = _bench(pipe_fullblock,
                (encoded, wb1, b1, g, eps, neighbors,
                 lr_w1, lr_b1, 12, lr_w2, lr_b2, N, g80), iters)
    print(f"{'full conv block + lowrank':<55} {us:>8.1f} {(us_curr-us)/us_curr*100:>+8.0f}% {'yes':>9} {fc_params+lr_params:>8}")

    us = _bench(pipe_fullblock_x2,
                (encoded, wb1, b1, g, wb2, b2, g2, eps, neighbors,
                 lr_w1, lr_b1, 12, lr_w2, lr_b2, N, g80), iters)
    print(f"{'full conv block × 2 + lowrank':<55} {us:>8.1f} {(us_curr-us)/us_curr*100:>+8.0f}% {'yes':>9} {2*fc_params+lr_params:>8}")

    us = _bench(pipe_dw_pw_dw,
                (encoded, conv_w, conv_b, pw_w, pw_b, g_mid,
                 conv_w2, conv_b2, g, eps, neighbors,
                 lr_w1, lr_b1, 12, lr_w2, lr_b2, N, g80), iters)
    print(f"{'dw→pw→dw + lowrank':<55} {us:>8.1f} {(us_curr-us)/us_curr*100:>+8.0f}% {'yes':>9} {2*dw_params+pw_params+lr_params:>8}")


if __name__ == "__main__":
    main()
