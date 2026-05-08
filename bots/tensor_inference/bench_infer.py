"""Microbenchmark the ActorModel.infer() hot path."""

from __future__ import annotations

import argparse
import random
import time

from model_constants import (
    GLOBAL_FEATURES,
    H_DIM,
    TILE_FEATURES,
    TILE_GRID_N,
)
from tensor_lib.actor import ActorModel

OUTPUT_HEADS = {
    "build_type": 11,
    "build_dir": 9,
    "conv_dir": 4,
    "bridge_target": 9,
    "move_dir": 9,
    "temperature": 1,
    "h_gate": H_DIM,
    "h_cand": H_DIM,
}

OBS_SIZE = TILE_GRID_N * TILE_FEATURES + GLOBAL_FEATURES + H_DIM


def _make_features(seed: int = 42) -> list[float]:
    random.seed(seed)
    feats = [0.0] * OBS_SIZE
    for t in range(TILE_GRID_N):
        base = t * TILE_FEATURES
        env = random.choice([0, 1, 2, 3])
        if env == 1:
            feats[base] = 1.0
        elif env == 2:
            feats[base + 1] = 1.0
        elif env == 3:
            feats[base + 2] = 1.0
        feats[base + 3] = float(random.randint(0, 1))
        bidx = random.randint(0, 9)
        feats[base + 4 + bidx] = 1.0
        feats[base + 14] = random.choice([-1.0, 0.0, 1.0])
        feats[base + 15] = float(random.randint(-1, 1))
        feats[base + 16] = float(random.randint(-1, 1))
        feats[base + 17] = random.random()
        feats[base + 18] = random.choice([-1.0, 0.0, 1.0])
        feats[base + 19] = float(random.randint(0, 1))
        feats[base + 20] = random.choice([-1.0, 0.0, 1.0])
    gstart = TILE_GRID_N * TILE_FEATURES
    for i in range(GLOBAL_FEATURES):
        feats[gstart + i] = random.gauss(0.0, 0.3)
    hstart = gstart + GLOBAL_FEATURES
    for i in range(H_DIM):
        feats[hstart + i] = random.gauss(0.0, 0.1)
    return feats


def bench_infer(iterations: int, profile: bool = False) -> float:
    random.seed(0)
    model = ActorModel(OUTPUT_HEADS)
    variants = [_make_features(seed=i) for i in range(10)]
    n = len(variants)
    # warmup
    for i in range(200):
        model.infer(variants[i % n])

    if profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        for i in range(iterations):
            model.infer(variants[i % n])
        pr.disable()
        pr.print_stats(sort="cumulative")
        return 0.0

    start = time.perf_counter()
    for i in range(iterations):
        model.infer(variants[i % n])
    elapsed = time.perf_counter() - start
    return elapsed * 1e6 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    us = bench_infer(args.iters, args.profile)
    if not args.profile:
        print(f"infer_us={us:.1f}")


if __name__ == "__main__":
    main()
