#!/usr/bin/env python3
"""Distill a trained teacher model into a production-sized student.

Loads the teacher from a session's weights, collects trajectories via
batched GPU self-play, then trains a student to match the teacher's
policy via KL divergence on logits.

Usage:
    python bots/tensor/torch/distill.py --session sp_20260316_...
    python bots/tensor/torch/distill.py --session sp_20260316_... --epochs 30 --games 12
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from model import (
    BuilderMLP, DEVICE, TEACHER_HIDDEN, WEIGHT_PATH,
    distill, _parse_trajectories,
)
from train_batched import collect_games_batched
from training_utils import (
    League, _init_session, PROJECT_ROOT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Distill teacher -> student")
    parser.add_argument("--session", type=str, required=True, help="Teacher session name")
    parser.add_argument("--epochs", type=int, default=20, help="Distillation epochs")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--games", type=int, default=6, help="Games to collect for distillation data")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="", help="Student weight output path")
    args = parser.parse_args()

    sess = _init_session(args.session)
    weight_path = str(sess["weight_path"])
    replay_dir = sess["replay_dir"]
    maps = [f.name for f in (PROJECT_ROOT / "maps").glob("*.map26")]

    # Load teacher
    teacher = BuilderMLP(hidden=TEACHER_HIDDEN).to(DEVICE)
    if not teacher.load_weights(weight_path):
        print(f"Could not load teacher weights from {weight_path}")
        return
    teacher.eval()
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher: {n_teacher:,} params loaded from {weight_path}")

    # Create student
    student = BuilderMLP().to(DEVICE)
    student_opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    n_student = sum(p.numel() for p in student.parameters())
    print(f"Student: {n_student:,} params")
    print(f"Device: {DEVICE}")

    # Load league for opponents
    league = League(sess["league_dir"])
    opponents = league.sample_opponents(min(3, len(league.models)))
    if not opponents:
        print("No opponents in league")
        return

    # Collect trajectories using teacher model
    print(f"\nCollecting {args.games} games for distillation data...")
    opp_paths = []
    opp_ids = []
    for opp in opponents:
        n_per = max(1, args.games // len(opponents))
        for _ in range(n_per):
            opp_paths.append(opp["path"])
            opp_ids.append(opp["id"])

    t0 = time.time()
    trajs, _all_winners, _per_opp_winners, _all_global_states, _traj_game_map, _game_origin, _game_delivery_totals, _game_specs = collect_games_batched(
        teacher, opp_paths, opp_ids, maps, 0, replay_dir,
        opp_hidden=TEACHER_HIDDEN,
    )
    t_collect = time.time() - t0
    print(f"Collected {len(trajs)} trajectories in {t_collect:.1f}s")

    data = _parse_trajectories(trajs)
    if data is None:
        print("No data collected")
        return

    obs, h = data["obs"], data["h"]
    print(f"Distilling on {obs.shape[0]:,} samples, {args.epochs} epochs, T={args.temperature}...")

    t0 = time.time()
    stats = distill(
        teacher, student, student_opt, obs, h,
        epochs=args.epochs,
        temperature=args.temperature,
    )
    t_distill = time.time() - t0
    print(f"Distillation loss: {stats['distill_loss']:.6f} ({stats['n_updates']} updates, {t_distill:.1f}s)")

    # Save student
    out_path = args.output or str(WEIGHT_PATH)
    student.save_weights(out_path)
    print(f"\nStudent weights saved to {out_path}")


if __name__ == "__main__":
    main()
