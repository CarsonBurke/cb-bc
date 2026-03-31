#!/usr/bin/env python3
"""Run actor BC, critic pretrain, then PPO sequentially.

This is a thin orchestrator over the existing training entry points. It keeps
checkpoint handoff explicit and gives the whole pipeline one base session name.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRETRAIN_DIR = ROOT / "pretrain"


def _default_base_session() -> str:
    return f"full_train_{time.strftime('%Y%m%d_%H%M%S')}"


def build_commands(args: argparse.Namespace) -> tuple[dict[str, str], list[list[str]]]:
    base = args.session or _default_base_session()
    critic_session = f"{base}_critic"
    actor_session = f"{base}_actor"
    ppo_session = f"{base}_ppo"

    critic_out = str(ROOT.parent / "sessions" / critic_session / "critic.pt")
    actor_out = str(ROOT.parent / "sessions" / actor_session / "weights_bc.json")

    actor_cmd = [
        sys.executable,
        str(PRETRAIN_DIR / "train_bc.py"),
        "--epochs", str(args.actor_epochs),
        "--games-per-batch", str(args.actor_games_per_batch),
        "--tbptt-len", str(args.actor_tbptt_len),
        "--chunk-groups-per-batch", str(args.actor_chunk_groups_per_batch),
        "--session", actor_session,
        "--output", actor_out,
    ]
    if args.actor_init_weights:
        actor_cmd.extend(["--init-weights", args.actor_init_weights])
    if args.actor_no_compile:
        actor_cmd.append("--no-compile")

    critic_cmd = [
        sys.executable,
        str(PRETRAIN_DIR / "train_critic.py"),
        "--epochs", str(args.critic_epochs),
        "--games-per-batch", str(args.critic_games_per_batch),
        "--rounds-per-group", str(args.critic_rounds_per_group),
        "--chunk-groups-per-batch", str(args.critic_chunk_groups_per_batch),
        "--session", critic_session,
        "--output", critic_out,
    ]
    if args.critic_init_weights:
        critic_cmd.extend(["--init-weights", args.critic_init_weights])
    critic_cmd.extend(["--actor-weights", actor_out])
    if args.critic_no_compile:
        critic_cmd.append("--no-compile")

    ppo_cmd = [
        sys.executable,
        str(ROOT / "train_batched.py"),
        "--episodes", str(args.ppo_episodes),
        "--opponents", str(args.ppo_opponents),
        "--games-per-opp", str(args.ppo_games_per_opp),
        "--checkpoint-every", str(args.ppo_checkpoint_every),
        "--session", ppo_session,
        "--init-weights", actor_out,
        "--critic-init-weights", critic_out,
    ]
    # PPO consumes the critic pretrain state_dict directly.
    if args.ppo_dummy_episodes:
        ppo_cmd.extend(["--dummy-episodes", str(args.ppo_dummy_episodes)])
    if args.ppo_critic_minibatch_size > 0:
        ppo_cmd.extend(["--critic-minibatch-size", str(args.ppo_critic_minibatch_size)])
    if args.ppo_critic_no_compile:
        ppo_cmd.append("--critic-no-compile")

    sessions = {
        "base": base,
        "critic": critic_session,
        "actor": actor_session,
        "ppo": ppo_session,
        "critic_out": critic_out,
        "actor_out": actor_out,
    }
    return sessions, [actor_cmd, critic_cmd, ppo_cmd]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential actor BC, critic pretrain, then PPO")
    parser.add_argument("--session", type=str, default="")

    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--critic-games-per-batch", type=int, default=240)
    parser.add_argument("--critic-rounds-per-group", type=int, default=32)
    parser.add_argument("--critic-chunk-groups-per-batch", type=int, default=96)
    parser.add_argument("--critic-init-weights", type=str, default="")
    parser.add_argument("--critic-no-compile", action="store_true")

    parser.add_argument("--actor-epochs", type=int, default=2)
    parser.add_argument("--actor-games-per-batch", type=int, default=800)
    parser.add_argument("--actor-tbptt-len", type=int, default=32)
    parser.add_argument("--actor-chunk-groups-per-batch", type=int, default=4)
    parser.add_argument("--actor-init-weights", type=str, default="")
    parser.add_argument("--actor-no-compile", action="store_true")

    parser.add_argument("--ppo-episodes", type=int, default=100)
    parser.add_argument("--ppo-opponents", type=int, default=4)
    parser.add_argument("--ppo-games-per-opp", type=int, default=4)
    parser.add_argument("--ppo-checkpoint-every", type=int, default=1)
    parser.add_argument("--ppo-dummy-episodes", type=int, default=0)
    parser.add_argument("--ppo-critic-minibatch-size", type=int, default=0)
    parser.add_argument("--ppo-critic-no-compile", action="store_true")

    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sessions, commands = build_commands(args)
    print(f"Base session: {sessions['base']}")
    print(f"Critic session: {sessions['critic']}")
    print(f"Actor session: {sessions['actor']}")
    print(f"PPO session: {sessions['ppo']}")

    for cmd in commands:
        print("+", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
