#!/usr/bin/env python3
"""CPU-only game worker for batched training collection."""

from __future__ import annotations

import argparse
import ast  # noqa: F401 - engine startup can expect ast in sys.modules
import json
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from replay_parser import ReplayData


def _run_game_proxy(args):
    bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed = args
    os.environ["TENSOR_TRAIN_LOG"] = log_path
    try:
        from cambc.cambc_engine import run_game

        run_game(bot_a, bot_b, engine_root, map_path, replay_path, seed)
        rd = ReplayData.from_file(replay_path, team=0)
        working = rd.working_harvesters_dict()
        ti_delivered = rd.titanium_delivered_dict()
        winner = rd.winner_str
        global_states = {
            "states_buf": rd.states_bytes,
            "scalars_buf": rd.scalars_bytes,
            "state_stride": rd.state_stride,
            "map_w": rd.map_w,
            "map_h": rd.map_h,
            "n_turns": rd.n_turns,
            "n_channels": rd.n_channels,
            "n_scalars": rd.n_scalars,
        }
    except BaseException as e:
        print(f"  Game error: {e}")
        return None, None, None, None, None
    finally:
        os.environ.pop("TENSOR_TRAIN_LOG", None)
    return (log_path if os.path.exists(log_path) else None), working, winner, global_states, ti_delivered


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batched training game worker")
    parser.add_argument("--result", required=True)
    parser.add_argument("--pipes", required=True)
    parser.add_argument("--game-args", required=True)
    args = parser.parse_args(argv)

    pipe_fds = tuple(int(fd) for fd in args.pipes.split(",") if fd)
    if len(pipe_fds) != 4:
        raise SystemExit("worker mode requires four pipe fds")

    obs_w_a, act_r_a, obs_w_b, act_r_b = pipe_fds
    os.environ["TENSOR_OBS_WFD"] = str(obs_w_a)
    os.environ["TENSOR_ACT_RFD"] = str(act_r_a)
    os.environ["TENSOR_OBS_WFD_B"] = str(obs_w_b)
    os.environ["TENSOR_ACT_RFD_B"] = str(act_r_b)
    try:
        result = _run_game_proxy(tuple(json.loads(args.game_args)))
    finally:
        for key in ("TENSOR_OBS_WFD", "TENSOR_ACT_RFD", "TENSOR_OBS_WFD_B", "TENSOR_ACT_RFD_B"):
            os.environ.pop(key, None)
        for fd in pipe_fds:
            try:
                os.close(fd)
            except OSError:
                pass

    try:
        with open(args.result, "wb") as f:
            pickle.dump(result, f)
    except OSError:
        pass


if __name__ == "__main__":
    main()
