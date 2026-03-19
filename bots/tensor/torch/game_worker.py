#!/usr/bin/env python3
"""CPU-only game worker for batched training collection."""

from __future__ import annotations

import argparse
import ast  # noqa: F401 - engine startup can expect ast in sys.modules
import json
import os
import pickle

from replay_parser import parse_replay, extract_working_harvesters_from_replay


def _inject_flow_bonus(log_path: str, flow_bonus: float) -> None:
    """Re-read the trajectory pickle, append flow_bonus as index 12 on each
    trajectory's terminal step (0.0 on all others), write back in-place."""
    if not log_path or not os.path.exists(log_path) or flow_bonus == 0.0:
        return
    try:
        trajs = []
        with open(log_path, "rb") as f:
            while True:
                try:
                    trajs.append(pickle.load(f))
                except EOFError:
                    break
        if not trajs:
            return
        # Distribute bonus equally across all trajectories from this game
        bonus_per_traj = flow_bonus / max(len(trajs), 1)
        for traj in trajs:
            if not traj:
                continue
            traj[-1] = tuple(traj[-1]) + (bonus_per_traj,)
            for i in range(len(traj) - 1):
                traj[i] = tuple(traj[i]) + (0.0,)
        with open(log_path, "wb") as f:
            for traj in trajs:
                pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _run_game_proxy(args):
    bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed = args
    os.environ["TENSOR_TRAIN_LOG"] = log_path
    try:
        from cambc.cambc_engine import run_game

        run_game(bot_a, bot_b, engine_root, map_path, replay_path, seed)
        replay = parse_replay(replay_path)
        working = extract_working_harvesters_from_replay(replay)
        # Winner: 0 = TEAM_A, 1 = TEAM_B
        winner_val = replay.get("winner")
        winner = "a" if winner_val == 0 else ("b" if winner_val == 1 else None)
    except BaseException as e:
        print(f"  Game error: {e}")
        return None, None, None
    finally:
        os.environ.pop("TENSOR_TRAIN_LOG", None)

    # Score flow field from completed replay and inject into trajectory file.
    # Done here (after run_game) so the replay is guaranteed to exist.
    actual_log = log_path if os.path.exists(log_path) else None
    if actual_log:
        try:
            from flow_field import score_conveyor_flow_from_data, REPLAY_DIR_DXY
            from replay_parser import score_conveyor_flow_from_replay
            flow_bonus = score_conveyor_flow_from_replay(replay_path, team=0)
            flow_bonus *= 3.0  # FLOW_BONUS_SCALE
            _inject_flow_bonus(actual_log, flow_bonus)
        except Exception:
            pass

    return actual_log, working, winner


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