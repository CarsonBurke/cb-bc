#!/usr/bin/env python3
"""CPU-only game worker for batched training collection."""

from __future__ import annotations

import argparse
import ast  # noqa: F401 - engine startup can expect ast in sys.modules
import json
import os
import pickle

from replay_parser import parse_replay, extract_working_harvesters_from_replay


def _inject_flow_bonus(
    log_path: str,
    turn_scores: dict[int, float],
    total_bonus: float,
) -> None:
    """Re-read the trajectory pickle and inject per-step flow weights.

    For each trajectory step where a conveyor was placed (bt == 2 or 8),
    look up the individual score for the conveyor placed that turn and use
    it as the weight for that step's share of the total bonus.

    Weight = individual_score / sum_of_all_scores * total_bonus.
    Steps with no conveyor placement get 0.0.
    Falls back to equal distribution if no turn_scores match.

    Appends the weighted bonus as index 12 on each step.
    """
    if not log_path or not os.path.exists(log_path) or total_bonus == 0.0:
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

        # Normalise turn_scores so weights sum to 1
        score_sum = sum(max(s, 0.0) for s in turn_scores.values())

        for traj in trajs:
            if not traj:
                continue
            n = len(traj)
            step_bonuses = [0.0] * n
            matched_any = False

            for step_i, step in enumerate(traj):
                s = step if isinstance(step, (list, tuple)) else None
                if s is None:
                    continue
                bt = s[2] if len(s) > 2 else -1
                if bt not in (2, 8):  # conveyor or armoured_conveyor
                    continue
                # step index ≈ round number for this builder
                turn = step_i
                raw_score = turn_scores.get(turn, 0.0)
                if raw_score > 0.0 and score_sum > 0.0:
                    step_bonuses[step_i] = (raw_score / score_sum) * total_bonus
                    matched_any = True

            # Fallback: equal distribution across conveyor steps
            if not matched_any:
                conv_steps = [i for i, s in enumerate(traj)
                              if isinstance(s, (list, tuple)) and len(s) > 2 and s[2] in (2, 8)]
                if conv_steps:
                    share = total_bonus / len(conv_steps)
                    for i in conv_steps:
                        step_bonuses[i] = share
                else:
                    step_bonuses[-1] = total_bonus

            for step_i, step in enumerate(traj):
                traj[step_i] = tuple(step) + (step_bonuses[step_i],)

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

    # Score flow field from completed replay and inject per-step weights.
    # Done here (after run_game) so the replay is guaranteed to exist.
    actual_log = log_path if os.path.exists(log_path) else None
    if actual_log:
        try:
            from flow_field import score_conveyor_flow_from_data_detailed, REPLAY_DIR_DXY
            from replay_parser import (
                score_conveyor_flow_from_replay,
                extract_conveyor_scores_by_turn,
            )
            # Parse replay once — reuse already-parsed replay from above
            map_data = replay.get("map", {})
            map_w = map_data.get("width", 0)
            map_h = map_data.get("height", 0)

            # Extract wall/ore/conveyor data (mirrors score_conveyor_flow_from_replay)
            _ENV_WALL, _ENV_ORE_TI, _ENV_ORE_AX = 1, 2, 3
            _CONV_KINDS = {"conveyor", "splitter", "armouredConveyor"}
            wall_set: set = set()
            ore_positions: list = []
            for y, row in enumerate(map_data.get("rows", [])):
                for x, tile_env in enumerate(row.get("tiles", [])):
                    if tile_env == _ENV_WALL:
                        wall_set.add((x, y))
                    elif tile_env in (_ENV_ORE_TI, _ENV_ORE_AX):
                        ore_positions.append((x, y))

            # Reconstruct final entity state
            entities: dict = {}
            for core in map_data.get("cores", []):
                eid = core["id"]
                entities[eid] = {
                    "id": eid, "kind": "core",
                    "team": core.get("team", 0),
                    "position": core.get("position", {}),
                }
            for turn in replay.get("turns", []):
                for update in turn.get("updates", []):
                    if "placeEntity" in update:
                        ent = update["placeEntity"].get("entity", {})
                        eid = ent.get("id", 0)
                        kind = None
                        extra = {}
                        for k in ("builderBot", "conveyor", "splitter", "armouredConveyor",
                                  "bridge", "harvester", "foundry", "road", "barrier",
                                  "marker", "core", "gunner", "sentinel", "breach", "launcher"):
                            if k in ent:
                                kind = k
                                extra = ent[k] if isinstance(ent[k], dict) else {}
                                break
                        entities[eid] = {
                            "id": eid, "kind": kind,
                            "team": ent.get("team", 0),
                            "position": ent.get("position", {}),
                            **extra,
                        }
                    elif "removeEntity" in update:
                        entities.pop(update["removeEntity"].get("id", 0), None)
                    elif "moveBuilderBot" in update:
                        mv = update["moveBuilderBot"]
                        eid = mv.get("id", 0)
                        if eid in entities:
                            entities[eid]["position"] = mv.get("to", {})

            # Find friendly core
            core_xy = None
            for ent in entities.values():
                if ent.get("kind") == "core" and ent.get("team", -1) == 0:
                    pos = ent.get("position", {})
                    core_xy = (pos.get("x", 0), pos.get("y", 0))
                    break

            if core_xy and ore_positions and map_w > 0:
                # Build conv_data facing vectors
                conv_data: dict = {}
                for ent in entities.values():
                    if ent.get("kind") not in _CONV_KINDS or ent.get("team", -1) != 0:
                        continue
                    pos = ent.get("position", {})
                    x, y = pos.get("x", -1), pos.get("y", -1)
                    if x < 0 or y < 0:
                        continue
                    dir_int = ent.get("direction", 0)
                    ddx, ddy = REPLAY_DIR_DXY.get(dir_int, (0, 0))
                    mag = (ddx * ddx + ddy * ddy) ** 0.5
                    conv_data[(x, y)] = (ddx / mag, ddy / mag) if mag > 0 else (0.0, 0.0)

                # Get total score and per-conveyor scores
                total_score, per_conv_scores = score_conveyor_flow_from_data_detailed(
                    wall_set=wall_set,
                    ore_positions=ore_positions,
                    conv_data=conv_data,
                    core_xy=core_xy,
                    map_w=map_w,
                    map_h=map_h,
                )
                FLOW_BONUS_SCALE = 3.0
                total_bonus = total_score * FLOW_BONUS_SCALE

                # Map per-conveyor scores to the turn each was placed
                turn_scores = extract_conveyor_scores_by_turn(replay, team=0, conv_scores=per_conv_scores)

                _inject_flow_bonus(actual_log, turn_scores, total_bonus)
        except Exception as e:
            print(f"  Flow scoring error: {e}")

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