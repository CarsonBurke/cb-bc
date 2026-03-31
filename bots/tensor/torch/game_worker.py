#!/usr/bin/env python3
"""CPU-only game worker for batched training collection."""

from __future__ import annotations

import argparse
import ast  # noqa: F401 - engine startup can expect ast in sys.modules
import json
import os
import pickle
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REPLAY_PARSER_PYTHON = PROJECT_ROOT / "packages" / "replay_parser" / "python"
if str(REPLAY_PARSER_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPLAY_PARSER_PYTHON))

from replay_parser import ReplayData


def _write_bytes(path: Path, data: bytes | bytearray) -> str | None:
    if not data:
        return None
    path.write_bytes(data)
    return str(path)


def _run_game_proxy(args, result_path: str):
    bot_a, bot_b, map_path, engine_root, replay_path, log_path, seed = args
    os.environ["TENSOR_TRAIN_LOG"] = log_path
    try:
        from cambc.cambc_engine import run_game

        run_game(bot_a, bot_b, engine_root, map_path, replay_path, seed)
        rd = ReplayData.from_file(replay_path, team=0)
        working = rd.working_harvesters_dict()
        winner = rd.winner_str
        base = Path(result_path)
        state_path = _write_bytes(base.with_suffix(".states.bin"), rd.states_bytes)
        scalars_path = _write_bytes(base.with_suffix(".scalars.bin"), rd.scalars_bytes)
        transfers_path = _write_bytes(
            base.with_suffix(".transfers.bin"),
            rd.resource_transfers_bytes,
        )
        ti_delivered_path = _write_bytes(
            base.with_suffix(".ti_delivered.bin"),
            rd.titanium_delivered_bytes,
        )
        # Extract team A core position from round 0 state grid
        # Channel 11 = core presence (both teams), channel 12 = team sign (+1 friendly)
        core_x, core_y = 0, 0
        try:
            n_ch, map_h, map_w = rd.n_channels, rd.map_h, rd.map_w
            turn0_bytes = rd.state_bytes_for_turn(0)
            plane = map_h * map_w
            ch11_offset = 11 * plane
            ch12_offset = 12 * plane
            for cy in range(map_h):
                for cx in range(map_w):
                    flat_idx = cy * map_w + cx
                    presence = struct.unpack_from("<f", turn0_bytes, (ch11_offset + flat_idx) * 4)[0]
                    team_sign = struct.unpack_from("<f", turn0_bytes, (ch12_offset + flat_idx) * 4)[0]
                    if presence > 0.5 and team_sign > 0.5:  # friendly core
                        core_x, core_y = cx, cy
                        break
                else:
                    continue
                break
        except Exception:
            pass

        global_states = {
            "states_path": state_path,
            "scalars_path": scalars_path,
            "transfer_events_path": transfers_path,
            "ti_delivered_path": ti_delivered_path,
            "state_stride": rd.state_stride,
            "map_w": rd.map_w,
            "map_h": rd.map_h,
            "n_turns": rd.n_turns,
            "n_channels": rd.n_channels,
            "n_scalars": rd.n_scalars,
            "core_x": core_x,
            "core_y": core_y,
        }
    except BaseException as e:
        print(f"  Game error: {e}")
        return None, None, None, None
    finally:
        os.environ.pop("TENSOR_TRAIN_LOG", None)
    return (
        (log_path if os.path.exists(log_path) else None),
        working,
        winner,
        global_states,
    )


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
        result = _run_game_proxy(tuple(json.loads(args.game_args)), args.result)
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
