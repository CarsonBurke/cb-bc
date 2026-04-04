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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPLAY_PARSER_PYTHON = PROJECT_ROOT / "packages" / "replay_parser" / "python"
if str(REPLAY_PARSER_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPLAY_PARSER_PYTHON))

from replay_parser import ReplayData
try:
    from trajectory_schema import WorkerGameResult
except ImportError:
    from bots.tensor.trajectory_schema import WorkerGameResult


def _write_bytes(path: Path, data: bytes | bytearray) -> str | None:
    if not data:
        return None
    path.write_bytes(data)
    return str(path)


def _run_game_proxy(args, result_path: str):
    bot_a, bot_b, map_path, engine_root, replay_path, seed = args
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
    ax_delivered_path = _write_bytes(
        base.with_suffix(".ax_delivered.bin"),
        rd.axionite_delivered_bytes,
    )
    core_x, core_y = 0, 0
    map_h, map_w = rd.map_h, rd.map_w
    turn0_bytes = rd.state_bytes_for_turn(0)
    plane = map_h * map_w
    ch11_offset = 11 * plane
    ch12_offset = 12 * plane
    for cy in range(map_h):
        for cx in range(map_w):
            flat_idx = cy * map_w + cx
            presence = struct.unpack_from("<f", turn0_bytes, (ch11_offset + flat_idx) * 4)[0]
            team_sign = struct.unpack_from("<f", turn0_bytes, (ch12_offset + flat_idx) * 4)[0]
            if presence > 0.5 and team_sign > 0.5:
                core_x, core_y = cx, cy
                break
        else:
            continue
        break

    global_states = {
        "states_path": state_path,
        "scalars_path": scalars_path,
        "transfer_events_path": transfers_path,
        "ti_delivered_path": ti_delivered_path,
        "ax_delivered_path": ax_delivered_path,
        "state_stride": rd.state_stride,
        "map_w": rd.map_w,
        "map_h": rd.map_h,
        "n_turns": rd.n_turns,
        "n_channels": rd.n_channels,
        "n_scalars": rd.n_scalars,
        "core_x": core_x,
        "core_y": core_y,
    }
    return WorkerGameResult(working=working, winner=winner, global_states=global_states)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batched training game worker")
    parser.add_argument("--result", required=True)
    parser.add_argument("--collector-fds", required=True)
    parser.add_argument("--game-args", required=True)
    args = parser.parse_args(argv)

    collector_fds = tuple(int(fd) for fd in args.collector_fds.split(",") if fd)
    if len(collector_fds) != 5:
        raise SystemExit("worker mode requires collector fd plus four notify pipe fds")

    collector_fd, req_w_a, resp_r_a, req_w_b, resp_r_b = collector_fds
    os.environ["TENSOR_COLLECTOR_FD"] = str(collector_fd)
    os.environ["TENSOR_REQ_NOTIFY_WFD_A"] = str(req_w_a)
    os.environ["TENSOR_RESP_NOTIFY_RFD_A"] = str(resp_r_a)
    if req_w_b >= 0 and resp_r_b >= 0:
        os.environ["TENSOR_REQ_NOTIFY_WFD_B"] = str(req_w_b)
        os.environ["TENSOR_RESP_NOTIFY_RFD_B"] = str(resp_r_b)
    try:
        result = _run_game_proxy(tuple(json.loads(args.game_args)), args.result)
    finally:
        for key in (
            "TENSOR_COLLECTOR_FD",
            "TENSOR_REQ_NOTIFY_WFD_A",
            "TENSOR_RESP_NOTIFY_RFD_A",
            "TENSOR_REQ_NOTIFY_WFD_B",
            "TENSOR_RESP_NOTIFY_RFD_B",
        ):
            os.environ.pop(key, None)
        for fd in collector_fds:
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
