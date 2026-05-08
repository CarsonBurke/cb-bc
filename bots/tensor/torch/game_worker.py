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


def _write_bytes(path: Path, data: bytes | bytearray | memoryview) -> str | None:
    if not data:
        return None
    path.write_bytes(data)
    return str(path)


def _training_turn_cap() -> int | None:
    raw = os.environ.get("TENSOR_TRAIN_MAX_TURNS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid TENSOR_TRAIN_MAX_TURNS: {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"TENSOR_TRAIN_MAX_TURNS must be > 0, got {value}")
    return value


def _turn_prefix(data: bytes | bytearray | None, *, per_turn_bytes: int, n_turns: int) -> bytes | bytearray:
    if not data or per_turn_bytes <= 0:
        return b""
    return memoryview(data)[: max(0, int(n_turns)) * per_turn_bytes]


def _build_global_states(rd: ReplayData, result_path: str, *, max_turns: int | None) -> dict:
    kept_turns = rd.n_turns if max_turns is None else min(rd.n_turns, int(max_turns))
    base = Path(result_path)
    state_path = _write_bytes(
        base.with_suffix(".states.bin"),
        _turn_prefix(rd.states_bytes, per_turn_bytes=rd.state_stride * 4, n_turns=kept_turns),
    )
    scalars_path = _write_bytes(
        base.with_suffix(".scalars.bin"),
        _turn_prefix(rd.scalars_bytes, per_turn_bytes=rd.n_scalars * 4, n_turns=kept_turns),
    )
    transfers_path = _write_bytes(
        base.with_suffix(".transfers.bin"),
        _turn_prefix(rd.resource_transfers_bytes, per_turn_bytes=8, n_turns=kept_turns),
    )
    ti_delivered_path = _write_bytes(
        base.with_suffix(".ti_delivered.bin"),
        _turn_prefix(rd.titanium_delivered_bytes, per_turn_bytes=8, n_turns=kept_turns),
    )
    ax_delivered_path = _write_bytes(
        base.with_suffix(".ax_delivered.bin"),
        _turn_prefix(rd.axionite_delivered_bytes, per_turn_bytes=8, n_turns=kept_turns),
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

    return {
        "states_path": state_path,
        "scalars_path": scalars_path,
        "transfer_events_path": transfers_path,
        "ti_delivered_path": ti_delivered_path,
        "ax_delivered_path": ax_delivered_path,
        "state_stride": rd.state_stride,
        "map_w": rd.map_w,
        "map_h": rd.map_h,
        "n_turns": kept_turns,
        "n_channels": rd.n_channels,
        "n_scalars": rd.n_scalars,
        "core_x": core_x,
        "core_y": core_y,
    }


def _run_game_proxy(args, result_path: str):
    bot_a, bot_b, map_path, engine_root, replay_path, seed, actor_team = args
    from cambc.cambc_engine import run_game

    run_game(bot_a, bot_b, engine_root, map_path, replay_path, seed)
    rd = ReplayData.from_file(replay_path, team=int(actor_team))
    working = rd.working_harvesters_dict()
    winner = rd.winner_str
    global_states = _build_global_states(rd, result_path, max_turns=_training_turn_cap())
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
