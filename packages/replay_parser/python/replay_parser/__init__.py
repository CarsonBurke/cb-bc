"""Fast replay parser for .replay26 protobuf files.

Native Rust implementation via PyO3. All state data is returned in contiguous
buffers suitable for zero-copy torch.frombuffer().

Primary API:
    replay = parse_replay_full(path, team=0)
    # replay["states"]  — bytes, f32[n_turns, 19, map_h, map_w]
    # replay["scalars"] — bytes, f32[n_turns, 17]
    # replay["working_harvesters"] — bytes, i32[n_turns, 2]
    # replay["resource_transfers"] — bytes, i32[n_turns, 2]
    # replay["titanium_delivered"] — bytes, i32[n_turns, 2]
    # replay["axionite_delivered"] — bytes, i32[n_turns, 2]
    # replay["map_w"], ["map_h"], ["n_turns"], ["state_stride"], ["winner"]
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from replay_parser._native import (
    parse_replay,
    parse_replay_full,
    parse_replay_working,
)

__all__ = [
    "parse_replay",
    "parse_replay_full",
    "parse_replay_working",
    "ReplayData",
    "extract_working_harvesters",
    "extract_global_states",
    "augment_trajectories_with_working_counts",
    "augment_trajectories_with_replay",
    "augment_trajectories_with_ti_delivery",
    "augment_trajectories_with_ax_delivery",
]


class ReplayData:
    """Wrapper around parse_replay_full result with efficient accessors.

    State data lives in contiguous bytes buffers. Access individual turns
    via slicing (zero-copy with torch.frombuffer) or iterate efficiently.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: dict):
        self._raw = raw

    @classmethod
    def from_file(cls, path: str | Path, team: int = 0) -> "ReplayData":
        return cls(parse_replay_full(str(path), team=team))

    @property
    def n_turns(self) -> int:
        return self._raw["n_turns"]

    @property
    def map_w(self) -> int:
        return self._raw["map_w"]

    @property
    def map_h(self) -> int:
        return self._raw["map_h"]

    @property
    def n_channels(self) -> int:
        return self._raw["n_channels"]

    @property
    def n_scalars(self) -> int:
        return self._raw["n_scalars"]

    @property
    def state_stride(self) -> int:
        """Number of f32 values per turn in the states buffer."""
        return self._raw["state_stride"]

    @property
    def winner(self) -> int | None:
        return self._raw["winner"]

    @property
    def winner_str(self) -> str | None:
        w = self._raw["winner"]
        if w == 0:
            return "a"
        elif w == 1:
            return "b"
        return None

    # ── Raw buffer access (for torch.frombuffer) ──

    @property
    def states_bytes(self) -> bytes:
        """Contiguous f32[n_turns, n_channels, map_h, map_w]."""
        return self._raw["states"]

    @property
    def scalars_bytes(self) -> bytes:
        """Contiguous f32[n_turns, n_scalars]."""
        return self._raw["scalars"]

    @property
    def working_harvesters_bytes(self) -> bytes:
        """Contiguous i32[n_turns, 2] — (team_a, team_b) per turn."""
        return self._raw["working_harvesters"]

    @property
    def resource_transfers_bytes(self) -> bytes:
        """Contiguous i32[n_turns, 2] — same-team non-core transfers per turn."""
        return self._raw.get("resource_transfers", b"")

    @property
    def titanium_delivered_bytes(self) -> bytes:
        """Contiguous i32[n_turns, 2] — per-turn ti delivered delta."""
        return self._raw.get("titanium_delivered", b"")

    @property
    def axionite_delivered_bytes(self) -> bytes:
        """Contiguous i32[n_turns, 2] — per-turn ax delivered delta."""
        return self._raw.get("axionite_delivered", b"")

    # ── Per-turn slice access (zero-copy with torch) ──

    def state_bytes_for_turn(self, turn: int) -> memoryview:
        """Return a memoryview slice for one turn's state (f32 * state_stride)."""
        byte_stride = self._raw["state_stride"] * 4
        start = turn * byte_stride
        return memoryview(self._raw["states"])[start:start + byte_stride]

    def scalars_bytes_for_turn(self, turn: int) -> memoryview:
        """Return a memoryview slice for one turn's scalars (f32 * n_scalars)."""
        byte_stride = self._raw["n_scalars"] * 4
        start = turn * byte_stride
        return memoryview(self._raw["scalars"])[start:start + byte_stride]

    def working_harvesters_for_turn(self, turn: int) -> tuple[int, int]:
        """Return (team_a_count, team_b_count) for a specific turn."""
        start = turn * 8  # 2 * i32
        a, b = struct.unpack_from("<ii", self._raw["working_harvesters"], start)
        return a, b

    # ── Dict-based access (compat with old callers) ──

    def working_harvesters_dict(self) -> dict[int, dict[int, int]]:
        """Return {turn: {0: team_a, 1: team_b}} for all turns."""
        buf = self._raw["working_harvesters"]
        n = self._raw["n_turns"]
        result = {}
        for t in range(n):
            off = t * 8
            a, b = struct.unpack_from("<ii", buf, off)
            result[t] = {0: a, 1: b}
        return result

    def titanium_delivered_dict(self) -> dict[int, dict[int, int]]:
        """Return {turn: {0: team_a, 1: team_b}} per-turn ti delivered delta."""
        buf = self._raw.get("titanium_delivered", b"")
        if not buf:
            return {}
        n = self._raw["n_turns"]
        result = {}
        for t in range(n):
            off = t * 8
            a, b = struct.unpack_from("<ii", buf, off)
            result[t] = {0: a, 1: b}
        return result

    def resource_transfers_dict(self) -> dict[int, dict[int, int]]:
        """Return {turn: {0: team_a, 1: team_b}} per-turn non-core transfer count."""
        buf = self._raw.get("resource_transfers", b"")
        if not buf:
            return {}
        n = self._raw["n_turns"]
        result = {}
        for t in range(n):
            off = t * 8
            a, b = struct.unpack_from("<ii", buf, off)
            result[t] = {0: a, 1: b}
        return result

    def axionite_delivered_dict(self) -> dict[int, dict[int, int]]:
        """Return {turn: {0: team_a, 1: team_b}} per-turn ax delivered delta."""
        buf = self._raw.get("axionite_delivered", b"")
        if not buf:
            return {}
        n = self._raw["n_turns"]
        result = {}
        for t in range(n):
            off = t * 8
            a, b = struct.unpack_from("<ii", buf, off)
            result[t] = {0: a, 1: b}
        return result

    def global_states_dict(self, max_w: int = 50, max_h: int = 50) -> dict[str, Any]:
        """Return global states in the legacy dict format.

        states[t] and scalars[t] are buffer objects compatible with
        torch.frombuffer(..., dtype=torch.float32).

        For bulk access, prefer states_bytes/scalars_bytes with a single
        torch.frombuffer + reshape to (n_turns, C, H, W).
        """
        from array import array as array_type
        stride_bytes = self._raw["state_stride"] * 4
        scalar_bytes = self._raw["n_scalars"] * 4
        states_buf = self._raw["states"]
        scalars_buf = self._raw["scalars"]
        n = self._raw["n_turns"]

        states = {}
        scalars = {}
        for t in range(n):
            a = array_type('f')
            a.frombytes(states_buf[t * stride_bytes:(t + 1) * stride_bytes])
            states[t] = a
            a = array_type('f')
            a.frombytes(scalars_buf[t * scalar_bytes:(t + 1) * scalar_bytes])
            scalars[t] = a

        return {
            "states": states,
            "scalars": scalars,
            "map_w": self._raw["map_w"],
            "map_h": self._raw["map_h"],
            "n_channels": self._raw["n_channels"],
            "n_scalars": self._raw["n_scalars"],
            "max_w": max_w,
            "max_h": max_h,
        }


# ── Convenience functions ──────────────────────────────────────────────────


def extract_working_harvesters(replay_path: str | Path) -> dict[int, dict[int, int]]:
    """Extract per-turn working harvester counts from a replay file."""
    return ReplayData.from_file(replay_path).working_harvesters_dict()


def extract_global_states(
    replay_path: str | Path,
    team: int = 0,
    max_w: int = 50,
    max_h: int = 50,
) -> dict[str, Any]:
    """Extract per-round global map state tensors from a replay file."""
    return ReplayData.from_file(replay_path, team).global_states_dict(max_w, max_h)


# ── Trajectory augmentation (pure Python) ─────────────────────────────────

_GLOBAL_START = 25 * 17  # GRID_SIZE^2 * TILE_FEATURES = 425
_ROUND_OBS_IDX = _GLOBAL_START


def _replay_aug_index(step: list | tuple, legacy_index: int, new_index: int) -> int:
    """Choose the correct mutable slot for old vs new trajectory tuple layouts."""
    return new_index if len(step) > 12 else legacy_index


def _ensure_step_index(step: list, target_index: int) -> None:
    """Extend a mutable trajectory step so a given index can be assigned."""
    if len(step) <= target_index:
        step.extend([0] * (target_index + 1 - len(step)))


def augment_trajectories_with_working_counts(
    trajectories: list[list],
    working: dict[int, dict[int, int]],
    team: int = 0,
) -> list[list]:
    """Override n_working in trajectory steps with precomputed replay ground truth."""
    if not working:
        return trajectories

    for traj in trajectories:
        for i, step in enumerate(traj):
            if not isinstance(step, (list, tuple)) or len(step) < 7:
                continue
            if isinstance(step, tuple):
                step = list(step)
                traj[i] = step
            obs = step[0]
            rnd = int(round(obs[_ROUND_OBS_IDX] * 2000.0))
            rnd = max(0, min(rnd, 1999))
            counts = working.get(rnd, {})
            n_working = counts.get(team, 0)
            step[_replay_aug_index(step, legacy_index=6, new_index=7)] = n_working

    return trajectories


def augment_trajectories_with_replay(
    trajectories: list[list],
    replay_path: str | Path,
    team: int = 0,
) -> list[list]:
    """Override n_working in trajectory steps with replay ground truth."""
    try:
        working = extract_working_harvesters(replay_path)
    except Exception:
        return trajectories
    return augment_trajectories_with_working_counts(trajectories, working, team)


def augment_trajectories_with_ti_delivery(
    trajectories: list[list],
    ti_delivered: dict[int, dict[int, int]],
    team: int = 0,
    step_index: int = 22,
) -> list[list]:
    """Write per-round titanium delivered delta into trajectory steps."""
    if not ti_delivered:
        return trajectories

    for traj in trajectories:
        for i, step in enumerate(traj):
            if not isinstance(step, (list, tuple)):
                continue
            if isinstance(step, tuple):
                step = list(step)
                traj[i] = step
            _ensure_step_index(step, step_index)
            obs = step[0]
            rnd = int(round(obs[_ROUND_OBS_IDX] * 2000.0))
            rnd = max(0, min(rnd, 1999))
            counts = ti_delivered.get(rnd, {})
            step[step_index] = counts.get(team, 0)

    return trajectories


def augment_trajectories_with_ax_delivery(
    trajectories: list[list],
    ax_delivered: dict[int, dict[int, int]],
    team: int = 0,
    step_index: int = 23,
) -> list[list]:
    """Write per-round axionite delivered delta into trajectory steps."""
    if not ax_delivered:
        return trajectories

    for traj in trajectories:
        for i, step in enumerate(traj):
            if not isinstance(step, (list, tuple)):
                continue
            if isinstance(step, tuple):
                step = list(step)
                traj[i] = step
            _ensure_step_index(step, step_index)
            obs = step[0]
            rnd = int(round(obs[_ROUND_OBS_IDX] * 2000.0))
            rnd = max(0, min(rnd, 1999))
            counts = ax_delivered.get(rnd, {})
            step[step_index] = counts.get(team, 0)

    return trajectories
