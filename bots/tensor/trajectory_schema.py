"""Strict rollout and worker result schema for tensor training."""

from __future__ import annotations

import math
from typing import NamedTuple


def _coerce_bytes(buf, *, field: str, expected_len: int) -> bytes:
    if isinstance(buf, bytes):
        out = buf
    elif isinstance(buf, (bytearray, memoryview)):
        out = bytes(buf)
    else:
        tobytes = getattr(buf, "tobytes", None)
        if not callable(tobytes):
            raise TypeError(f"{field} must be bytes-like, got {type(buf).__name__}")
        out = tobytes()
    if len(out) != expected_len:
        raise ValueError(f"{field} must be exactly {expected_len} bytes, got {len(out)}")
    return out


def _require_int(value, *, field: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{field} must be int, got {type(value).__name__}")
    return value


def _require_finite_float(value, *, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return out


class TrajectoryStep(NamedTuple):
    obs_bytes: bytes
    h_prev_bytes: bytes
    round_idx: int
    bt: int
    bd: int
    cd: int
    brt: int
    md: int
    reward: float
    old_log_prob: float
    enemy_core_dmg: float
    own_core_dmg: float
    harv_attempt: int
    harv_fail: int
    harv_prox_gate: int
    harv_prox_paid: int
    conv_attempt: int
    conv_fail: int
    conv_prox_gate: int
    conv_prox_paid: int

    def with_reward(self, reward: float) -> "TrajectoryStep":
        return self._replace(reward=_require_finite_float(reward, field="reward"))


def make_trajectory_step(
    *,
    obs,
    h_prev,
    obs_bytes_len: int,
    h_bytes_len: int,
    round_idx: int,
    bt: int,
    bd: int,
    cd: int,
    brt: int,
    md: int,
    reward: float,
    old_log_prob: float,
    enemy_core_dmg: float,
    own_core_dmg: float,
    harv_attempt: int,
    harv_fail: int,
    harv_prox_gate: int,
    harv_prox_paid: int,
    conv_attempt: int,
    conv_fail: int,
    conv_prox_gate: int,
    conv_prox_paid: int,
) -> TrajectoryStep:
    round_idx = _require_int(round_idx, field="round_idx")
    if round_idx < 0:
        raise ValueError(f"round_idx must be >= 0, got {round_idx}")
    return TrajectoryStep(
        obs_bytes=_coerce_bytes(obs, field="obs", expected_len=obs_bytes_len),
        h_prev_bytes=_coerce_bytes(h_prev, field="h_prev", expected_len=h_bytes_len),
        round_idx=round_idx,
        bt=_require_int(bt, field="bt"),
        bd=_require_int(bd, field="bd"),
        cd=_require_int(cd, field="cd"),
        brt=_require_int(brt, field="brt"),
        md=_require_int(md, field="md"),
        reward=_require_finite_float(reward, field="reward"),
        old_log_prob=_require_finite_float(old_log_prob, field="old_log_prob"),
        enemy_core_dmg=_require_finite_float(enemy_core_dmg, field="enemy_core_dmg"),
        own_core_dmg=_require_finite_float(own_core_dmg, field="own_core_dmg"),
        harv_attempt=_require_int(harv_attempt, field="harv_attempt"),
        harv_fail=_require_int(harv_fail, field="harv_fail"),
        harv_prox_gate=_require_int(harv_prox_gate, field="harv_prox_gate"),
        harv_prox_paid=_require_int(harv_prox_paid, field="harv_prox_paid"),
        conv_attempt=_require_int(conv_attempt, field="conv_attempt"),
        conv_fail=_require_int(conv_fail, field="conv_fail"),
        conv_prox_gate=_require_int(conv_prox_gate, field="conv_prox_gate"),
        conv_prox_paid=_require_int(conv_prox_paid, field="conv_prox_paid"),
    )


def validate_trajectory_step(
    step,
    *,
    obs_bytes_len: int,
    h_bytes_len: int,
) -> TrajectoryStep:
    if not isinstance(step, TrajectoryStep):
        raise TypeError(f"Expected TrajectoryStep, got {type(step).__name__}")
    if len(step.obs_bytes) != obs_bytes_len:
        raise ValueError(
            f"TrajectoryStep.obs_bytes must be {obs_bytes_len} bytes, got {len(step.obs_bytes)}"
        )
    if len(step.h_prev_bytes) != h_bytes_len:
        raise ValueError(
            f"TrajectoryStep.h_prev_bytes must be {h_bytes_len} bytes, got {len(step.h_prev_bytes)}"
        )
    if step.round_idx < 0:
        raise ValueError(f"TrajectoryStep.round_idx must be >= 0, got {step.round_idx}")
    _require_finite_float(step.reward, field="reward")
    _require_finite_float(step.old_log_prob, field="old_log_prob")
    _require_finite_float(step.enemy_core_dmg, field="enemy_core_dmg")
    _require_finite_float(step.own_core_dmg, field="own_core_dmg")
    return step


class WorkerGameResult(NamedTuple):
    working: dict
    winner: str | None
    global_states: dict


def validate_worker_game_result(result) -> WorkerGameResult:
    if not isinstance(result, WorkerGameResult):
        raise TypeError(f"Expected WorkerGameResult, got {type(result).__name__}")
    if not isinstance(result.working, dict):
        raise TypeError("WorkerGameResult.working must be a dict")
    if result.winner not in {None, "a", "b"}:
        raise ValueError(f"Unexpected winner value: {result.winner!r}")
    if not isinstance(result.global_states, dict):
        raise TypeError("WorkerGameResult.global_states must be a dict")
    return result
