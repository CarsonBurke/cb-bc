"""Regression coverage for train_batched hot-path helpers."""

from __future__ import annotations

from array import array
from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
TENSOR_DIR = PROJECT_ROOT / "bots" / "tensor"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))
if str(TENSOR_DIR) not in sys.path:
    sys.path.insert(0, str(TENSOR_DIR))

import train_batched as train_batched_mod
from trajectory_schema import make_trajectory_step


def test_tensor_byte_view_shares_tensor_storage():
    tensor = torch.arange(8, dtype=torch.uint8)

    view = train_batched_mod._tensor_byte_view(tensor)

    assert bytes(view[:4]) == bytes([0, 1, 2, 3])
    tensor[0] = 99
    assert bytes(view[:4]) == bytes([99, 1, 2, 3])


def test_parse_obs_packet_a_returns_metadata_and_payload_view():
    obs = array("f", [float(i) for i in range(train_batched_mod.OBS_SIZE)])
    h_prev = array("f", [float(1000 + i) for i in range(train_batched_mod.H_DIM)])
    unit_id = 17
    round_idx = 23
    packet = (
        train_batched_mod._OBS_A_HEAD.pack(unit_id, round_idx)
        + obs.tobytes()
        + h_prev.tobytes()
    )

    parsed_unit_id, parsed_round_idx, payload = train_batched_mod._parse_obs_packet_a(packet)

    assert parsed_unit_id == unit_id
    assert parsed_round_idx == round_idx
    assert payload.readonly
    assert bytes(payload[: train_batched_mod.OBS_SIZE * 4]) == obs.tobytes()
    assert bytes(payload[train_batched_mod.OBS_SIZE * 4 :]) == h_prev.tobytes()


def test_parse_obs_packet_a_rejects_wrong_size():
    with pytest.raises(ValueError, match="side-A obs packet must be"):
        train_batched_mod._parse_obs_packet_a(b"\x00" * 7)


def test_server_side_trajectory_step_uses_packet_payload_and_behavior_log_prob():
    obs = array("f", [float(i) for i in range(train_batched_mod.OBS_SIZE)])
    h_prev = array("f", [float(500 + i) for i in range(train_batched_mod.H_DIM)])
    packet = (
        train_batched_mod._OBS_A_HEAD.pack(99, 41)
        + obs.tobytes()
        + h_prev.tobytes()
    )
    _, round_idx, payload = train_batched_mod._parse_obs_packet_a(packet)

    step = make_trajectory_step(
        obs=payload[: train_batched_mod.OBS_SIZE * 4],
        h_prev=payload[train_batched_mod.OBS_SIZE * 4 :],
        obs_bytes_len=train_batched_mod.OBS_SIZE * 4,
        h_bytes_len=train_batched_mod.H_DIM * 4,
        round_idx=round_idx,
        bt=1,
        bd=2,
        cd=3,
        brt=4,
        md=5,
        reward=0.0,
        old_log_prob=-0.75,
        enemy_core_dmg=0.0,
        own_core_dmg=0.0,
        harv_attempt=0,
        harv_fail=0,
        harv_prox_gate=0,
        harv_prox_paid=0,
        conv_attempt=0,
        conv_fail=0,
        conv_prox_gate=0,
        conv_prox_paid=0,
    )

    assert step.round_idx == 41
    assert step.old_log_prob == -0.75
    assert step.obs_bytes == obs.tobytes()
    assert step.h_prev_bytes == h_prev.tobytes()
