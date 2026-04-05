"""Regression coverage for train_batched hot-path helpers."""

from __future__ import annotations

from array import array
import mmap
import os
from pathlib import Path
import sys
import tempfile
import threading

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
from hidden_state_bank import HiddenStateBank
from shared_collector import (
    COLLECTOR_RING_SLOTS_DEFAULT,
    CollectorClient,
    CollectorServerSide,
    OBS_BYTES,
    REQUEST_META,
    collector_bytes,
    initialize_collector,
)
from trajectory_schema import make_trajectory_step


def _make_collector_mmap(slot_count: int = COLLECTOR_RING_SLOTS_DEFAULT):
    fd, path = tempfile.mkstemp(prefix="collector_test_")
    os.unlink(path)
    os.ftruncate(fd, collector_bytes(slot_count))
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE)
    initialize_collector(mm, slot_count)
    return fd, mm


def test_tensor_byte_view_shares_tensor_storage():
    tensor = torch.arange(8, dtype=torch.uint8)

    view = train_batched_mod._tensor_byte_view(tensor)

    assert bytes(view[:4]) == bytes([0, 1, 2, 3])
    tensor[0] = 99
    assert bytes(view[:4]) == bytes([99, 1, 2, 3])


def test_collector_server_side_pops_multiple_ring_requests():
    obs0 = array("f", [float(i) for i in range(train_batched_mod.OBS_SIZE)])
    obs1 = array("f", [float(1000 + i) for i in range(train_batched_mod.OBS_SIZE)])
    fd, mm = _make_collector_mmap(slot_count=4)
    slot = CollectorServerSide(mm, "a")
    REQUEST_META.pack_into(mm, slot._layout.request_meta_offset(0), 17, 23, 0)
    REQUEST_META.pack_into(mm, slot._layout.request_meta_offset(1), 18, 24, 0)
    slot.request_obs_buffer(0)[:] = obs0.tobytes()
    slot.request_obs_buffer(1)[:] = obs1.tobytes()
    slot._set_req_tail(2)

    requests = slot.pop_requests()

    assert [(req.unit_id, req.round_idx, req.slot_index) for req in requests] == [(17, 23, 0), (18, 24, 1)]
    assert bytes(requests[0].obs_buffer) == obs0.tobytes()
    assert bytes(requests[1].obs_buffer) == obs1.tobytes()
    assert slot._req_head() == 2

    slot.close()
    mm.close()
    os.close(fd)


def test_collector_server_side_rejects_unknown_side():
    fd, mm = _make_collector_mmap()
    with pytest.raises(ValueError, match="unsupported collector side"):
        CollectorServerSide(mm, "z")
    mm.close()
    os.close(fd)


def test_server_side_trajectory_step_uses_shared_slot_obs_and_server_hidden_state():
    obs = array("f", [float(i) for i in range(train_batched_mod.OBS_SIZE)])
    h_prev = array("f", [float(500 + i) for i in range(train_batched_mod.H_DIM)])
    fd, mm = _make_collector_mmap()
    slot = CollectorServerSide(mm, "a")
    REQUEST_META.pack_into(mm, slot._layout.request_meta_offset(0), 99, 41, 0)
    slot.request_obs_buffer(0)[:] = obs.tobytes()

    step = make_trajectory_step(
        obs=slot.request_obs_buffer(0),
        h_prev=h_prev.tobytes(),
        obs_bytes_len=OBS_BYTES,
        h_bytes_len=train_batched_mod.H_DIM * 4,
        round_idx=41,
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

    slot.close()
    mm.close()
    os.close(fd)


def test_collector_client_and_server_exchange_over_ring():
    obs = array("f", [float(i) for i in range(train_batched_mod.OBS_SIZE)])
    fd, mm = _make_collector_mmap(slot_count=4)
    req_r, req_w = os.pipe()
    resp_r, resp_w = os.pipe()
    client = CollectorClient(fd, req_w, resp_r, "a")
    server = CollectorServerSide(mm, "a")
    response = {}

    def _run_client():
        response["value"] = client.request(7, 19, obs)

    thread = threading.Thread(target=_run_client)
    thread.start()
    assert os.read(req_r, 1) == b"\x01"
    requests = server.pop_requests()
    assert len(requests) == 1
    assert requests[0].unit_id == 7
    assert requests[0].round_idx == 19
    assert bytes(requests[0].obs_buffer) == obs.tobytes()
    server.write_response(1, 2, 3, 4, 5)
    os.write(resp_w, b"\x01")
    thread.join(timeout=2.0)

    assert response["value"] == (1, 2, 3, 4, 5)

    client.close()
    server.close()
    mm.close()
    for pipe_fd in (req_r, req_w, resp_r, resp_w, fd):
        os.close(pipe_fd)


def test_resolve_benchmark_maps_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown benchmark map"):
        train_batched_mod._resolve_benchmark_maps(["a.map26", "b.map26"], "a.map26,missing.map26")


def test_build_game_launch_specs_is_deterministic_for_benchmark_seed(tmp_path):
    kwargs = dict(
        opp_weight_paths=["opp_a.pkl", train_batched_mod.DUMMY_OPP_PATH, "opp_b.pkl"],
        opp_ids=["opp-a", "dummy", "opp-b"],
        maps=["zeta.map26", "alpha.map26", "gamma.map26"],
        episode_num=1,
        replay_dir=tmp_path,
        bot_a="bot_a.py",
        bot_b_proxy="bot_b_proxy.py",
        bot_b_dummy="bot_b_dummy.py",
        engine_root="/engine",
        benchmark_seed=12345,
        benchmark_maps=None,
    )

    first = train_batched_mod._build_game_launch_specs(**kwargs)
    second = train_batched_mod._build_game_launch_specs(**kwargs)

    assert first == second
    assert [Path(spec.map_path).name for spec in first] == [Path(spec.map_path).name for spec in second]
    assert [spec.seed for spec in first] == [spec.seed for spec in second]
    assert first[1].bot_b == "bot_b_dummy.py"
    assert first[0].bot_b == "bot_b_proxy.py"


def test_build_game_launch_specs_uses_requested_benchmark_maps(tmp_path):
    specs = train_batched_mod._build_game_launch_specs(
        opp_weight_paths=["opp_a.pkl", "opp_b.pkl"],
        opp_ids=["opp-a", "opp-b"],
        maps=["alpha.map26", "beta.map26", "gamma.map26"],
        episode_num=2,
        replay_dir=tmp_path,
        bot_a="bot_a.py",
        bot_b_proxy="bot_b_proxy.py",
        bot_b_dummy="bot_b_dummy.py",
        engine_root="/engine",
        benchmark_seed=7,
        benchmark_maps=["beta.map26"],
    )

    assert [Path(spec.map_path).name for spec in specs] == ["beta.map26", "beta.map26"]


def test_hidden_state_bank_reuses_updated_rows_without_aliasing():
    bank = HiddenStateBank(hidden_dim=4, device=torch.device("cpu"), initial_capacity=2, growth_capacity=2)

    first = bank.batch([(1, 10), (1, 11), (2, 10)])
    assert torch.equal(first.h_prev, torch.zeros(3, 4))

    updated = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=torch.float32,
    )
    bank.update(first.row_index, updated)

    second = bank.batch([(1, 11), (2, 10), (1, 10)])
    assert torch.equal(second.h_prev, updated[[1, 2, 0]])


def test_hidden_state_bank_release_game_zeroes_reused_rows():
    bank = HiddenStateBank(hidden_dim=2, device=torch.device("cpu"), initial_capacity=1, growth_capacity=1)

    first = bank.batch([(5, 99)])
    bank.update(first.row_index, torch.tensor([[3.0, 4.0]], dtype=torch.float32))
    bank.release_game(5)

    second = bank.batch([(6, 99)])
    assert torch.equal(second.h_prev, torch.zeros(1, 2))


def test_hidden_state_bank_gather_into_fills_preallocated_buffer():
    bank = HiddenStateBank(hidden_dim=3, device=torch.device("cpu"), initial_capacity=2, growth_capacity=2)
    first = bank.batch([(1, 10), (1, 11)])
    bank.update(
        first.row_index,
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )

    out = torch.empty(2, 3, dtype=torch.float32)
    row_index = bank.gather_into([(1, 11), (1, 10)], out)

    assert torch.equal(row_index, torch.tensor([1, 0], dtype=torch.long))
    assert torch.equal(
        out,
        torch.tensor(
            [
                [4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        ),
    )
