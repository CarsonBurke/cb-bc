"""Regression coverage for persistent replay pretrain caches."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = ROOT / "bots" / "tensor" / "torch"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))


def _load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dataset_mod = _load_module("tensor_pretrain_dataset_for_tests", "bots/tensor/torch/pretrain/dataset.py")


def _write_index(
    replay_dir: Path,
    replay_name: str,
    *,
    winner_side: str = "a",
    turns: int | None = None,
) -> None:
    entry = {
        "file": replay_name,
        "team_a_rating": 2000,
        "team_b_rating": 2000,
        "winner_side": winner_side,
    }
    if turns is not None:
        entry["turns"] = turns
    (replay_dir / "index.jsonl").write_text(json.dumps(entry) + "\n")


def _actor_parse_result(*, n_samples: int = 2, obs_size: int = 498) -> dict:
    obs_vals = [float(i) for i in range(n_samples * obs_size)]
    bot_ids = struct.pack(f"<{n_samples}i", *range(n_samples))
    turns = struct.pack(f"<{n_samples}I", *range(n_samples))
    return {
        "n_samples": n_samples,
        "obs_size": obs_size,
        "obs": struct.pack(f"<{len(obs_vals)}f", *obs_vals),
        "actions": bytes([1, 0, 2, 3, 4, 0, 5, 6, 7, 8]),
        "bot_ids": bot_ids,
        "turns": turns,
    }


def _critic_parse_result(*, n_turns: int = 2, map_w: int = 2, map_h: int = 1, n_channels: int = 19, n_scalars: int = 17) -> dict:
    state_vals = [float(i) for i in range(n_turns * n_channels * map_h * map_w)]
    scalar_vals = [float(i) for i in range(n_turns * n_scalars)]
    zero_i32 = struct.pack(f"<{n_turns * 2}i", *([0] * (n_turns * 2)))
    zero_f32 = struct.pack(f"<{n_turns * 2}f", *([0.0] * (n_turns * 2)))
    return {
        "n_turns": n_turns,
        "map_w": map_w,
        "map_h": map_h,
        "n_channels": n_channels,
        "n_scalars": n_scalars,
        "states": struct.pack(f"<{len(state_vals)}f", *state_vals),
        "scalars": struct.pack(f"<{len(scalar_vals)}f", *scalar_vals),
        "working_harvesters": zero_i32,
        "resource_transfers": zero_f32,
        "titanium_delivered": zero_i32,
        "axionite_delivered": zero_i32,
    }


def test_critic_step_features_match_ppo_core_relative_xy():
    obs = torch.zeros(2, 498, dtype=torch.float32)
    turns = torch.tensor([0, 1], dtype=torch.int32)
    obs[:, dataset_mod._OBS_X_NORM] = torch.tensor([0.0, 0.4])
    obs[:, dataset_mod._OBS_Y_NORM] = torch.tensor([0.0, -0.25])

    step_rounds, unit_xy, unit_ctx, round_offsets = dataset_mod._critic_step_features_from_obs(
        obs,
        turns,
        map_w=10,
        map_h=8,
        n_turns=2,
        core_x=3,
        core_y=4,
    )

    assert torch.equal(step_rounds, torch.tensor([0, 1], dtype=torch.int16))
    assert torch.equal(unit_xy, torch.tensor([[3, 4], [7, 2]], dtype=torch.int16))
    assert unit_ctx.shape == (2, dataset_mod.UNIT_CONTEXT_DIM)
    assert torch.equal(round_offsets, torch.tensor([0, 1, 2], dtype=torch.long))


def test_extract_friendly_core_xy_reads_core_channel():
    state0 = torch.zeros(19, 4, 5, dtype=torch.float16)
    state0[11, 2, 4] = 1.0
    assert dataset_mod._extract_friendly_core_xy(state0) == (4, 2)


def test_actor_cache_reuses_disk_payload_between_dataset_instances(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "sample.replay26"
    replay_path.write_bytes(b"replay-a")
    _write_index(replay_dir, replay_path.name)

    calls = {"working": 0, "sequential": 0}

    def fake_working(_path: str):
        calls["working"] += 1
        return {"winner": 0, "n_turns": 2}

    def fake_working_batch(paths: list[str]):
        return [fake_working(path) for path in paths]

    def fake_sequential(_path: str, team: int):
        calls["sequential"] += 1
        assert team == 0
        return _actor_parse_result()

    monkeypatch.setattr(dataset_mod, "parse_replay_working", fake_working)
    monkeypatch.setattr(dataset_mod, "parse_replays_working_batch", fake_working_batch)
    monkeypatch.setattr(dataset_mod, "parse_replay_sequential", fake_sequential)

    ds1 = dataset_mod.ActorBCDataset(replay_dir)
    obs1, act1 = ds1.load_game_batch(n_games=1, device="cpu")

    ds2 = dataset_mod.ActorBCDataset(replay_dir)
    obs2, act2 = ds2.load_game_batch(n_games=1, device="cpu")

    assert calls["sequential"] == 1
    assert torch.equal(obs1, obs2)
    assert torch.equal(act1, act2)


def test_actor_metadata_scan_uses_index_metadata_without_reparse(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "sample.replay26"
    replay_path.write_bytes(b"replay-a")
    _write_index(replay_dir, replay_path.name, winner_side="b", turns=7)

    def fail_working(_path: str):
        raise AssertionError("parse_replay_working should not be called when index has turns and winner_side")

    monkeypatch.setattr(dataset_mod, "parse_replay_working", fail_working)

    ds = dataset_mod.ActorBCDataset(replay_dir)

    assert ds.n_games == 1
    assert ds.total_samples == 7
    assert ds.games[0]["team"] == 1
    assert ds.games[0]["n_turns"] == 7


def test_actor_metadata_scan_batches_missing_working_queries(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_a = replay_dir / "a.replay26"
    replay_b = replay_dir / "b.replay26"
    replay_a.write_bytes(b"a")
    replay_b.write_bytes(b"b")
    index = replay_dir / "index.jsonl"
    index.write_text(
        "\n".join([
            json.dumps({"file": replay_a.name, "team_a_rating": 2000, "team_b_rating": 2000}),
            json.dumps({"file": replay_b.name, "team_a_rating": 2000, "team_b_rating": 2000}),
        ]),
    )

    calls = {"batch": []}

    def fake_batch(paths: list[str]):
        calls["batch"].append(paths)
        return [
            {"winner": 0, "n_turns": 11},
            {"winner": 1, "n_turns": 13},
        ]

    def fail_working(_path: str):
        raise AssertionError("parse_replay_working should not be called when batch metadata is available")

    monkeypatch.setattr(dataset_mod, "parse_replays_working_batch", fake_batch)
    monkeypatch.setattr(dataset_mod, "parse_replay_working", fail_working)

    ds = dataset_mod.ActorBCDataset(replay_dir, winner_only=False)

    assert calls["batch"] == [[str(replay_a), str(replay_b)]]
    assert ds.n_games == 4
    assert ds.total_samples == 48


def test_actor_cache_invalidates_when_replay_file_changes(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "sample.replay26"
    replay_path.write_bytes(b"replay-a")
    _write_index(replay_dir, replay_path.name)

    calls = {"sequential": 0}

    monkeypatch.setattr(dataset_mod, "parse_replay_working", lambda _path: {"winner": 0, "n_turns": 2})
    monkeypatch.setattr(
        dataset_mod,
        "parse_replays_working_batch",
        lambda paths: [{"winner": 0, "n_turns": 2} for _ in paths],
    )

    def fake_sequential(_path: str, team: int):
        calls["sequential"] += 1
        assert team == 0
        return _actor_parse_result()

    monkeypatch.setattr(dataset_mod, "parse_replay_sequential", fake_sequential)

    ds1 = dataset_mod.ActorBCDataset(replay_dir)
    ds1.load_game_batch(n_games=1, device="cpu")

    replay_path.write_bytes(b"replay-b-updated")

    ds2 = dataset_mod.ActorBCDataset(replay_dir)
    ds2.load_game_batch(n_games=1, device="cpu")

    assert calls["sequential"] == 2


def test_critic_cache_reuses_disk_payload_between_dataset_instances(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "sample.replay26"
    replay_path.write_bytes(b"replay-a")
    _write_index(replay_dir, replay_path.name)

    calls = {"full": 0, "sequential": 0}

    monkeypatch.setattr(dataset_mod, "parse_replay_working", lambda _path: {"winner": 0, "n_turns": 2})

    def fake_full(_path: str, team: int):
        calls["full"] += 1
        assert team == 0
        return _critic_parse_result()

    def fake_sequential(_path: str, team: int):
        calls["sequential"] += 1
        assert team == 0
        return _actor_parse_result()

    monkeypatch.setattr(dataset_mod, "parse_replay_full", fake_full)
    monkeypatch.setattr(dataset_mod, "parse_replay_sequential", fake_sequential)

    ds1 = dataset_mod.CriticBCDataset(replay_dir)
    buf1 = ds1._get_buffers(str(replay_path), team=0)

    ds2 = dataset_mod.CriticBCDataset(replay_dir)
    buf2 = ds2._get_buffers(str(replay_path), team=0)

    assert calls["full"] == 1
    assert calls["sequential"] == 1
    assert torch.equal(buf1["states"], buf2["states"])
    assert torch.equal(buf1["scalars"], buf2["scalars"])
    assert torch.equal(buf1["targets"], buf2["targets"])
    assert torch.equal(buf1["step_rounds"], buf2["step_rounds"])
    assert torch.equal(buf1["step_unit_xy"], buf2["step_unit_xy"])
    assert torch.equal(buf1["step_unit_ctx"], buf2["step_unit_ctx"])
    assert torch.equal(buf1["step_targets"], buf2["step_targets"])
    assert torch.equal(buf1["round_offsets"], buf2["round_offsets"])


def test_critic_buffers_throw_on_stale_turn_metadata(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "sample.replay26"
    replay_path.write_bytes(b"replay-a")
    _write_index(replay_dir, replay_path.name, winner_side="a", turns=9)

    monkeypatch.setattr(dataset_mod, "parse_replay_working", lambda _path: {"winner": 0, "n_turns": 9})
    monkeypatch.setattr(dataset_mod, "parse_replay_full", lambda _path, team: _critic_parse_result(n_turns=7))

    ds = dataset_mod.CriticBCDataset(replay_dir)

    with pytest.raises(RuntimeError, match="Stale replay metadata"):
        ds._get_buffers(str(replay_path), team=0)


def test_critic_metadata_scan_uses_index_metadata_without_reparse(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "sample.replay26"
    replay_path.write_bytes(b"replay-a")
    _write_index(replay_dir, replay_path.name, winner_side="a", turns=9)

    def fail_working(_path: str):
        raise AssertionError("parse_replay_working should not be called when index has turns and winner_side")

    monkeypatch.setattr(dataset_mod, "parse_replay_working", fail_working)

    ds = dataset_mod.CriticBCDataset(replay_dir)

    assert ds.n_games == 2
    assert ds.total_turns == 18
    assert ds.games[0]["n_turns"] == 9
    assert ds.games[0]["base_value"] == 1.0
    assert ds.games[1]["base_value"] == -1.0


def test_critic_metadata_scan_batches_missing_working_queries(tmp_path, monkeypatch):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_a = replay_dir / "a.replay26"
    replay_b = replay_dir / "b.replay26"
    replay_a.write_bytes(b"a")
    replay_b.write_bytes(b"b")
    index = replay_dir / "index.jsonl"
    index.write_text(
        "\n".join([
            json.dumps({"file": replay_a.name, "team_a_rating": 2000, "team_b_rating": 2000}),
            json.dumps({"file": replay_b.name, "team_a_rating": 2000, "team_b_rating": 2000}),
        ]),
    )

    calls = {"batch": []}

    def fake_batch(paths: list[str]):
        calls["batch"].append(paths)
        return [
            {"winner": 0, "n_turns": 5},
            {"winner": 1, "n_turns": 7},
        ]

    def fail_working(_path: str):
        raise AssertionError("parse_replay_working should not be called when batch metadata is available")

    monkeypatch.setattr(dataset_mod, "parse_replays_working_batch", fake_batch)
    monkeypatch.setattr(dataset_mod, "parse_replay_working", fail_working)

    ds = dataset_mod.CriticBCDataset(replay_dir)

    assert calls["batch"] == [[str(replay_a), str(replay_b)]]
    assert ds.n_games == 4
    assert ds.total_turns == 24
