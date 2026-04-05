"""Regression coverage for rollout horizon fallback/truncation helpers."""

from __future__ import annotations

from pathlib import Path
import struct
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))

import bots.tensor_proxy.main as tensor_proxy_main
import game_worker as game_worker_mod


def _build_obs() -> list[float]:
    obs = [0.0] * tensor_proxy_main.INPUT_SIZE
    g = tensor_proxy_main.GRID_SIZE * tensor_proxy_main.GRID_SIZE * tensor_proxy_main.TILE_FEATURES
    obs[g + 3] = 1.0  # titanium
    obs[g + 4] = 1.0  # axionite
    obs[g + 5] = 0.2  # scale percent 100 -> multiplier 1
    return obs


def _mark_buildable(obs: list[float], dir_idx: int, *, ore: bool = False) -> None:
    tile_idx = tensor_proxy_main._dir_tile_index(dir_idx)
    ts = tile_idx * tensor_proxy_main.TILE_FEATURES
    obs[ts + 4] = 1.0
    if ore:
        obs[ts + 1] = 1.0


def test_scripted_fallback_prefers_adjacent_ore():
    obs = _build_obs()
    _mark_buildable(obs, 1, ore=True)  # north
    _mark_buildable(obs, 5)  # east

    bt, bd, cd, brt, md = tensor_proxy_main._scripted_fallback_actions(
        obs,
        unit_id=0,
        round_idx=200,
    )

    assert (bt, bd, cd, brt) == (1, 1, 0, 0)
    assert md in tensor_proxy_main._FALLBACK_MOVE_DIR_ORDER


def test_scripted_fallback_uses_road_when_no_ore():
    obs = _build_obs()
    _mark_buildable(obs, 5)  # east only

    bt, bd, cd, brt, md = tensor_proxy_main._scripted_fallback_actions(
        obs,
        unit_id=0,
        round_idx=200,
    )

    assert (bt, bd, cd, brt) == (3, 5, 0, 0)
    assert md in tensor_proxy_main._FALLBACK_MOVE_DIR_ORDER


class _FakeController:
    def get_current_round(self):
        return 200


def test_player_uses_local_fallback_after_horizon(monkeypatch):
    player = tensor_proxy_main.Player()
    player._training_turn_cap = 200
    player.my_team = object()
    player.core_center = object()

    obs = _build_obs()
    _mark_buildable(obs, 1, ore=True)
    monkeypatch.setattr(player, "_encode_state_buf", lambda ct: obs)
    monkeypatch.setattr(
        tensor_proxy_main,
        "_get_collector",
        lambda: (_ for _ in ()).throw(AssertionError("collector should not be used")),
    )

    seen = {}

    def _fake_build(ct, action, direction, conv_dir, my_team, bridge_target_idx):
        seen["build"] = (action, direction, conv_dir, bridge_target_idx)
        return 0.0, 0.0, [0] * 8

    def _fake_move(ct, direction):
        seen["move"] = direction

    monkeypatch.setattr(tensor_proxy_main, "execute_build", _fake_build)
    monkeypatch.setattr(tensor_proxy_main, "execute_move", _fake_move)

    player._run_builder(_FakeController())

    assert seen["build"][0] == "harvester"
    assert seen["move"] in tensor_proxy_main.DIR_MAP_9


def test_player_flushes_one_collector_request_at_horizon(monkeypatch):
    player = tensor_proxy_main.Player()
    player._training_turn_cap = 200
    player.my_team = object()
    player.core_center = object()
    player._has_collected_step = True
    player._dbg_flags = 37

    obs = _build_obs()
    _mark_buildable(obs, 1, ore=True)
    monkeypatch.setattr(player, "_encode_state_buf", lambda ct: obs)

    class _Collector:
        def __init__(self):
            self.calls = []

        def request(self, unit_id, round_idx, sent_obs, dbg_flags):
            self.calls.append((unit_id, round_idx, list(sent_obs), dbg_flags))
            return 0, 4, 0, 0, 4

    collector = _Collector()
    monkeypatch.setattr(tensor_proxy_main, "_get_collector", lambda: collector)
    monkeypatch.setattr(
        tensor_proxy_main,
        "execute_build",
        lambda ct, action, direction, conv_dir, my_team, bridge_target_idx: (0.0, 0.0, [0] * 8),
    )
    monkeypatch.setattr(tensor_proxy_main, "execute_move", lambda ct, direction: None)

    player._run_builder(_FakeController())
    player._run_builder(type("_Round201", (), {"get_current_round": lambda self: 201})())

    assert [(round_idx, dbg_flags) for _unit_id, round_idx, _sent_obs, dbg_flags in collector.calls] == [(200, 37)]
    assert player._post_horizon_flushed is True


def test_invalid_training_turn_cap_env_raises(monkeypatch):
    monkeypatch.setenv("TENSOR_TRAIN_MAX_TURNS", "abc")
    with pytest.raises(RuntimeError, match="invalid TENSOR_TRAIN_MAX_TURNS"):
        tensor_proxy_main._training_turn_cap()
    with pytest.raises(RuntimeError, match="invalid TENSOR_TRAIN_MAX_TURNS"):
        game_worker_mod._training_turn_cap()


class _FakeReplayData:
    def __init__(self):
        self.state_stride = 13
        self.n_scalars = 3
        self.n_turns = 4
        self.n_channels = 13
        self.map_w = 1
        self.map_h = 1
        self.states_bytes = bytes(range(self.state_stride * 4 * self.n_turns))
        self.scalars_bytes = bytes(range(self.n_scalars * 4 * self.n_turns))
        self.resource_transfers_bytes = bytes(range(8 * self.n_turns))
        self.titanium_delivered_bytes = bytes(range(8 * self.n_turns))
        self.axionite_delivered_bytes = bytes(range(8 * self.n_turns))

        turn0 = bytearray(self.state_stride * 4)
        struct.pack_into("<f", turn0, 11 * 4, 1.0)
        struct.pack_into("<f", turn0, 12 * 4, 1.0)
        self._turn0 = bytes(turn0)

    def state_bytes_for_turn(self, turn_idx: int) -> bytes:
        assert turn_idx == 0
        return self._turn0


def test_game_worker_build_global_states_truncates_artifacts(tmp_path):
    result_path = tmp_path / "worker.pkl"
    rd = _FakeReplayData()

    global_states = game_worker_mod._build_global_states(rd, str(result_path), max_turns=2)

    assert global_states["n_turns"] == 2
    assert global_states["core_x"] == 0
    assert global_states["core_y"] == 0
    assert Path(global_states["states_path"]).read_bytes() == rd.states_bytes[: rd.state_stride * 4 * 2]
    assert Path(global_states["scalars_path"]).read_bytes() == rd.scalars_bytes[: rd.n_scalars * 4 * 2]
    assert Path(global_states["transfer_events_path"]).read_bytes() == rd.resource_transfers_bytes[: 16]
    assert Path(global_states["ti_delivered_path"]).read_bytes() == rd.titanium_delivered_bytes[: 16]
    assert Path(global_states["ax_delivered_path"]).read_bytes() == rd.axionite_delivered_bytes[: 16]
