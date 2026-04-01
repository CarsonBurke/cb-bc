"""Integration tests for pretraining: dataset loading, loss computation, and short training loops."""

from __future__ import annotations

import collections
import json
import sys
import tempfile
from pathlib import Path
from threading import Lock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
PRETRAIN_DIR = TORCH_DIR / "pretrain"
REPLAY_DIR = PROJECT_ROOT / "replay_data"
INDEX_PATH = REPLAY_DIR / "index.jsonl"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))
if str(PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_DIR))

from model import (
    BuilderMLP, CentralizedCritic, DEVICE, H_DIM, N_BUILD, N_DIR,
    GLOBAL_MAP_CHANNELS, GLOBAL_MAP_MAX_W, GLOBAL_MAP_MAX_H, GLOBAL_SCALAR_DIM,
    UNIT_CONTEXT_DIM, ACTOR_LATENT_DIM,
    twohot_encode, _make_bucket_centers,
)
from dataset import ActorBCDataset, CriticBCDataset, _load_index
from dataset import _critic_winloss_targets
import train_batched as train_batched_mod
from train_critic import train_critic
import train_critic as train_critic_mod
import full_train as full_train_mod

# Skip the whole module if no replay data is available
pytestmark = pytest.mark.skipif(
    not INDEX_PATH.exists() or not any(REPLAY_DIR.glob("*.replay26")),
    reason="No replay data available",
)


def _small_index(tmp_path: Path, n_entries: int = 3) -> Path:
    """Write a truncated index with only n_entries so scans are fast."""
    entries = _load_index(INDEX_PATH, min_rating=0)[:n_entries]
    index = tmp_path / "index.jsonl"
    index.write_text("\n".join(json.dumps(e) for e in entries))
    return index


# ---------------------------------------------------------------------------
# Dataset / index tests
# ---------------------------------------------------------------------------

class TestLoadIndex:
    def test_filters_by_min_rating(self):
        entries = _load_index(INDEX_PATH, min_rating=0)
        high = _load_index(INDEX_PATH, min_rating=9999)
        assert len(entries) > 0
        assert len(high) < len(entries)

    def test_missing_file_returns_empty(self, tmp_path):
        assert _load_index(tmp_path / "nonexistent.jsonl") == []


class TestActorBCDataset:
    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("actor")
        idx = _small_index(tmp, n_entries=3)
        return ActorBCDataset(
            REPLAY_DIR, index_path=idx, min_rating=0,
            winner_only=True, cache_size=4,
        )

    def test_loads_games(self, ds):
        assert ds.n_games > 0
        assert ds.total_samples > 0

    def test_trajectories_shape_and_ranges(self, ds):
        trajs = ds.load_all_trajectories()
        assert len(trajs) > 0
        for t in trajs[:5]:  # spot-check a few
            obs, actions = t["obs"], t["actions"]
            assert obs.ndim == 2
            assert actions.ndim == 2 and actions.shape[1] == 5
            assert obs.shape[0] == actions.shape[0] and obs.shape[0] > 0
            bt = actions[:, 0]
            assert bt.min() >= 0 and bt.max() < N_BUILD


class TestCriticBCDataset:
    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("critic")
        idx = _small_index(tmp, n_entries=2)
        return CriticBCDataset(
            REPLAY_DIR, index_path=idx, min_rating=0, cache_size=4,
        )

    def test_loads_both_teams(self, ds):
        assert ds.n_games >= 2

    def test_batch_shapes_and_value_range(self, ds):
        states, masks, scalars, values = ds.load_game_batch(
            n_games=1, max_turns_per_game=32,
            max_map_w=GLOBAL_MAP_MAX_W, max_map_h=GLOBAL_MAP_MAX_H,
            n_channels=GLOBAL_MAP_CHANNELS, n_scalars=GLOBAL_SCALAR_DIM,
        )
        N = states.shape[0]
        assert N > 0 and N <= 32
        assert states.shape == (N, GLOBAL_MAP_CHANNELS, GLOBAL_MAP_MAX_H, GLOBAL_MAP_MAX_W)
        assert masks.shape == (N, 1, GLOBAL_MAP_MAX_H, GLOBAL_MAP_MAX_W)
        assert scalars.shape == (N, GLOBAL_SCALAR_DIM)
        assert values.shape == (N,)
        assert torch.isfinite(values).all()
        assert values.abs().max() > 0.0

    def test_winloss_targets_backpropagate_terminal_outcome(self):
        targets = _critic_winloss_targets(
            3,
            base_value=1.0,
            gamma=0.5,
        )
        assert torch.allclose(targets, torch.tensor([0.25, 0.5, 1.0]))

    def test_winloss_targets_zero_draw_signal(self):
        targets = _critic_winloss_targets(
            4,
            base_value=0.0,
            gamma=0.9,
        )
        assert torch.equal(targets, torch.zeros(4))

    def test_load_staged_games_reuses_cached_tensor_refs(self, ds):
        game = ds.games[0]
        bufs = ds._get_buffers(game["path"], game["team"])
        staged = ds.load_staged_games([game], pin_memory=True)

        assert len(staged) == 1
        assert staged[0]["states"].data_ptr() == bufs["states"].data_ptr()
        assert staged[0]["scalars"].data_ptr() == bufs["scalars"].data_ptr()
        assert staged[0]["step_targets"].data_ptr() == bufs["step_targets"].data_ptr()
        assert staged[0]["round_offsets"].data_ptr() == bufs["round_offsets"].data_ptr()


# ---------------------------------------------------------------------------
# Short training loop smoke tests
# ---------------------------------------------------------------------------

class TestActorBCTraining:
    """Verify actor BC loss decreases over a few steps on real data."""

    def test_loss_decreases(self, tmp_path):
        idx = _small_index(tmp_path, n_entries=2)
        ds = ActorBCDataset(REPLAY_DIR, index_path=idx, min_rating=0, cache_size=4)
        trajs = ds.load_all_trajectories()
        # Concatenate a few trajectories for a flat batch
        obs = torch.cat([t["obs"] for t in trajs[:20]])
        actions = torch.cat([t["actions"] for t in trajs[:20]]).long()
        N = obs.shape[0]

        model = BuilderMLP()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        h = torch.zeros(N, H_DIM)

        losses = []
        for _ in range(5):
            bt_logits, bd_logits, cd_logits, brt_logits, md_logits, _eps, _h = model(obs, h)
            loss = (
                torch.nn.functional.cross_entropy(bt_logits, actions[:, 0])
                + torch.nn.functional.cross_entropy(md_logits, actions[:, 4])
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"


class TestCriticPretraining:
    """Verify critic pretrain loss decreases over a few steps on real data."""

    @staticmethod
    def _frozen_actor():
        actor = BuilderMLP().to(DEVICE).eval()
        for p in actor.parameters():
            p.requires_grad_(False)
        return actor

    def test_step_actor_latent_materialization_releases_replay_actor_staging(self, monkeypatch):
        actor = self._frozen_actor()
        actor_latent_bt = torch.arange(3 * ACTOR_LATENT_DIM, dtype=torch.float16).reshape(3, ACTOR_LATENT_DIM)
        expected = actor_latent_bt.index_select(0, torch.tensor([2, 0], dtype=torch.long))
        calls: list[str] = []

        def fake_obs_pre(_actor, obs_bt_cpu, chunk_size=4096):
            calls.append("obs_pre")
            assert chunk_size == 4096
            assert obs_bt_cpu.shape == (3, 7)
            return torch.ones(3, 5, dtype=torch.float16)

        def fake_latents(_actor, obs_pre_bt_cpu, bot_ids_bt_cpu):
            calls.append("latent")
            assert obs_pre_bt_cpu.shape == (3, 5)
            assert torch.equal(bot_ids_bt_cpu, torch.tensor([10, 10, 42], dtype=torch.int32))
            return actor_latent_bt

        monkeypatch.setattr(train_critic_mod, "_compute_actor_obs_pre_for_replay", fake_obs_pre)
        monkeypatch.setattr(train_critic_mod, "_compute_actor_latents_for_replay", fake_latents)

        bufs = {
            "actor_obs": torch.zeros((3, 7), dtype=torch.float16),
            "actor_bot_ids": torch.tensor([10, 10, 42], dtype=torch.int32),
            "actor_step_order": torch.tensor([2, 0], dtype=torch.long),
        }

        step_actor_latent = train_critic_mod._ensure_step_actor_latents(bufs, actor)

        assert torch.equal(step_actor_latent, expected)
        assert bufs["step_actor_latent"] is step_actor_latent
        assert "actor_obs" not in bufs
        assert "actor_obs_pre" not in bufs
        assert "actor_bot_ids" not in bufs
        assert "actor_step_order" not in bufs

        cached = train_critic_mod._ensure_step_actor_latents(bufs, actor)
        assert cached is step_actor_latent
        assert calls == ["obs_pre", "latent"]

    def test_materialize_critic_batch_skips_game_log_tensors_when_disabled(self):
        actor = self._frozen_actor()

        class TinyCriticDataset:
            def _get_buffers(self, path: str, team: int):
                assert path == "g0"
                assert team == 0
                return {
                    "n_turns": 2,
                    "states": torch.zeros((2, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                    "scalars": torch.zeros((2, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                    "targets": torch.zeros(2, dtype=torch.float32),
                    "step_rounds": torch.tensor([0, 1], dtype=torch.int16),
                    "step_unit_xy": torch.zeros((2, 2), dtype=torch.int16),
                    "step_unit_ctx": torch.zeros((2, UNIT_CONTEXT_DIM), dtype=torch.float16),
                    "step_actor_latent": torch.zeros((2, ACTOR_LATENT_DIM), dtype=torch.float16),
                    "step_targets": torch.zeros(2, dtype=torch.float32),
                    "round_offsets": torch.tensor([0, 1, 2], dtype=torch.long),
                    "map_w": 1,
                    "map_h": 1,
                    "n_channels": GLOBAL_MAP_CHANNELS,
                    "n_scalars": GLOBAL_SCALAR_DIM,
                }

        games = [{"path": "g0", "team": 0, "n_turns": 2}]
        batch_groups = [{"game_idx": 0, "round_start": 0, "round_end": 2}]
        buffers = train_critic_mod._CriticBatchBuffers()

        out = train_critic_mod._materialize_critic_batch(
            batch_groups,
            games,
            TinyCriticDataset(),
            buffers,
            actor,
            pin_memory=False,
            need_game_log=False,
        )

        assert out[-2] is None
        assert out[-1] is None
        assert buffers.step_game_idx is None
        assert buffers.step_round_num is None

    def test_precompute_all_actor_latents_batches_multiple_games(self, monkeypatch, tmp_path):
        actor = self._frozen_actor()
        obs_pre_calls: list[int] = []
        latent_bot_ids: list[int] = []
        saved_paths: list[Path] = []

        def fake_obs_pre(_actor, obs_bt_cpu, device, chunk_size=4096):
            del _actor, device, chunk_size
            obs_pre_calls.append(int(obs_bt_cpu.shape[0]))
            return torch.zeros((obs_bt_cpu.shape[0], 4), dtype=torch.float16)

        def fake_latents(_actor, obs_pre_bt, bot_ids_bt, device):
            del _actor, device
            latent_bot_ids[:] = bot_ids_bt.tolist()
            rows = obs_pre_bt.shape[0]
            data = torch.arange(rows * ACTOR_LATENT_DIM, dtype=torch.float16)
            return data.reshape(rows, ACTOR_LATENT_DIM)

        monkeypatch.setattr(train_critic_mod, "_compute_actor_obs_pre", fake_obs_pre)
        monkeypatch.setattr(train_critic_mod, "_compute_actor_latents_for_replay", fake_latents)
        monkeypatch.setattr(train_critic_mod, "_atomic_torch_save", lambda payload, path: saved_paths.append(path))

        class TinyDataset:
            def __init__(self):
                self.cache_dir = tmp_path / "cache"
                self.cache_dir.mkdir()
                self._cache = {}
                self._cache_size = 1
                self._cache_order = collections.deque()
                self._cache_lock = Lock()
                self.games = []
                self._buffers = {}

                specs = [
                    (tmp_path / "g0.replay26", 0, 2, torch.tensor([7, 7], dtype=torch.int32), torch.tensor([1, 0], dtype=torch.long)),
                    (tmp_path / "g1.replay26", 1, 3, torch.tensor([7, 7, 9], dtype=torch.int32), torch.tensor([2, 0, 1], dtype=torch.long)),
                ]
                for path, team, n_steps, bot_ids, step_order in specs:
                    path.write_bytes(b"replay")
                    game = {"path": str(path), "team": team}
                    self.games.append(game)
                    self._buffers[(str(path), team)] = {
                        "n_turns": n_steps,
                        "states": torch.zeros((n_steps, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                        "scalars": torch.zeros((n_steps, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                        "targets": torch.zeros(n_steps, dtype=torch.float32),
                        "step_rounds": torch.arange(n_steps, dtype=torch.int16),
                        "step_unit_xy": torch.zeros((n_steps, 2), dtype=torch.int16),
                        "step_unit_ctx": torch.zeros((n_steps, UNIT_CONTEXT_DIM), dtype=torch.float16),
                        "step_targets": torch.zeros(n_steps, dtype=torch.float32),
                        "round_offsets": torch.arange(n_steps + 1, dtype=torch.long),
                        "actor_obs": torch.zeros((n_steps, 7), dtype=torch.float16),
                        "actor_bot_ids": bot_ids,
                        "actor_step_order": step_order,
                        "map_w": 1,
                        "map_h": 1,
                        "core_x": 0,
                        "core_y": 0,
                        "n_channels": GLOBAL_MAP_CHANNELS,
                        "n_scalars": GLOBAL_SCALAR_DIM,
                    }

            def _get_buffers(self, path: str, team: int):
                return self._buffers[(path, team)]

        dataset = TinyDataset()
        train_critic_mod.precompute_all_actor_latents(
            dataset,
            actor,
            torch.device("cpu"),
            target_samples_per_batch=16,
        )

        assert obs_pre_calls == [5]
        assert latent_bot_ids[0] == latent_bot_ids[1]
        assert latent_bot_ids[1] != latent_bot_ids[2]
        assert latent_bot_ids[2] == latent_bot_ids[3]
        assert len(saved_paths) == 2

        g0 = dataset._get_buffers(dataset.games[0]["path"], dataset.games[0]["team"])
        g1 = dataset._get_buffers(dataset.games[1]["path"], dataset.games[1]["team"])
        step = float(ACTOR_LATENT_DIM)
        assert torch.equal(g0["step_actor_latent"][:, 0], torch.tensor([step, 0.0], dtype=torch.float16))
        assert torch.equal(g1["step_actor_latent"][:, 0], torch.tensor([4 * step, 2 * step, 3 * step], dtype=torch.float16))
        assert "actor_obs" not in g0 and "actor_bot_ids" not in g0 and "actor_step_order" not in g0
        assert "actor_obs" not in g1 and "actor_bot_ids" not in g1 and "actor_step_order" not in g1

    def test_train_critic_flushes_weights_once_per_epoch_not_per_step(self, monkeypatch, tmp_path):
        save_calls: list[Path] = []

        monkeypatch.setattr(
            train_critic_mod,
            "_atomic_torch_save",
            lambda payload, path: save_calls.append(path),
        )

        class TinyCriticDataset:
            def __init__(self):
                self.games = [{"path": "g0", "team": 0, "n_turns": 4}]
                self.n_games = 1

            def _get_buffers(self, path: str, team: int):
                assert path == "g0"
                assert team == 0
                return {
                    "n_turns": 4,
                    "states": torch.zeros((4, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                    "scalars": torch.zeros((4, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                    "targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "step_rounds": torch.tensor([0, 1, 2, 3], dtype=torch.int16),
                    "step_unit_xy": torch.zeros((4, 2), dtype=torch.int16),
                    "step_unit_ctx": torch.zeros((4, UNIT_CONTEXT_DIM), dtype=torch.float16),
                    "step_actor_latent": torch.zeros((4, ACTOR_LATENT_DIM), dtype=torch.float16),
                    "step_targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "round_offsets": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                    "map_w": 1,
                    "map_h": 1,
                    "n_channels": GLOBAL_MAP_CHANNELS,
                    "n_scalars": GLOBAL_SCALAR_DIM,
                }

        critic = CentralizedCritic().to(DEVICE)
        stats = train_critic(
            critic,
            TinyCriticDataset(),
            epochs=1,
            lr=1e-3,
            games_per_batch=1,
            rounds_per_group=2,
            chunk_groups_per_batch=1,
            step_log_every=1,
            weights_path=tmp_path / "critic_weights.pt",
            enable_compile=False,
        )

        assert stats["optimizer_steps"] == 2
        assert save_calls == [tmp_path / "critic_weights.pt"]

    def test_loss_decreases(self, tmp_path):
        idx = _small_index(tmp_path, n_entries=2)
        ds = CriticBCDataset(REPLAY_DIR, index_path=idx, min_rating=0, cache_size=4)
        critic = CentralizedCritic().to(DEVICE)
        critic.train()
        opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

        bucket_centers = _make_bucket_centers().to(DEVICE)
        states, masks, scalars, values = ds.load_game_batch(
            n_games=1, max_turns_per_game=16,
            max_map_w=GLOBAL_MAP_MAX_W, max_map_h=GLOBAL_MAP_MAX_H,
            n_channels=GLOBAL_MAP_CHANNELS, n_scalars=GLOBAL_SCALAR_DIM,
        )
        states = states.to(DEVICE)
        masks = masks.to(DEVICE)
        scalars = scalars.to(DEVICE)
        values = values.to(DEVICE)
        N = states.shape[0]
        targets = twohot_encode(values, bucket_centers)
        dummy_xy = torch.zeros(N, 2, dtype=torch.long, device=DEVICE)
        dummy_ctx = torch.zeros(N, UNIT_CONTEXT_DIM, device=DEVICE)
        dummy_lat = torch.zeros(N, ACTOR_LATENT_DIM, device=DEVICE)

        losses = []
        for _ in range(5):
            feat = critic.encode_map(states, masks)
            round_idx = torch.arange(N, device=DEVICE)
            _vals, logits = critic.unit_value_logits(
                feat, scalars, round_idx, dummy_xy, dummy_ctx, dummy_lat,
            )
            loss = torch.nn.functional.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"

    def test_writes_step_value_loss_log(self, tmp_path):
        class TinyCriticDataset:
            def __init__(self):
                self.games = [{"path": "g0", "team": 0, "n_turns": 4}]
                self.n_games = 1

            def _get_buffers(self, path: str, team: int):
                assert path == "g0"
                assert team == 0
                return {
                    "n_turns": 4,
                    "states": torch.zeros((4, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                    "scalars": torch.zeros((4, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                    "targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "step_rounds": torch.tensor([0, 1, 2, 3], dtype=torch.int16),
                    "step_unit_xy": torch.zeros((4, 2), dtype=torch.int16),
                    "step_unit_ctx": torch.zeros((4, UNIT_CONTEXT_DIM), dtype=torch.float16),
                    "step_actor_latent": torch.zeros((4, ACTOR_LATENT_DIM), dtype=torch.float16),
                    "step_targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "round_offsets": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                    "map_w": 1,
                    "map_h": 1,
                    "n_channels": GLOBAL_MAP_CHANNELS,
                    "n_scalars": GLOBAL_SCALAR_DIM,
                }

        critic = CentralizedCritic().to(DEVICE)
        train_log = tmp_path / "train_log.jsonl"
        step_log = tmp_path / "train_step_log.jsonl"
        stats = train_critic(
            critic,
            TinyCriticDataset(),
            epochs=1,
            lr=1e-3,
            games_per_batch=1,
            rounds_per_group=2,
            chunk_groups_per_batch=1,
            step_log_every=1,
            log_path=train_log,
            step_log_path=step_log,
            enable_compile=False,
        )

        assert stats["optimizer_steps"] == 2
        train_rows = [json.loads(line) for line in train_log.read_text().splitlines() if line.strip()]
        step_rows = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
        assert len(train_rows) == 1
        assert train_rows[0]["value_loss"] == pytest.approx(train_rows[0]["loss"])
        assert "explained_var" in train_rows[0]
        assert len(step_rows) == 1
        assert step_rows[0]["optimizer_step"] == 1
        assert step_rows[0]["value_loss"] == pytest.approx(step_rows[0]["loss"])
        assert "explained_var" in step_rows[0]
        assert step_rows[0]["n_samples"] == 2

    def test_logged_explained_var_matches_ppo_residual_variance(self, monkeypatch, tmp_path):
        unit_ctx = torch.zeros((4, UNIT_CONTEXT_DIM), dtype=torch.float16)
        unit_ctx[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float16)

        class TinyCriticDataset:
            def __init__(self):
                self.games = [{"path": "g0", "team": 0, "n_turns": 4}]
                self.n_games = 1

            def _get_buffers(self, path: str, team: int):
                assert path == "g0"
                assert team == 0
                return {
                    "n_turns": 4,
                    "states": torch.zeros((4, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                    "scalars": torch.zeros((4, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                    "targets": torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
                    "step_rounds": torch.tensor([0, 1, 2, 3], dtype=torch.int16),
                    "step_unit_xy": torch.zeros((4, 2), dtype=torch.int16),
                    "step_unit_ctx": unit_ctx.clone(),
                    "step_actor_latent": torch.zeros((4, ACTOR_LATENT_DIM), dtype=torch.float16),
                    "step_targets": torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32),
                    "round_offsets": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                    "map_w": 1,
                    "map_h": 1,
                    "n_channels": GLOBAL_MAP_CHANNELS,
                    "n_scalars": GLOBAL_SCALAR_DIM,
                }

        class FakeTrainStep(torch.nn.Module):
            def forward(
                self,
                states: torch.Tensor,
                masks: torch.Tensor,
                scalars: torch.Tensor,
                round_idx: torch.Tensor,
                unit_xy: torch.Tensor,
                unit_ctx: torch.Tensor,
                actor_latent: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                pred_values = unit_ctx[:, 0].to(dtype=torch.float32) + 10.0
                logits = torch.zeros(
                    (round_idx.shape[0], train_critic_mod._make_bucket_centers().numel()),
                    device=states.device,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                return pred_values, logits

        monkeypatch.setattr(
            train_critic_mod,
            "maybe_compile",
            lambda _module, _cuda_cfg, label="": FakeTrainStep(),
        )

        critic = CentralizedCritic().to(DEVICE)
        train_log = tmp_path / "train_log.jsonl"
        step_log = tmp_path / "train_step_log.jsonl"
        stats = train_critic(
            critic,
            TinyCriticDataset(),
            epochs=1,
            lr=1e-3,
            games_per_batch=1,
            rounds_per_group=2,
            chunk_groups_per_batch=1,
            step_log_every=1,
            log_path=train_log,
            step_log_path=step_log,
            enable_compile=False,
        )

        assert stats["optimizer_steps"] == 2
        train_rows = [json.loads(line) for line in train_log.read_text().splitlines() if line.strip()]
        step_rows = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
        assert len(train_rows) == 1
        assert len(step_rows) == 1
        assert train_rows[0]["explained_var"] == pytest.approx(1.0)
        assert step_rows[0]["explained_var"] == pytest.approx(1.0)

    def test_global_round_group_shuffle_uses_locality_blocks(self, monkeypatch):
        class TinyCriticDataset:
            def __init__(self):
                self.games = [
                    {"path": f"g{i}", "team": 0, "n_turns": 4}
                    for i in range(3)
                ]
                self.n_games = len(self.games)

            def _get_buffers(self, path: str, team: int):
                assert team == 0
                return {
                    "states": torch.zeros((4, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                    "scalars": torch.zeros((4, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                    "targets": torch.zeros(4, dtype=torch.float32),
                    "step_rounds": torch.tensor([0, 1, 2, 3], dtype=torch.int16),
                    "step_unit_xy": torch.zeros((4, 2), dtype=torch.int16),
                    "step_unit_ctx": torch.zeros((4, UNIT_CONTEXT_DIM), dtype=torch.float16),
                    "step_actor_latent": torch.zeros((4, ACTOR_LATENT_DIM), dtype=torch.float16),
                    "step_targets": torch.zeros(4, dtype=torch.float32),
                    "round_offsets": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                    "map_w": 1,
                    "map_h": 1,
                    "n_channels": GLOBAL_MAP_CHANNELS,
                    "n_scalars": GLOBAL_SCALAR_DIM,
                }

        shuffle_sizes: list[int] = []
        real_shuffle = train_critic_mod.random.shuffle

        def recording_shuffle(seq):
            shuffle_sizes.append(len(seq))
            real_shuffle(seq)

        monkeypatch.setattr(train_critic_mod.random, "shuffle", recording_shuffle)

        critic = CentralizedCritic().to(DEVICE)
        stats = train_critic(
            critic,
            self._frozen_actor(),
            TinyCriticDataset(),
            epochs=1,
            lr=1e-3,
            games_per_batch=1,
            rounds_per_group=2,
            chunk_groups_per_batch=1,
            enable_compile=False,
        )

        assert stats["optimizer_steps"] == 6
        assert shuffle_sizes == [3, 3, 2, 2, 2]

    def test_resume_continues_optimizer_steps_in_same_logs(self, tmp_path):
        class TinyCriticDataset:
            def __init__(self):
                self.games = [{"path": "g0", "team": 0, "n_turns": 4}]
                self.n_games = 1

            def _get_buffers(self, path: str, team: int):
                assert path == "g0"
                assert team == 0
                return {
                    "n_turns": 4,
                    "states": torch.zeros((4, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                    "scalars": torch.zeros((4, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                    "targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "step_rounds": torch.tensor([0, 1, 2, 3], dtype=torch.int16),
                    "step_unit_xy": torch.zeros((4, 2), dtype=torch.int16),
                    "step_unit_ctx": torch.zeros((4, UNIT_CONTEXT_DIM), dtype=torch.float16),
                    "step_actor_latent": torch.zeros((4, ACTOR_LATENT_DIM), dtype=torch.float16),
                    "step_targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "round_offsets": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                    "map_w": 1,
                    "map_h": 1,
                    "n_channels": GLOBAL_MAP_CHANNELS,
                    "n_scalars": GLOBAL_SCALAR_DIM,
                }

        critic = CentralizedCritic().to(DEVICE)
        train_log = tmp_path / "train_log.jsonl"
        step_log = tmp_path / "train_step_log.jsonl"
        state_path = tmp_path / "trainer_state.pt"

        train_log.write_text("")
        step_log.write_text("")

        partial = train_critic(
            critic,
            self._frozen_actor(),
            TinyCriticDataset(),
            epochs=1,
            lr=1e-3,
            games_per_batch=1,
            rounds_per_group=2,
            chunk_groups_per_batch=1,
            step_log_every=1,
            log_path=train_log,
            step_log_path=step_log,
            resume_state_path=state_path,
            stop_after_optimizer_steps=1,
            enable_compile=False,
        )

        assert state_path.exists()
        assert partial["optimizer_steps"] == 1
        step_rows = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
        assert [row["optimizer_step"] for row in step_rows] == [1]

        resumed = train_critic(
            critic,
            self._frozen_actor(),
            TinyCriticDataset(),
            epochs=1,
            lr=1e-3,
            games_per_batch=1,
            rounds_per_group=2,
            chunk_groups_per_batch=1,
            step_log_every=1,
            log_path=train_log,
            step_log_path=step_log,
            resume_state_path=state_path,
            resume=True,
            enable_compile=False,
        )

        assert resumed["optimizer_steps"] == 2
        assert not state_path.exists()
        step_rows = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
        assert [row["optimizer_step"] for row in step_rows] == [1, 2]
        train_rows = [json.loads(line) for line in train_log.read_text().splitlines() if line.strip()]
        assert len(train_rows) == 1
        assert train_rows[0]["n_batches"] == 2

    def test_log_offsets_append_precursor_history(self, tmp_path):
        class TinyCriticDataset:
            def __init__(self):
                self.games = [{"path": "g0", "team": 0, "n_turns": 4}]
                self.n_games = 1

            def _get_buffers(self, path: str, team: int):
                assert path == "g0"
                assert team == 0
                return {
                    "n_turns": 4,
                    "states": torch.zeros((4, GLOBAL_MAP_CHANNELS, 1, 1), dtype=torch.float32),
                    "scalars": torch.zeros((4, GLOBAL_SCALAR_DIM), dtype=torch.float32),
                    "targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "step_rounds": torch.tensor([0, 1, 2, 3], dtype=torch.int16),
                    "step_unit_xy": torch.zeros((4, 2), dtype=torch.int16),
                    "step_unit_ctx": torch.zeros((4, UNIT_CONTEXT_DIM), dtype=torch.float16),
                    "step_actor_latent": torch.zeros((4, ACTOR_LATENT_DIM), dtype=torch.float16),
                    "step_targets": torch.linspace(0.0, 1.0, 4, dtype=torch.float32),
                    "round_offsets": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                    "map_w": 1,
                    "map_h": 1,
                    "n_channels": GLOBAL_MAP_CHANNELS,
                    "n_scalars": GLOBAL_SCALAR_DIM,
                }

        critic = CentralizedCritic().to(DEVICE)
        train_log = tmp_path / "train_log.jsonl"
        step_log = tmp_path / "train_step_log.jsonl"
        precursor_train = {
            "episode": 1,
            "loss": 1.0,
            "value_loss": 1.0,
            "explained_var": 0.5,
            "elapsed": 10.0,
            "n_batches": 2,
            "n_samples": 4,
        }
        precursor_step = {
            "episode": 1,
            "optimizer_step": 2,
            "epoch_optimizer_step": 2,
            "loss": 1.0,
            "value_loss": 1.0,
            "explained_var": 0.5,
            "elapsed": 10.0,
            "n_samples": 4,
        }
        train_log.write_text(json.dumps(precursor_train) + "\n")
        step_log.write_text(json.dumps(precursor_step) + "\n")

        stats = train_critic(
            critic,
            self._frozen_actor(),
            TinyCriticDataset(),
            epochs=1,
            lr=1e-3,
            games_per_batch=1,
            rounds_per_group=2,
            chunk_groups_per_batch=1,
            step_log_every=1,
            log_path=train_log,
            step_log_path=step_log,
            log_episode_offset=1,
            log_optimizer_step_offset=2,
            log_elapsed_offset=10.0,
            enable_compile=False,
        )

        assert stats["optimizer_steps"] == 4
        train_rows = [json.loads(line) for line in train_log.read_text().splitlines() if line.strip()]
        step_rows = [json.loads(line) for line in step_log.read_text().splitlines() if line.strip()]
        assert [row["episode"] for row in train_rows] == [1, 2]
        assert [row["optimizer_step"] for row in step_rows] == [2, 3, 4]
        assert step_rows[1]["elapsed"] >= 10.0


class TestFullTrainScript:
    def test_critic_pretrain_cli_requires_actor_weights(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_critic_mod, "DEVICE", torch.device("cpu"))
        monkeypatch.setattr(train_critic_mod, "SESSIONS_DIR", tmp_path / "sessions")

        class TinyDataset:
            total_turns = 0
            n_games = 0

        monkeypatch.setattr(train_critic_mod, "CriticBCDataset", lambda *_args, **_kwargs: TinyDataset())
        monkeypatch.setattr(train_critic_mod, "CentralizedCritic", lambda: torch.nn.Linear(1, 1))
        monkeypatch.setattr(train_critic_mod, "BuilderMLP", lambda: torch.nn.Linear(1, 1))
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_critic.py", "--session", "critic_cli_guard", "--replay-dir", str(tmp_path / "replays")],
        )

        with pytest.raises(SystemExit, match="requires --actor-weights"):
            train_critic_mod.main()

    def test_builds_sequential_checkpoint_handoff(self, monkeypatch):
        monkeypatch.setattr(full_train_mod, "_default_base_session", lambda: "full_seq")
        args = full_train_mod.argparse.Namespace(
            session="",
            critic_epochs=1,
            critic_games_per_batch=240,
            critic_rounds_per_group=32,
            critic_chunk_groups_per_batch=8,
            critic_init_weights="critic_init.pt",
            critic_no_compile=False,
            actor_epochs=2,
            actor_games_per_batch=800,
            actor_tbptt_len=32,
            actor_chunk_groups_per_batch=1,
            actor_init_weights="actor_init.json",
            actor_no_compile=False,
            ppo_episodes=100,
            ppo_opponents=3,
            ppo_games_per_opp=4,
            ppo_checkpoint_every=1,
            ppo_dummy_episodes=0,
            ppo_critic_minibatch_size=16384,
            ppo_critic_no_compile=True,
            dry_run=False,
        )

        sessions, commands = full_train_mod.build_commands(args)

        assert sessions["critic"] == "full_seq_critic"
        assert sessions["actor"] == "full_seq_actor"
        assert sessions["ppo"] == "full_seq_ppo"

        actor_cmd, critic_cmd, ppo_cmd = commands
        assert actor_cmd[0] == sys.executable
        assert actor_cmd[actor_cmd.index("--output") + 1] == sessions["actor_out"]
        assert critic_cmd[0] == sys.executable
        assert critic_cmd[critic_cmd.index("--output") + 1] == sessions["critic_out"]
        assert critic_cmd[critic_cmd.index("--actor-weights") + 1] == sessions["actor_out"]
        assert ppo_cmd[ppo_cmd.index("--init-weights") + 1] == sessions["actor_out"]
        assert ppo_cmd[ppo_cmd.index("--critic-init-weights") + 1] == sessions["critic_out"]
        assert ppo_cmd[ppo_cmd.index("--critic-minibatch-size") + 1] == "16384"
        assert "--critic-no-compile" in ppo_cmd


class TestPPOCriticPlumbing:
    def test_resume_critic_checkpoint_uses_resolved_lr(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(
            train_batched_mod,
            "load_centralized_critic_checkpoint",
            lambda path, lr: calls.append((Path(path), lr)) or True,
        )

        assert train_batched_mod._load_resume_critic_checkpoint(tmp_path / "weights.json", lr=1.25e-4)
        assert calls == [(tmp_path / "weights.json", 1.25e-4)]

    def test_load_critic_init_weights_accepts_plain_state_dict(self, monkeypatch, tmp_path):
        reset_calls = []
        loaded_states = []
        loaded_opt_states = []

        class DummyCritic:
            def load_state_dict(self, state):
                loaded_states.append(state)

        class DummyOpt:
            def load_state_dict(self, state):
                loaded_opt_states.append(state)

        monkeypatch.setattr(train_batched_mod, "reset_centralized_critic", lambda: reset_calls.append(True))
        monkeypatch.setattr(
            train_batched_mod,
            "get_centralized_critic",
            lambda lr: (DummyCritic(), DummyOpt()),
        )

        state = {"layer.weight": torch.tensor([1.0]), "layer.bias": torch.tensor([0.0])}
        ckpt_path = tmp_path / "critic.pt"
        torch.save(state, ckpt_path)

        assert train_batched_mod._load_critic_init_weights(ckpt_path, lr=7.5e-5)
        assert reset_calls == [True]
        assert len(loaded_states) == 1
        assert loaded_states[0].keys() == state.keys()
        assert loaded_opt_states == []
