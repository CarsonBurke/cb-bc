"""Regression coverage for actor BC truncated BPTT batching."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
PRETRAIN_DIR = TORCH_DIR / "pretrain"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))
if str(PRETRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_DIR))

from model import ACTOR_LATENT_DIM, BuilderMLP, OBS_SIZE, TEACHER_HIDDEN, _OBS_AX, _OBS_SCALE, _OBS_TI
from pretrain import train_bc


class _DummyDataset:
    def __init__(self):
        self.games = [{"path": "synthetic.replay26", "team": 0}]

    @property
    def n_games(self) -> int:
        return len(self.games)


def test_load_game_batch_respects_staging_game_budget(monkeypatch):
    class _BudgetDataset:
        def __init__(self):
            self.games = [
                {"path": "g1.replay26", "team": 0, "n_turns": 2},
                {"path": "g2.replay26", "team": 0, "n_turns": 2},
            ]

        def _get_parsed(self, path: str, team: int):
            del team
            if path == "g1.replay26":
                return (
                    torch.randn(3, OBS_SIZE),
                    torch.zeros(3, 5, dtype=torch.long),
                    torch.tensor([0, 0, 1], dtype=torch.int32),
                    torch.tensor([0, 1, 1], dtype=torch.int32),
                )
            return (
                torch.randn(4, OBS_SIZE),
                torch.zeros(4, 5, dtype=torch.long),
                torch.tensor([0, 0, 1, 1], dtype=torch.int32),
                torch.tensor([0, 1, 0, 1], dtype=torch.int32),
            )

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    games, total_slots, max_rounds, used_games = train_bc._load_game_batch(
        _BudgetDataset(),
        _BudgetDataset().games,
        max_staging_games=1,
    )

    assert used_games == 1
    assert len(games) == 1
    assert games[0]["obs"].shape[0] == 3
    assert total_slots == 2
    assert max_rounds == 2


def test_load_game_batch_keeps_per_bot_labels_separate(monkeypatch):
    class _PerBotDataset:
        def __init__(self):
            self.games = [{"path": "g1.replay26", "team": 0, "n_turns": 2}]

        def _get_parsed(self, _path: str, _team: int):
            obs = torch.randn(4, OBS_SIZE)
            actions = torch.tensor([
                [2, 1, 0, 0, 4],   # bot 10, turn 0
                [1, 7, 0, 0, 4],   # bot 20, turn 0
                [0, 4, 0, 0, 3],   # bot 10, turn 1
                [12, 5, 0, 9, 4],  # bot 20, turn 1
            ], dtype=torch.long)
            bot_ids = torch.tensor([10, 20, 10, 20], dtype=torch.int32)
            turns = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
            return obs, actions, bot_ids, turns

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    games, total_slots, max_rounds, used_games = train_bc._load_game_batch(
        _PerBotDataset(),
        _PerBotDataset().games,
        max_staging_games=1,
    )

    assert used_games == 1
    assert total_slots == 2
    assert max_rounds == 2
    assert games[0]["round_offsets"] == [0, 2, 4]
    assert games[0]["bot_slots"].tolist() == [0, 1, 0, 1]
    assert games[0]["global_slots"].tolist() == [0, 1, 0, 1]
    assert games[0]["act"].tolist() == [
        [2, 1, 0, 0, 4],
        [1, 7, 0, 0, 4],
        [0, 4, 0, 0, 3],
        [12, 5, 0, 9, 4],
    ]


def test_load_game_batch_round_sort_handles_large_bot_ids(monkeypatch):
    class _LargeBotIdDataset:
        def __init__(self):
            self.games = [{"path": "g1.replay26", "team": 0, "n_turns": 2}]

        def _get_parsed(self, _path: str, _team: int):
            obs = torch.arange(2 * OBS_SIZE, dtype=torch.float32).reshape(2, OBS_SIZE)
            actions = torch.tensor([
                [1, 1, 0, 0, 4],  # bot 100000, turn 1
                [2, 2, 0, 0, 4],  # bot 200000, turn 0
            ], dtype=torch.long)
            bot_ids = torch.tensor([100000, 200000], dtype=torch.int32)
            turns = torch.tensor([1, 0], dtype=torch.int32)
            return obs, actions, bot_ids, turns

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    games, total_slots, max_rounds, used_games = train_bc._load_game_batch(
        _LargeBotIdDataset(),
        _LargeBotIdDataset().games,
        max_staging_games=1,
    )

    assert used_games == 1
    assert total_slots == 2
    assert max_rounds == 2
    assert games[0]["round_offsets"] == [0, 1, 2]
    assert games[0]["act"].tolist() == [
        [2, 2, 0, 0, 4],
        [1, 1, 0, 0, 4],
    ]


def test_load_game_batch_throws_on_stale_turn_metadata(monkeypatch):
    class _StaleMetadataDataset:
        def __init__(self):
            self.games = [{"path": "g1.replay26", "team": 0, "n_turns": 1}]

        def _get_parsed(self, _path: str, _team: int):
            obs = torch.randn(2, OBS_SIZE)
            actions = torch.zeros(2, 5, dtype=torch.long)
            bot_ids = torch.tensor([10, 10], dtype=torch.int32)
            turns = torch.tensor([0, 1], dtype=torch.int32)
            return obs, actions, bot_ids, turns

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))

    with pytest.raises(RuntimeError, match="Stale replay metadata"):
        train_bc._load_game_batch(
            _StaleMetadataDataset(),
            _StaleMetadataDataset().games,
            max_staging_games=1,
        )


def test_load_game_batch_uses_round_major_cache_when_available(monkeypatch):
    class _RoundMajorDataset:
        def __init__(self):
            self.games = [{"path": "g1.replay26", "team": 0, "n_turns": 3}]
            self.round_major_calls = 0

        def _get_round_major(self, _path: str, _team: int, *, n_rounds_hint: int | None = None):
            self.round_major_calls += 1
            assert n_rounds_hint == 3
            return (
                torch.arange(4 * OBS_SIZE, dtype=torch.float16).reshape(4, OBS_SIZE),
                torch.tensor([
                    [2, 1, 0, 0, 4],
                    [1, 7, 0, 0, 4],
                    [0, 4, 0, 0, 3],
                    [12, 5, 0, 9, 4],
                ], dtype=torch.int16),
                torch.tensor([0, 1, 0, 1], dtype=torch.long),
                torch.tensor([0, 2, 4, 4], dtype=torch.long),
                2,
            )

        def _get_parsed(self, _path: str, _team: int):
            raise AssertionError("_get_parsed should not be used when round-major cache is available")

    ds = _RoundMajorDataset()
    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))

    games, total_slots, max_rounds, used_games = train_bc._load_game_batch(
        ds,
        ds.games,
        max_staging_games=1,
    )

    assert used_games == 1
    assert total_slots == 2
    assert max_rounds == 3
    assert ds.round_major_calls == 1
    assert games[0]["round_offsets"] == [0, 2, 4, 4]
    assert games[0]["bot_slots"].tolist() == [0, 1, 0, 1]


def test_forward_from_obs_pre_step_fused_matches_model_path():
    model = BuilderMLP()
    obs = torch.randn(4, OBS_SIZE, dtype=torch.float32)
    h_prev = torch.randn(4, train_bc.H_DIM, dtype=torch.float32)
    obs_pre = model._obs_encoder_pre(obs)
    combined_w, combined_b = train_bc._fused_actor_head_params(model)

    # Reference: use model's own forward (returns 8 values: 5 logits + eps + h_gate + h_cand)
    bt_e, bd_e, cd_e, brt_e, md_e, _eps, _hg, _hc = model._forward_from_obs_pre(obs_pre, h_prev)
    # Fused BC path (returns 6 values: 5 logits + h_next, no eps)
    bt_a, bd_a, cd_a, brt_a, md_a, _h_a = train_bc._forward_from_obs_pre_step_fused(
        model, obs_pre, h_prev, combined_w, combined_b,
    )

    for lhs, rhs in zip([bt_e, bd_e, cd_e, brt_e, md_e], [bt_a, bd_a, cd_a, brt_a, md_a]):
        assert torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_fused_actor_bc_path_ignores_head_logit_scale():
    model = BuilderMLP()
    obs = torch.randn(3, OBS_SIZE, dtype=torch.float32)
    h_prev = torch.randn(3, train_bc.H_DIM, dtype=torch.float32)
    obs_pre = model._obs_encoder_pre(obs)

    combined_w_a, combined_b_a = train_bc._fused_actor_head_params(model)
    actual_a = train_bc._forward_from_obs_pre_step_fused(model, obs_pre, h_prev, combined_w_a, combined_b_a)

    with torch.no_grad():
        model.head_logit_scale.weight.normal_(mean=0.0, std=50.0)
        model.head_logit_scale.bias.normal_(mean=0.0, std=50.0)

    combined_w_b, combined_b_b = train_bc._fused_actor_head_params(model)
    actual_b = train_bc._forward_from_obs_pre_step_fused(model, obs_pre, h_prev, combined_w_b, combined_b_b)

    assert torch.allclose(combined_w_a, combined_w_b)
    assert torch.allclose(combined_b_a, combined_b_b)
    for lhs, rhs in zip(actual_a[:5], actual_b[:5]):
        assert torch.allclose(lhs, rhs, atol=1e-5, rtol=1e-5)
    # h_next should also be identical
    assert torch.allclose(actual_a[5], actual_b[5], atol=1e-5, rtol=1e-5)


def test_actor_bc_optimizer_excludes_head_logit_scale():
    model = BuilderMLP()
    optimizer = train_bc._actor_bc_optimizer(model, lr=3e-4)
    opt_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }

    assert id(model.head_logit_scale.weight) not in opt_param_ids
    assert id(model.head_logit_scale.bias) not in opt_param_ids
    assert id(model.head_build_type.weight) in opt_param_ids
    assert id(model.head_move_dir.bias) in opt_param_ids


def test_hidden_update_from_obs_pre_step_fused_matches_model_path():
    model = BuilderMLP()
    obs = torch.randn(4, OBS_SIZE, dtype=torch.float32)
    h_prev = torch.randn(4, train_bc.H_DIM, dtype=torch.float32)
    obs_pre = model._obs_encoder_pre(obs)
    recurrent_w, recurrent_b = train_bc._fused_recurrent_head_params(model)

    combined_w, combined_b = train_bc._fused_actor_head_params(model)
    expected = train_bc._forward_from_obs_pre_step_fused(model, obs_pre, h_prev, combined_w, combined_b)[-1]
    actual = train_bc._hidden_update_from_obs_pre_step_fused(
        model,
        obs_pre,
        h_prev,
        recurrent_w,
        recurrent_b,
    )

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)


def test_recurrent_candidate_is_tanh_bounded():
    model = BuilderMLP()
    obs = torch.randn(4, OBS_SIZE, dtype=torch.float32)
    h_prev = torch.randn(4, train_bc.H_DIM, dtype=torch.float32)

    obs_pre = model._obs_encoder_pre(obs)
    *_unused, h_gate, h_cand = model._forward_from_obs_pre(obs_pre, h_prev)

    assert h_gate.shape == h_cand.shape
    assert torch.all(h_cand <= 1.0 + 1e-6)
    assert torch.all(h_cand >= -1.0 - 1e-6)


def test_default_actor_uses_32d_policy_latent():
    model = BuilderMLP()
    obs = torch.randn(2, OBS_SIZE, dtype=torch.float32)
    h_prev = torch.randn(2, train_bc.H_DIM, dtype=torch.float32)

    latent = model.actor_latent(obs, h_prev)

    assert latent.shape[-1] == ACTOR_LATENT_DIM
    assert model.head_build_type.in_features == ACTOR_LATENT_DIM
    assert model.head_h_gate.in_features == ACTOR_LATENT_DIM + train_bc.H_DIM


def test_teacher_hidden_widens_actor_latent():
    model = BuilderMLP(hidden=TEACHER_HIDDEN)
    obs = torch.randn(2, OBS_SIZE, dtype=torch.float32)
    h_prev = torch.randn(2, train_bc.H_DIM, dtype=torch.float32)

    latent = model.actor_latent(obs, h_prev)

    assert latent.shape[-1] > ACTOR_LATENT_DIM
    assert model.head_build_type.in_features == latent.shape[-1]
    assert model.head_h_gate.in_features == latent.shape[-1] + train_bc.H_DIM


def test_masked_cross_entropy_ignores_inactive_targets():
    logits = torch.tensor([
        [4.0, 0.0, -1.0, 2.0],
        [0.1, 0.2, 0.3, 0.4],
        [1.0, 3.0, 2.0, 0.0],
    ])
    targets_a = torch.tensor([0, 1, 2])
    targets_b = torch.tensor([0, 3, 2])
    mask = torch.tensor([True, False, True])

    loss_a = train_bc._masked_cross_entropy(logits, targets_a, mask)
    loss_b = train_bc._masked_cross_entropy(logits, targets_b, mask)

    assert torch.allclose(loss_a, loss_b)


def test_bc_step_losses_ignore_noncausal_head_targets():
    obs = torch.zeros(2, OBS_SIZE, dtype=torch.float32)
    obs[:, _OBS_SCALE] = 1.0
    obs[:, _OBS_TI] = 0.0
    obs[:, _OBS_AX] = 0.0
    # bt_act=0 (no build) → bd, cd, brt are causally masked.
    # md is always active (matching RL post-training), so md targets must match.
    actions_a = torch.tensor([
        [0, 4, 0, 0, 4],
        [0, 4, 0, 0, 4],
    ], dtype=torch.long)
    actions_b = torch.tensor([
        [0, 1, 3, 9, 4],  # bd/cd/brt differ, md same
        [0, 8, 2, 7, 4],
    ], dtype=torch.long)

    bt_l = torch.tensor([[2.0] + [-1.0] * 12, [1.5] + [-0.5] * 12], dtype=torch.float32)
    bd_l = torch.randn(2, train_bc.N_DIR)
    cd_l = torch.randn(2, train_bc.N_CONV_DIR)
    brt_l = torch.randn(2, train_bc.N_BRIDGE_TARGETS)
    md_l = torch.randn(2, train_bc.N_DIR)

    losses_a = train_bc._bc_step_losses(obs, actions_a, bt_l, bd_l, cd_l, brt_l, md_l)
    losses_b = train_bc._bc_step_losses(obs, actions_b, bt_l, bd_l, cd_l, brt_l, md_l)

    # Compare NLL sums (index 0-5) — bd/cd/brt should match, bt and md should match
    for i in range(6):
        assert torch.allclose(losses_a[i], losses_b[i])
    assert losses_a[6] == losses_b[6]  # sample count


def test_bc_step_losses_respect_build_type_legality_mask():
    obs = torch.zeros(1, OBS_SIZE, dtype=torch.float32)
    obs[:, _OBS_SCALE] = 1.0
    obs[:, _OBS_TI] = 0.0
    obs[:, _OBS_AX] = 0.0
    actions = torch.tensor([[0, 4, 0, 0, 4]], dtype=torch.long)

    bt_good = torch.tensor([[10.0] + [-10.0] * 12], dtype=torch.float32)
    bt_bad_invalid = torch.tensor([[-10.0, 20.0] + [-10.0] * 11], dtype=torch.float32)
    bd_l = torch.zeros(1, train_bc.N_DIR)
    cd_l = torch.zeros(1, train_bc.N_CONV_DIR)
    brt_l = torch.zeros(1, train_bc.N_BRIDGE_TARGETS)
    md_l = torch.zeros(1, train_bc.N_DIR)

    good = train_bc._bc_step_losses(obs, actions, bt_good, bd_l, cd_l, brt_l, md_l)
    bad = train_bc._bc_step_losses(obs, actions, bt_bad_invalid, bd_l, cd_l, brt_l, md_l)

    # bt_nll (index 1) should be small for both: the illegal high-logit action
    # (index 1) is masked out by legality, so bt_act=0 gets high probability.
    assert good[1] < 1e-4  # bt_nll with correct high logit for action 0
    assert bad[1] < 1e-4   # bt_nll: illegal action 1 masked → action 0 still wins


def test_materialize_training_batch_interleaves_groups_by_tbptt_step():
    group_a = {
        "h0": torch.zeros(1, train_bc.H_DIM),
        "obs": torch.tensor([[10.0] * OBS_SIZE, [11.0] * OBS_SIZE], dtype=torch.float16),
        "act": torch.tensor([[1, 0, 0, 0, 0], [2, 0, 0, 0, 0]], dtype=torch.int16),
        "train_slots": torch.tensor([0, 0], dtype=torch.long),
        "step_sizes": [1, 1],
        "step_offsets": [0, 1],
    }
    group_b = {
        "h0": torch.zeros(2, train_bc.H_DIM),
        "obs": torch.tensor([[20.0] * OBS_SIZE, [21.0] * OBS_SIZE, [22.0] * OBS_SIZE], dtype=torch.float16),
        "act": torch.tensor([[3, 0, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]], dtype=torch.int16),
        "train_slots": torch.tensor([0, 1, 1], dtype=torch.long),
        "step_sizes": [2, 1],
        "step_offsets": [0, 2],
    }

    (
        _h_live,
        batch_obs,
        batch_act,
        step_offsets,
        step_sizes,
        batch_train_slots,
        _batch_bt_mask,
        _batch_bd_mask,
        _batch_harv_bd_mask,
    ) = train_bc._materialize_training_batch(
        [group_a, group_b],
        games=[],
    )

    assert step_offsets == [0, 3]
    assert step_sizes == [3, 2]
    assert batch_obs[:, 0].tolist() == [10.0, 20.0, 21.0, 11.0, 22.0]
    assert batch_act[:, 0].tolist() == [1, 3, 4, 2, 5]
    assert batch_train_slots.tolist() == [0, 1, 2, 0, 2]


def test_prepare_tbptt_groups_defers_reusable_group_tensors_until_prefetch(monkeypatch):
    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    model = BuilderMLP()
    obs = torch.randn(4, OBS_SIZE, dtype=torch.float32)
    actions = torch.tensor([
        [1, 0, 0, 0, 0],
        [2, 1, 0, 0, 1],
        [3, 2, 0, 0, 2],
        [4, 3, 0, 0, 3],
    ], dtype=torch.int16)
    bt_mask, bd_mask = train_bc.compute_action_masks(obs)
    harv_bd_mask = train_bc.compute_harvester_build_dir_mask(obs, bd_mask)
    bt_mask = bt_mask.clone()
    bt_mask[:, train_bc.HARVESTER_BUILD_TYPE] &= harv_bd_mask.any(dim=1)
    game = {
        "obs": obs.to(torch.float16),
        "act": actions,
        "bot_slots": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "global_slots": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "bt_mask": bt_mask,
        "bd_mask": bd_mask,
        "harv_bd_mask": harv_bd_mask,
        "round_offsets": [0, 2, 4],
        "n_rounds": 2,
        "n_slots": 2,
        "slot_base": 0,
    }

    groups = train_bc._prepare_tbptt_groups(
        model,
        [game],
        total_slots=2,
        max_rounds=2,
        tbptt_len=2,
        cuda_cfg=train_bc.resolve_cuda_config(torch.device("cpu"), spatial_inputs=False, enable_compile=False),
    )

    assert len(groups) == 1
    group = groups[0]
    assert "obs" not in group
    assert "act" not in group
    assert "prep_slots" not in group
    assert "train_slots" not in group
    assert "bt_mask" not in group
    assert "bd_mask" not in group
    assert "harv_bd_mask" not in group

    (
        group_obs,
        group_act,
        group_prep_slots,
        group_train_slots,
        group_bt_mask,
        group_bd_mask,
        group_harv_bd_mask,
    ) = train_bc._materialize_group_training_tensors(group, [game])

    assert torch.equal(group_obs, game["obs"])
    assert torch.equal(group_act, game["act"])
    assert torch.equal(group_prep_slots, game["global_slots"])
    assert torch.equal(group_train_slots, game["bot_slots"])
    assert torch.equal(group_bt_mask, game["bt_mask"])
    assert torch.equal(group_bd_mask, game["bd_mask"])
    assert torch.equal(group_harv_bd_mask, game["harv_bd_mask"])


def test_release_group_training_tensors_clears_cached_staging():
    h0 = torch.zeros(2, train_bc.H_DIM)
    group = {
        "obs": torch.randn(2, OBS_SIZE, dtype=torch.float16),
        "act": torch.zeros(2, 5, dtype=torch.int16),
        "prep_slots": torch.tensor([0, 1], dtype=torch.long),
        "train_slots": torch.tensor([0, 1], dtype=torch.long),
        "bt_mask": torch.zeros(2, train_bc.N_BUILD, dtype=torch.bool),
        "bd_mask": torch.zeros(2, train_bc.N_DIR, dtype=torch.bool),
        "harv_bd_mask": torch.zeros(2, train_bc.N_DIR, dtype=torch.bool),
        "batch_h_base": 7,
        "h0": h0,
    }

    train_bc._release_group_training_tensors([group])

    assert "obs" not in group
    assert "act" not in group
    assert "prep_slots" not in group
    assert "train_slots" not in group
    assert "bt_mask" not in group
    assert "bd_mask" not in group
    assert "harv_bd_mask" not in group
    assert "batch_h_base" not in group
    assert group["h0"] is h0


def test_train_actor_bc_uses_one_optimizer_step_per_tbptt_chunk(monkeypatch):
    model = BuilderMLP()
    dataset = _DummyDataset()

    obs = torch.randn(8, OBS_SIZE, dtype=torch.float32)
    actions = torch.randint(0, 1, (8, 5), dtype=torch.long)
    actions[:, 0] = torch.randint(0, train_bc.N_BUILD, (8,))
    actions[:, 1] = torch.randint(0, train_bc.N_DIR, (8,))
    actions[:, 2] = torch.randint(0, train_bc.N_CONV_DIR, (8,))
    actions[:, 3] = torch.randint(0, train_bc.N_BRIDGE_TARGETS, (8,))
    actions[:, 4] = torch.randint(0, train_bc.N_DIR, (8,))
    game = {
        "obs": obs,
        "act": actions.to(torch.int16),
        "bot_slots": torch.tensor([0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long),
        "global_slots": torch.tensor([0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long),
        "round_offsets": [0, 2, 4, 6, 7, 8],
        "n_rounds": 5,
        "n_slots": 2,
        "slot_base": 0,
    }

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        train_bc,
        "_load_game_batch",
        lambda _dataset, _game_list, **_kwargs: ([game], 2, 5, 1),
    )

    class SpyAdamW(torch.optim.AdamW):
        last_instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0
            SpyAdamW.last_instance = self

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    monkeypatch.setattr(train_bc.torch.optim, "AdamW", SpyAdamW)

    stats = train_bc.train_actor_bc(
        model,
        dataset,
        epochs=1,
        lr=1e-3,
        games_per_batch=1,
        tbptt_len=2,
        chunk_groups_per_batch=1,
        enable_compile=False,
    )

    assert SpyAdamW.last_instance is not None
    assert SpyAdamW.last_instance.step_calls == 3
    assert stats["optimizer_steps"] == 3
    assert stats["total_samples"] == 8


def test_train_actor_bc_writes_step_log_per_optimizer_step(monkeypatch, tmp_path):
    model = BuilderMLP()
    dataset = _DummyDataset()

    obs = torch.randn(8, OBS_SIZE, dtype=torch.float32)
    actions = torch.zeros(8, 5, dtype=torch.long)
    actions[:, 0] = torch.tensor([0, 1, 2, 0, 7, 12, 0, 3])
    actions[:, 1] = 4
    actions[:, 2] = torch.tensor([0, 1, 2, 0, 3, 0, 0, 0])
    actions[:, 3] = torch.tensor([0, 0, 0, 0, 0, 5, 0, 0])
    actions[:, 4] = torch.tensor([4, 4, 1, 4, 4, 2, 4, 4])
    game = {
        "obs": obs,
        "act": actions.to(torch.int16),
        "bot_slots": torch.tensor([0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long),
        "global_slots": torch.tensor([0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long),
        "round_offsets": [0, 2, 4, 6, 7, 8],
        "n_rounds": 5,
        "n_slots": 2,
        "slot_base": 0,
    }

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        train_bc,
        "_load_game_batch",
        lambda _dataset, _game_list, **_kwargs: ([game], 2, 5, 1),
    )

    step_log_path = tmp_path / "train_step_log.jsonl"
    stats = train_bc.train_actor_bc(
        model,
        dataset,
        epochs=1,
        lr=1e-3,
        games_per_batch=1,
        tbptt_len=2,
        chunk_groups_per_batch=1,
        step_log_path=step_log_path,
        step_log_every=1,
        enable_compile=False,
    )

    rows = [json.loads(line) for line in step_log_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert [row["optimizer_step"] for row in rows] == [1, 2, 3]
    assert [row["episode"] for row in rows] == [1, 2, 3]
    assert all("loss_cd" in row and "loss_brt" in row for row in rows)
    assert stats["optimizer_steps"] == 3


def test_train_actor_bc_releases_group_staging_after_consumption(monkeypatch):
    model = BuilderMLP()
    dataset = _DummyDataset()

    obs = torch.randn(8, OBS_SIZE, dtype=torch.float32)
    actions = torch.zeros(8, 5, dtype=torch.long)
    game = {
        "obs": obs,
        "act": actions.to(torch.int16),
        "bot_slots": torch.tensor([0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long),
        "global_slots": torch.tensor([0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.long),
        "round_offsets": [0, 2, 4, 6, 7, 8],
        "n_rounds": 5,
        "n_slots": 2,
        "slot_base": 0,
    }

    original_prepare = train_bc._prepare_tbptt_groups
    captured_groups = []

    def spy_prepare(*args, **kwargs):
        groups = original_prepare(*args, **kwargs)
        captured_groups.extend(groups)
        return groups

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        train_bc,
        "_load_game_batch",
        lambda _dataset, _game_list, **_kwargs: ([game], 2, 5, 1),
    )
    monkeypatch.setattr(train_bc, "_prepare_tbptt_groups", spy_prepare)

    stats = train_bc.train_actor_bc(
        model,
        dataset,
        epochs=1,
        lr=1e-3,
        games_per_batch=1,
        tbptt_len=2,
        chunk_groups_per_batch=1,
        enable_compile=False,
    )

    assert stats["optimizer_steps"] == 3
    assert len(captured_groups) == 3
    for group in captured_groups:
        assert "obs" not in group
        assert "act" not in group
        assert "prep_slots" not in group
        assert "train_slots" not in group
        assert "bt_mask" not in group
        assert "bd_mask" not in group
        assert "harv_bd_mask" not in group


def test_train_actor_bc_disables_step_logging_by_default(monkeypatch, tmp_path):
    model = BuilderMLP()
    dataset = _DummyDataset()

    obs = torch.randn(4, OBS_SIZE, dtype=torch.float32)
    actions = torch.zeros(4, 5, dtype=torch.long)
    game = {
        "obs": obs,
        "act": actions.to(torch.int16),
        "bot_slots": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "global_slots": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        "round_offsets": [0, 2, 4],
        "n_rounds": 2,
        "n_slots": 2,
        "slot_base": 0,
    }

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        train_bc,
        "_load_game_batch",
        lambda _dataset, _game_list, **_kwargs: ([game], 2, 2, 1),
    )

    step_log_path = tmp_path / "train_step_log.jsonl"
    train_bc.train_actor_bc(
        model,
        dataset,
        epochs=1,
        lr=1e-3,
        games_per_batch=1,
        tbptt_len=2,
        chunk_groups_per_batch=1,
        step_log_path=step_log_path,
        enable_compile=False,
    )

    assert not step_log_path.exists()


def test_train_actor_bc_randomizes_tbptt_group_order(monkeypatch):
    model = BuilderMLP()
    dataset = _DummyDataset()

    def _group_game(obs_tag: float):
        obs = torch.zeros(1, OBS_SIZE, dtype=torch.float32)
        obs[0, 0] = obs_tag
        return {
            "obs": obs,
            "act": torch.zeros(1, 5, dtype=torch.int16),
            "bot_slots": torch.tensor([0], dtype=torch.long),
            "global_slots": torch.tensor([0], dtype=torch.long),
            "round_offsets": [0, 1],
            "n_rounds": 1,
            "n_slots": 1,
            "slot_base": 0,
        }

    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        train_bc,
        "_load_game_batch",
        lambda _dataset, _game_list, **_kwargs: ([_group_game(0.0)], 1, 1, 1),
    )
    monkeypatch.setattr(
        train_bc,
        "_prepare_tbptt_groups",
        lambda *_args, **_kwargs: [
            {
                "chunk_start": 0,
                "chunk_lens": [1],
                "step_sizes": [1],
                "step_offsets": [0],
                "h0": torch.zeros(1, train_bc.H_DIM),
                "obs": _group_game(10.0)["obs"],
                "act": _group_game(10.0)["act"],
                "prep_slots": torch.tensor([0], dtype=torch.long),
                "train_slots": torch.tensor([0], dtype=torch.long),
            },
            {
                "chunk_start": 0,
                "chunk_lens": [1],
                "step_sizes": [1],
                "step_offsets": [0],
                "h0": torch.zeros(1, train_bc.H_DIM),
                "obs": _group_game(20.0)["obs"],
                "act": _group_game(20.0)["act"],
                "prep_slots": torch.tensor([0], dtype=torch.long),
                "train_slots": torch.tensor([0], dtype=torch.long),
            },
            {
                "chunk_start": 0,
                "chunk_lens": [1],
                "step_sizes": [1],
                "step_offsets": [0],
                "h0": torch.zeros(1, train_bc.H_DIM),
                "obs": _group_game(30.0)["obs"],
                "act": _group_game(30.0)["act"],
                "prep_slots": torch.tensor([0], dtype=torch.long),
                "train_slots": torch.tensor([0], dtype=torch.long),
            },
        ],
    )
    monkeypatch.setattr(train_bc.random, "shuffle", lambda xs: xs.reverse())

    seen = []
    orig_obs_encoder_pre = BuilderMLP._obs_encoder_pre

    def spy_obs_encoder_pre(self, x):
        seen.append(float(x[0, 0].item()))
        return orig_obs_encoder_pre(self, x)

    monkeypatch.setattr(BuilderMLP, "_obs_encoder_pre", spy_obs_encoder_pre)

    train_bc.train_actor_bc(
        model,
        dataset,
        epochs=1,
        lr=1e-3,
        games_per_batch=1,
        tbptt_len=1,
        chunk_groups_per_batch=1,
        enable_compile=False,
    )

    # First three chunk encodes should respect the reversed group order.
    assert seen[-3:] == [30.0, 20.0, 10.0]


def test_tbptt_gradient_flows_across_time_steps(monkeypatch):
    """Verify that h_gate/h_cand heads receive gradient from future steps.

    If BPTT is working, the loss at step 1 should produce gradient on the
    hidden-state heads (h_gate, h_cand) because h_new at step 0 feeds into
    step 1's forward pass.  If the gradient is severed (e.g. by in-place
    index_copy_ into a detached buffer), these heads get zero grad from
    step 1's loss alone.
    """
    model = BuilderMLP()
    model.train()
    monkeypatch.setattr(train_bc, "DEVICE", torch.device("cpu"))

    # Two bots, two rounds — minimal BPTT scenario.
    obs = torch.randn(4, OBS_SIZE)
    obs[:, _OBS_SCALE] = 1.0
    actions = torch.zeros(4, 5, dtype=torch.long)
    bot_ids = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    turns = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    # Sort round-major while preserving bot-major order within each round.
    order = turns.argsort(stable=True)
    obs = obs[order]
    actions = actions[order]
    bot_ids = bot_ids[order]
    turns = turns[order]

    n_slots = 2
    bot_slots = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    game = {
        "obs": obs.half().contiguous(),
        "act": actions.to(torch.int16).contiguous(),
        "bot_slots": bot_slots,
        "global_slots": bot_slots.clone(),
        "round_offsets": [0, 2, 4],
        "n_rounds": 2,
        "n_slots": n_slots,
        "slot_base": 0,
    }

    # Run the inner training loop manually with tbptt_len=2 (one chunk).
    h_live = torch.zeros(n_slots, train_bc.H_DIM)
    combined_w, combined_b = train_bc._fused_actor_head_params(model)

    obs_all = obs.float()
    obs_pre_all = model._obs_encoder_pre(obs_all)

    # Step 0: rounds 0
    slot_ids_0 = bot_slots[:2]
    b_h_0 = h_live[slot_ids_0]
    bt_l, bd_l, cd_l, brt_l, md_l, h_new_0 = train_bc._forward_from_obs_pre_step_fused(
        model, obs_pre_all[:2], b_h_0, combined_w, combined_b,
    )
    loss_0, *_ = train_bc._bc_step_losses(obs_all[:2], actions[:2], bt_l, bd_l, cd_l, brt_l, md_l)

    # Differentiable scatter (same as train loop).
    h_live = train_bc._DiffScatter.apply(h_live, slot_ids_0, h_new_0.float())

    # Step 1: round 1
    slot_ids_1 = bot_slots[2:]
    b_h_1 = h_live[slot_ids_1]
    bt_l, bd_l, cd_l, brt_l, md_l, h_new_1 = train_bc._forward_from_obs_pre_step_fused(
        model, obs_pre_all[2:], b_h_1, combined_w, combined_b,
    )
    loss_1, *_ = train_bc._bc_step_losses(obs_all[2:], actions[2:], bt_l, bd_l, cd_l, brt_l, md_l)

    # Backward through step 1 only — if BPTT works, h_gate/h_cand get grad.
    model.zero_grad()
    loss_1.backward()

    h_gate_grad = model.head_h_gate.weight.grad
    h_cand_grad = model.head_h_cand.weight.grad
    assert h_gate_grad is not None, "h_gate should receive gradient from step 1 via BPTT"
    assert h_cand_grad is not None, "h_cand should receive gradient from step 1 via BPTT"
    assert h_gate_grad.abs().sum() > 0, "h_gate gradient should be nonzero (BPTT signal)"
    assert h_cand_grad.abs().sum() > 0, "h_cand gradient should be nonzero (BPTT signal)"
