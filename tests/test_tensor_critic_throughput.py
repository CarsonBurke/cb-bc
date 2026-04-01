"""Regression coverage for PPO critic throughput helpers."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))

import model as model_mod


def test_critic_chunk_plan_handles_sparse_chunk_ids():
    chunk_ids = torch.tensor([0, 0, 0, 2, 2, 5, 5, 5], dtype=torch.long)

    actual_ids, offsets = model_mod._critic_chunk_plan(chunk_ids)

    assert actual_ids.tolist() == [0, 2, 5]
    assert offsets.tolist() == [0, 3, 5, 8]


def test_critic_round_grouped_perm_keeps_rounds_contiguous(monkeypatch):
    monkeypatch.setattr(
        model_mod.torch,
        "randperm",
        lambda n, device=None: torch.tensor([2, 0, 1], device=device),
    )

    step_round_idx = torch.tensor([4, 4, 7, 7, 9, 9], dtype=torch.long)
    perm = model_mod._critic_round_grouped_perm(step_round_idx)

    assert perm.tolist() == [2, 3, 4, 5, 0, 1]


def test_compute_all_values_batches_only_sparse_unique_rounds():
    device = model_mod.DEVICE

    class FakeCritic:
        def __init__(self):
            self.round_idx_calls: list[torch.Tensor] = []
            self.map_batch_sizes: list[int] = []

        def encode_map(self, map_tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            del valid_mask
            self.map_batch_sizes.append(int(map_tensor.shape[0]))
            return torch.zeros(map_tensor.shape[0], 1, 1, 1, device=map_tensor.device)

        def unit_values(
            self,
            feat: torch.Tensor,
            round_scalars: torch.Tensor,
            round_idx: torch.Tensor,
            unit_xy: torch.Tensor,
            unit_ctx: torch.Tensor,
            actor_latent: torch.Tensor,
        ) -> torch.Tensor:
            del feat, round_scalars, unit_xy, unit_ctx, actor_latent
            self.round_idx_calls.append(round_idx.detach().cpu())
            return torch.zeros(round_idx.shape[0], device=round_idx.device)

    fake_critic = FakeCritic()
    data = {
        "gpu_obs": torch.zeros(9001, model_mod.GLOBAL_MAP_CHANNELS, 1, 1, device=device),
        "gpu_mask": torch.ones(9001, 1, 1, 1, device=device),
        "gpu_scalars": torch.zeros(9001, model_mod.GLOBAL_SCALAR_DIM, device=device),
        "step_round_idx": torch.tensor([0, 9000], dtype=torch.long, device=device),
        "step_unit_xy": torch.zeros(2, 2, dtype=torch.long, device=device),
        "step_unit_ctx": torch.zeros(2, model_mod.UNIT_CONTEXT_DIM, device=device),
        "step_actor_latent": torch.zeros(2, model_mod.ACTOR_LATENT_DIM, device=device),
        "n_steps": 2,
    }

    values = model_mod._compute_all_values(fake_critic, data, cnn_chunk_size=4096, unit_chunk_size=16)

    assert values.shape == (2,)
    assert fake_critic.map_batch_sizes == [2]
    assert len(fake_critic.round_idx_calls) == 1
    assert fake_critic.round_idx_calls[0].tolist() == [0, 1]


def test_critic_round_batch_plan_respects_step_target_and_round_cap(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "_critic_round_grouped_perm",
        lambda rounds: torch.arange(rounds.shape[0], device=rounds.device),
    )

    step_round_idx = torch.tensor([1, 1, 2, 2, 2, 5, 5, 9], dtype=torch.long)
    perm, unique_rounds, round_offsets, batch_ranges = model_mod._critic_round_batch_plan(
        step_round_idx,
        target_steps=3,
        max_rounds=2,
    )

    assert perm.tolist() == list(range(8))
    assert unique_rounds.tolist() == [1, 2, 5, 9]
    assert round_offsets.tolist() == [0, 2, 5, 7, 8]
    assert batch_ranges == [(0, 1), (1, 2), (2, 4)]


def test_freeze_critic_skips_critic_singleton_and_compile(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("critic singleton should not be constructed when frozen")

    monkeypatch.setattr(model_mod, "get_centralized_critic", fail_if_called)
    monkeypatch.setattr(model_mod, "get_centralized_critic_step", fail_if_called)
    monkeypatch.setattr(model_mod, "get_centralized_critic_loss_step", fail_if_called)

    obs = torch.zeros(2, model_mod.OBS_SIZE)
    h = torch.zeros(2, model_mod.H_DIM)
    bt = torch.zeros(2, dtype=torch.long)
    bd = torch.zeros(2, dtype=torch.long)
    cd = torch.zeros(2, dtype=torch.long)
    brt = torch.zeros(2, dtype=torch.long)
    md = torch.zeros(2, dtype=torch.long)
    old_log_probs = torch.zeros(2)
    advantages = torch.zeros(2)
    returns = torch.zeros(2)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=0.1)

    stats = model_mod.ppo_update(
        model=torch.nn.Identity(),
        optimizer=optimizer,
        obs=obs,
        h=h,
        bt_act=bt,
        bd_act=bd,
        cd_act=cd,
        brt_act=brt,
        md_act=md,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        centralized_data={},
        ppo_epochs=0,
        freeze_critic=True,
    )

    assert stats["n_updates"] == 0
    assert stats["vf_loss"] == 0.0
