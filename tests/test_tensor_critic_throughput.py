"""Regression coverage for PPO critic throughput helpers."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import warnings

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
TENSOR_DIR = PROJECT_ROOT / "bots" / "tensor"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))
if str(TENSOR_DIR) not in sys.path:
    sys.path.insert(0, str(TENSOR_DIR))

import model as model_mod
import replay_buffers as replay_buffers_mod
from trajectory_schema import make_trajectory_step


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
            actions: torch.Tensor,
        ) -> torch.Tensor:
            del feat, round_scalars, unit_xy, unit_ctx, actor_latent, actions
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
        "step_actions": torch.zeros(2, 5, dtype=torch.long, device=device),
        "n_steps": 2,
    }

    values = model_mod._compute_all_values(fake_critic, data, cnn_chunk_size=4096)

    assert values.shape == (2,)
    assert fake_critic.map_batch_sizes == [2]
    assert len(fake_critic.round_idx_calls) == 1
    assert fake_critic.round_idx_calls[0].tolist() == [0, 1]


def test_compute_all_values_reports_nonfinite_batch_value_context(monkeypatch):
    device = model_mod.DEVICE

    class FakeFeat:
        def __init__(self, device: torch.device):
            self.patches = torch.zeros(1, 1, 1, 1, device=device)
            self.global_cls = torch.zeros(1, 1, device=device)

    class FakeCritic:
        def encode_map(self, map_tensor: torch.Tensor, valid_mask: torch.Tensor) -> FakeFeat:
            del map_tensor, valid_mask
            return FakeFeat(device)

        def unit_values(
            self,
            feat: FakeFeat,
            round_scalars: torch.Tensor,
            round_idx: torch.Tensor,
            unit_xy: torch.Tensor,
            unit_ctx: torch.Tensor,
            actor_latent: torch.Tensor,
            actions: torch.Tensor,
        ) -> torch.Tensor:
            del feat, round_scalars, round_idx, unit_xy, unit_ctx, actor_latent, actions
            return torch.tensor([float("nan")], device=device)

    monkeypatch.setattr(
        model_mod,
        "_load_critic_round_batch",
        lambda data, c_rounds: (
            torch.zeros(c_rounds.shape[0], model_mod.GLOBAL_MAP_CHANNELS, 1, 1, device=device),
            torch.ones(c_rounds.shape[0], 1, 1, 1, device=device),
            torch.zeros(c_rounds.shape[0], model_mod.GLOBAL_SCALAR_DIM, device=device),
        ),
    )

    data = {
        "step_round_idx": torch.tensor([1], dtype=torch.long, device=device),
        "step_unit_xy": torch.tensor([[3, 4]], dtype=torch.long, device=device),
        "step_unit_ctx": torch.zeros(1, model_mod.UNIT_CONTEXT_DIM, device=device),
        "step_actor_latent": torch.zeros(1, model_mod.ACTOR_LATENT_DIM, device=device),
        "step_actions": torch.zeros(1, 5, dtype=torch.long, device=device),
        "n_steps": 1,
    }

    with pytest.raises(RuntimeError, match=r"step_index=0 .*round_id=1 .*local_round=0 .*unit_xy=\[3, 4\]"):
        model_mod._compute_all_values(FakeCritic(), data, cnn_chunk_size=4096)


def test_critic_global_layout_keeps_padded_unit_queries_nonempty():
    device = torch.device("cpu")
    layer = model_mod.CriticGlobalSA().to(device)
    unit_cls = torch.zeros(3, layer.dim, device=device)
    unit_xy = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long, device=device)
    unit_round_idx = torch.tensor([0, 0, 1], dtype=torch.long, device=device)

    layout = layer.prepare_layout(
        n_patches=4,
        pH=2,
        pW=2,
        unit_cls=unit_cls,
        unit_xy=unit_xy,
        unit_round_idx=unit_round_idx,
        n_rounds=2,
    )

    assert layout.unit_attn_bias is not None
    invalid_queries = ~layout.valid_unit_mask
    assert invalid_queries.any()
    # Padded unit key positions should have -inf bias
    unit_bias_slice = layout.unit_attn_bias[:, 0, 0, layout.unit_start:]
    assert (unit_bias_slice[invalid_queries] == float('-inf')).all()


def test_critic_windowed_sa_grouped_flash_layout_matches_flattened_windows():
    torch.manual_seed(0)
    device = torch.device("cpu")
    layer = model_mod.CriticWindowedSA().to(device).eval()
    pH, pW = 7, 9
    n_rounds = 3
    dim = layer.attn.n_heads * layer.attn.head_dim
    patches = torch.randn(n_rounds, pH * pW, dim, device=device)
    anchor_patches = torch.randn(n_rounds, pH * pW, dim, device=device)

    with torch.no_grad():
        actual = layer(patches, pH, pW, anchor_patches)

        x = patches.reshape(n_rounds, pH, pW, dim)
        x0 = anchor_patches.reshape(n_rounds, pH, pW, dim)
        grouped_windows, pad_hw = model_mod.critic_window_partition(x, layer.window_size)
        anchor_windows, _ = model_mod.critic_window_partition(x0, layer.window_size)
        flat_windows = grouped_windows.reshape(-1, grouped_windows.shape[-2], grouped_windows.shape[-1])
        flat_anchor_windows = anchor_windows.reshape_as(flat_windows)
        mixed_windows = layer.attn.mix_residual_input(flat_windows, flat_anchor_windows)
        q, k, v = layer.attn.project_qkv(
            mixed_windows,
            rope_cos=layer._local_rope_cos,
            rope_sin=layer._local_rope_sin,
        )
        flat_out = layer.attn.attend(q, k, v, self_value=v)
        expected_windows = layer.attn.apply_residual_mlp(mixed_windows, flat_out)
        expected = model_mod.critic_window_unpartition(
            expected_windows.reshape_as(grouped_windows),
            layer.window_size,
            pad_hw,
            (pH, pW),
        ).reshape(n_rounds, pH * pW, dim)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_critic_global_layout_initializes_padded_unit_rope_to_identity():
    device = torch.device("cpu")
    layer = model_mod.CriticGlobalSA().to(device)
    unit_cls = torch.zeros(3, layer.dim, device=device)
    unit_xy = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long, device=device)
    unit_round_idx = torch.tensor([0, 0, 1], dtype=torch.long, device=device)

    layout = layer.prepare_layout(
        n_patches=4,
        pH=2,
        pW=2,
        unit_cls=unit_cls,
        unit_xy=unit_xy,
        unit_round_idx=unit_round_idx,
        n_rounds=2,
    )

    invalid_queries = ~layout.valid_unit_mask
    # Unit rope lives in the tail of full_rope_cos/sin; padded unit positions
    # must stay at the RoPE identity (cos=1, sin=0) so invariants downstream
    # — and the -inf attention bias that masks them — line up correctly.
    unit_cos = layout.full_rope_cos[:, layout.unit_start:]
    unit_sin = layout.full_rope_sin[:, layout.unit_start:]
    assert torch.equal(unit_cos[invalid_queries], torch.ones_like(unit_cos[invalid_queries]))
    assert torch.equal(unit_sin[invalid_queries], torch.zeros_like(unit_sin[invalid_queries]))


def test_critic_global_sa_uses_single_shared_prefix_attend_for_patch_and_global_tokens(monkeypatch):
    device = torch.device("cpu")
    layer = model_mod.CriticGlobalSA(dim=8, n_heads=2, n_kv_heads=1, n_global_cls=2, max_grid_size=2).to(device).eval()

    with torch.no_grad():
        layer.attn.out_proj.weight.copy_(torch.eye(8))
        for norm in (
            layer.attn.attn_input_norm,
            layer.attn.attn_output_norm,
            layer.attn.mlp_input_norm,
            layer.attn.mlp_output_norm,
            layer.attn.q_norm,
            layer.attn.k_norm,
        ):
            norm.weight.fill_(1.0)
        layer.attn.mlp_fc1.weight.zero_()
        layer.attn.mlp_fc2.weight.zero_()

    attend_calls: list[tuple[int, int]] = []

    def fake_attend(q, k, v, attn_mask=None, self_value=None):
        del v, attn_mask, self_value
        B, _, N_q, _ = q.shape
        attend_calls.append((N_q, k.shape[-2]))
        token_vectors = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            device=q.device,
            dtype=q.dtype,
        )
        return token_vectors[:N_q].unsqueeze(0).expand(B, -1, -1)

    monkeypatch.setattr(layer.attn, "attend", fake_attend)

    patches = torch.zeros(1, 4, layer.dim, device=device)
    anchor_patches = torch.zeros_like(patches)
    global_cls = torch.zeros(1, 2, layer.dim, device=device)
    anchor_global_cls = torch.zeros_like(global_cls)
    scalar_tok = torch.zeros(1, 1, layer.dim, device=device)
    anchor_scalar_tok = torch.zeros_like(scalar_tok)
    unit_cls = torch.zeros(0, layer.dim, device=device)
    anchor_unit_cls = torch.zeros_like(unit_cls)
    layout = layer.prepare_layout(
        n_patches=4,
        pH=2,
        pW=2,
        unit_cls=unit_cls,
        unit_xy=torch.zeros(0, 2, dtype=torch.long, device=device),
        unit_round_idx=torch.zeros(0, dtype=torch.long, device=device),
        n_rounds=1,
    )

    out_patches, out_global_cls, out_unit_cls = layer(
        patches,
        anchor_patches,
        global_cls,
        anchor_global_cls,
        scalar_tok,
        anchor_scalar_tok,
        unit_cls,
        anchor_unit_cls,
        layout,
    )

    assert attend_calls == [(layout.unit_start, layout.unit_start)]
    expected_patch = layer.attn.attn_output_norm(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            device=device,
        ).unsqueeze(0),
    )
    expected_global = layer.attn.attn_output_norm(
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            device=device,
        ).unsqueeze(0),
    )
    torch.testing.assert_close(out_patches, expected_patch, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_global_cls, expected_global, atol=1e-6, rtol=1e-6)
    assert out_unit_cls.shape == (0, layer.dim)


def test_critic_global_layout_keeps_shared_rope_prefix_round_independent():
    device = torch.device("cpu")
    layer = model_mod.CriticGlobalSA().to(device)
    unit_cls = torch.zeros(3, layer.dim, device=device)
    unit_xy = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long, device=device)
    unit_round_idx = torch.tensor([0, 0, 1], dtype=torch.long, device=device)

    layout = layer.prepare_layout(
        n_patches=4,
        pH=2,
        pW=2,
        unit_cls=unit_cls,
        unit_xy=unit_xy,
        unit_round_idx=unit_round_idx,
        n_rounds=2,
    )

    # Shared prefix RoPE must be identical across rounds so the merged
    # apply_rotary_2d call doesn't introduce round-dependent drift on
    # patch/scalar/CLS positions.
    prefix_cos = layout.full_rope_cos[:, :layout.unit_start]
    prefix_sin = layout.full_rope_sin[:, :layout.unit_start]
    assert prefix_cos.shape[0] == layout.n_rounds
    assert prefix_cos.shape[1] == layout.unit_start
    for r in range(1, layout.n_rounds):
        assert torch.equal(prefix_cos[r], prefix_cos[0])
        assert torch.equal(prefix_sin[r], prefix_sin[0])


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


def test_build_critic_epoch_plan_packs_round_major_slices(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "_critic_round_grouped_perm",
        lambda rounds: torch.arange(rounds.shape[0], device=rounds.device),
    )

    step_round_idx = torch.tensor([1, 1, 2, 2, 3], dtype=torch.long)
    step_unit_xy = torch.tensor([[10, 0], [11, 0], [20, 0], [21, 0], [30, 0]], dtype=torch.long)
    step_unit_float = torch.tensor([[1.0], [1.1], [2.0], [2.1], [3.0]], dtype=torch.float32)
    step_rv = torch.tensor([[10.0, 100.0], [11.0, 101.0], [20.0, 200.0], [21.0, 201.0], [30.0, 300.0]])
    step_actions = torch.zeros(5, 5, dtype=torch.long)

    plan = model_mod._build_critic_epoch_plan(
        step_round_idx,
        step_unit_xy,
        step_unit_float,
        step_rv,
        step_actions,
        target_steps=3,
        max_rounds=2,
    )

    assert plan.round_ids.tolist() == [1, 2, 3]
    assert plan.round_offsets.tolist() == [0, 2, 4, 5]
    assert plan.batch_ranges == [(0, 1), (1, 3)]
    assert plan.packed_xy.tolist() == [[10, 0], [11, 0], [20, 0], [21, 0], [30, 0]]
    assert torch.allclose(
        plan.packed_unit_float[:, 0],
        torch.tensor([1.0, 1.1, 2.0, 2.1, 3.0], dtype=plan.packed_unit_float.dtype),
    )
    assert plan.local_round_idx_by_batch[0].tolist() == [0, 0]
    assert plan.local_round_idx_by_batch[1].tolist() == [0, 0, 1]


def test_build_critic_epoch_plan_spreads_positive_rounds_without_splitting_rounds(monkeypatch):
    monkeypatch.setattr(
        model_mod.torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )

    step_round_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    step_unit_xy = torch.tensor(
        [[10, 0], [11, 0], [20, 0], [21, 0], [30, 0], [31, 0], [40, 0], [41, 0]],
        dtype=torch.long,
    )
    step_unit_float = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    step_rv = torch.stack([torch.arange(8, dtype=torch.float32), torch.arange(100, 108, dtype=torch.float32)], dim=1)
    step_actions = torch.zeros(8, 5, dtype=torch.long)
    positive_mask = torch.tensor([False, False, False, False, True, True, True, True], dtype=torch.bool)

    plan = model_mod._build_critic_epoch_plan(
        step_round_idx,
        step_unit_xy,
        step_unit_float,
        step_rv,
        step_actions,
        target_steps=4,
        positive_mask=positive_mask,
    )

    assert plan.round_ids.tolist() == [2, 0, 3, 1]
    assert plan.batch_ranges == [(0, 2), (2, 4)]
    assert plan.round_offsets.tolist() == [0, 2, 4, 6, 8]
    assert torch.equal(plan.packed_xy[:, 0], torch.tensor([30, 31, 10, 11, 40, 41, 20, 21]))
    assert plan.local_round_idx_by_batch[0].tolist() == [0, 0, 1, 1]
    assert plan.local_round_idx_by_batch[1].tolist() == [0, 0, 1, 1]


def test_build_critic_epoch_plan_positive_path_respects_round_cap(monkeypatch):
    monkeypatch.setattr(
        model_mod.torch,
        "randperm",
        lambda n, device=None: torch.arange(n, device=device),
    )

    step_round_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    step_unit_xy = torch.tensor(
        [[10, 0], [11, 0], [20, 0], [21, 0], [30, 0], [31, 0], [40, 0], [41, 0]],
        dtype=torch.long,
    )
    step_unit_float = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    step_rv = torch.stack([torch.arange(8, dtype=torch.float32), torch.arange(100, 108, dtype=torch.float32)], dim=1)
    step_actions = torch.zeros(8, 5, dtype=torch.long)
    positive_mask = torch.tensor([False, False, False, False, True, True, True, True], dtype=torch.bool)

    plan = model_mod._build_critic_epoch_plan(
        step_round_idx,
        step_unit_xy,
        step_unit_float,
        step_rv,
        step_actions,
        target_steps=8,
        positive_mask=positive_mask,
        max_rounds=1,
    )

    assert plan.batch_ranges == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert plan.round_ids.tolist() == [2, 3, 0, 1]


def test_coerce_replay_buffer_reuses_writable_bytearray():
    buf = bytearray(range(8))
    out = model_mod._coerce_replay_buffer(buf)

    assert out is buf


def test_load_torch_replay_buffer_only_copies_torch_backed_specs(tmp_path):
    data_path = tmp_path / "buffer.bin"
    data_path.write_bytes(bytes(range(8)))
    global_states = {
        "states_path": str(data_path),
        "transfer_events_path": str(data_path),
    }

    transfer_buf = replay_buffers_mod.load_replay_buffer(
        global_states,
        replay_buffers_mod.TRANSFER_EVENTS_BUFFER,
    )
    state_buf = replay_buffers_mod.load_torch_replay_buffer(
        global_states,
        replay_buffers_mod.STATE_BUFFER,
    )

    assert isinstance(transfer_buf, bytes)
    assert global_states["transfer_events_buf"] is transfer_buf
    assert isinstance(state_buf, bytearray)
    assert global_states["states_buf"] is state_buf


def test_parse_trajectories_reads_strict_step_old_log_probs():
    obs_bytes = struct.pack(f"<{model_mod.OBS_SIZE}f", *([0.0] * model_mod.OBS_SIZE))
    h_bytes = struct.pack(f"<{model_mod.H_DIM}f", *([0.0] * model_mod.H_DIM))
    trajs = [[
        make_trajectory_step(
            obs=obs_bytes,
            h_prev=h_bytes,
            obs_bytes_len=model_mod.OBS_SIZE * 4,
            h_bytes_len=model_mod.H_DIM * 4,
            round_idx=0,
            bt=0,
            bd=0,
            cd=0,
            brt=0,
            md=0,
            reward=1.25,
            old_log_prob=2.5,
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
    ]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = model_mod._parse_trajectories(trajs)

    assert data is not None
    assert data["old_log_probs"].shape == (1,)
    assert data["old_log_probs"][0].item() == 2.5
    assert not any("not writable" in str(w.message) for w in caught)


def test_compose_action_log_prob_ignores_noncausal_negative_infinities():
    # bd is always included (upstream); cd and brt are zeroed when non-causal
    lp = model_mod._compose_action_log_prob(
        torch.tensor([-0.5]),   # bt_lp
        torch.tensor([-0.3]),   # bd_lp (always included)
        torch.tensor([float("-inf")]),  # cd_lp (zeroed)
        torch.tensor([float("-inf")]),  # brt_lp (zeroed)
        torch.tensor([-0.25]),  # md_lp
        cd_causal=torch.tensor([False]),
        brt_causal=torch.tensor([False]),
    )

    assert torch.isfinite(lp).all()
    # bd_lp (-0.3) + bt_lp (-0.5) + cd_lp (0, zeroed) + brt_lp (0, zeroed) + md_lp (-0.25)
    assert torch.allclose(lp, torch.tensor([-1.05]))


def test_sample_actions_from_policy_heads_returns_finite_behavior_log_prob():
    obs = torch.zeros(2, model_mod.OBS_SIZE)
    bt_l = torch.zeros(2, model_mod.N_BUILD)
    bd_l = torch.zeros(2, model_mod.N_DIR)
    cd_l = torch.zeros(2, model_mod.N_CONV_DIR)
    brt_l = torch.zeros(2, model_mod.N_BRIDGE_TARGETS)
    md_l = torch.zeros(2, model_mod.N_DIR)
    head_temperature = torch.ones((2, model_mod.N_ACTION_TEMPERATURES))
    h_next = torch.zeros(2, model_mod.H_DIM)

    torch.manual_seed(0)
    bt_act, bd_act, cd_act, brt_act, md_act, lp, out_h = model_mod._sample_actions_from_policy_heads(
        obs,
        bt_l,
        bd_l,
        cd_l,
        brt_l,
        md_l,
        head_temperature,
        h_next,
        temperature=1.0,
        return_log_prob=True,
    )

    assert bt_act.shape == (2,)
    assert bd_act.shape == (2,)
    assert cd_act.shape == (2,)
    assert brt_act.shape == (2,)
    assert md_act.shape == (2,)
    assert lp is not None
    assert torch.isfinite(lp).all()
    assert out_h.shape == h_next.shape


def test_materialize_critic_inputs_packs_cpu_slabs_for_staged_round_loading(monkeypatch):
    monkeypatch.setattr(model_mod, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        model_mod,
        "_get_critic_cuda_config",
        lambda: type("Cfg", (), {"channels_last": True})(),
    )

    round0 = torch.arange(
        model_mod.GLOBAL_MAP_CHANNELS * 2 * 3,
        dtype=torch.float32,
    ).reshape(model_mod.GLOBAL_MAP_CHANNELS, 2, 3)
    round1 = torch.full((model_mod.GLOBAL_MAP_CHANNELS, 1, 1), 7.0, dtype=torch.float32)
    global_scalars = torch.arange(
        2 * model_mod.GLOBAL_SCALAR_DIM,
        dtype=torch.float32,
    ).reshape(2, model_mod.GLOBAL_SCALAR_DIM)
    data = {
        "global_states": [round0, round1],
        "global_sizes": torch.tensor([[3, 2], [1, 1]], dtype=torch.int16),
        "global_scalars": global_scalars.clone(),
    }

    model_mod._materialize_critic_inputs(data)
    g_obs, g_mask, g_scalars = model_mod._load_critic_round_batch(
        data, torch.tensor([0, 1], dtype=torch.long),
    )

    assert data["global_states"] is None
    assert data["critic_input_staging"] is None
    assert data["critic_obs_cpu"].shape == (2, model_mod.GLOBAL_MAP_CHANNELS, 2, 3)
    assert data["critic_mask_cpu"].shape == (2, 1, 2, 3)
    assert data["critic_obs_cpu"].is_contiguous(memory_format=torch.channels_last)
    assert data["critic_mask_cpu"].is_contiguous(memory_format=torch.channels_last)
    assert torch.allclose(g_obs[0], round0.to(dtype=g_obs.dtype))
    assert torch.allclose(
        g_obs[1, :, 0, 0],
        torch.full((model_mod.GLOBAL_MAP_CHANNELS,), 7.0, dtype=g_obs.dtype),
    )
    assert torch.count_nonzero(g_obs[1, :, 0, 1:]) == 0
    assert torch.count_nonzero(g_obs[1, :, 1:, :]) == 0
    assert torch.equal(g_mask[0, 0], torch.ones(2, 3, dtype=g_mask.dtype))
    assert torch.equal(
        g_mask[1, 0],
        torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=g_mask.dtype),
    )
    assert torch.allclose(g_scalars, global_scalars)


def test_resolve_critic_minibatch_size_uses_configured_epoch_batch_target():
    assert model_mod._resolve_critic_minibatch_size(32000) == 4000
    assert model_mod._resolve_critic_minibatch_size(8) == 1
    assert model_mod._resolve_critic_minibatch_size(9) == 2


def test_build_critic_turn_minibatches_preserves_coverage_and_target_steps():
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            model_mod.torch,
            "randperm",
            lambda n, device=None: torch.arange(n, device=device),
        )
        step_round_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        positive_mask = torch.tensor([False, False, False, False, True, True, True, True], dtype=torch.bool)
        batches = model_mod._build_critic_turn_minibatches(
            step_round_idx,
            target_steps=4,
            positive_mask=positive_mask,
        )
    finally:
        monkeypatch.undo()

    flattened = torch.cat(batches)
    assert flattened.numel() == step_round_idx.numel()
    assert torch.equal(flattened.sort().values, torch.arange(step_round_idx.numel()))
    assert all(batch.numel() <= 4 for batch in batches)
    assert all(positive_mask[batch].any().item() for batch in batches)


def test_build_critic_turn_minibatches_handles_insufficient_positive_turns_without_duplication():
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            model_mod.torch,
            "randperm",
            lambda n, device=None: torch.arange(n, device=device),
        )
        step_round_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        positive_mask = torch.tensor([False, False, False, False, False, False, True, True], dtype=torch.bool)
        batches = model_mod._build_critic_turn_minibatches(
            step_round_idx,
            target_steps=4,
            positive_mask=positive_mask,
        )
    finally:
        monkeypatch.undo()

    flattened = torch.cat(batches)
    assert flattened.numel() == step_round_idx.numel()
    assert torch.equal(flattened.sort().values, torch.arange(step_round_idx.numel()))
    assert all(batch.numel() <= 4 for batch in batches)
    assert sum(positive_mask[batch].any().item() for batch in batches) == 1


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
        obs_pre_all=None,
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


def test_ppo_update_reuses_cached_obs_pre_for_actor_minibatches(monkeypatch):
    class TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def _obs_encoder_pre(self, _obs):
            raise AssertionError("_obs_encoder_pre should not be called when obs_pre_all is cached")

    captured_obs_pre: list[torch.Tensor] = []

    def fake_recurrent_actor_pass(
        model,
        obs_pre,
        h0,
        bt_act,
        bd_act,
        cd_act,
        brt_act,
        md_act,
        bt_mask,
        bd_mask,
        cd_causal,
    ):
        del h0, bt_act, bd_act, cd_act, brt_act, md_act, bt_mask, bd_mask, cd_causal
        captured_obs_pre.append(obs_pre.detach().cpu())
        new_lps = obs_pre[..., 0] * model.scale
        entropies = torch.zeros_like(new_lps)
        head_temperature = torch.ones(*new_lps.shape, model_mod.N_ACTION_TEMPERATURES, device=new_lps.device)
        return new_lps, entropies, head_temperature

    monkeypatch.setattr(model_mod, "_recurrent_actor_pass", fake_recurrent_actor_pass)

    model = TinyActor()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    obs = torch.zeros(2, model_mod.OBS_SIZE)
    h = torch.zeros(2, model_mod.H_DIM)
    obs_pre_all = torch.tensor([[1.0], [2.0]])
    bt = torch.zeros(2, dtype=torch.long)
    bd = torch.zeros(2, dtype=torch.long)
    cd = torch.zeros(2, dtype=torch.long)
    brt = torch.zeros(2, dtype=torch.long)
    md = torch.zeros(2, dtype=torch.long)
    old_log_probs = torch.zeros(2)
    advantages = torch.ones(2)
    returns = torch.zeros(2)

    stats = model_mod.ppo_update(
        model=model,
        optimizer=optimizer,
        obs=obs,
        h=h,
        obs_pre_all=obs_pre_all,
        bt_act=bt,
        bd_act=bd,
        cd_act=cd,
        brt_act=brt,
        md_act=md,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        centralized_data={},
        ep_boundaries=[(0, 2)],
        ppo_epochs=1,
        bptt_k=2,
        actor_minibatch_size=2,
        freeze_critic=True,
    )

    assert stats["n_updates"] == 1
    assert len(captured_obs_pre) == 1
    assert torch.equal(captured_obs_pre[0], obs_pre_all.unsqueeze(0))
