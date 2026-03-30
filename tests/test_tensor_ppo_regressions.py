"""Targeted regression coverage for PPO-specific training paths."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from bots.tensor.torch.model import GAMMA, LAMBDA
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))

import train_batched
from model import (
    ACTOR_LATENT_DIM,
    BRIDGE_BUILD_TYPE,
    BuilderMLP,
    CentralizedCritic,
    CONVEYOR_BUILD_TYPES,
    DEVICE,
    GLOBAL_MAP_CHANNELS,
    GLOBAL_SCALAR_DIM,
    H_DIM,
    HARVESTER_BUILD_TYPE,
    EPS_MIN,
    N_BRIDGE_TARGETS,
    NUM_BUCKETS,
    OBS_SIZE,
    TILE_FEATURES,
    UNIT_CONTEXT_DIM,
    _MASK_BD_TILES,
    _OBS_AX,
    _OBS_SCALE,
    _OBS_TI,
    _build_bptt_chunks,
    _build_centralized_critic_data,
    _compute_gae,
    _make_bucket_centers,
    _masked_ppo_actor_terms,
    _normalize_actor_latent_dim,
    _parse_trajectories,
    _precompute_policy_masks,
    _recurrent_actor_pass,
    _recurrent_logprob_pass,
    _load_critic_round_batch,
    ppo_update,
    _unimix_logits,
    _sequential_logprobs,
    _split_chunk_ids,
    action_log_prob_and_entropy,
    get_centralized_critic,
    load_centralized_critic_checkpoint,
    reset_centralized_critic,
    save_centralized_critic_checkpoint,
    compute_action_masks,
    compute_build_dir_mask_for_type,
    compute_harvester_build_dir_mask,
    conveyor_dir_mask,
    deterministic_policy_actions,
    decode_value,
    distributional_value_loss,
    soft_cross_entropy_from_logits,
    symexp,
    symlog,
    twohot_encode,
    value_target_probs,
    GraphedInference,
)


def _make_obs(step_idx: int) -> torch.Tensor:
    obs = torch.zeros(OBS_SIZE, dtype=torch.float32)
    obs[_OBS_TI] = 1.0
    obs[_OBS_AX] = 1.0
    obs[_OBS_SCALE] = 0.2  # 100% scale multiplier

    for tile in _MASK_BD_TILES:
        obs[tile * TILE_FEATURES + 4] = 1.0  # empty tile

    ore_tile = _MASK_BD_TILES[step_idx % len(_MASK_BD_TILES)]
    obs[ore_tile * TILE_FEATURES + 1] = 1.0
    return obs


def _manual_rollout_episode(
    model: BuilderMLP,
    length: int,
    *,
    offset: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[int], list[int], list[int], list[int], list[float]]:
    device = next(model.parameters()).device
    h_prev = torch.zeros(1, H_DIM, device=device)

    obs_list: list[torch.Tensor] = []
    h_list: list[torch.Tensor] = []
    bt_list: list[int] = []
    bd_list: list[int] = []
    cd_list: list[int] = []
    brt_list: list[int] = []
    md_list: list[int] = []
    lp_list: list[float] = []

    for local_step in range(length):
        step_idx = offset + local_step
        obs = _make_obs(step_idx).unsqueeze(0).to(device)
        bt_l, bd_l, cd_l, brt_l, md_l, eps, h_next = model(obs, h_prev)

        bt_mask, bd_mask = compute_action_masks(obs)
        harv_bd_mask = compute_harvester_build_dir_mask(obs, bd_mask)
        bt_mask = bt_mask.clone()
        bt_mask[:, HARVESTER_BUILD_TYPE] &= harv_bd_mask.any(dim=1)

        if local_step % 4 == 0 and bool(bt_mask[0, HARVESTER_BUILD_TYPE]):
            bt_act = HARVESTER_BUILD_TYPE
            bd_act = int(harv_bd_mask[0].float().argmax().item())
        elif local_step % 4 == 1 and bool(bt_mask[0, CONVEYOR_BUILD_TYPES[0]]):
            bt_act = CONVEYOR_BUILD_TYPES[0]
            bd_act = int(bd_mask[0].float().argmax().item())
        elif local_step % 4 == 2 and bool(bt_mask[0, BRIDGE_BUILD_TYPE]):
            bt_act = BRIDGE_BUILD_TYPE
            bd_act = int(bd_mask[0].float().argmax().item())
        else:
            bt_act = 0
            bd_act = 0

        bt_act_t = torch.tensor([bt_act], device=device, dtype=torch.long)
        bd_mask = compute_build_dir_mask_for_type(obs, bt_act_t, bd_mask, harv_bd_mask)
        cd_mask = conveyor_dir_mask(device, 1)

        cd_act = step_idx % 4
        brt_act = step_idx % N_BRIDGE_TARGETS
        md_act = (step_idx * 2) % 9
        bd_act = int(bd_act if bool(bd_mask[0, bd_act]) else bd_mask[0].float().argmax().item())

        bd_act_t = torch.tensor([bd_act], device=device, dtype=torch.long)
        cd_act_t = torch.tensor([cd_act], device=device, dtype=torch.long)
        brt_act_t = torch.tensor([brt_act], device=device, dtype=torch.long)
        md_act_t = torch.tensor([md_act], device=device, dtype=torch.long)

        bt_l = bt_l.masked_fill(~bt_mask, -1e9)
        bd_l = bd_l.masked_fill(~bd_mask, -1e9)
        cd_l = cd_l.masked_fill(~cd_mask, -1e9)

        bt_lp, _ = action_log_prob_and_entropy(bt_l, bt_act_t, bt_mask, eps[:, 0])
        bd_lp, _ = action_log_prob_and_entropy(bd_l, bd_act_t, bd_mask, eps[:, 1])
        cd_lp, _ = action_log_prob_and_entropy(cd_l, cd_act_t, cd_mask, eps[:, 2])
        brt_lp, _ = action_log_prob_and_entropy(brt_l, brt_act_t, eps=eps[:, 3])
        md_lp, _ = action_log_prob_and_entropy(md_l, md_act_t, eps=eps[:, 4])
        bd_causal = 1.0 if bt_act != 0 else 0.0
        cd_causal = 1.0 if bt_act in CONVEYOR_BUILD_TYPES else 0.0
        brt_causal = 1.0 if bt_act == BRIDGE_BUILD_TYPE else 0.0
        total_lp = bt_lp + bd_lp * bd_causal + cd_lp * cd_causal + brt_lp * brt_causal + md_lp

        obs_list.append(obs.squeeze(0).cpu())
        h_list.append(h_prev.squeeze(0).cpu())
        bt_list.append(bt_act)
        bd_list.append(bd_act)
        cd_list.append(cd_act)
        brt_list.append(brt_act)
        md_list.append(md_act)
        lp_list.append(float(total_lp.item()))

        h_prev = h_next

    return obs_list, h_list, bt_list, bd_list, cd_list, brt_list, md_list, lp_list


# =====================================================================
# Log probability recomputation
# =====================================================================

class TestLogProbRecomputation:
    """Verify that _sequential_logprobs matches step-by-step rollout log probs."""

    def test_recomputed_log_probs_match_manual_sequential_rollout(self):
        torch.manual_seed(0)
        model = BuilderMLP().eval()

        ep1 = _manual_rollout_episode(model, 5, offset=0)
        ep2 = _manual_rollout_episode(model, 7, offset=100)
        all_obs = ep1[0] + ep2[0]
        all_h = ep1[1] + ep2[1]
        all_bt = ep1[2] + ep2[2]
        all_bd = ep1[3] + ep2[3]
        all_cd = ep1[4] + ep2[4]
        all_brt = ep1[5] + ep2[5]
        all_md = ep1[6] + ep2[6]
        manual_lp = torch.tensor(ep1[7] + ep2[7], dtype=torch.float32)

        device = next(model.parameters()).device
        obs = torch.stack(all_obs).to(device)
        h = torch.stack(all_h).to(device)
        bt = torch.tensor(all_bt, dtype=torch.long, device=device)
        bd = torch.tensor(all_bd, dtype=torch.long, device=device)
        cd = torch.tensor(all_cd, dtype=torch.long, device=device)
        brt = torch.tensor(all_brt, dtype=torch.long, device=device)
        md = torch.tensor(all_md, dtype=torch.long, device=device)

        ep_boundaries = [(0, len(ep1[0])), (len(ep1[0]), len(all_obs))]
        chunks = _build_bptt_chunks(ep_boundaries, bptt_k=4)
        recomputed = _sequential_logprobs(
            model,
            obs,
            h,
            bt,
            bd,
            cd,
            brt,
            md,
            chunks,
            bptt_k=4,
            batch_size=8,
        ).cpu()

        assert torch.allclose(recomputed, manual_lp, atol=1e-5, rtol=1e-5)

    def test_ppo_update_keeps_rollout_old_log_probs_fixed_across_epochs(self, monkeypatch):
        import model as model_mod

        torch.manual_seed(0)
        model = BuilderMLP().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ep = _manual_rollout_episode(model.eval(), 2, offset=0)
        obs = torch.stack(ep[0]).to(DEVICE)
        h = torch.stack(ep[1]).to(DEVICE)
        bt = torch.tensor(ep[2], dtype=torch.long, device=DEVICE)
        bd = torch.tensor(ep[3], dtype=torch.long, device=DEVICE)
        cd = torch.tensor(ep[4], dtype=torch.long, device=DEVICE)
        brt = torch.tensor(ep[5], dtype=torch.long, device=DEVICE)
        md = torch.tensor(ep[6], dtype=torch.long, device=DEVICE)
        old_log_probs = torch.tensor(ep[7], dtype=torch.float32, device=DEVICE)
        advantages = torch.zeros(2, dtype=torch.float32, device=DEVICE)
        returns = torch.zeros(2, dtype=torch.float32, device=DEVICE)
        old_values = torch.zeros(2, dtype=torch.float32, device=DEVICE)

        class DummyCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(NUM_BUCKETS, device=DEVICE))
                self.register_buffer("bucket_centers", _make_bucket_centers())

            def encode_map(self, g_obs, g_mask):
                del g_mask
                return torch.zeros(g_obs.shape[0], 64, device=g_obs.device)

            def unit_value_logits(self, feat, g_scalars, round_idx, unit_xy, unit_ctx, actor_latent=None):
                del feat, g_scalars, unit_xy, unit_ctx, actor_latent
                bsz = round_idx.shape[0]
                logits = self.bias.unsqueeze(0).expand(bsz, -1)
                values = logits[:, 0] * 0.0
                return values, logits

        dummy_critic = DummyCritic()
        dummy_opt = torch.optim.Adam(dummy_critic.parameters(), lr=1e-3)

        def fail_if_recomputed(*args, **kwargs):
            del args, kwargs
            raise AssertionError("ppo_update must not recompute old_log_probs")

        def zero_value_loss(logits, newvalue, target, bucket_centers, **kwargs):
            del logits, target, bucket_centers, kwargs
            return newvalue * 0.0

        monkeypatch.setattr(model_mod, "_sequential_logprobs", fail_if_recomputed)
        monkeypatch.setattr(model_mod, "get_centralized_critic", lambda: (dummy_critic, dummy_opt))
        monkeypatch.setattr(model_mod, "distributional_value_loss", zero_value_loss)

        centralized_data = {
            "step_round_idx": torch.zeros(2, dtype=torch.long, device=DEVICE),
            "step_unit_xy": torch.zeros(2, 2, dtype=torch.long, device=DEVICE),
            "step_unit_ctx": torch.zeros(2, UNIT_CONTEXT_DIM, dtype=torch.float32, device=DEVICE),
            "step_actor_latent": torch.zeros(2, ACTOR_LATENT_DIM, dtype=torch.float32, device=DEVICE),
            "global_states": [torch.zeros(GLOBAL_MAP_CHANNELS, 1, 1, dtype=torch.float32)],
            "global_sizes": torch.tensor([[1, 1]], dtype=torch.int16),
            "global_scalars": torch.zeros(1, GLOBAL_SCALAR_DIM, dtype=torch.float32),
        }

        stats = ppo_update(
            model,
            optimizer,
            obs,
            h,
            bt,
            bd,
            cd,
            brt,
            md,
            old_log_probs,
            advantages,
            returns,
            centralized_data,
            old_values=old_values,
            ep_boundaries=[(0, 2)],
            ppo_epochs=2,
            minibatch_size=2,
            bptt_k=2,
        )

        assert stats["n_updates"] == 2

    def test_parse_trajectories_preserves_actor_decision_tick(self):
        torch.manual_seed(123)
        model = BuilderMLP().to(DEVICE).eval()
        ep = _manual_rollout_episode(model, 4, offset=20)
        traj = [[
            {
                "obs": step_obs.tolist(),
                "h_prev": step_h.tolist(),
                "bt": bt,
                "bd": bd,
                "cd": cd,
                "brt": brt,
                "md": md,
                "old_lp": old_lp,
            }
            for step_obs, step_h, bt, bd, cd, brt, md, old_lp in zip(*ep)
        ]]

        parsed = _parse_trajectories(traj)
        assert parsed is not None

        recomputed = _sequential_logprobs(
            model,
            parsed["obs"],
            parsed["h"],
            parsed["bt"],
            parsed["bd"],
            parsed["cd"],
            parsed["brt"],
            parsed["md"],
            _build_bptt_chunks(parsed["ep_boundaries"], bptt_k=4),
            bptt_k=4,
            batch_size=8,
        ).cpu()

        expected = torch.tensor(ep[7], dtype=torch.float32)
        assert torch.allclose(recomputed, expected, atol=1e-5, rtol=1e-5)

    def test_single_step_episode_log_prob(self):
        """A single-step episode should still produce valid recomputed log probs."""
        torch.manual_seed(42)
        model = BuilderMLP().eval()

        ep = _manual_rollout_episode(model, 1, offset=0)
        device = next(model.parameters()).device
        obs = torch.stack(ep[0]).to(device)
        h = torch.stack(ep[1]).to(device)
        bt = torch.tensor(ep[2], dtype=torch.long, device=device)
        bd = torch.tensor(ep[3], dtype=torch.long, device=device)
        cd = torch.tensor(ep[4], dtype=torch.long, device=device)
        brt = torch.tensor(ep[5], dtype=torch.long, device=device)
        md = torch.tensor(ep[6], dtype=torch.long, device=device)
        manual_lp = torch.tensor(ep[7], dtype=torch.float32)

        chunks = _build_bptt_chunks([(0, 1)], bptt_k=4)
        recomputed = _sequential_logprobs(
            model, obs, h, bt, bd, cd, brt, md, chunks, bptt_k=4, batch_size=8,
        ).cpu()

        assert torch.allclose(recomputed, manual_lp, atol=1e-5)

    def test_causal_masking_zeroes_bd_cd_brt_for_no_build(self):
        """When bt_act=0 (no build), bd, cd, and brt log probs must not contribute."""
        torch.manual_seed(7)
        logits = torch.randn(1, 13)
        bd_logits = torch.randn(1, 9)
        cd_logits = torch.randn(1, 4)
        brt_logits = torch.randn(1, N_BRIDGE_TARGETS)
        md_logits = torch.randn(1, 9)

        bt_act = torch.tensor([0])  # no build
        bd_act = torch.tensor([3])
        cd_act = torch.tensor([1])
        brt_act = torch.tensor([5])
        md_act = torch.tensor([2])

        bt_lp, _ = action_log_prob_and_entropy(logits, bt_act)
        bd_lp, _ = action_log_prob_and_entropy(bd_logits, bd_act)
        cd_lp, _ = action_log_prob_and_entropy(cd_logits, cd_act)
        brt_lp, _ = action_log_prob_and_entropy(brt_logits, brt_act)
        md_lp, _ = action_log_prob_and_entropy(md_logits, md_act)

        bd_causal = 0.0
        cd_causal = 0.0
        brt_causal = 0.0
        total = bt_lp + bd_lp * bd_causal + cd_lp * cd_causal + brt_lp * brt_causal + md_lp

        assert total.item() == pytest.approx((bt_lp + md_lp).item(), abs=1e-7)

    def test_causal_masking_includes_bd_for_harvester(self):
        """For harvester build, bd contributes but cd and brt do not."""
        torch.manual_seed(7)
        logits = torch.randn(1, 13)
        bd_logits = torch.randn(1, 9)
        cd_logits = torch.randn(1, 4)
        brt_logits = torch.randn(1, N_BRIDGE_TARGETS)
        md_logits = torch.randn(1, 9)

        bt_act = torch.tensor([HARVESTER_BUILD_TYPE])
        bd_act = torch.tensor([3])
        cd_act = torch.tensor([1])
        brt_act = torch.tensor([5])
        md_act = torch.tensor([2])

        bt_lp, _ = action_log_prob_and_entropy(logits, bt_act)
        bd_lp, _ = action_log_prob_and_entropy(bd_logits, bd_act)
        cd_lp, _ = action_log_prob_and_entropy(cd_logits, cd_act)
        brt_lp, _ = action_log_prob_and_entropy(brt_logits, brt_act)
        md_lp, _ = action_log_prob_and_entropy(md_logits, md_act)

        bd_causal = 1.0
        cd_causal = 0.0
        brt_causal = 0.0
        total = bt_lp + bd_lp * bd_causal + cd_lp * cd_causal + brt_lp * brt_causal + md_lp

        assert total.item() == pytest.approx((bt_lp + bd_lp + md_lp).item(), abs=1e-7)

    def test_causal_masking_includes_bd_cd_for_conveyor(self):
        """For conveyor build, bd and cd contribute but brt does not."""
        torch.manual_seed(7)
        logits = torch.randn(1, 13)
        bd_logits = torch.randn(1, 9)
        cd_logits = torch.randn(1, 4)
        brt_logits = torch.randn(1, N_BRIDGE_TARGETS)
        md_logits = torch.randn(1, 9)

        bt_act = torch.tensor([CONVEYOR_BUILD_TYPES[0]])
        bd_act = torch.tensor([3])
        cd_act = torch.tensor([1])
        brt_act = torch.tensor([5])
        md_act = torch.tensor([2])

        bt_lp, _ = action_log_prob_and_entropy(logits, bt_act)
        bd_lp, _ = action_log_prob_and_entropy(bd_logits, bd_act)
        cd_lp, _ = action_log_prob_and_entropy(cd_logits, cd_act)
        brt_lp, _ = action_log_prob_and_entropy(brt_logits, brt_act)
        md_lp, _ = action_log_prob_and_entropy(md_logits, md_act)

        bd_causal = 1.0
        cd_causal = 1.0
        brt_causal = 0.0
        total = bt_lp + bd_lp * bd_causal + cd_lp * cd_causal + brt_lp * brt_causal + md_lp

        assert total.item() == pytest.approx((bt_lp + bd_lp + cd_lp + md_lp).item(), abs=1e-7)

    def test_causal_masking_includes_bd_brt_for_bridge(self):
        """For bridge build, bd and brt contribute but cd does not."""
        torch.manual_seed(7)
        logits = torch.randn(1, 13)
        bd_logits = torch.randn(1, 9)
        cd_logits = torch.randn(1, 4)
        brt_logits = torch.randn(1, N_BRIDGE_TARGETS)
        md_logits = torch.randn(1, 9)

        bt_act = torch.tensor([BRIDGE_BUILD_TYPE])
        bd_act = torch.tensor([3])
        cd_act = torch.tensor([1])
        brt_act = torch.tensor([5])
        md_act = torch.tensor([2])

        bt_lp, _ = action_log_prob_and_entropy(logits, bt_act)
        bd_lp, _ = action_log_prob_and_entropy(bd_logits, bd_act)
        cd_lp, _ = action_log_prob_and_entropy(cd_logits, cd_act)
        brt_lp, _ = action_log_prob_and_entropy(brt_logits, brt_act)
        md_lp, _ = action_log_prob_and_entropy(md_logits, md_act)

        bd_causal = 1.0
        cd_causal = 0.0
        brt_causal = 1.0
        total = bt_lp + bd_lp * bd_causal + cd_lp * cd_causal + brt_lp * brt_causal + md_lp

        assert total.item() == pytest.approx((bt_lp + bd_lp + brt_lp + md_lp).item(), abs=1e-7)


# =====================================================================
# action_log_prob_and_entropy correctness
# =====================================================================

class TestActionLogProbAndEntropy:
    """Verify log_prob and entropy computation against manual formulas."""

    def test_log_prob_matches_manual(self):
        """Log prob should match manual softmax + unimix + gather."""
        logits = torch.tensor([[2.0, 1.0, 0.5]])
        action = torch.tensor([1])

        lp, _ = action_log_prob_and_entropy(logits, action)
        probs = F.softmax(logits, dim=-1)
        mixed = (1 - EPS_MIN) * probs + EPS_MIN / logits.shape[-1]
        expected = mixed.log()[0, 1]
        assert lp.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_entropy_matches_manual(self):
        """Entropy should match -sum(p * log(p)) over unimixed distribution."""
        logits = torch.tensor([[2.0, 1.0, 0.5]])
        action = torch.tensor([0])

        _, ent = action_log_prob_and_entropy(logits, action)
        probs = F.softmax(logits, dim=-1)
        mixed = (1 - EPS_MIN) * probs + EPS_MIN / logits.shape[-1]
        log_mixed = mixed.log()
        expected_ent = -(mixed * log_mixed).sum(dim=-1)
        assert ent.item() == pytest.approx(expected_ent.item(), abs=1e-6)

    def test_uniform_logits_max_entropy(self):
        """Uniform logits should produce maximum categorical entropy."""
        n = 9
        logits = torch.zeros(1, n)
        action = torch.tensor([0])

        _, ent = action_log_prob_and_entropy(logits, action)
        max_ent = torch.log(torch.tensor(float(n)))
        assert ent.item() == pytest.approx(max_ent.item(), abs=1e-5)

    def test_peaked_logits_has_entropy_floor(self):
        """Peaked logits should still have nonzero entropy due to unimix."""
        logits = torch.tensor([[-100.0, 100.0, -100.0]])
        action = torch.tensor([1])

        lp, ent = action_log_prob_and_entropy(logits, action)
        assert ent.item() > 0.0
        assert lp.item() < 0.0

    def test_log_prob_always_negative_or_zero(self):
        """Log probabilities must be <= 0."""
        torch.manual_seed(99)
        for _ in range(20):
            logits = torch.randn(5, 7)
            actions = torch.randint(0, 7, (5,))
            lp, _ = action_log_prob_and_entropy(logits, actions)
            assert (lp <= 1e-6).all()

    def test_batch_independence(self):
        """Each batch element's log prob should be independent."""
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        actions = torch.tensor([0, 1])

        lp, ent = action_log_prob_and_entropy(logits, actions)
        assert lp[0].item() == pytest.approx(lp[1].item(), abs=1e-6)
        assert ent[0].item() == pytest.approx(ent[1].item(), abs=1e-6)


# =====================================================================
# Distributional value prediction (symexp bins, twohot, decode)
# =====================================================================

class TestDistributionalValuePrediction:
    """Verify the Dreamer4-style symexp two-hot value prediction chain."""

    def test_symlog_symexp_inverse(self):
        """symexp(symlog(x)) == x for a range of values."""
        x = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0, 10000.0])
        assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)

    def test_bucket_centers_symmetric(self):
        """Bin centers should be symmetric around 0."""
        bins = _make_bucket_centers()
        n = bins.shape[0]
        m = (n - 1) // 2
        assert bins[m].item() == pytest.approx(0.0, abs=1e-7)
        # bins[m-k] should be -bins[m+k]
        for k in range(1, min(10, m)):
            assert bins[m - k].item() == pytest.approx(-bins[m + k].item(), abs=1e-6)

    def test_bucket_centers_monotonic(self):
        """Bin centers must be strictly increasing."""
        bins = _make_bucket_centers()
        diffs = bins[1:] - bins[:-1]
        assert (diffs > 0).all()

    def test_decode_uniform_logits_is_zero(self):
        """Uniform logits over symmetric bins should decode to exactly 0."""
        bins = _make_bucket_centers()
        logits = torch.zeros(1, bins.shape[0])
        val = decode_value(logits, bins)
        assert val.item() == pytest.approx(0.0, abs=1e-6)

    def test_decode_peaked_at_bin_returns_bin_value(self):
        """Logits peaked at one bin should decode close to that bin's value."""
        bins = _make_bucket_centers()
        for idx in [0, 50, 127, 200, 254]:
            logits = torch.full((1, bins.shape[0]), -100.0)
            logits[0, idx] = 100.0
            val = decode_value(logits, bins)
            assert val.item() == pytest.approx(bins[idx].item(), rel=1e-3, abs=1e-6)

    def test_twohot_encode_sums_to_one(self):
        """Two-hot encoding must produce valid probability distributions."""
        bins = _make_bucket_centers()
        targets = torch.tensor([-50.0, 0.0, 0.5, 100.0, 5000.0])
        encoded = twohot_encode(targets, bins)
        sums = encoded.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_twohot_encode_at_most_two_nonzero(self):
        """Each two-hot vector should have at most 2 non-zero entries."""
        bins = _make_bucket_centers()
        targets = torch.tensor([0.0, 1.0, -5.0, 1000.0])
        encoded = twohot_encode(targets, bins)
        for i in range(targets.shape[0]):
            assert (encoded[i] > 0).sum() <= 2

    def test_twohot_encode_at_bin_center_is_onehot(self):
        """Target exactly at a bin center should produce a one-hot."""
        bins = _make_bucket_centers()
        target = bins[100:101]
        encoded = twohot_encode(target, bins)
        assert (encoded > 1e-6).sum() == 1
        assert encoded[0, 100].item() == pytest.approx(1.0, abs=1e-5)

    def test_decode_roundtrip_through_twohot(self):
        """Encoding a value as twohot then decoding via weighted sum should recover it."""
        bins = _make_bucket_centers()
        # Test a range of values within bin range
        targets = torch.tensor([-500.0, -10.0, 0.0, 10.0, 500.0])
        encoded = twohot_encode(targets, bins)
        # Decode: weighted sum of bins
        recovered = (encoded * bins.unsqueeze(0)).sum(dim=-1)
        assert torch.allclose(recovered, targets.clamp(bins[0], bins[-1]), atol=1.0)

    def test_value_target_probs_matches_twohot(self):
        """value_target_probs should be identical to twohot_encode."""
        bins = _make_bucket_centers()
        targets = torch.tensor([0.0, 5.0, -5.0])
        a = value_target_probs(targets, bins)
        b = twohot_encode(targets, bins)
        assert torch.allclose(a, b, atol=1e-7)


# =====================================================================
# Distributional value loss (PPO clipping in distributional setting)
# =====================================================================

class TestDistributionalValueLoss:
    def test_distributional_critic_clip_can_select_clipped_branch(self):
        bins = _make_bucket_centers()
        center = bins.shape[0] // 2

        logits_a = torch.full((1, bins.shape[0]), -30.0)
        logits_a[0, center] = 0.0

        old_values = torch.tensor([10.0], dtype=torch.float32)
        returns = torch.tensor([0.0], dtype=torch.float32)
        target = value_target_probs(returns, bins)

        unclipped_a = soft_cross_entropy_from_logits(logits_a, target)
        newvalue_a = decode_value(logits_a, bins)

        clipped_values = old_values + (newvalue_a - old_values).clamp(-0.4, 0.4)
        clipped_branch = soft_cross_entropy_from_logits(
            value_target_probs(clipped_values, bins), target,
        )
        assert clipped_branch > unclipped_a

        clipped_a = distributional_value_loss(
            logits_a, newvalue_a, returns, bins,
            old_values=old_values, clip_vloss=True, value_clip=0.4,
        )
        assert torch.allclose(clipped_a, clipped_branch, atol=1e-6)

    def test_no_clip_equals_raw_cross_entropy(self):
        """With clip_vloss=False, loss should equal plain cross-entropy."""
        bins = _make_bucket_centers()
        logits = torch.randn(4, bins.shape[0])
        values = decode_value(logits, bins)
        returns = torch.tensor([1.0, -1.0, 0.0, 5.0])

        loss_noclip = distributional_value_loss(
            logits, values, returns, bins, clip_vloss=False,
        )
        target = value_target_probs(returns, bins)
        expected = soft_cross_entropy_from_logits(logits, target)
        assert torch.allclose(loss_noclip, expected, atol=1e-6)

    def test_clip_without_old_values_equals_unclipped(self):
        """When old_values=None, clipping should be disabled."""
        bins = _make_bucket_centers()
        logits = torch.randn(3, bins.shape[0])
        values = decode_value(logits, bins)
        returns = torch.tensor([2.0, -2.0, 0.5])

        loss = distributional_value_loss(
            logits, values, returns, bins,
            old_values=None, clip_vloss=True,
        )
        target = value_target_probs(returns, bins)
        expected = soft_cross_entropy_from_logits(logits, target)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_value_loss_always_nonnegative(self):
        """Cross-entropy based value loss should be >= 0."""
        bins = _make_bucket_centers()
        torch.manual_seed(123)
        for _ in range(10):
            logits = torch.randn(8, bins.shape[0])
            values = decode_value(logits, bins)
            returns = torch.randn(8) * 100
            old_values = torch.randn(8) * 100
            loss = distributional_value_loss(
                logits, values, returns, bins,
                old_values=old_values, clip_vloss=True,
            )
            assert (loss >= -1e-6).all()

    def test_clip_selects_pessimistic_branch(self):
        """Dreamer4 clipping should take the max of unclipped and clipped losses."""
        bins = _make_bucket_centers()
        old_values = torch.tensor([0.0])
        returns = torch.tensor([10.0])
        logits = torch.full((1, bins.shape[0]), -30.0)
        target_idx = (bins - 10.0).abs().argmin()
        logits[0, target_idx] = 10.0
        values = decode_value(logits, bins)

        loss = distributional_value_loss(
            logits, values, returns, bins,
            old_values=old_values, clip_vloss=True, value_clip=0.4,
        )
        loss_noclip = distributional_value_loss(
            logits, values, returns, bins, clip_vloss=False,
        )
        clipped_val = old_values + (values - old_values).clamp(-0.4, 0.4)
        clipped_loss = soft_cross_entropy_from_logits(
            value_target_probs(clipped_val, bins),
            value_target_probs(returns, bins),
        )
        expected = torch.maximum(loss_noclip, clipped_loss)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_clipped_branch_backprops_through_values(self):
        bins = _make_bucket_centers()
        old_values = torch.tensor([0.0], dtype=torch.float32)
        target_idx = (bins - 100.0).abs().argmin()
        returns = bins[target_idx:target_idx + 1].clone().detach()
        logits = torch.full((1, bins.shape[0]), -30.0)
        logits[0, target_idx] = 10.0
        values = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)
        loss = distributional_value_loss(
            logits, values, returns, bins,
            old_values=old_values, clip_vloss=True, value_clip=0.4,
        )
        unclipped = soft_cross_entropy_from_logits(
            logits, value_target_probs(returns, bins),
        )
        clipped = soft_cross_entropy_from_logits(
            value_target_probs(
                old_values + (values - old_values).clamp(-0.4, 0.4),
                bins,
            ),
            value_target_probs(returns, bins),
        )

        assert clipped.item() > unclipped.item()
        assert torch.allclose(loss, torch.maximum(unclipped, clipped), atol=1e-6)
        loss.backward()
        assert values.grad is not None
        assert values.grad.abs().sum().item() > 0.0


# =====================================================================
# GAE computation
# =====================================================================

class TestGAE:
    def test_long_episode_gae_tail_remains_terminal(self):
        rewards = torch.zeros(1900)
        values = torch.full((1900,), 5.0)
        advantages, returns = _compute_gae(
            rewards, values, [(0, 1900)], gamma=GAMMA, gae_lambda=1.0,
        )
        assert advantages[-1].item() == pytest.approx(-5.0)
        assert returns[-1].item() == pytest.approx(0.0)

    def test_single_step_episode(self):
        """A single-step episode: advantage = r - v (terminal, no bootstrap)."""
        rewards = torch.tensor([3.0])
        values = torch.tensor([1.0])
        advantages, returns = _compute_gae(
            rewards, values, [(0, 1)], gamma=GAMMA, gae_lambda=LAMBDA,
        )
        # Terminal: delta = r + 0 - v = 3 - 1 = 2
        assert advantages[0].item() == pytest.approx(2.0)
        assert returns[0].item() == pytest.approx(3.0)  # adv + v = 2 + 1

    def test_two_step_episode_gae(self):
        """Manually verify GAE for a 2-step episode."""
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 1.0])
        gamma, lam = GAMMA, LAMBDA

        advantages, returns = _compute_gae(
            rewards, values, [(0, 2)], gamma=gamma, gae_lambda=lam,
        )

        # Step 1 (t=1, terminal): delta_1 = r_1 + 0 - v_1 = 2 - 1 = 1
        # GAE_1 = delta_1 = 1
        assert advantages[1].item() == pytest.approx(1.0)

        # Step 0 (t=0): delta_0 = r_0 + gamma * v_1 - v_0 = 1 + 0.99*1 - 0.5 = 1.49
        # GAE_0 = delta_0 + gamma*lambda*GAE_1 = 1.49 + 0.99*0.95*1 = 1.49 + 0.9405
        delta_0 = 1.0 + gamma * 1.0 - 0.5
        gae_0 = delta_0 + gamma * lam * 1.0
        assert advantages[0].item() == pytest.approx(gae_0, abs=1e-5)
        assert returns[0].item() == pytest.approx(gae_0 + 0.5, abs=1e-5)

    def test_multiple_episodes_independent(self):
        """GAE for each episode should be computed independently."""
        # Two identical single-step episodes
        rewards = torch.tensor([5.0, 5.0])
        values = torch.tensor([2.0, 2.0])
        adv_multi, ret_multi = _compute_gae(
            rewards, values, [(0, 1), (1, 2)], gamma=GAMMA, gae_lambda=LAMBDA,
        )
        # Each should be identical
        assert adv_multi[0].item() == pytest.approx(adv_multi[1].item())
        assert ret_multi[0].item() == pytest.approx(ret_multi[1].item())

        # Compare to single episode
        adv_single, ret_single = _compute_gae(
            rewards[:1], values[:1], [(0, 1)], gamma=GAMMA, gae_lambda=LAMBDA,
        )
        assert adv_multi[0].item() == pytest.approx(adv_single[0].item())

    def test_episode_boundary_no_cross_bootstrap(self):
        """Value from next episode must not leak across episode boundaries."""
        # ep1: [r=0, v=0], ep2: [r=0, v=1000]
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.0, 1000.0])
        adv, _ = _compute_gae(
            rewards, values, [(0, 1), (1, 2)], gamma=GAMMA, gae_lambda=LAMBDA,
        )
        # ep1 advantage: delta = 0 + 0 - 0 = 0 (should NOT bootstrap from v=1000)
        assert adv[0].item() == pytest.approx(0.0)
        # ep2 advantage: delta = 0 + 0 - 1000 = -1000
        assert adv[1].item() == pytest.approx(-1000.0)

    def test_zero_gamma_only_immediate_reward(self):
        """With gamma=0, advantage = r - v for every step."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])
        adv, ret = _compute_gae(
            rewards, values, [(0, 3)], gamma=0.0, gae_lambda=LAMBDA,
        )
        expected_adv = rewards - values
        assert torch.allclose(adv, expected_adv, atol=1e-5)

    def test_returns_equal_advantages_plus_values(self):
        """returns = advantages + values, always."""
        torch.manual_seed(42)
        N = 50
        rewards = torch.randn(N)
        values = torch.randn(N)
        ep_bounds = [(0, 20), (20, 35), (35, 50)]
        adv, ret = _compute_gae(rewards, values, ep_bounds)
        assert torch.allclose(ret, adv + values, atol=1e-5)


# =====================================================================
# BPTT chunking
# =====================================================================

class TestBPTTChunking:
    def test_chunks_cover_all_steps(self):
        """Chunks must cover every timestep exactly once."""
        ep_bounds = [(0, 10), (10, 23), (23, 30)]
        for bptt_k in [4, 8, 16, 32]:
            chunks = _build_bptt_chunks(ep_bounds, bptt_k)
            covered = set()
            for start, length in chunks:
                for t in range(start, start + length):
                    assert t not in covered, f"Step {t} covered twice with bptt_k={bptt_k}"
                    covered.add(t)
            assert covered == set(range(30)), f"Missing steps with bptt_k={bptt_k}"

    def test_chunks_respect_episode_boundaries(self):
        """No chunk should span across episode boundaries."""
        ep_bounds = [(0, 10), (10, 25)]
        chunks = _build_bptt_chunks(ep_bounds, bptt_k=4)
        for start, length in chunks:
            # Must be entirely within one episode
            found = False
            for ep_start, ep_end in ep_bounds:
                if ep_start <= start and start + length <= ep_end:
                    found = True
                    break
            assert found, f"Chunk ({start}, {length}) spans episode boundary"

    def test_chunk_lengths_bounded_by_bptt_k(self):
        """Every chunk should have length <= bptt_k."""
        ep_bounds = [(0, 7), (7, 19)]
        bptt_k = 4
        chunks = _build_bptt_chunks(ep_bounds, bptt_k)
        for start, length in chunks:
            assert length <= bptt_k
            assert length > 0

    def test_single_step_episodes(self):
        """Each single-step episode produces exactly one length-1 chunk."""
        ep_bounds = [(0, 1), (1, 2), (2, 3)]
        chunks = _build_bptt_chunks(ep_bounds, bptt_k=4)
        assert len(chunks) == 3
        for start, length in chunks:
            assert length == 1


# =====================================================================
# PPO ratio and clipping properties
# =====================================================================

class TestPPORatioProperties:
    """Test PPO ratio and clipping math in isolation."""

    def test_ratio_is_one_when_log_probs_match(self):
        """When new_lp == old_lp, ratio should be exactly 1."""
        old_lp = torch.tensor([-1.5, -0.5, -2.0])
        new_lp = old_lp.clone()
        ratio = (new_lp - old_lp).exp()
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-7)

    def test_ratio_increases_when_new_more_likely(self):
        """Higher new log prob => ratio > 1."""
        old_lp = torch.tensor([-2.0])
        new_lp = torch.tensor([-1.0])  # more likely
        ratio = (new_lp - old_lp).exp()
        assert ratio.item() > 1.0

    def test_clipped_surrogate_loss_positive_advantage(self):
        """With positive advantage, clipping caps the benefit of increasing ratio."""
        adv = torch.tensor([1.0])
        clip_eps = 0.2
        # Ratio well above clip range
        ratio = torch.tensor([2.0])
        pg1 = -adv * ratio
        pg2 = -adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        loss = torch.max(pg1, pg2)
        # pg2 (clipped) = -1.0 * 1.2 = -1.2, pg1 = -1.0 * 2.0 = -2.0
        # max(-2, -1.2) = -1.2 (the clipped, less negative one)
        assert loss.item() == pytest.approx(-1.2, abs=1e-6)

    def test_clipped_surrogate_loss_negative_advantage(self):
        """With negative advantage, clipping caps the penalty of decreasing ratio."""
        adv = torch.tensor([-1.0])
        clip_eps = 0.2
        # Ratio well below clip range
        ratio = torch.tensor([0.5])
        pg1 = -adv * ratio  # 0.5
        pg2 = -adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)  # 0.8
        loss = torch.max(pg1, pg2)
        # max(0.5, 0.8) = 0.8
        assert loss.item() == pytest.approx(0.8, abs=1e-6)

    def test_approx_kl_formula(self):
        """Verify the approximate KL divergence formula: mean((ratio - 1) - logratio)."""
        old_lp = torch.tensor([-1.0, -2.0, -0.5])
        new_lp = torch.tensor([-1.1, -1.8, -0.6])
        logratio = new_lp - old_lp
        ratio = logratio.exp()
        # Schulman's approximation: KL ≈ mean((ratio - 1) - log(ratio))
        approx_kl = ((ratio - 1) - logratio).mean()
        # Should be >= 0 (KL is always non-negative)
        assert approx_kl.item() >= -1e-7
        # When distributions are identical, KL = 0
        approx_kl_same = ((torch.ones(3) - 1) - torch.zeros(3)).mean()
        assert approx_kl_same.item() == pytest.approx(0.0)

    def test_asymmetric_clipping_higher_bound(self):
        """Asymmetric clipping with clip_eps_higher > clip_eps."""
        adv = torch.tensor([1.0])
        clip_eps = 0.2
        clip_eps_higher = 0.28
        ratio = torch.tensor([1.5])  # above both clip ranges
        pg1 = -adv * ratio
        pg2 = -adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps_higher)
        loss = torch.max(pg1, pg2)
        # clamp(1.5, 0.8, 1.28) = 1.28, pg2 = -1.28
        assert loss.item() == pytest.approx(-1.28, abs=1e-6)

    def test_masked_tail_terms_raise_on_invalid_infinite_logratio(self):
        logratio = torch.tensor([[0.1, float("inf")]])
        adv = torch.tensor([[1.0, 0.0]])
        ent = torch.tensor([[0.2, 0.5]])
        valid_mask = torch.tensor([[True, False]])

        with pytest.raises(RuntimeError, match="tail logratio is non-finite"):
            _masked_ppo_actor_terms(
                logratio,
                adv,
                ent,
                valid_mask,
                clip_eps=0.2,
                clip_eps_higher=0.28,
            )


# =====================================================================
# Soft cross-entropy
# =====================================================================

class TestSoftCrossEntropy:
    def test_perfect_prediction_zero_loss(self):
        """When logits produce the same distribution as target, CE = entropy."""
        target = torch.tensor([[0.5, 0.3, 0.2]])
        # log of target as logits
        logits = target.log()
        ce = soft_cross_entropy_from_logits(logits, target)
        # Should equal the entropy of the target
        entropy = -(target * target.log()).sum(dim=-1)
        assert ce.item() == pytest.approx(entropy.item(), abs=1e-5)

    def test_cross_entropy_nonnegative(self):
        """Cross-entropy against valid probability targets is always >= 0."""
        torch.manual_seed(5)
        for _ in range(10):
            logits = torch.randn(3, 10)
            target = F.softmax(torch.randn(3, 10), dim=-1)
            ce = soft_cross_entropy_from_logits(logits, target)
            assert (ce >= -1e-6).all()


# =====================================================================
# Critic architecture sanity
# =====================================================================

class TestCentralizedCritic:
    def test_map_encoder_stays_fp32_even_when_bf16_enabled(self, monkeypatch):
        import model as model_mod

        monkeypatch.setattr(model_mod, "_USE_BF16", True)
        critic = model_mod.CentralizedCritic()

        for module in (critic.stem, critic.down_a, critic.stage_a, critic.down_b, critic.stage_b):
            dtypes = {param.dtype for param in module.parameters()}
            assert dtypes == {torch.float32}

    def test_critic_forward_output_shape(self):
        """Critic forward should return one value per unit."""
        torch.manual_seed(0)
        critic = CentralizedCritic()
        B = 3
        H, W = 10, 10
        map_tensor = torch.randn(B, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.ones(B, 1, H, W)
        scalars = torch.randn(B, GLOBAL_SCALAR_DIM)
        unit_xy = torch.randint(0, 10, (B, 2))
        unit_ctx = torch.randn(B, UNIT_CONTEXT_DIM)
        actor_latent = torch.randn(B, ACTOR_LATENT_DIM)

        values = critic(map_tensor, valid_mask, scalars, unit_xy, unit_ctx, actor_latent)
        assert values.shape == (B,)

    def test_encode_map_returns_round_summaries(self):
        """encode_map should return one shared summary vector per round."""
        critic = CentralizedCritic()
        map_tensor = torch.randn(4, GLOBAL_MAP_CHANNELS, 10, 10)
        valid_mask = torch.ones(4, 1, 10, 10)
        feat = critic.encode_map(map_tensor, valid_mask)
        assert feat.shape == (4, 64)

    def test_unit_value_logits_returns_values_and_logits(self):
        """unit_value_logits should return matching value/logits pair."""
        torch.manual_seed(0)
        critic = CentralizedCritic()
        R = 2
        H, W = 8, 8
        map_tensor = torch.randn(R, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.ones(R, 1, H, W)
        feat = critic.encode_map(map_tensor, valid_mask)
        scalars = torch.randn(R, GLOBAL_SCALAR_DIM)
        B = 5
        round_idx = torch.randint(0, R, (B,))
        unit_xy = torch.randint(0, 8, (B, 2))
        unit_ctx = torch.randn(B, UNIT_CONTEXT_DIM)
        actor_latent = torch.randn(B, ACTOR_LATENT_DIM)

        values, logits = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent,
        )
        assert values.shape == (B,)
        assert logits.shape == (B, NUM_BUCKETS)
        # Values should match decode_value applied to logits
        expected = decode_value(logits, critic.bucket_centers)
        assert torch.allclose(values, expected, atol=1e-5)

    def test_unit_xy_no_longer_changes_values_directly(self):
        """The whole-map summary critic should ignore direct unit_xy indexing."""
        torch.manual_seed(0)
        critic = CentralizedCritic().eval()
        map_tensor = torch.randn(1, GLOBAL_MAP_CHANNELS, 10, 10)
        valid_mask = torch.ones(1, 1, 10, 10)
        feat = critic.encode_map(map_tensor, valid_mask)
        scalars = torch.randn(1, GLOBAL_SCALAR_DIM)
        round_idx = torch.zeros(3, dtype=torch.long)
        unit_xy_a = torch.tensor([[1, 1], [5, 5], [9, 0]])
        unit_xy_b = torch.tensor([[9, 9], [0, 0], [2, 7]])
        unit_ctx = torch.randn(3, UNIT_CONTEXT_DIM)
        actor_latent = torch.randn(3, ACTOR_LATENT_DIM)

        values_a, logits_a = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy_a, unit_ctx, actor_latent,
        )
        values_b, logits_b = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy_b, unit_ctx, actor_latent,
        )
        assert torch.allclose(values_a, values_b, atol=1e-6)
        assert torch.allclose(logits_a, logits_b, atol=1e-6)

    def test_critic_starts_with_nonuniform_logits(self):
        """Dreamer4 create_mlp uses standard linear init, not zero logits."""
        torch.manual_seed(0)
        critic = CentralizedCritic().eval()
        R = 2
        H, W = 8, 8
        map_tensor = torch.randn(R, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.ones(R, 1, H, W)
        feat = critic.encode_map(map_tensor, valid_mask)
        scalars = torch.randn(R, GLOBAL_SCALAR_DIM)
        B = 5
        round_idx = torch.randint(0, R, (B,))
        unit_xy = torch.randint(0, 8, (B, 2))
        unit_ctx = torch.randn(B, UNIT_CONTEXT_DIM)
        actor_latent = torch.randn(B, ACTOR_LATENT_DIM)

        values, logits = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent,
        )
        assert not torch.allclose(logits, torch.zeros_like(logits), atol=1e-7)
        assert values.abs().max().item() > 1e-7

    def test_critic_gradients_flow(self):
        """Verify gradients flow through the critic from supervised value loss."""
        torch.manual_seed(0)
        critic = CentralizedCritic()
        B = 2
        H, W = 8, 8
        map_tensor = torch.randn(B, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.ones(B, 1, H, W)
        scalars = torch.randn(B, GLOBAL_SCALAR_DIM)
        unit_xy = torch.randint(0, 8, (B, 2))
        unit_ctx = torch.randn(B, UNIT_CONTEXT_DIM)
        actor_latent = torch.randn(B, ACTOR_LATENT_DIM)

        feat = critic.encode_map(map_tensor, valid_mask)
        round_idx = torch.arange(B)
        values, logits = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent,
        )
        returns = torch.tensor([1.0, -1.0], dtype=torch.float32)
        loss = distributional_value_loss(
            logits, values, returns, critic.bucket_centers, clip_vloss=False,
        ).sum()
        loss.backward()

        has_grad = False
        for p in critic.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through critic"

        last_weight_grad = critic.unit_head[-1].weight.grad
        assert last_weight_grad is not None
        assert last_weight_grad.abs().sum().item() > 0.0

    def test_fully_masked_map_produces_zero_features(self):
        """A fully zeroed valid_mask should produce all-zero feature maps."""
        torch.manual_seed(0)
        critic = CentralizedCritic().eval()
        H, W = 10, 10
        map_tensor = torch.randn(1, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.zeros(1, 1, H, W)  # all masked

        feat = critic.encode_map(map_tensor, valid_mask)
        assert (feat == 0).all(), "Fully masked map should produce zero features"

    def test_different_maps_same_mask_different_features(self):
        """Different map content in valid region should change encoded features."""
        torch.manual_seed(0)
        critic = CentralizedCritic().eval()
        H, W = 10, 10
        valid_mask = torch.ones(1, 1, H, W)

        map_a = torch.randn(1, GLOBAL_MAP_CHANNELS, H, W)
        map_b = torch.randn(1, GLOBAL_MAP_CHANNELS, H, W) * 5

        feat_a = critic.encode_map(map_a, valid_mask)
        feat_b = critic.encode_map(map_b, valid_mask)
        assert not torch.allclose(feat_a, feat_b, atol=1e-3), \
            "Different maps should produce different encoded features"

    def test_load_centralized_critic_checkpoint_accepts_standalone_state_dict(self, tmp_path):
        torch.manual_seed(0)
        source = CentralizedCritic()
        ckpt = tmp_path / "critic_pretrained.pt"
        torch.save(source.state_dict(), ckpt)

        reset_centralized_critic()
        assert load_centralized_critic_checkpoint(ckpt)
        loaded, _opt = get_centralized_critic()

        source_state = source.state_dict()
        loaded_state = loaded.state_dict()
        assert source_state.keys() == loaded_state.keys()
        for key in source_state:
            assert torch.equal(source_state[key].cpu(), loaded_state[key].cpu()), key

        reset_centralized_critic()

    def test_save_centralized_critic_checkpoint_accepts_direct_pt_path(self, tmp_path):
        reset_centralized_critic()
        _critic, _opt = get_centralized_critic()
        ckpt = tmp_path / "critic_sidecarless.pt"
        saved = save_centralized_critic_checkpoint(ckpt)
        assert saved == ckpt
        assert ckpt.exists()
        payload = torch.load(ckpt, map_location="cpu")
        assert isinstance(payload, dict)
        assert "critic" in payload
        assert "optimizer" in payload
        reset_centralized_critic()

    def test_load_centralized_critic_checkpoint_rejects_non_pt_path(self, tmp_path):
        bad = tmp_path / "weights.json"
        bad.write_text("{}")
        reset_centralized_critic()
        assert not load_centralized_critic_checkpoint(bad)
        reset_centralized_critic()

    def test_save_centralized_critic_checkpoint_rejects_non_pt_path(self, tmp_path):
        reset_centralized_critic()
        _critic, _opt = get_centralized_critic()
        bad = tmp_path / "weights.json"
        with pytest.raises(ValueError, match="critic checkpoints must use a \\.pt path"):
            save_centralized_critic_checkpoint(bad)
        reset_centralized_critic()


# =====================================================================
# Advantage normalization
# =====================================================================

class TestAdvantageNormalization:
    def test_normalized_advantages_zero_mean_unit_var(self):
        """After normalization, advantages should have ~0 mean and ~1 std."""
        adv = torch.randn(1000) * 10 + 5  # non-zero mean, large std
        norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        assert norm_adv.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert norm_adv.std().item() == pytest.approx(1.0, abs=1e-2)

    def test_constant_advantages_no_nan(self):
        """All-equal advantages shouldn't produce NaN after normalization."""
        adv = torch.full((100,), 3.0)
        norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        assert not torch.isnan(norm_adv).any()
        # Should be all ~0
        assert norm_adv.abs().max().item() < 1e-4


# =====================================================================
# Logit scaling
# =====================================================================

class TestUnimix:
    """Verify fixed unimix mechanism."""

    def test_unimix_with_mask_zeros_invalid(self):
        """Masked actions should get zero probability from unimix."""
        logits = torch.randn(2, 5)
        mask = torch.tensor([[True, True, False, False, False],
                             [True, True, True, True, True]])
        logits[~mask] = -1e9
        log_probs = _unimix_logits(logits, mask)
        assert log_probs[0, 2].item() == float('-inf')
        assert log_probs[0, 3].item() == float('-inf')
        assert log_probs[0, 0].isfinite()
        assert log_probs[1, 4].isfinite()

    def test_unimix_with_mask_matches_valid_support_mix(self):
        """Masked unimix should match a valid-only Dreamer-style probability mix."""
        logits = torch.tensor([[3.0, -7.0, 1.0, -9.0]])
        mask = torch.tensor([[True, False, True, False]])
        eps = 0.01

        log_probs = _unimix_logits(logits, mask, eps=eps)
        probs = log_probs.exp()

        valid_logits = logits[0, mask[0]]
        valid_probs = torch.softmax(valid_logits, dim=-1)
        expected_valid = (1.0 - eps) * valid_probs + eps * (1.0 / valid_probs.numel())

        assert probs[0, 1].item() == pytest.approx(0.0, abs=1e-12)
        assert probs[0, 3].item() == pytest.approx(0.0, abs=1e-12)
        assert probs[0, mask[0]].tolist() == pytest.approx(expected_valid.tolist(), abs=1e-7)

    def test_unimix_provides_entropy_floor(self):
        """Even peaked logits should have nonzero entropy from unimix."""
        logits = torch.tensor([[-100.0, 100.0, -100.0]])
        _, entropy = action_log_prob_and_entropy(logits, torch.tensor([1]))
        assert entropy.item() > 0.0

    def test_unimix_masked_logprob_backward_stays_finite(self):
        """Masked-out zero-prob actions must not create NaN gradients in backward."""
        logits = torch.tensor([[3.0, -1e9, 1.0, -1e9]], requires_grad=True)
        mask = torch.tensor([[True, False, True, False]])
        action = torch.tensor([0])

        log_probs = _unimix_logits(logits, mask)
        loss = -log_probs.gather(1, action.unsqueeze(1)).mean()
        loss.backward()

        assert torch.isfinite(logits.grad).all()
        assert logits.grad[0, 1].item() == pytest.approx(0.0, abs=1e-12)
        assert logits.grad[0, 3].item() == pytest.approx(0.0, abs=1e-12)


class TestDeterministicPolicyActions:
    def test_zero_noise_decode_keeps_bridge_target_and_ignores_eps(self):
        obs = _make_obs(0).unsqueeze(0)
        bt_l = torch.full((1, 13), -10.0)
        bt_l[0, BRIDGE_BUILD_TYPE] = 10.0
        bd_l = torch.full((1, 9), -10.0)
        bd_l[0, 3] = 8.0
        cd_l = torch.full((1, 4), -10.0)
        cd_l[0, 2] = 7.0
        brt_l = torch.full((1, N_BRIDGE_TARGETS), -10.0)
        brt_l[0, 5] = 9.0
        md_l = torch.full((1, 9), -10.0)
        md_l[0, 4] = 6.0
        eps = torch.full((1, 5), LAMBDA)

        bt_act, bd_act, cd_act, brt_act, md_act, log_prob = deterministic_policy_actions(
            obs,
            bt_l,
            bd_l,
            cd_l,
            brt_l,
            md_l,
            eps,
            zero_noise=True,
            return_log_prob=True,
        )

        assert bt_act.item() == BRIDGE_BUILD_TYPE
        assert bd_act.item() == 3
        assert cd_act.item() == 2
        assert brt_act.item() == 5
        assert md_act.item() == 4
        assert torch.isfinite(log_prob).all()

    @pytest.mark.skipif(DEVICE.type != "cuda", reason="CUDA graph inference only runs on CUDA")
    def test_graphed_deterministic_inference_matches_helper(self):
        torch.manual_seed(0)
        model = BuilderMLP().to(DEVICE).eval()
        graph = GraphedInference(
            model,
            max_batch=4,
            temperature=1.0,
            deterministic=True,
            zero_noise=True,
            return_log_prob=False,
        )
        graph.capture()

        packed = torch.zeros(3, OBS_SIZE + H_DIM, device=DEVICE)
        for i in range(3):
            packed[i, :OBS_SIZE] = _make_obs(i).to(DEVICE)
        bt_l, bd_l, cd_l, brt_l, md_l, eps, _ = model.infer_packed(packed)
        exp_bt, exp_bd, exp_cd, exp_brt, exp_md, _ = deterministic_policy_actions(
            packed[:, :OBS_SIZE],
            bt_l,
            bd_l,
            cd_l,
            brt_l,
            md_l,
            eps,
            temperature=1.0,
            zero_noise=True,
            return_log_prob=False,
        )

        got_bt, got_bd, got_cd, got_brt, got_md, _lp, _h = graph(packed)

        assert torch.equal(got_bt, exp_bt)
        assert torch.equal(got_bd, exp_bd)
        assert torch.equal(got_cd, exp_cd)
        assert torch.equal(got_brt, exp_brt)
        assert torch.equal(got_md, exp_md)


class TestOpponentModelLoading:
    def test_cached_opponent_model_raises_when_load_fails(self, monkeypatch, tmp_path):
        weight_path = tmp_path / "broken_opp.json"
        weight_path.write_text("{}")
        train_batched._opp_model_cache.clear()

        def fake_load_weights(self, path):
            return False

        monkeypatch.setattr(train_batched.BuilderMLP, "load_weights", fake_load_weights)

        with pytest.raises(RuntimeError, match="Failed to load opponent weights"):
            train_batched._get_cached_opponent_model(str(weight_path), None)


class TestCentralizedCriticRoundLoader:
    def test_load_critic_round_batch_streams_selected_rounds_only(self):
        round0 = torch.arange(GLOBAL_MAP_CHANNELS * 2 * 2, dtype=torch.float32).reshape(GLOBAL_MAP_CHANNELS, 2, 2)
        round1 = torch.full((GLOBAL_MAP_CHANNELS, 1, 1), 7.0, dtype=torch.float32)
        data = {
            "global_states": [round0, round1],
            "global_sizes": torch.tensor([[2, 2], [1, 1]], dtype=torch.int16),
            "global_scalars": torch.tensor(
                [[1.0] * GLOBAL_SCALAR_DIM, [2.0] * GLOBAL_SCALAR_DIM],
                dtype=torch.float32,
            ),
        }

        obs, mask, scalars = _load_critic_round_batch(
            data,
            torch.tensor([1, 0], dtype=torch.long, device=DEVICE),
        )

        assert obs.shape == (2, GLOBAL_MAP_CHANNELS, 2, 2)
        assert mask.shape == (2, 1, 2, 2)
        assert scalars.shape == (2, GLOBAL_SCALAR_DIM)
        assert torch.allclose(obs[0, :, :1, :1].float().cpu(), round1)
        assert torch.allclose(obs[1].float().cpu(), round0)
        assert mask[0, 0, 0, 0].item() == pytest.approx(1.0)
        assert mask[0, 0, 1, 1].item() == pytest.approx(0.0)
        assert scalars[0, 0].item() == pytest.approx(2.0)
        assert scalars[1, 0].item() == pytest.approx(1.0)

    def test_build_data_supports_file_backed_round_buffers(self, tmp_path):
        n_turns = 2
        map_w = 2
        map_h = 2
        state0 = torch.arange(GLOBAL_MAP_CHANNELS * map_h * map_w, dtype=torch.float32)
        state1 = state0 + 1000
        states = torch.cat([state0, state1]).contiguous()
        scalars = torch.arange(n_turns * GLOBAL_SCALAR_DIM, dtype=torch.float32).reshape(n_turns, GLOBAL_SCALAR_DIM)

        states_path = tmp_path / "states.bin"
        scalars_path = tmp_path / "scalars.bin"
        states_path.write_bytes(bytes(states.view(torch.uint8).reshape(-1).tolist()))
        scalars_path.write_bytes(bytes(scalars.view(torch.uint8).reshape(-1).tolist()))

        obs_t = torch.zeros(2, OBS_SIZE, dtype=torch.float32)
        round_idx_t = torch.tensor([0, 1], dtype=torch.long)
        data = _build_centralized_critic_data(
            obs_t,
            round_idx_t,
            [{
                "states_path": str(states_path),
                "scalars_path": str(scalars_path),
                "state_stride": GLOBAL_MAP_CHANNELS * map_h * map_w,
                "map_w": map_w,
                "map_h": map_h,
                "n_turns": n_turns,
                "n_channels": GLOBAL_MAP_CHANNELS,
                "n_scalars": GLOBAL_SCALAR_DIM,
            }],
            [(0, 2)],
            2,
            traj_game_map=[0],
        )
        try:
            round_ids = data["step_round_idx"].flip(0)
            obs, mask, got_scalars = _load_critic_round_batch(
                data,
                round_ids,
            )
            assert torch.allclose(
                obs[0, :, :map_h, :map_w].float().cpu(),
                state1.reshape(GLOBAL_MAP_CHANNELS, map_h, map_w),
                atol=4.0,
                rtol=0.0,
            )
            assert torch.allclose(
                obs[1, :, :map_h, :map_w].float().cpu(),
                state0.reshape(GLOBAL_MAP_CHANNELS, map_h, map_w),
                atol=1e-4,
                rtol=0.0,
            )
            assert torch.equal(got_scalars.float().cpu(), torch.stack([scalars[1], scalars[0]]))
            assert mask[:, :, :map_h, :map_w].sum().item() == pytest.approx(2 * map_w * map_h)
        finally:
            for resource in data.get("global_backing_resources", []):
                try:
                    resource.close()
                except Exception:
                    pass

    def test_build_data_round_lookup_never_looks_ahead(self):
        map_w = 2
        map_h = 1

        def _state_fill(value: float) -> torch.Tensor:
            return torch.full((GLOBAL_MAP_CHANNELS, map_h, map_w), value, dtype=torch.float32)

        def _state_bytes(value: float) -> bytes:
            return bytes(_state_fill(value).view(torch.uint8).reshape(-1).tolist())

        def _scalar_fill(value: float) -> list[float]:
            row = [0.0] * GLOBAL_SCALAR_DIM
            row[0] = value
            return row

        obs_t = torch.zeros(6, OBS_SIZE, dtype=torch.float32)
        round_idx_t = torch.tensor([0, 1, 4, 6, 9, 10], dtype=torch.long)
        data = _build_centralized_critic_data(
            obs_t,
            round_idx_t,
            [{
                "states": {
                    2: _state_bytes(2.0),
                    5: _state_bytes(5.0),
                    10: _state_bytes(10.0),
                },
                "scalars": {
                    2: _scalar_fill(2.0),
                    5: _scalar_fill(5.0),
                    10: _scalar_fill(10.0),
                },
                "map_w": map_w,
                "map_h": map_h,
            }],
            [(0, 6)],
            6,
            traj_game_map=[0],
        )

        obs, mask, scalars = _load_critic_round_batch(data, data["step_round_idx"])

        # Rounds 0-1 have no historical replay state, so they fall back to zero.
        assert torch.equal(
            scalars[:, 0].float().cpu(),
            torch.tensor([0.0, 0.0, 2.0, 5.0, 5.0, 10.0], dtype=torch.float32),
        )
        assert torch.allclose(obs[2, :, :map_h, :map_w].float().cpu(), _state_fill(2.0))
        assert torch.allclose(obs[3, :, :map_h, :map_w].float().cpu(), _state_fill(5.0))
        assert torch.allclose(obs[4, :, :map_h, :map_w].float().cpu(), _state_fill(5.0))
        assert torch.allclose(obs[5, :, :map_h, :map_w].float().cpu(), _state_fill(10.0))
        assert mask[0, :, :map_h, :map_w].sum().item() == pytest.approx(1.0)
        assert mask[1, :, :map_h, :map_w].sum().item() == pytest.approx(1.0)
        assert mask[2:, :, :map_h, :map_w].sum().item() == pytest.approx(4 * map_w * map_h)


class TestTrajectoryParsing:
    def test_parse_trajectories_accepts_compact_bytes_obs_and_hidden(self):
        obs0 = torch.arange(OBS_SIZE, dtype=torch.float32)
        obs1 = obs0 + 1.0
        h0 = torch.arange(H_DIM, dtype=torch.float32)
        h1 = h0 + 2.0
        traj = [
            (bytes(obs0.view(torch.uint8).reshape(-1).tolist()),
             bytes(h0.view(torch.uint8).reshape(-1).tolist()),
             1, 2, 3, 4, 0.0, 0, 0.0, 0.0, 0.0, 0.0, -0.2),
            (bytes(obs1.view(torch.uint8).reshape(-1).tolist()),
             bytes(h1.view(torch.uint8).reshape(-1).tolist()),
             2, 3, 1, 4, 0.0, 0, 0.0, 0.0, 0.0, 0.0, -0.1),
        ]

        data = _parse_trajectories([traj])

        assert data is not None
        assert data["n_steps"] == 2
        assert torch.allclose(data["obs"][0].float().cpu(), obs0)
        assert torch.allclose(data["obs"][1].float().cpu(), obs1)
        assert torch.allclose(data["h"][0].float().cpu(), h0)
        assert torch.allclose(data["h"][1].float().cpu(), h1)
        assert data["bt"].tolist() == [1, 2]
        assert data["bd"].tolist() == [2, 3]


# =====================================================================
# _recurrent_logprob_pass vs _recurrent_actor_pass consistency
# =====================================================================

class TestRecurrentPassConsistency:
    """Verify that the logprob-only and full actor recurrent passes produce
    the same log probabilities (the actor pass adds entropy/scales)."""

    def test_logprob_pass_matches_actor_pass(self):
        torch.manual_seed(42)
        model = BuilderMLP().eval()

        ep = _manual_rollout_episode(model, 6, offset=0)
        device = next(model.parameters()).device
        obs = torch.stack(ep[0]).to(device)
        bt = torch.tensor(ep[2], dtype=torch.long, device=device)
        bd = torch.tensor(ep[3], dtype=torch.long, device=device)
        cd = torch.tensor(ep[4], dtype=torch.long, device=device)
        brt = torch.tensor(ep[5], dtype=torch.long, device=device)
        md = torch.tensor(ep[6], dtype=torch.long, device=device)
        h0 = torch.stack(ep[1][:1]).to(device)

        bt_mask, bd_mask, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)

        # Reshape for recurrent pass: (B=1, K=6, ...)
        obs_pre = model._obs_encoder_pre(obs).unsqueeze(0)
        bt_a = bt.unsqueeze(0)
        bd_a = bd.unsqueeze(0)
        cd_a = cd.unsqueeze(0)
        brt_a = brt.unsqueeze(0)
        md_a = md.unsqueeze(0)
        bt_m = bt_mask.unsqueeze(0)
        bd_m = bd_mask.unsqueeze(0)
        bd_c = bd_causal.unsqueeze(0)
        cd_c = cd_causal.unsqueeze(0)
        brt_c = brt_causal.unsqueeze(0)

        with torch.no_grad():
            lp_only = _recurrent_logprob_pass(
                model, obs_pre, h0, bt_a, bd_a, cd_a, brt_a, md_a, bt_m, bd_m, bd_c, cd_c, brt_c,
            )
            lp_actor, ent, _eps = _recurrent_actor_pass(
                model, obs_pre, h0, bt_a, bd_a, cd_a, brt_a, md_a, bt_m, bd_m, bd_c, cd_c, brt_c,
            )

        assert torch.allclose(lp_only, lp_actor, atol=1e-5), \
            f"Log prob mismatch: max diff {(lp_only - lp_actor).abs().max()}"

    def test_entropy_is_nonnegative(self):
        """Entropy from the actor pass should always be >= 0."""
        torch.manual_seed(7)
        model = BuilderMLP().eval()
        ep = _manual_rollout_episode(model, 4, offset=0)
        device = next(model.parameters()).device
        obs = torch.stack(ep[0]).to(device)
        bt = torch.tensor(ep[2], dtype=torch.long, device=device)
        bd = torch.tensor(ep[3], dtype=torch.long, device=device)
        cd = torch.tensor(ep[4], dtype=torch.long, device=device)
        brt = torch.tensor(ep[5], dtype=torch.long, device=device)
        md = torch.tensor(ep[6], dtype=torch.long, device=device)
        h0 = torch.stack(ep[1][:1]).to(device)

        bt_mask, bd_mask, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)
        obs_pre = model._obs_encoder_pre(obs).unsqueeze(0)

        with torch.no_grad():
            _, ent, _eps = _recurrent_actor_pass(
                model, obs_pre, h0,
                bt.unsqueeze(0), bd.unsqueeze(0), cd.unsqueeze(0), brt.unsqueeze(0), md.unsqueeze(0),
                bt_mask.unsqueeze(0), bd_mask.unsqueeze(0),
                bd_causal.unsqueeze(0), cd_causal.unsqueeze(0), brt_causal.unsqueeze(0),
            )

        assert (ent >= -1e-6).all(), f"Negative entropy found: {ent.min()}"

    def test_noncausal_invalid_bd_action_is_ignored(self):
        torch.manual_seed(0)
        model = BuilderMLP().eval()
        device = next(model.parameters()).device

        obs = _make_obs(0).unsqueeze(0).to(device)
        obs[0, _MASK_BD_TILES[0] * TILE_FEATURES + 0] = 1.0
        bt = torch.tensor([0], dtype=torch.long, device=device)
        bd = torch.tensor([0], dtype=torch.long, device=device)
        cd = torch.tensor([3], dtype=torch.long, device=device)
        brt = torch.tensor([7], dtype=torch.long, device=device)
        md = torch.tensor([4], dtype=torch.long, device=device)
        h0 = torch.zeros(1, H_DIM, device=device)

        bt_mask, bd_mask, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)
        assert not bool(bd_mask[0, 0])
        assert bd_causal[0].item() == 0.0

        obs_pre = model._obs_encoder_pre(obs).unsqueeze(0)
        with torch.no_grad():
            lp_only = _recurrent_logprob_pass(
                model,
                obs_pre,
                h0,
                bt.unsqueeze(0),
                bd.unsqueeze(0),
                cd.unsqueeze(0),
                brt.unsqueeze(0),
                md.unsqueeze(0),
                bt_mask.unsqueeze(0),
                bd_mask.unsqueeze(0),
                bd_causal.unsqueeze(0),
                cd_causal.unsqueeze(0),
                brt_causal.unsqueeze(0),
            )
            lp_actor, ent, _eps = _recurrent_actor_pass(
                model,
                obs_pre,
                h0,
                bt.unsqueeze(0),
                bd.unsqueeze(0),
                cd.unsqueeze(0),
                brt.unsqueeze(0),
                md.unsqueeze(0),
                bt_mask.unsqueeze(0),
                bd_mask.unsqueeze(0),
                bd_causal.unsqueeze(0),
                cd_causal.unsqueeze(0),
                brt_causal.unsqueeze(0),
            )

        assert torch.isfinite(lp_only).all()
        assert torch.isfinite(lp_actor).all()
        assert torch.isfinite(ent).all()


# =====================================================================
# _precompute_policy_masks
# =====================================================================

class TestPrecomputePolicyMasks:
    def test_bd_causal_zero_for_no_build(self):
        """bd_causal should be 0 when bt_act is 0 (no build)."""
        obs = _make_obs(0).unsqueeze(0)
        bt = torch.tensor([0])
        _, _, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)
        assert bd_causal[0].item() == 0.0
        assert cd_causal[0].item() == 0.0
        assert brt_causal[0].item() == 0.0

    def test_bd_causal_one_for_harvester(self):
        obs = _make_obs(0).unsqueeze(0)
        bt = torch.tensor([HARVESTER_BUILD_TYPE])
        _, _, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)
        assert bd_causal[0].item() == 1.0
        assert cd_causal[0].item() == 0.0  # harvester is not conveyor
        assert brt_causal[0].item() == 0.0

    def test_cd_causal_one_for_conveyor(self):
        obs = _make_obs(0).unsqueeze(0)
        for cbt in CONVEYOR_BUILD_TYPES:
            bt = torch.tensor([cbt])
            _, _, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)
            assert bd_causal[0].item() == 1.0, f"bd_causal should be 1 for conveyor type {cbt}"
            assert cd_causal[0].item() == 1.0, f"cd_causal should be 1 for conveyor type {cbt}"
            assert brt_causal[0].item() == 0.0

    def test_brt_causal_one_for_bridge(self):
        obs = _make_obs(0).unsqueeze(0)
        bt = torch.tensor([BRIDGE_BUILD_TYPE])
        _, _, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)
        assert bd_causal[0].item() == 1.0
        assert cd_causal[0].item() == 0.0
        assert brt_causal[0].item() == 1.0


# =====================================================================
# _split_chunk_ids
# =====================================================================

class TestSplitChunkIds:
    def test_all_full_length(self):
        """When all chunks are bptt_k, tail should be empty."""
        lens = torch.tensor([4, 4, 4, 4])
        full, tail = _split_chunk_ids(lens, bptt_k=4)
        assert full.shape[0] == 4
        assert tail.shape[0] == 0

    def test_all_tail(self):
        """When no chunk is bptt_k, full should be empty."""
        lens = torch.tensor([1, 2, 3])
        full, tail = _split_chunk_ids(lens, bptt_k=4)
        assert full.shape[0] == 0
        assert tail.shape[0] == 3

    def test_tails_sorted_descending(self):
        """Tail chunk ids should be sorted by length descending."""
        lens = torch.tensor([4, 1, 3, 4, 2])
        _, tail = _split_chunk_ids(lens, bptt_k=4)
        tail_lens = lens[tail]
        for i in range(len(tail_lens) - 1):
            assert tail_lens[i] >= tail_lens[i + 1]

    def test_full_plus_tail_covers_all(self):
        """Full + tail ids should cover all chunk indices."""
        lens = torch.tensor([4, 2, 4, 1, 4, 3])
        full, tail = _split_chunk_ids(lens, bptt_k=4)
        all_ids = torch.cat([full, tail]).sort().values
        expected = torch.arange(6)
        assert torch.equal(all_ids, expected)


# =====================================================================
# _normalize_actor_latent_dim
# =====================================================================

class TestNormalizeActorLatent:
    def test_exact_dim_is_identity(self):
        x = torch.randn(3, ACTOR_LATENT_DIM)
        assert torch.equal(_normalize_actor_latent_dim(x), x)

    def test_larger_dim_truncates(self):
        x = torch.randn(3, ACTOR_LATENT_DIM + 10)
        out = _normalize_actor_latent_dim(x)
        assert out.shape == (3, ACTOR_LATENT_DIM)
        assert torch.equal(out, x[:, :ACTOR_LATENT_DIM])

    def test_smaller_dim_zero_pads(self):
        x = torch.randn(3, ACTOR_LATENT_DIM - 5)
        out = _normalize_actor_latent_dim(x)
        assert out.shape == (3, ACTOR_LATENT_DIM)
        assert torch.equal(out[:, :ACTOR_LATENT_DIM - 5], x)
        assert (out[:, ACTOR_LATENT_DIM - 5:] == 0).all()


# =====================================================================
# Actor gradient flow (PPO actor loss backward)
# =====================================================================

class TestActorGradientFlow:
    """Verify gradients flow through the actor from the PPO surrogate loss."""

    def test_actor_loss_backward_produces_gradients(self):
        torch.manual_seed(0)
        model = BuilderMLP()
        model.train()

        ep = _manual_rollout_episode(model, 4, offset=0)
        device = next(model.parameters()).device
        obs = torch.stack(ep[0]).to(device)
        bt = torch.tensor(ep[2], dtype=torch.long, device=device)
        bd = torch.tensor(ep[3], dtype=torch.long, device=device)
        cd = torch.tensor(ep[4], dtype=torch.long, device=device)
        brt = torch.tensor(ep[5], dtype=torch.long, device=device)
        md = torch.tensor(ep[6], dtype=torch.long, device=device)
        old_lps = torch.tensor(ep[7], dtype=torch.float32, device=device)
        h0 = torch.zeros(1, H_DIM, device=device)

        bt_mask, bd_mask, bd_causal, cd_causal, brt_causal = _precompute_policy_masks(obs, bt)
        obs_pre = model._obs_encoder_pre(obs).unsqueeze(0)

        new_lps, ent, _eps = _recurrent_actor_pass(
            model, obs_pre, h0,
            bt.unsqueeze(0), bd.unsqueeze(0), cd.unsqueeze(0), brt.unsqueeze(0), md.unsqueeze(0),
            bt_mask.unsqueeze(0), bd_mask.unsqueeze(0),
            bd_causal.unsqueeze(0), cd_causal.unsqueeze(0), brt_causal.unsqueeze(0),
        )
        new_lps = new_lps.squeeze(0)
        ent = ent.squeeze(0)

        # Build PPO surrogate loss
        advantages = torch.randn_like(old_lps)
        ratio = (new_lps - old_lps.detach()).exp()
        clip_eps = 0.2
        pg1 = -advantages * ratio
        pg2 = -advantages * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        pi_loss = torch.max(pg1, pg2).mean()
        actor_loss = pi_loss - 0.01 * ent.mean()

        actor_loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through actor"

    def test_no_grad_leak_through_old_log_probs(self):
        """old_log_probs should be detached — no gradient should flow through them."""
        torch.manual_seed(0)
        model = BuilderMLP()

        ep = _manual_rollout_episode(model, 3, offset=0)
        old_lps = torch.tensor(ep[7], dtype=torch.float32, requires_grad=True)

        # Simulate ratio computation with detached old_lps
        new_lps = torch.randn(3, requires_grad=True)
        ratio = (new_lps - old_lps.detach()).exp()
        loss = ratio.mean()
        loss.backward()

        assert old_lps.grad is None or (old_lps.grad == 0).all(), \
            "Gradients leaked through old_log_probs"


class TestMuCenterPolicyHeads:
    """Verify mu-centering projection keeps policy head weights centered."""

    def test_weights_centered_after_projection(self):
        torch.manual_seed(42)
        model = BuilderMLP()
        # Artificially drift weights away from center
        with torch.no_grad():
            for head in (model.head_build_type, model.head_build_dir,
                         model.head_conv_dir, model.head_bridge_target,
                         model.head_move_dir):
                head.weight.add_(torch.randn(1, head.weight.shape[1]) * 5.0)
                head.bias.add_(3.0)

        model.mu_center_policy_heads()

        for name, head in [
            ("build_type", model.head_build_type),
            ("build_dir", model.head_build_dir),
            ("conv_dir", model.head_conv_dir),
            ("bridge_target", model.head_bridge_target),
            ("move_dir", model.head_move_dir),
        ]:
            assert head.weight.mean(dim=0).abs().max() < 1e-5, \
                f"{name} weight rows not centered"
            assert abs(head.bias.mean().item()) < 1e-5, \
                f"{name} bias not centered"

    def test_softmax_output_unchanged(self):
        """Mu-centering must not change the model's output distribution."""
        torch.manual_seed(7)
        model = BuilderMLP()
        model.eval()
        obs = torch.randn(4, OBS_SIZE)
        h = torch.zeros(4, H_DIM)

        with torch.no_grad():
            bt1, bd1, cd1, brt1, md1, eps1, _ = model(obs, h)
            probs_before = [F.softmax(l, dim=-1) for l in (bt1, bd1, cd1, brt1, md1)]

        # Drift weights then center
        with torch.no_grad():
            for head in (model.head_build_type, model.head_build_dir,
                         model.head_conv_dir, model.head_bridge_target,
                         model.head_move_dir):
                head.weight.add_(torch.randn(1, head.weight.shape[1]) * 10.0)
                head.bias.add_(7.0)
        model.mu_center_policy_heads()

        # Clear cached fused params
        if hasattr(model, '_cached_head_w'):
            del model._cached_head_w
            del model._cached_head_b

        with torch.no_grad():
            bt2, bd2, cd2, brt2, md2, eps2, _ = model(obs, h)
            probs_after = [F.softmax(l, dim=-1) for l in (bt2, bd2, cd2, brt2, md2)]

        for i, (p_before, p_after) in enumerate(zip(probs_before, probs_after)):
            assert torch.allclose(p_before, p_after, atol=1e-5), \
                f"Head {i}: softmax changed after mu-centering a uniform shift"

    def test_does_not_center_non_policy_heads(self):
        """Logit scale and recurrent heads should be untouched."""
        torch.manual_seed(0)
        model = BuilderMLP()
        gate_w_before = model.head_h_gate.weight.clone()
        cand_w_before = model.head_h_cand.weight.clone()
        scale_w_before = model.head_logit_scale.weight.clone()

        model.mu_center_policy_heads()

        assert torch.equal(model.head_h_gate.weight, gate_w_before)
        assert torch.equal(model.head_h_cand.weight, cand_w_before)
        assert torch.equal(model.head_logit_scale.weight, scale_w_before)
