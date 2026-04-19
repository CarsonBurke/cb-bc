"""Targeted regression coverage for PPO-specific training paths."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))

from model import (
    ACTOR_LATENT_DIM,
    BRIDGE_BUILD_TYPE,
    BuilderMLP,
    CentralizedCritic,
    CriticAttentionBlock,
    CONVEYOR_BUILD_TYPES,
    GLOBAL_MAP_CHANNELS,
    GLOBAL_SCALAR_DIM,
    H_DIM,
    HARVESTER_BUILD_TYPE,
    NUM_BUCKETS,
    OBS_SIZE,
    RMSNorm,
    TEMP_MIN,
    TEMP_MAX,
    TILE_FEATURES,
    UNIT_CONTEXT_DIM,
    _MASK_BD_TILES,
    _OBS_AX,
    _OBS_SCALE,
    _OBS_TI,
    _build_bptt_chunks,
    _compute_gae,
    _make_bucket_centers,
    _normalize_advantages,
    _normalize_advantages_per_group,
    _normalize_actor_latent_dim,
    _parameter_golf_ortho_init,
    _policy_weight_from_advantages,
    _temperature_head_init_bias,
    _precompute_policy_masks,
    _recurrent_actor_pass,
    _recurrent_logprob_pass,
    _temperature_scaled_log_probs,
    _sequential_logprobs,
    _split_chunk_ids,
    action_log_prob_and_entropy,
    apply_rotary_2d,
    compute_action_masks,
    bd_has_ore,
    conveyor_dir_mask,
    decode_value,
    distributional_value_loss,
    exclusive_self_attention,
    residual_mix_input,
    soft_cross_entropy_from_logits,
    symexp,
    symlog,
    value_target_probs,
)
from normuon import (
    SingleDeviceNorMuonWithAuxAdam,
    split_normuon_params,
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
        bt_l, bd_l, cd_l, brt_l, md_l, head_temperature, h_next = model(obs, h_prev)

        bt_mask, bd_mask = compute_action_masks(obs)
        cd_mask = conveyor_dir_mask(device, 1)

        # Direction first, then condition build-type on ore.
        bd_act = int(bd_mask[0].float().argmax().item())
        bd_act_t = torch.tensor([bd_act], device=device, dtype=torch.long)
        has_ore = bool(bd_has_ore(obs, bd_act_t)[0])
        bt_mask = bt_mask.clone()
        if not has_ore:
            bt_mask[:, HARVESTER_BUILD_TYPE] = False

        if local_step % 3 == 0 and bool(bt_mask[0, HARVESTER_BUILD_TYPE]):
            bt_act = HARVESTER_BUILD_TYPE
        elif local_step % 3 == 1 and bool(bt_mask[0, CONVEYOR_BUILD_TYPES[0]]):
            bt_act = CONVEYOR_BUILD_TYPES[0]
        else:
            bt_act = 0

        cd_act = step_idx % 4
        brt_act = step_idx % 28
        md_act = (step_idx * 2) % 9

        bt_act_t = torch.tensor([bt_act], device=device, dtype=torch.long)
        cd_act_t = torch.tensor([cd_act], device=device, dtype=torch.long)
        brt_act_t = torch.tensor([brt_act], device=device, dtype=torch.long)
        md_act_t = torch.tensor([md_act], device=device, dtype=torch.long)

        bt_l = bt_l.masked_fill(~bt_mask, -1e9)
        bd_l = bd_l.masked_fill(~bd_mask, -1e9)
        cd_l = cd_l.masked_fill(~cd_mask, -1e9)

        bt_lp, _ = action_log_prob_and_entropy(bt_l, bt_act_t, bt_mask, head_temperature[:, 0])
        bd_lp, _ = action_log_prob_and_entropy(bd_l, bd_act_t, bd_mask, head_temperature[:, 1])
        cd_lp, _ = action_log_prob_and_entropy(cd_l, cd_act_t, cd_mask, head_temperature[:, 2])
        brt_lp, _ = action_log_prob_and_entropy(brt_l, brt_act_t, temperature=head_temperature[:, 3])
        md_lp, _ = action_log_prob_and_entropy(md_l, md_act_t, temperature=head_temperature[:, 4])
        cd_causal = 1.0 if bt_act in CONVEYOR_BUILD_TYPES else 0.0
        brt_causal = 1.0 if bt_act == BRIDGE_BUILD_TYPE else 0.0
        total_lp = bd_lp + bt_lp + cd_lp * cd_causal + brt_lp * brt_causal + md_lp

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

    def test_causal_masking_zeroes_bd_cd_for_no_build(self):
        """When bt_act=0 (no build), bd and cd log probs must not contribute."""
        torch.manual_seed(7)
        logits = torch.randn(1, 12)  # build type logits
        bd_logits = torch.randn(1, 9)
        cd_logits = torch.randn(1, 4)
        md_logits = torch.randn(1, 9)

        bt_act = torch.tensor([0])  # no build
        bd_act = torch.tensor([3])
        cd_act = torch.tensor([1])
        md_act = torch.tensor([2])

        bt_lp, _ = action_log_prob_and_entropy(logits, bt_act)
        bd_lp, _ = action_log_prob_and_entropy(bd_logits, bd_act)
        cd_lp, _ = action_log_prob_and_entropy(cd_logits, cd_act)
        md_lp, _ = action_log_prob_and_entropy(md_logits, md_act)

        # With no build: bd always included (upstream), cd zeroed
        cd_causal = 0.0  # bt_act not in CONVEYOR_BUILD_TYPES
        total = bd_lp + bt_lp + cd_lp * cd_causal + md_lp

        assert total.item() == pytest.approx((bd_lp + bt_lp + md_lp).item(), abs=1e-7)

    def test_causal_masking_includes_bd_for_harvester(self):
        """For harvester build, bd contributes but cd does not."""
        torch.manual_seed(7)
        logits = torch.randn(1, 12)
        bd_logits = torch.randn(1, 9)
        cd_logits = torch.randn(1, 4)
        md_logits = torch.randn(1, 9)

        bt_act_val = HARVESTER_BUILD_TYPE  # =1
        bt_act = torch.tensor([bt_act_val])
        bd_act = torch.tensor([3])
        cd_act = torch.tensor([1])
        md_act = torch.tensor([2])

        bt_lp, _ = action_log_prob_and_entropy(logits, bt_act)
        bd_lp, _ = action_log_prob_and_entropy(bd_logits, bd_act)
        cd_lp, _ = action_log_prob_and_entropy(cd_logits, cd_act)
        md_lp, _ = action_log_prob_and_entropy(md_logits, md_act)

        cd_causal = 0.0  # harvester not in CONVEYOR_BUILD_TYPES
        total = bd_lp + bt_lp + cd_lp * cd_causal + md_lp

        assert total.item() == pytest.approx((bd_lp + bt_lp + md_lp).item(), abs=1e-7)

    def test_causal_masking_includes_all_for_conveyor(self):
        """For conveyor build, all heads contribute."""
        torch.manual_seed(7)
        logits = torch.randn(1, 12)
        bd_logits = torch.randn(1, 9)
        cd_logits = torch.randn(1, 4)
        md_logits = torch.randn(1, 9)

        bt_act_val = CONVEYOR_BUILD_TYPES[0]  # conveyor
        bt_act = torch.tensor([bt_act_val])
        bd_act = torch.tensor([3])
        cd_act = torch.tensor([1])
        md_act = torch.tensor([2])

        bt_lp, _ = action_log_prob_and_entropy(logits, bt_act)
        bd_lp, _ = action_log_prob_and_entropy(bd_logits, bd_act)
        cd_lp, _ = action_log_prob_and_entropy(cd_logits, cd_act)
        md_lp, _ = action_log_prob_and_entropy(md_logits, md_act)

        cd_causal = 1.0  # conveyor type
        total = bd_lp + bt_lp + cd_lp * cd_causal + md_lp

        assert total.item() == pytest.approx((bd_lp + bt_lp + cd_lp + md_lp).item(), abs=1e-7)


# =====================================================================
# action_log_prob_and_entropy correctness
# =====================================================================

class TestActionLogProbAndEntropy:
    """Verify log_prob and entropy computation against manual formulas."""

    def test_log_prob_matches_manual(self):
        """Log prob should match manual temperature-scaled softmax + gather."""
        logits = torch.tensor([[2.0, 1.0, 0.5]])
        action = torch.tensor([1])

        lp, _ = action_log_prob_and_entropy(logits, action)
        probs = F.softmax(logits, dim=-1)
        expected = probs.log()[0, 1]
        assert lp.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_entropy_matches_manual(self):
        """Entropy should match -sum(p * log(p)) over the scaled distribution."""
        logits = torch.tensor([[2.0, 1.0, 0.5]])
        action = torch.tensor([0])

        _, ent = action_log_prob_and_entropy(logits, action)
        probs = F.softmax(logits, dim=-1)
        log_probs = probs.log()
        expected_ent = -(probs * log_probs).sum(dim=-1)
        assert ent.item() == pytest.approx(expected_ent.item(), abs=1e-6)

    def test_uniform_logits_max_entropy(self):
        """Uniform logits should produce maximum categorical entropy."""
        n = 9
        logits = torch.zeros(1, n)
        action = torch.tensor([0])

        _, ent = action_log_prob_and_entropy(logits, action)
        max_ent = torch.log(torch.tensor(float(n)))
        assert ent.item() == pytest.approx(max_ent.item(), abs=1e-5)

    def test_higher_temperature_increases_entropy(self):
        """A higher temperature should flatten the categorical distribution."""
        logits = torch.tensor([[4.0, 1.0, -1.0]])
        action = torch.tensor([1])

        lp_cold, ent_cold = action_log_prob_and_entropy(logits, action, temperature=TEMP_MIN)
        lp_hot, ent_hot = action_log_prob_and_entropy(logits, action, temperature=2.0)
        assert ent_hot.item() > ent_cold.item()
        assert lp_hot.item() > lp_cold.item()

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
        """Logits peaked at one bin should decode close to symexp(center)."""
        bins = _make_bucket_centers()
        for idx in [0, 50, 127, 200, 254]:
            logits = torch.full((1, bins.shape[0]), -100.0)
            logits[0, idx] = 100.0
            val = decode_value(logits, bins)
            expected = symexp(bins[idx:idx+1]).item()
            assert val.item() == pytest.approx(expected, rel=1e-3, abs=1e-6)

    def test_ghl_target_probs_sum_to_one(self):
        """GHL encoding must produce valid probability distributions."""
        bins = _make_bucket_centers()
        targets = torch.tensor([-50.0, 0.0, 0.5, 100.0, 5000.0])
        encoded = value_target_probs(targets, bins)
        sums = encoded.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_ghl_target_probs_nonnegative(self):
        """GHL probabilities must be non-negative."""
        bins = _make_bucket_centers()
        targets = torch.tensor([0.0, 1.0, -5.0, 1000.0])
        encoded = value_target_probs(targets, bins)
        assert (encoded >= -1e-7).all()

    def test_ghl_target_probs_stay_fp32_for_bf16_targets(self):
        """Target encoding should avoid bf16 precision loss under autocast."""
        bins = _make_bucket_centers()
        targets = torch.tensor([0.0, 0.5, 1.0, 5.0], dtype=torch.bfloat16)
        encoded = value_target_probs(targets, bins)
        assert encoded.dtype == torch.float32
        assert torch.allclose(encoded.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_ghl_target_probs_spread_across_bins(self):
        """GHL should spread mass across multiple bins (unlike two-hot)."""
        bins = _make_bucket_centers()
        target = torch.tensor([1.0])
        encoded = value_target_probs(target, bins)
        assert (encoded[0] > 0.001).sum() > 2

    def test_ghl_symmetric_for_symmetric_targets(self):
        """GHL probs for +x and -x should be mirror images."""
        bins = _make_bucket_centers()
        pos = value_target_probs(torch.tensor([5.0]), bins)
        neg = value_target_probs(torch.tensor([-5.0]), bins)
        assert torch.allclose(pos[0], neg[0].flip(0), atol=1e-5)


# =====================================================================
# Distributional value loss (GHL)
# =====================================================================

class TestDistributionalValueLoss:
    def test_equals_raw_cross_entropy(self):
        """Loss should equal cross-entropy of logits against GHL target."""
        bins = _make_bucket_centers()
        logits = torch.randn(4, bins.shape[0])
        returns = torch.tensor([1.0, -1.0, 0.0, 5.0])

        loss = distributional_value_loss(logits, returns, bins)
        target = value_target_probs(returns, bins)
        expected = soft_cross_entropy_from_logits(logits, target)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_value_loss_always_nonnegative(self):
        """Cross-entropy based value loss should be >= 0."""
        bins = _make_bucket_centers()
        torch.manual_seed(123)
        for _ in range(10):
            logits = torch.randn(8, bins.shape[0])
            returns = torch.randn(8) * 100
            loss = distributional_value_loss(logits, returns, bins)
            assert (loss >= -1e-6).all()

    def test_loss_decreases_when_logits_match_target(self):
        """Loss should be lower when logits align with the GHL target."""
        bins = _make_bucket_centers()
        returns = torch.tensor([3.0])
        target = value_target_probs(returns, bins)
        good_logits = target.clamp(min=1e-10).log()
        bad_logits = torch.randn(1, bins.shape[0])
        loss_good = distributional_value_loss(good_logits, returns, bins)
        loss_bad = distributional_value_loss(bad_logits, returns, bins)
        assert loss_good.item() < loss_bad.item()

    def test_clipped_value_loss_uses_pessimistic_branch(self):
        """PPO critic loss should not improve by moving far from old values in one step."""
        bins = _make_bucket_centers()
        returns = torch.tensor([5.0])
        old_values = torch.tensor([0.0])
        # A large jump toward the target should be clipped to 0.4 for the pessimistic branch.
        jumped_values = torch.tensor([5.0])
        target = value_target_probs(returns, bins)
        jumped_logits = target.clamp(min=1e-10).log()

        unclipped = distributional_value_loss(
            jumped_logits, returns, bins, values=jumped_values, old_values=old_values, clip_vloss=False,
        )
        clipped = distributional_value_loss(
            jumped_logits, returns, bins, values=jumped_values, old_values=old_values, clip_vloss=True,
        )

        assert clipped.item() > unclipped.item()


# =====================================================================
# GAE computation
# =====================================================================

class TestGAE:
    def test_long_episode_gae_tail_remains_terminal(self):
        rewards = torch.zeros(1900)
        values = torch.full((1900,), 5.0)
        advantages, returns = _compute_gae(
            rewards, values, [(0, 1900)], gamma=0.99, gae_lambda=1.0,
        )
        assert advantages[-1].item() == pytest.approx(-5.0)
        assert returns[-1].item() == pytest.approx(0.0)

    def test_single_step_episode(self):
        """A single-step episode: advantage = r - v (terminal, no bootstrap)."""
        rewards = torch.tensor([3.0])
        values = torch.tensor([1.0])
        advantages, returns = _compute_gae(
            rewards, values, [(0, 1)], gamma=0.99, gae_lambda=0.95,
        )
        # Terminal: delta = r + 0 - v = 3 - 1 = 2
        assert advantages[0].item() == pytest.approx(2.0)
        assert returns[0].item() == pytest.approx(3.0)  # adv + v = 2 + 1

    def test_two_step_episode_gae(self):
        """Manually verify GAE for a 2-step episode."""
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 1.0])
        gamma, lam = 0.99, 0.95

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
            rewards, values, [(0, 1), (1, 2)], gamma=0.99, gae_lambda=0.95,
        )
        # Each should be identical
        assert adv_multi[0].item() == pytest.approx(adv_multi[1].item())
        assert ret_multi[0].item() == pytest.approx(ret_multi[1].item())

        # Compare to single episode
        adv_single, ret_single = _compute_gae(
            rewards[:1], values[:1], [(0, 1)], gamma=0.99, gae_lambda=0.95,
        )
        assert adv_multi[0].item() == pytest.approx(adv_single[0].item())

    def test_episode_boundary_no_cross_bootstrap(self):
        """Value from next episode must not leak across episode boundaries."""
        # ep1: [r=0, v=0], ep2: [r=0, v=1000]
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.0, 1000.0])
        adv, _ = _compute_gae(
            rewards, values, [(0, 1), (1, 2)], gamma=0.99, gae_lambda=0.95,
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
            rewards, values, [(0, 3)], gamma=0.0, gae_lambda=0.95,
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


def test_project_policy_heads_keep_raw_categorical_logits():
    model = BuilderMLP().eval()
    latent = torch.zeros(1, ACTOR_LATENT_DIM)

    with torch.no_grad():
        for head in (
            model.head_build_type,
            model.head_build_dir,
            model.head_conv_dir,
            model.head_bridge_target,
            model.head_move_dir,
            model.head_temperature,
        ):
            head.weight.zero_()
            head.bias.zero_()
        model.head_build_type.bias[:3] = torch.tensor([100.0, -100.0, 0.0])
        model.head_build_dir.bias[:3] = torch.tensor([100.0, -100.0, 0.0])
        model.head_conv_dir.bias[:3] = torch.tensor([100.0, -100.0, 0.0])
        model.head_bridge_target.bias[:3] = torch.tensor([100.0, -100.0, 0.0])
        model.head_move_dir.bias[:3] = torch.tensor([100.0, -100.0, 0.0])

    bt_l, bd_l, cd_l, brt_l, md_l, _head_temperature = model._project_policy_heads(latent)

    for logits in (bt_l, bd_l, cd_l, brt_l, md_l):
        assert logits[0, 0].item() == pytest.approx(100.0, abs=1e-5)
        assert logits[0, 1].item() == pytest.approx(-100.0, abs=1e-5)
        assert logits[0, 2].item() == pytest.approx(0.0, abs=1e-7)


def test_mu_center_policy_heads_preserves_categorical_distributions():
    model = BuilderMLP().eval()
    latent = torch.randn(5, ACTOR_LATENT_DIM)

    with torch.no_grad():
        before = model._project_policy_heads(latent)[:5]
        before_probs = [torch.softmax(logits, dim=-1) for logits in before]
        model.mu_center_policy_heads()
        after = model._project_policy_heads(latent)[:5]
        after_probs = [torch.softmax(logits, dim=-1) for logits in after]

    for lhs, rhs in zip(before_probs, after_probs, strict=True):
        torch.testing.assert_close(lhs, rhs, rtol=1e-6, atol=1e-6)


def test_project_policy_heads_temperature_initializes_near_1p0():
    model = BuilderMLP().eval()
    latent = torch.randn(32, ACTOR_LATENT_DIM)

    with torch.no_grad():
        for head in (
            model.head_build_type,
            model.head_build_dir,
            model.head_conv_dir,
            model.head_bridge_target,
            model.head_move_dir,
        ):
            head.weight.zero_()
            head.bias.zero_()
        model.head_temperature.weight.zero_()
        model.head_temperature.bias.fill_(_temperature_head_init_bias())

    _bt_l, _bd_l, _cd_l, _brt_l, _md_l, head_temperature = model._project_policy_heads(latent)

    torch.testing.assert_close(
        head_temperature[0],
        torch.full((model.head_temperature.out_features,), 1.0000011, dtype=head_temperature.dtype),
        rtol=1e-5,
        atol=1e-5,
    )


def test_entropy_gradient_does_not_flow_into_temperature_when_detached():
    logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
    temperature = torch.tensor([1.5], requires_grad=True)
    actions = torch.tensor([0], dtype=torch.long)

    _chosen, entropy = action_log_prob_and_entropy(
        logits,
        actions,
        temperature=temperature,
        entropy_temperature=temperature.detach(),
    )
    entropy.sum().backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0
    assert temperature.grad is None or temperature.grad.abs().max().item() == pytest.approx(0.0, abs=1e-12)


# =====================================================================
# Critic architecture sanity
# =====================================================================

class TestCentralizedCritic:
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
        actions = torch.zeros(B, 5, dtype=torch.long)

        values = critic(map_tensor, valid_mask, scalars, unit_xy, unit_ctx, actor_latent, actions)
        assert values.shape == (B,)

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
        actions = torch.zeros(B, 5, dtype=torch.long)

        values, logits = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent, actions,
        )
        assert values.shape == (B,)
        assert logits.shape == (B, NUM_BUCKETS)
        # Values should match decode_value applied to logits
        expected = decode_value(logits, critic.bucket_centers)
        assert torch.allclose(values, expected, atol=1e-5)

    def test_critic_starts_with_zero_logits(self):
        """Critic value head starts at zero logits by design for stable bootstrapping."""
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
        actions = torch.zeros(B, 5, dtype=torch.long)

        values, logits = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent, actions,
        )
        assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-7)
        assert values.abs().max().item() <= 1e-7

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
        actions = torch.zeros(B, 5, dtype=torch.long)

        feat = critic.encode_map(map_tensor, valid_mask)
        round_idx = torch.arange(B)
        values, logits = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent, actions,
        )
        returns = torch.tensor([1.0, -1.0], dtype=torch.float32)
        loss = distributional_value_loss(
            logits, returns, critic.bucket_centers,
        ).sum()
        loss.backward()

        has_grad = False
        for p in critic.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through critic"

        last_weight_grad = critic.value_head.mlp[-1].weight.grad
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
        assert (feat.patches == 0).all(), "Fully masked map should produce zero patch features"

    def test_attention_block_applies_peri_norm_to_both_residual_branches(self):
        """Critic residual branches apply branch-side norms before the residual add."""
        block = CriticAttentionBlock(dim=8, n_heads=2, n_kv_heads=1, mlp_ratio=2).eval()
        with torch.no_grad():
            for layer in (block.q_proj, block.k_proj, block.v_proj, block.mlp_fc1, block.mlp_fc2):
                layer.weight.zero_()
            block.out_proj.weight.copy_(torch.eye(8))
            for norm in (
                block.attn_input_norm,
                block.attn_output_norm,
                block.mlp_input_norm,
                block.mlp_output_norm,
                block.q_norm,
                block.k_norm,
            ):
                norm.weight.fill_(1.0)

        x = torch.randn(2, 4, 8)
        attn_out = torch.randn_like(x)
        out = block.apply_residual_mlp(x, attn_out)
        expected = x + block.attn_output_norm(attn_out)
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_exclusive_self_attention_removes_self_value_component(self):
        attn_out = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        self_value = torch.tensor([[[0.0, 0.0, 4.0]]], dtype=torch.float32)

        out = exclusive_self_attention(attn_out, self_value)

        torch.testing.assert_close(
            out,
            torch.tensor([[[1.0, 2.0, 0.0]]], dtype=torch.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_residual_mix_input_blends_running_state_with_anchor(self):
        x = torch.tensor([[[2.0, 4.0]]], dtype=torch.float32)
        x0 = torch.tensor([[[10.0, 20.0]]], dtype=torch.float32)
        mix = torch.tensor([[0.25, 0.5], [0.75, 0.5]], dtype=torch.float32)

        out = residual_mix_input(x, x0, mix)

        torch.testing.assert_close(
            out,
            torch.tensor([[[8.0, 12.0]]], dtype=torch.float32),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_partial_rotary_preserves_unrotated_tail(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 9.0, 8.0]]]] , dtype=torch.float32)
        cos = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        sin = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        out = apply_rotary_2d(x, cos, sin)

        torch.testing.assert_close(out[..., 4:], x[..., 4:], atol=1e-6, rtol=1e-6)

    def test_rmsnorm_cuda_bf16_matches_fp32_reference(self):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            pytest.skip("requires CUDA bf16")

        norm = RMSNorm(32).cuda().eval()
        x = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16) * 100

        with torch.no_grad():
            actual = norm(x)
            expected = F.rms_norm(
                x.float(),
                (32,),
                norm.weight.float(),
                norm.eps,
            ).to(dtype=torch.bfloat16)

        assert torch.isfinite(actual).all()
        assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)

    def test_parameter_golf_ortho_init_gates_large_matrices_and_scales_projections(self):
        torch.manual_seed(0)
        large_proj = torch.nn.Linear(128, 128, bias=False)
        small_rect = torch.nn.Linear(32, 64, bias=False)

        _parameter_golf_ortho_init(large_proj, n_residual_layers=8, scale_proj=True)
        _parameter_golf_ortho_init(small_rect)

        large_expected = (1.0 / math.sqrt(128.0)) * (1.0 / math.sqrt(16.0))
        small_expected = math.sqrt(2.0 / (32.0 + 64.0))

        assert large_proj.weight.std().item() == pytest.approx(large_expected, rel=0.2)
        assert small_rect.weight.std().item() == pytest.approx(small_expected, rel=0.25)

    def test_critic_transformer_linears_follow_parameter_golf_large_matrix_gate(self):
        torch.manual_seed(0)
        critic = CentralizedCritic().eval()

        weights = [
            (critic.windowed_sa[0].attn.q_proj.weight, math.sqrt(2.0 / (32.0 + 32.0))),
            (critic.windowed_sa[0].attn.out_proj.weight, math.sqrt(2.0 / (32.0 + 32.0))),
            (critic.value_head.mlp[0].weight, 1.0 / math.sqrt(128.0)),
            (critic.scalar_proj[0].weight, math.sqrt(2.0 / (GLOBAL_SCALAR_DIM + 32.0))),
            (critic.global_cls, math.sqrt(2.0 / (critic.global_cls.shape[0] + critic.global_cls.shape[1]))),
        ]

        for weight, expected_std in weights:
            assert weight.std().item() == pytest.approx(expected_std, rel=0.3)

    def test_actor_attention_and_ffn_linears_follow_parameter_golf_large_matrix_gate(self):
        torch.manual_seed(0)
        actor = BuilderMLP().eval()

        expected = [
            math.sqrt(2.0 / (actor.tile_sa_q_proj.in_features + actor.tile_sa_q_proj.out_features)),
            math.sqrt(2.0 / (actor.cls_ca_q_proj.in_features + actor.cls_ca_q_proj.out_features)),
            math.sqrt(2.0 / (actor.latent_ffn_up.in_features + actor.latent_ffn_up.out_features)),
            math.sqrt(2.0 / (actor.latent_ffn_down.in_features + actor.latent_ffn_down.out_features)),
        ]

        actual_weights = [
            actor.tile_sa_q_proj.weight,
            actor.cls_ca_q_proj.weight,
            actor.latent_ffn_up.weight,
            actor.latent_ffn_down.weight,
        ]

        for weight, expected_std in zip(actual_weights, expected, strict=True):
            assert weight.std().item() == pytest.approx(expected_std, rel=0.3)

    def test_normuon_split_keeps_actor_heads_on_aux_adam(self):
        actor = BuilderMLP().eval()
        muon_named, aux_named = split_normuon_params(actor.named_parameters())
        muon_names = {name for name, _ in muon_named}
        aux_names = {name for name, _ in aux_named}

        assert "tile_sa_q_proj.weight" in muon_names
        assert "latent_ffn_up.weight" in muon_names
        assert "head_build_type.weight" in aux_names
        assert "head_h_gate.weight" in aux_names
        assert "actor_cls" in aux_names
        assert "context_tau" in aux_names

    def test_normuon_split_keeps_critic_output_head_on_aux_adam(self):
        critic = CentralizedCritic().eval()
        muon_named, aux_named = split_normuon_params(critic.named_parameters())
        muon_names = {name for name, _ in muon_named}
        aux_names = {name for name, _ in aux_named}

        assert "windowed_sa.0.attn.q_proj.weight" in muon_names
        assert "value_head.mlp.0.weight" in muon_names
        assert "value_head.mlp.4.weight" in aux_names
        assert "global_cls" in aux_names
        assert "value_head.cls_weights" in aux_names

    def test_normuon_optimizer_step_updates_muon_and_aux_parameters(self):
        matrix = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32))
        scalar = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        optimizer = SingleDeviceNorMuonWithAuxAdam([
            {"params": [matrix], "lr": 1e-2, "use_muon": True},
            {"params": [scalar], "lr": 1e-2, "use_muon": False},
        ])
        matrix.grad = torch.tensor([[0.1, -0.2], [0.3, -0.4]], dtype=torch.float32)
        scalar.grad = torch.tensor([0.25], dtype=torch.float32)

        before_matrix = matrix.detach().clone()
        before_scalar = scalar.detach().clone()
        optimizer.step()

        assert torch.isfinite(matrix).all()
        assert torch.isfinite(scalar).all()
        assert not torch.allclose(matrix, before_matrix)
        assert not torch.allclose(scalar, before_scalar)

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
        assert not torch.allclose(feat_a.patches, feat_b.patches, atol=1e-3), \
            "Different maps should produce different encoded features"

    def test_unit_value_logits_are_order_equivariant_within_round(self):
        """Reordering unit tokens should only reorder outputs, not change them."""
        torch.manual_seed(0)
        critic = CentralizedCritic().eval()
        R = 3
        H, W = 10, 10
        map_tensor = torch.randn(R, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.ones(R, 1, H, W)
        feat = critic.encode_map(map_tensor, valid_mask)
        scalars = torch.randn(R, GLOBAL_SCALAR_DIM)
        round_idx = torch.tensor([0, 1, 1, 2, 0, 2, 1], dtype=torch.long)
        unit_xy = torch.randint(0, 10, (round_idx.shape[0], 2))
        unit_ctx = torch.randn(round_idx.shape[0], UNIT_CONTEXT_DIM)
        actor_latent = torch.randn(round_idx.shape[0], ACTOR_LATENT_DIM)
        actions = torch.zeros(round_idx.shape[0], 5, dtype=torch.long)

        values_a, logits_a = critic.unit_value_logits(
            feat, scalars, round_idx, unit_xy, unit_ctx, actor_latent, actions,
        )

        perm = torch.tensor([4, 2, 6, 0, 5, 1, 3], dtype=torch.long)
        values_b, logits_b = critic.unit_value_logits(
            feat,
            scalars,
            round_idx[perm],
            unit_xy[perm],
            unit_ctx[perm],
            actor_latent[perm],
            actions[perm],
        )
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel())

        assert torch.allclose(values_a, values_b[inv_perm], atol=1e-5)
        assert torch.allclose(logits_a, logits_b[inv_perm], atol=1e-5)


# =====================================================================
# Advantage normalization
# =====================================================================

class TestAdvantageNormalization:
    def test_normalized_advantages_zero_mean_unit_var(self):
        """After normalization, advantages should have ~0 mean and ~1 std."""
        adv = torch.randn(1000) * 10 + 5  # non-zero mean, large std
        norm_adv = _normalize_advantages(adv)
        assert norm_adv.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert norm_adv.std().item() == pytest.approx(1.0, abs=1e-2)

    def test_constant_advantages_no_nan(self):
        """All-equal advantages shouldn't produce NaN after normalization."""
        adv = torch.full((100,), 3.0)
        norm_adv = _normalize_advantages(adv)
        assert not torch.isnan(norm_adv).any()
        # Should be all ~0
        assert norm_adv.abs().max().item() < 1e-4

    def test_pmpo_policy_weight_uses_advantage_sign(self):
        adv = torch.tensor([-3.0, -0.2, 0.0, 0.7, 5.0])
        weights = _policy_weight_from_advantages(adv, use_pmpo=True)
        assert torch.equal(weights, torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]))

    def test_plain_ppo_policy_weight_keeps_advantage_magnitude(self):
        adv = torch.tensor([-3.0, -0.2, 0.0, 0.7, 5.0])
        weights = _policy_weight_from_advantages(adv, use_pmpo=False)
        assert torch.equal(weights, adv)


class TestPerGroupAdvantageNormalization:
    def test_zero_mean_unit_var_within_each_group(self):
        torch.manual_seed(0)
        g1 = torch.randn(32) * 3.0 + 10.0
        g2 = torch.randn(32) * 0.5 - 4.0
        adv = torch.cat([g1, g2])
        gid = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)])
        out = _normalize_advantages_per_group(adv, gid, min_group_size=8)
        for g in (0, 1):
            sel = gid == g
            assert out[sel].mean().abs().item() < 1e-5
            assert abs(out[sel].std(unbiased=False).item() - 1.0) < 1e-3

    def test_shared_reward_component_cancels(self):
        """For transitions sharing a scalar reward, per-group norm must retain
        the per-unit V-differential while the shared reward cancels out."""
        # Simulate A_i = G - V(s_i): G shared within group, V varies per unit.
        V = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        G_a, G_b = 100.0, -50.0
        adv_a = G_a - V
        adv_b = G_b - V
        adv = torch.cat([adv_a, adv_b])
        gid = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)])
        out = _normalize_advantages_per_group(adv, gid, min_group_size=8)
        # After per-group norm, both groups should produce the SAME normalized
        # advantages (the G term cancels). Highest-V unit has the lowest
        # normalized advantage in its group.
        out_a, out_b = out[:8], out[8:]
        assert torch.allclose(out_a, out_b, atol=1e-5)
        assert out_a.argmin().item() == V.argmax().item()
        assert out_a.argmax().item() == V.argmin().item()

    def test_global_norm_crushes_per_unit_differential(self):
        """Without per-group norm, a big shared-reward swing between groups
        crushes the per-unit signal; per-group norm must restore it."""
        V = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        adv_win = 100.0 - V    # team won this game, big positive
        adv_loss = -100.0 - V  # team lost this game, big negative
        adv = torch.cat([adv_win, adv_loss])
        gid = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)])

        global_norm = _normalize_advantages(adv)
        per_group = _normalize_advantages_per_group(adv, gid, min_group_size=8)

        # Within the winning group, the spread of per-unit advantages after
        # global norm is dominated by cross-game variance, so the per-unit
        # range collapses relative to the per-group regime.
        gw_global = global_norm[:8].max() - global_norm[:8].min()
        gw_per_group = per_group[:8].max() - per_group[:8].min()
        assert gw_per_group.item() > 10.0 * gw_global.item()

    def test_small_group_falls_back_when_no_large_groups(self):
        """When no group meets the threshold, fallback degenerates to
        whole-batch stats so no transition produces NaN or 0."""
        adv = torch.tensor([5.0, 5.0, 5.0, -5.0, -5.0, -5.0, 100.0])
        gid = torch.tensor([0, 0, 0, 1, 1, 1, 2], dtype=torch.long)
        out = _normalize_advantages_per_group(adv, gid, min_group_size=4)
        assert torch.isfinite(out).all()
        expected = (adv[6] - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        assert abs(out[6].item() - expected.item()) < 1e-5

    def test_constant_within_group_no_nan(self):
        adv = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        gid = torch.zeros(8, dtype=torch.long)
        out = _normalize_advantages_per_group(adv, gid, min_group_size=4)
        assert not torch.isnan(out).any()
        assert out.abs().max().item() < 1e-3

    def test_bf16_large_mean_small_variance_stable(self):
        """bf16 advantages with large mean relative to std must not explode
        after normalization — reductions run in fp32 internally."""
        torch.manual_seed(0)
        adv_fp32 = torch.randn(128) * 3.0 + 100.0
        adv_bf16 = adv_fp32.to(torch.bfloat16)
        gid = torch.zeros(128, dtype=torch.long)
        out = _normalize_advantages_per_group(adv_bf16, gid, min_group_size=8)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()
        # Post-normalization spread should still be ~O(1), not ~O(1e4).
        assert out.float().abs().max().item() < 20.0

    def test_bf16_counts_dont_saturate_on_large_group(self):
        """Internal counts must be fp32, not bf16 (which saturates above 256)."""
        adv = torch.zeros(512, dtype=torch.bfloat16)
        adv[0] = 1.0
        adv[-1] = -1.0
        gid = torch.zeros(512, dtype=torch.long)
        out = _normalize_advantages_per_group(adv, gid, min_group_size=8)
        # If counts saturated at 256 instead of 512, the divisor would be
        # halved and the output scale would be off. Mean-zero invariant must
        # hold exactly regardless of group size.
        assert abs(out.float().mean().item()) < 1e-2

    def test_fallback_excludes_small_group_outliers(self):
        """Small-group fallback stats must be computed over large groups only,
        so a small-group outlier doesn't inflate the fallback denominator."""
        # Large group: 16 mild values centered on 0.
        big = torch.linspace(-1.0, 1.0, 16)
        # Small group: 2 wild outliers that would dominate whole-batch std.
        small = torch.tensor([1000.0, -1000.0])
        adv = torch.cat([big, small])
        gid = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(2, dtype=torch.long)])
        out = _normalize_advantages_per_group(adv, gid, min_group_size=8)
        # The small-group transitions are normalized with the large-group std
        # (~0.6), so they should end up at ~±1600σ, not ~±1σ (which would be
        # the whole-batch std case).
        assert out[16].item() > 500.0
        assert out[17].item() < -500.0

    def test_device_mismatch_raises(self):
        adv = torch.zeros(8)
        gid = torch.zeros(8, dtype=torch.long)
        if torch.cuda.is_available():
            with pytest.raises(ValueError, match="device"):
                _normalize_advantages_per_group(adv, gid.cuda(), min_group_size=4)

    def test_shape_mismatch_raises(self):
        adv = torch.zeros(8)
        gid = torch.zeros(7, dtype=torch.long)
        with pytest.raises(ValueError, match="shape"):
            _normalize_advantages_per_group(adv, gid, min_group_size=4)


# =====================================================================
# Temperature scaling
# =====================================================================

class TestTemperatureScaledLogProbs:
    """Verify masked temperature-scaled categorical helpers."""

    def test_masked_temperature_scaled_log_probs_zero_invalid(self):
        """Masked actions should get zero probability mass."""
        logits = torch.randn(2, 5)
        mask = torch.tensor([[True, True, False, False, False],
                             [True, True, True, True, True]])
        logits[~mask] = -1e9
        log_probs = _temperature_scaled_log_probs(logits, mask)
        assert log_probs[0, 2].item() == float('-inf')
        assert log_probs[0, 3].item() == float('-inf')
        assert log_probs[0, 0].isfinite()
        assert log_probs[1, 4].isfinite()

    def test_masked_temperature_matches_valid_support_softmax(self):
        """Masked logits should normalize only over the valid support."""
        logits = torch.tensor([[3.0, -7.0, 1.0, -9.0]])
        mask = torch.tensor([[True, False, True, False]])

        log_probs = _temperature_scaled_log_probs(logits, mask, temperature=1.0)
        probs = log_probs.exp()

        valid_logits = logits[0, mask[0]]
        expected_valid = torch.softmax(valid_logits, dim=-1)

        assert probs[0, 1].item() == pytest.approx(0.0, abs=1e-12)
        assert probs[0, 3].item() == pytest.approx(0.0, abs=1e-12)
        assert probs[0, mask[0]].tolist() == pytest.approx(expected_valid.tolist(), abs=1e-7)

    def test_higher_temperature_flattens_distribution(self):
        """Higher temperature should move mass away from the argmax action."""
        logits = torch.tensor([[3.0, 1.0, -1.0]])
        cold = _temperature_scaled_log_probs(logits, temperature=TEMP_MIN).exp()
        hot = _temperature_scaled_log_probs(logits, temperature=2.0).exp()
        assert hot[0, 0].item() < cold[0, 0].item()
        assert hot[0, 2].item() > cold[0, 2].item()

    def test_masked_temperature_logprob_backward_stays_finite(self):
        """Masked-out zero-prob actions must not create NaN gradients in backward."""
        logits = torch.tensor([[3.0, -1e9, 1.0, -1e9]], requires_grad=True)
        mask = torch.tensor([[True, False, True, False]])
        action = torch.tensor([0])

        log_probs = _temperature_scaled_log_probs(logits, mask)
        loss = -log_probs.gather(1, action.unsqueeze(1)).mean()
        loss.backward()

        assert torch.isfinite(logits.grad).all()
        assert logits.grad[0, 1].item() == pytest.approx(0.0, abs=1e-12)
        assert logits.grad[0, 3].item() == pytest.approx(0.0, abs=1e-12)


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

        bt_mask, bd_mask, cd_causal = _precompute_policy_masks(obs, bt, bd)

        # Reshape for recurrent pass: (B=1, K=6, ...)
        obs_pre = model._obs_encoder_pre(obs).unsqueeze(0)
        bt_a = bt.unsqueeze(0)
        bd_a = bd.unsqueeze(0)
        cd_a = cd.unsqueeze(0)
        brt_a = brt.unsqueeze(0)
        md_a = md.unsqueeze(0)
        bt_m = bt_mask.unsqueeze(0)
        bd_m = bd_mask.unsqueeze(0)
        cd_c = cd_causal.unsqueeze(0)

        with torch.no_grad():
            lp_only = _recurrent_logprob_pass(
                model, obs_pre, h0, bt_a, bd_a, cd_a, brt_a, md_a, bt_m, bd_m, cd_c,
            )
            lp_actor, ent, _head_ents, _head_w, _eps = _recurrent_actor_pass(
                model, obs_pre, h0, bt_a, bd_a, cd_a, brt_a, md_a, bt_m, bd_m, cd_c,
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

        bt_mask, bd_mask, cd_causal = _precompute_policy_masks(obs, bt, bd)
        obs_pre = model._obs_encoder_pre(obs).unsqueeze(0)

        with torch.no_grad():
            _, ent, _head_ents, _head_w, _eps = _recurrent_actor_pass(
                model, obs_pre, h0,
                bt.unsqueeze(0), bd.unsqueeze(0), cd.unsqueeze(0),
                brt.unsqueeze(0), md.unsqueeze(0),
                bt_mask.unsqueeze(0), bd_mask.unsqueeze(0),
                cd_causal.unsqueeze(0),
            )

        assert (ent >= -1e-6).all(), f"Negative entropy found: {ent.min()}"


# =====================================================================
# _precompute_policy_masks
# =====================================================================

class TestPrecomputePolicyMasks:
    def test_cd_causal_zero_for_no_build(self):
        """cd_causal should be 0 when bt_act is not a conveyor type."""
        obs = _make_obs(0).unsqueeze(0)
        bt = torch.tensor([0])
        _, _, cd_causal = _precompute_policy_masks(obs, bt, torch.tensor([0]))
        assert cd_causal[0].item() == 0.0

    def test_cd_causal_zero_for_harvester(self):
        obs = _make_obs(0).unsqueeze(0)
        bt = torch.tensor([HARVESTER_BUILD_TYPE])
        _, _, cd_causal = _precompute_policy_masks(obs, bt, torch.tensor([0]))
        assert cd_causal[0].item() == 0.0  # harvester is not conveyor

    def test_cd_causal_one_for_conveyor(self):
        obs = _make_obs(0).unsqueeze(0)
        for cbt in CONVEYOR_BUILD_TYPES:
            bt = torch.tensor([cbt])
            _, _, cd_causal = _precompute_policy_masks(obs, bt, torch.tensor([0]))
            assert cd_causal[0].item() == 1.0, f"cd_causal should be 1 for conveyor type {cbt}"


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

        bt_mask, bd_mask, cd_causal = _precompute_policy_masks(obs, bt, bd)
        obs_pre = model._obs_encoder_pre(obs).unsqueeze(0)

        new_lps, ent, _head_ents, _head_w, _eps = _recurrent_actor_pass(
            model, obs_pre, h0,
            bt.unsqueeze(0), bd.unsqueeze(0), cd.unsqueeze(0),
            brt.unsqueeze(0), md.unsqueeze(0),
            bt_mask.unsqueeze(0), bd_mask.unsqueeze(0),
            cd_causal.unsqueeze(0),
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
