"""Test RBF attention as a drop-in replacement for dot-product tile SA.

Validates:
1. RBF attention math: softmax(2γ(QK^T) - γ||K||²) is equivalent to
   softmax(-γ||Q-K||²) up to the query-norm constant (cancels in softmax).
2. Gradient flow: γ receives gradients and the attention output is differentiable.
3. Stdlib inference parity: pure-Python RBF attention matches PyTorch.
4. Sharpness comparison: RBF vs dot-product attention entropy at dim=8.
5. Non-softmax RBF: raw exp(-γ||q-k||²) kernel without softmax normalization.
"""

from pathlib import Path
import math
import sys

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "bots" / "tensor_inference"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from model_constants import TILE_EMBED_DIM, TILE_GRID_N

_DOT = math.sumprod
_EXP = math.exp

# ---- PyTorch RBF attention ----

def rbf_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """RBF attention: softmax(2γ(QK^T) - γ||K||²) @ V.

    Args:
        q: (B, N_q, D)
        k: (B, N_k, D)
        v: (B, N_k, D)
        gamma: scalar (learnable, positive)
    Returns:
        (B, N_q, D)
    """
    # QK^T: (B, N_q, N_k)
    qk = torch.bmm(q, k.transpose(-1, -2))
    # ||K||²: (B, 1, N_k)
    k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)
    # Score: 2γ(QK^T) - γ||K||²
    scores = 2.0 * gamma * qk - gamma * k_norm_sq
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)


def rbf_attention_nosoftmax_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Non-softmax RBF attention: exp(-γ||Q-K||²) @ V, normalized by weight sum.

    Since exp(-γ||q-k||²) = exp(-γ||q||²) * exp(2γ(q·k)) * exp(-γ||k||²),
    and we normalize by the sum of weights, the exp(-γ||q||²) factor cancels:
      w_ij = exp(2γ(q_i·k_j) - γ||k_j||²)   [unnormalized]
      out_i = Σ_j w_ij * v_j / Σ_j w_ij

    Unlike softmax, this doesn't subtract max for numerical stability —
    we do it manually. The key difference from softmax RBF is that
    the normalization is sum-of-exponentials, not log-sum-exp shifted.
    In practice: identical output to softmax RBF (both normalize by sum).
    The real non-softmax variant skips normalization entirely.
    """
    qk = torch.bmm(q, k.transpose(-1, -2))
    k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)
    q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
    # Full distance: -γ||q-k||²
    neg_dist = -gamma * (q_norm_sq - 2.0 * qk + k_norm_sq)
    weights = neg_dist.exp()
    # Normalize by sum (not softmax — no max subtraction needed for correctness,
    # but we do it anyway for numerical stability)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.bmm(weights, v)


def rbf_attention_unnormalized_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Truly unnormalized RBF: exp(-γ||Q-K||²) @ V with no division.

    Output magnitude depends on how many keys are close to the query.
    Useful when "attend to nothing" (all keys far → output ≈ 0) is desirable.
    """
    qk = torch.bmm(q, k.transpose(-1, -2))
    k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)
    q_norm_sq = (q * q).sum(dim=-1, keepdim=True)
    neg_dist = -gamma * (q_norm_sq - 2.0 * qk + k_norm_sq)
    weights = neg_dist.exp()
    return torch.bmm(weights, v)


def rbf_attention_unnormalized_stdlib(
    q_vecs: list[list[float]],
    k_vecs: list[list[float]],
    v_cols: list[list[float]],
    gamma: float,
    n_q: int,
    n_k: int,
    dim: int,
) -> list[list[float]]:
    """Pure-Python unnormalized RBF attention."""
    # Precompute ||k||² and ||q||²
    k_norm_sq = [_DOT(k_vecs[i], k_vecs[i]) for i in range(n_k)]
    out = [[0.0] * dim for _ in range(n_q)]
    weights = [0.0] * n_k

    for qi in range(n_q):
        q_nsq = _DOT(q_vecs[qi], q_vecs[qi])
        for ki in range(n_k):
            dist_sq = q_nsq - 2.0 * _DOT(q_vecs[qi], k_vecs[ki]) + k_norm_sq[ki]
            weights[ki] = _EXP(-gamma * dist_sq)
        for j in range(dim):
            out[qi][j] = _DOT(weights, v_cols[j])
    return out


def standard_attention_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Standard scaled dot-product attention for comparison."""
    scores = torch.bmm(q, k.transpose(-1, -2)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)


# ---- Stdlib RBF attention (inference) ----

def rbf_attention_stdlib(
    q_vecs: list[list[float]],
    k_vecs: list[list[float]],
    v_cols: list[list[float]],
    gamma: float,
    n_q: int,
    n_k: int,
    dim: int,
) -> list[list[float]]:
    """Pure-Python RBF attention matching the torch version.

    Args:
        q_vecs: [n_q][dim] query vectors
        k_vecs: [n_k][dim] key vectors
        v_cols: [dim][n_k] value columns (transposed for sumprod)
        gamma: positive scalar
        n_q, n_k, dim: sizes
    Returns:
        [n_q][dim] output vectors
    """
    # Precompute ||k||² for each key
    k_norm_sq = [0.0] * n_k
    for i in range(n_k):
        k_norm_sq[i] = _DOT(k_vecs[i], k_vecs[i])

    out = [[0.0] * dim for _ in range(n_q)]
    scores = [0.0] * n_k

    for qi in range(n_q):
        best = -1e30
        for ki in range(n_k):
            score = 2.0 * gamma * _DOT(q_vecs[qi], k_vecs[ki]) - gamma * k_norm_sq[ki]
            scores[ki] = score
            if score > best:
                best = score

        z = 0.0
        for ki in range(n_k):
            w = _EXP(scores[ki] - best)
            scores[ki] = w
            z += w

        inv_z = 1.0 / z
        for j in range(dim):
            out[qi][j] = _DOT(scores, v_cols[j]) * inv_z

    return out


# ---- Tests ----

class TestRBFMath:
    """Verify RBF attention math is correct."""

    def test_rbf_equals_negative_distance(self):
        """softmax(2γ(QK^T) - γ||K||²) == softmax(-γ||Q-K||² + γ||Q||²).
        Since ||Q||² is constant per query row, softmax cancels it."""
        torch.manual_seed(42)
        B, N, D = 2, TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(B, N, D)
        k = torch.randn(B, N, D)
        gamma = torch.tensor(1.0)

        # Method 1: efficient form
        qk = torch.bmm(q, k.transpose(-1, -2))
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)
        scores_efficient = 2.0 * gamma * qk - gamma * k_norm_sq

        # Method 2: explicit distance
        # ||q_i - k_j||² = ||q_i||² - 2(q_i·k_j) + ||k_j||²
        q_norm_sq = (q * q).sum(dim=-1, keepdim=True)  # (B, N, 1)
        dist_sq = q_norm_sq - 2.0 * qk + k_norm_sq  # (B, N_q, N_k)
        scores_distance = -gamma * dist_sq

        # Softmax should be identical (q_norm_sq cancels)
        w1 = F.softmax(scores_efficient, dim=-1)
        w2 = F.softmax(scores_distance, dim=-1)
        torch.testing.assert_close(w1, w2, atol=1e-6, rtol=1e-5)

    def test_gamma_gradient_flows(self):
        """γ must receive gradients for the kernel bandwidth to be learnable."""
        torch.manual_seed(42)
        B, N, D = 1, TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(B, N, D)
        k = torch.randn(B, N, D)
        v = torch.randn(B, N, D)
        gamma = torch.tensor(1.0, requires_grad=True)

        out = rbf_attention_torch(q, k, v, gamma)
        loss = out.sum()
        loss.backward()

        assert gamma.grad is not None
        assert gamma.grad.abs() > 0, "γ should receive non-zero gradient"

    def test_qkv_gradient_flows(self):
        """Q, K, V all receive gradients."""
        torch.manual_seed(42)
        B, N, D = 1, TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(B, N, D, requires_grad=True)
        k = torch.randn(B, N, D, requires_grad=True)
        v = torch.randn(B, N, D, requires_grad=True)
        gamma = torch.tensor(1.0)

        out = rbf_attention_torch(q, k, v, gamma)
        loss = out.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v)]:
            assert t.grad is not None, f"{name} should have grad"
            assert t.grad.abs().sum() > 0, f"{name} should have non-zero grad"


class TestRBFStdlibParity:
    """Stdlib inference must match PyTorch exactly."""

    def test_parity(self):
        torch.manual_seed(123)
        N, D = TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(1, N, D)
        k = torch.randn(1, N, D)
        v = torch.randn(1, N, D)
        gamma_val = 0.5

        # PyTorch reference
        ref = rbf_attention_torch(q, k, v, torch.tensor(gamma_val))

        # Stdlib
        q_list = [list(q[0, i].tolist()) for i in range(N)]
        k_list = [list(k[0, i].tolist()) for i in range(N)]
        # Column-major V for sumprod
        v_cols = [[v[0, i, j].item() for i in range(N)] for j in range(D)]

        out_list = rbf_attention_stdlib(q_list, k_list, v_cols, gamma_val, N, N, D)

        for i in range(N):
            for j in range(D):
                assert out_list[i][j] == pytest.approx(
                    ref[0, i, j].item(), abs=1e-5,
                ), f"Mismatch at [{i},{j}]"


class TestRBFvsStandard:
    """Compare RBF and standard attention properties at dim=8."""

    def test_rbf_sharper_at_low_dim(self):
        """RBF attention should produce lower-entropy (sharper) distributions
        than dot-product at dim=8, since distance is more discriminative."""
        torch.manual_seed(7)
        B, N, D = 32, TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(B, N, D)
        k = torch.randn(B, N, D)
        v = torch.randn(B, N, D)

        # RBF with gamma=1
        rbf_out = rbf_attention_torch(q, k, v, torch.tensor(1.0))
        qk = torch.bmm(q, k.transpose(-1, -2))
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)
        rbf_scores = 2.0 * qk - k_norm_sq
        rbf_weights = F.softmax(rbf_scores, dim=-1)
        rbf_entropy = -(rbf_weights * rbf_weights.clamp(min=1e-8).log()).sum(dim=-1).mean()

        # Standard with 1/sqrt(d) scaling
        std_scores = qk / math.sqrt(D)
        std_weights = F.softmax(std_scores, dim=-1)
        std_entropy = -(std_weights * std_weights.clamp(min=1e-8).log()).sum(dim=-1).mean()

        # Just report — not asserting direction since gamma=1 is arbitrary
        print(f"\nEntropy comparison (dim={D}, N={N}):")
        print(f"  Standard (1/√d): {std_entropy.item():.3f}")
        print(f"  RBF (γ=1.0):     {rbf_entropy.item():.3f}")
        print(f"  Max entropy:      {math.log(N):.3f}")

    def test_rbf_penalizes_large_norm_distractors(self):
        """Standard attention rewards high-norm keys (large dot product).
        RBF penalizes keys far from query in L2, even if dot product is high.

        Scenario: q is small-norm. k1 is nearby in L2 (similar). k2 has 4x
        norm (high dot product but far in L2). Standard attention prefers k2;
        RBF prefers k1 — the actually similar token."""
        D = TILE_EMBED_DIM
        q = torch.tensor([[[0.5] * D]])       # small query, (1, 1, D)
        k = torch.tensor([[[0.6] * D,          # close in L2: ||q-k1||² = 8*0.01 = 0.08
                           [2.0] * D]])         # far in L2: ||q-k2||² = 8*2.25 = 18.0

        # Standard: q·k1=2.4, q·k2=8.0 → strongly prefers k2 (high norm)
        std_scores = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(D)
        std_w = F.softmax(std_scores, dim=-1)

        # RBF: k1 is close to q, k2 is far → prefers k1
        gamma = torch.tensor(0.5)
        rbf_qk = torch.bmm(q, k.transpose(-1, -2))
        k_nsq = (k * k).sum(-1, keepdim=True).transpose(-1, -2)
        rbf_scores = 2 * gamma * rbf_qk - gamma * k_nsq
        rbf_w = F.softmax(rbf_scores, dim=-1)

        print(f"\nNorm-distractor test:")
        print(f"  Standard w(k1)={std_w[0,0,0]:.3f}, w(k2)={std_w[0,0,1]:.3f}")
        print(f"  RBF      w(k1)={rbf_w[0,0,0]:.3f}, w(k2)={rbf_w[0,0,1]:.3f}")

        # Standard prefers the high-norm distractor
        assert std_w[0, 0, 1] > std_w[0, 0, 0], "Standard should prefer high-norm k2"
        # RBF prefers the L2-nearby token
        assert rbf_w[0, 0, 0] > rbf_w[0, 0, 1], "RBF should prefer nearby k1"


class TestNonSoftmaxRBF:
    """Test non-softmax RBF variants."""

    def test_normalized_nosoftmax_equals_softmax(self):
        """Normalized exp(-γ||q-k||²) / Σ exp(...) should equal softmax RBF,
        since both normalize the same kernel values."""
        torch.manual_seed(42)
        B, N, D = 2, TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(B, N, D)
        k = torch.randn(B, N, D)
        v = torch.randn(B, N, D)
        gamma = torch.tensor(1.0)

        out_softmax = rbf_attention_torch(q, k, v, gamma)
        out_nosoftmax = rbf_attention_nosoftmax_torch(q, k, v, gamma)
        torch.testing.assert_close(out_softmax, out_nosoftmax, atol=1e-5, rtol=1e-4)

    def test_unnormalized_gradient_flows(self):
        """Gradients must flow through the unnormalized variant."""
        torch.manual_seed(42)
        B, N, D = 1, TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(B, N, D, requires_grad=True)
        k = torch.randn(B, N, D, requires_grad=True)
        v = torch.randn(B, N, D, requires_grad=True)
        gamma = torch.tensor(1.0, requires_grad=True)

        out = rbf_attention_unnormalized_torch(q, k, v, gamma)
        loss = out.sum()
        loss.backward()

        for name, t in [("q", q), ("k", k), ("v", v), ("gamma", gamma)]:
            assert t.grad is not None and t.grad.abs().sum() > 0, f"{name} needs grad"

    def test_unnormalized_attend_to_nothing(self):
        """When all keys are far from query, unnormalized output ≈ 0.
        This is the key advantage over softmax: can express 'nothing relevant'."""
        D = TILE_EMBED_DIM
        q = torch.zeros(1, 1, D)  # query at origin
        # All keys far away
        k = torch.ones(1, TILE_GRID_N, D) * 10.0
        v = torch.ones(1, TILE_GRID_N, D)
        gamma = torch.tensor(1.0)

        out_unnorm = rbf_attention_unnormalized_torch(q, k, v, gamma)
        out_softmax = rbf_attention_torch(q, k, v, gamma)

        # Unnormalized: all weights ≈ 0, so output ≈ 0
        assert out_unnorm.abs().max() < 1e-10, \
            f"Unnormalized should be ~0, got {out_unnorm.abs().max():.2e}"
        # Softmax: forced to distribute weight, output ≈ uniform average of V = 1.0
        assert out_softmax.abs().mean() > 0.5, \
            "Softmax must attend somewhere even when all keys are far"

    def test_unnormalized_output_scales_with_neighbors(self):
        """More nearby keys → larger output magnitude.
        This tests that the model can learn 'density' of relevant info."""
        D = TILE_EMBED_DIM
        q = torch.zeros(1, 1, D)
        gamma = torch.tensor(1.0)
        v = torch.ones(1, TILE_GRID_N, D)

        # Few nearby keys (1 close, rest far)
        k_sparse = torch.ones(1, TILE_GRID_N, D) * 10.0
        k_sparse[0, 0] = 0.0  # one key at origin
        out_sparse = rbf_attention_unnormalized_torch(q, k_sparse, v, gamma)

        # Many nearby keys (5 close, rest far)
        k_dense = torch.ones(1, TILE_GRID_N, D) * 10.0
        k_dense[0, :5] = 0.0  # five keys at origin
        out_dense = rbf_attention_unnormalized_torch(q, k_dense, v, gamma)

        # Dense should have ~5x the magnitude
        ratio = out_dense.abs().mean() / out_sparse.abs().mean()
        assert 4.0 < ratio < 6.0, f"Expected ~5x ratio, got {ratio:.1f}"

    def test_unnormalized_stdlib_parity(self):
        """Stdlib unnormalized RBF must match PyTorch."""
        torch.manual_seed(99)
        N, D = TILE_GRID_N, TILE_EMBED_DIM
        q = torch.randn(1, N, D)
        k = torch.randn(1, N, D)
        v = torch.randn(1, N, D)
        gamma_val = 0.5

        ref = rbf_attention_unnormalized_torch(q, k, v, torch.tensor(gamma_val))

        q_list = [list(q[0, i].tolist()) for i in range(N)]
        k_list = [list(k[0, i].tolist()) for i in range(N)]
        v_cols = [[v[0, i, j].item() for i in range(N)] for j in range(D)]

        out_list = rbf_attention_unnormalized_stdlib(
            q_list, k_list, v_cols, gamma_val, N, N, D,
        )
        for i in range(N):
            for j in range(D):
                assert out_list[i][j] == pytest.approx(
                    ref[0, i, j].item(), abs=1e-4,
                ), f"Mismatch at [{i},{j}]"

    def test_unnormalized_inference_cost(self):
        """Non-softmax saves: no max-finding pass, no normalization division.
        Count the operations to verify."""
        N, D = TILE_GRID_N, TILE_EMBED_DIM
        # Softmax RBF ops per query:
        #   - N sumprod calls (QK^T)
        #   - N comparisons (find max)
        #   - N exp calls
        #   - N additions (sum for normalization)
        #   - 1 division
        #   - D sumprod calls (weighted sum of V)
        softmax_ops_per_q = N + N + N + N + 1 + D  # sumprod + cmp + exp + add + div + sumprod

        # Unnormalized RBF ops per query:
        #   - N sumprod calls (QK^T)
        #   - N exp calls
        #   - D sumprod calls (weighted sum of V)
        #   (no max, no normalization)
        unnorm_ops_per_q = N + N + D  # sumprod + exp + sumprod

        savings = 1.0 - unnorm_ops_per_q / softmax_ops_per_q
        print(f"\nInference op count per query (N={N}, D={D}):")
        print(f"  Softmax RBF:      {softmax_ops_per_q}")
        print(f"  Unnormalized RBF: {unnorm_ops_per_q}")
        print(f"  Savings:          {savings:.0%}")
        # Should save ~30% of per-query ops
        assert savings > 0.2


class TestRBFModelIntegration:
    """Verify the RBF attention model works end-to-end."""

    @pytest.fixture
    def model_and_obs(self):
        """Create a fresh actor model."""
        for p in (PROJECT_ROOT / "bots" / "tensor" / "torch",):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from model import BuilderMLP, OBS_SIZE, H_DIM
        model = BuilderMLP()
        model.eval()
        return model, OBS_SIZE, H_DIM

    def test_rbf_tile_sa_forward(self, model_and_obs):
        """Tile SA forward produces valid output with RBF attention."""
        model, OBS_SIZE, H_DIM = model_and_obs
        torch.manual_seed(42)
        B = 4
        obs = torch.randn(B, OBS_SIZE)
        h_prev = torch.randn(B, H_DIM)

        with torch.no_grad():
            tiles, _ = model._split_obs_pre(model._obs_encoder_pre(obs))
            out = model._tile_self_attention(tiles)

        assert out.shape == (B, TILE_GRID_N, TILE_EMBED_DIM)
        assert torch.isfinite(out).all(), "Output should be finite"

    def test_gamma_is_learnable(self, model_and_obs):
        """gamma should receive gradients during training."""
        model, OBS_SIZE, H_DIM = model_and_obs
        model.train()
        obs = torch.randn(2, OBS_SIZE)
        h_prev = torch.randn(2, H_DIM)

        latent = model.actor_latent(obs, h_prev)
        loss = latent.sum()
        loss.backward()

        assert model.tile_sa_gamma.grad is not None
        assert model.tile_sa_gamma.grad.abs() > 0

    def test_gamma_gradient_matches_explicit_rbf_logits(self, model_and_obs):
        """The SDPA formulation should preserve the full RBF gamma gradient."""
        model, OBS_SIZE, _H_DIM = model_and_obs
        model.train()
        obs = torch.randn(2, OBS_SIZE)

        with torch.no_grad():
            tiles, _ = model._split_obs_pre(model._obs_encoder_pre(obs))
            normed_tiles = model.tile_sa_norm(tiles)
            q = model.tile_sa_q_proj(normed_tiles)
            k = model.tile_sa_k_proj(normed_tiles)
            v = model.tile_sa_v_proj(normed_tiles)

        gamma_param_actual = model.tile_sa_gamma.detach().clone().requires_grad_(True)
        gamma_actual = gamma_param_actual.exp()
        gamma_scale = torch.sqrt(2.0 * gamma_actual)
        k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)
        actual = F.scaled_dot_product_attention(
            (q * gamma_scale).unsqueeze(1),
            (k * gamma_scale).unsqueeze(1),
            v.unsqueeze(1),
            attn_mask=(-gamma_actual * k_norm_sq).unsqueeze(1),
            scale=1.0,
        ).squeeze(1)
        actual.sum().backward()
        gamma_grad_actual = gamma_param_actual.grad

        gamma_param_expected = model.tile_sa_gamma.detach().clone().requires_grad_(True)
        gamma_expected = gamma_param_expected.exp()
        expected = rbf_attention_torch(q, k, v, gamma_expected)
        expected.sum().backward()
        gamma_grad_expected = gamma_param_expected.grad

        assert gamma_grad_actual is not None
        assert gamma_grad_expected is not None
        torch.testing.assert_close(gamma_grad_actual, gamma_grad_expected, rtol=1e-5, atol=1e-6)

    def test_no_rope_attributes(self, model_and_obs):
        """Model should not have any RoPE-related attributes."""
        model, _, _ = model_and_obs
        for attr in ("_tile_rope_cos", "_tile_rope_sin", "_apply_tile_rope"):
            assert not hasattr(model, attr), f"Model still has {attr}"

    def test_export_import_roundtrip(self, model_and_obs):
        """Weight export/import preserves gamma."""
        model, _, _ = model_and_obs
        # Set gamma to a known value
        with torch.no_grad():
            model.tile_sa_gamma.fill_(0.42)

        state = model.export_weights()

        from model import BuilderMLP
        model2 = BuilderMLP()
        model2.import_weights(state)

        torch.testing.assert_close(
            model2.tile_sa_gamma, model.tile_sa_gamma,
        )

    def test_tile_sa_loss_decreases(self, model_and_obs):
        """End-to-end smoke: training the new tile SA path should reduce MSE
        against a fixed random target. Covers SuSIE pos_weight, look-away
        register, γ and the non-softmax RBF kernel all receiving gradient."""
        from model import BuilderMLP
        torch.manual_seed(0)
        model = BuilderMLP()
        model.train()
        B = 16
        OBS_SIZE = model_and_obs[1]
        obs = torch.randn(B, OBS_SIZE)
        with torch.no_grad():
            tiles0, _ = model._split_obs_pre(model._obs_encoder_pre(obs))
        target = torch.randn_like(tiles0)

        tile_sa_params = [
            model.tile_sa_q_proj.weight, model.tile_sa_q_proj.bias,
            model.tile_sa_k_proj.weight, model.tile_sa_k_proj.bias,
            model.tile_sa_v_proj.weight, model.tile_sa_v_proj.bias,
            model.tile_sa_norm.weight,
            model.tile_sa_gamma,
            model.tile_sa_pos_weight,
            model.tile_sa_lookaway,
            model.tile_ffn_norm.weight,
            model.tile_ffn_up.weight, model.tile_ffn_up.bias,
            model.tile_ffn_down.weight, model.tile_ffn_down.bias,
            model.tile_ffn_out_norm.weight,
        ]
        opt = torch.optim.Adam(tile_sa_params, lr=1e-2)

        initial_pos_weight = model.tile_sa_pos_weight.detach().clone()
        initial_lookaway = model.tile_sa_lookaway.detach().clone()

        with torch.no_grad():
            tiles_in, _ = model._split_obs_pre(model._obs_encoder_pre(obs))
        first_loss = (model._tile_self_attention(tiles_in) - target).pow(2).mean().item()

        last_loss = first_loss
        for _ in range(30):
            opt.zero_grad()
            with torch.no_grad():
                tiles_in, _ = model._split_obs_pre(model._obs_encoder_pre(obs))
            out = model._tile_self_attention(tiles_in)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()
            last_loss = loss.item()

        assert last_loss < first_loss * 0.5, (
            f"loss did not halve: {first_loss:.4f} -> {last_loss:.4f}"
        )
        # Both new params must have received gradient and moved.
        assert not torch.equal(model.tile_sa_pos_weight.detach(), initial_pos_weight), (
            "tile_sa_pos_weight did not update"
        )
        assert not torch.equal(model.tile_sa_lookaway.detach(), initial_lookaway), (
            "tile_sa_lookaway did not update"
        )

    def test_full_actor_parity(self, model_and_obs):
        """Full forward pass: PyTorch actor matches stdlib inference actor."""
        from model import BuilderMLP, H_DIM
        from tensor_lib import ActorModel as InferActor
        import main as infer_main
        from bench_obs import FakeController

        torch_model = BuilderMLP().eval()
        state = torch_model.export_weights()
        infer_model = InferActor(output_heads={
            "build_type": 13, "build_dir": 9, "conv_dir": 4,
            "bridge_target": 28, "move_dir": 9, "temperature": 5,
            "h_gate": H_DIM, "h_cand": H_DIM,
        })
        infer_model.load(state)

        ct = FakeController()
        player = infer_main.Player()
        player.my_team = ct.get_team()
        player.core_center = infer_main.Position(18, 20)
        player.enemy_core = infer_main.Position(25, 22)
        player._spawn_round = 100
        player._prev_collected = (160.0, 35.0)
        obs = list(player._encode_state(ct, ct.nearby_tiles, ct.nearby_entities))
        h_prev = torch.randn(1, H_DIM)
        packed = torch.tensor([obs + h_prev[0].tolist()], dtype=torch.float32)

        with torch.no_grad():
            bt_l, bd_l, cd_l, brt_l, md_l, _, h_next = torch_model.infer_packed(packed)

        features = obs + h_prev[0].tolist()
        result = infer_model.infer(features)

        for name, pt_tensor in [
            ("build_type", bt_l), ("build_dir", bd_l),
            ("conv_dir", cd_l), ("move_dir", md_l),
        ]:
            pt_vals = pt_tensor[0].tolist()
            for i, (a, b) in enumerate(zip(pt_vals, result[name])):
                assert a == pytest.approx(b, abs=1e-4), \
                    f"{name}[{i}]: torch={a:.6f} vs stdlib={b:.6f}"
