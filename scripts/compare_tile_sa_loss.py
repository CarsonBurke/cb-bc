"""Loss comparison across tile-SA attention variants.

Trains each variant on the same synthetic MSE regression target using identical
projection init (via seed) and optimizer settings, then reports the loss
trajectory. Each variant is trained once with Adam and once with NorMuon so we
can see how the optimizer interacts with the attention formulation.

Run from the repo root: `python3 scripts/compare_tile_sa_loss.py`.

What the loss measures:
  - obs: fixed random tensor of shape (B, TILE_GRID_N, TILE_EMBED_DIM), same
    layout as the model's tile embeddings after `_obs_encoder_pre`/`_split_obs_pre`.
  - target: fixed random tensor of the same shape.
  - forward: variant's tile self-attention (norm → Q/K/V → attention → +residual
    → FFN block), matching the prod structure of `_tile_self_attention` in
    bots/tensor/torch/model.py.
  - loss: plain mean-squared-error between the variant's output and the random
    target. This is a stationary regression toy with no RL dynamics — it
    measures how well the attention block can memorize an arbitrary mapping
    under a fixed optimizer/step budget. Lower = the formulation fits this
    toy better; it does NOT directly predict RL return.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bots" / "tensor" / "torch"))

from model import (
    TILE_EMBED_DIM,
    TILE_GRID_N,
    RMSNorm,
    _make_tile_susie,
    _parameter_golf_ortho_init,
    ffn_activation,
)
from normuon import make_normuon_optimizer

TILE_FFN_DIM = TILE_EMBED_DIM * 4  # same pattern as prod
N_STEPS = 15000
LR = 1e-2
B = 16
SEED = 0


def _build_rope(n_tokens, dim, theta=10000.0):
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(n_tokens, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs).repeat(1, 2), torch.sin(freqs).repeat(1, 2)


def _apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class TileSABase(nn.Module):
    """Shared projections/FFN, identical init seed across variants."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.q = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.k = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.v = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_EMBED_DIM))
        self.norm_in = RMSNorm(TILE_EMBED_DIM)
        self.ffn_norm = RMSNorm(TILE_EMBED_DIM)
        self.ffn_up = _parameter_golf_ortho_init(nn.Linear(TILE_EMBED_DIM, TILE_FFN_DIM))
        self.ffn_down = _parameter_golf_ortho_init(nn.Linear(TILE_FFN_DIM, TILE_EMBED_DIM))
        self.ffn_out_norm = RMSNorm(TILE_EMBED_DIM)

    def _ffn(self, x):
        return x + self.ffn_out_norm(
            self.ffn_down(ffn_activation(self.ffn_up(self.ffn_norm(x))))
        )


class QKNormRoPESoftmax(TileSABase):
    def __init__(self):
        super().__init__()
        cos, sin = _build_rope(TILE_GRID_N, TILE_EMBED_DIM)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.q_norm = RMSNorm(TILE_EMBED_DIM)
        self.k_norm = RMSNorm(TILE_EMBED_DIM)
        self.tau = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(TILE_EMBED_DIM))))

    def forward(self, tiles):
        normed = self.norm_in(tiles)
        q = self.q_norm(self.q(normed))
        k = self.k_norm(self.k(normed))
        v = self.v(normed)
        q = _apply_rope(q, self.rope_cos, self.rope_sin)
        k = _apply_rope(k, self.rope_cos, self.rope_sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.tau.exp()
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self._ffn(tiles + attn)


class RBFSoftmaxBare(TileSABase):
    """RBF softmax without SuSIE or register."""
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, tiles):
        normed = self.norm_in(tiles)
        q = self.q(normed); k = self.k(normed); v = self.v(normed)
        gamma = self.gamma.exp()
        k_norm_sq = (k * k).sum(-1, keepdim=True).transpose(-1, -2)
        qk = torch.matmul(q, k.transpose(-1, -2))
        scores = 2.0 * gamma * qk - gamma * k_norm_sq
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self._ffn(tiles + attn)


class RBFSoftmaxSuSIE(TileSABase):
    """RBF softmax + SuSIE, no register."""
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.pos_weight = nn.Parameter(torch.full((TILE_EMBED_DIM // 2,), 0.5))
        self.register_buffer("susie", _make_tile_susie(TILE_GRID_N, TILE_EMBED_DIM), persistent=False)

    def forward(self, tiles):
        normed = self.norm_in(tiles)
        q = self.q(normed); k = self.k(normed); v = self.v(normed)
        pos = self.susie * self.pos_weight.repeat(2)
        q = q + pos; k = k + pos
        gamma = self.gamma.exp()
        k_norm_sq = (k * k).sum(-1, keepdim=True).transpose(-1, -2)
        qk = torch.matmul(q, k.transpose(-1, -2))
        scores = 2.0 * gamma * qk - gamma * k_norm_sq
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self._ffn(tiles + attn)


class RBFSoftmaxProdFull(TileSABase):
    """Production: RBF softmax + SuSIE + look-away register."""
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.pos_weight = nn.Parameter(torch.full((TILE_EMBED_DIM // 2,), 0.5))
        self.lookaway = nn.Parameter(torch.zeros(TILE_EMBED_DIM))
        self.register_buffer("susie", _make_tile_susie(TILE_GRID_N, TILE_EMBED_DIM), persistent=False)

    def forward(self, tiles):
        B_, N, D = tiles.shape
        la = self.lookaway.view(1, 1, D).expand(B_, 1, D)
        seq = torch.cat([la, tiles], dim=1)
        normed = self.norm_in(seq)
        q = self.q(normed); k = self.k(normed); v = self.v(normed)
        pos_real = self.susie * self.pos_weight.repeat(2)
        pos_emb = F.pad(pos_real, (0, 0, 1, 0))
        q = q + pos_emb; k = k + pos_emb
        gamma = self.gamma.exp()
        k_norm_sq = (k * k).sum(-1, keepdim=True).transpose(-1, -2)
        qk = torch.matmul(q, k.transpose(-1, -2))
        scores = 2.0 * gamma * qk - gamma * k_norm_sq
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)[:, 1:, :]
        return self._ffn(tiles + attn)


class RBFUnnormProdFull(TileSABase):
    """Abandoned target: RBF non-softmax + SuSIE + look-away (no post-norm)."""
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.pos_weight = nn.Parameter(torch.full((TILE_EMBED_DIM // 2,), 0.5))
        self.lookaway = nn.Parameter(torch.zeros(TILE_EMBED_DIM))
        self.register_buffer("susie", _make_tile_susie(TILE_GRID_N, TILE_EMBED_DIM), persistent=False)

    def forward(self, tiles):
        B_, N, D = tiles.shape
        la = self.lookaway.view(1, 1, D).expand(B_, 1, D)
        seq = torch.cat([la, tiles], dim=1)
        normed = self.norm_in(seq)
        q = self.q(normed); k = self.k(normed); v = self.v(normed)
        pos_real = self.susie * self.pos_weight.repeat(2)
        pos_emb = F.pad(pos_real, (0, 0, 1, 0))
        q = q + pos_emb; k = k + pos_emb
        gamma = self.gamma.exp()
        diff = q.unsqueeze(-2) - k.unsqueeze(-3)  # (B,Nq,Nk,D)
        dist_sq = (diff * diff).sum(-1)
        weights = torch.exp(-gamma * dist_sq)
        attn = torch.matmul(weights, v)[:, 1:, :]
        return self._ffn(tiles + attn)


class RBFSoftmaxProdOutNorm(TileSABase):
    """PROD + post-attn out-norm (the removed tile_sa_out_norm)."""
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.pos_weight = nn.Parameter(torch.full((TILE_EMBED_DIM // 2,), 0.5))
        self.lookaway = nn.Parameter(torch.zeros(TILE_EMBED_DIM))
        self.out_norm = RMSNorm(TILE_EMBED_DIM)
        self.register_buffer("susie", _make_tile_susie(TILE_GRID_N, TILE_EMBED_DIM), persistent=False)

    def forward(self, tiles):
        B_, N, D = tiles.shape
        la = self.lookaway.view(1, 1, D).expand(B_, 1, D)
        seq = torch.cat([la, tiles], dim=1)
        normed = self.norm_in(seq)
        q = self.q(normed); k = self.k(normed); v = self.v(normed)
        pos_real = self.susie * self.pos_weight.repeat(2)
        pos_emb = F.pad(pos_real, (0, 0, 1, 0))
        q = q + pos_emb; k = k + pos_emb
        gamma = self.gamma.exp()
        k_norm_sq = (k * k).sum(-1, keepdim=True).transpose(-1, -2)
        qk = torch.matmul(q, k.transpose(-1, -2))
        scores = 2.0 * gamma * qk - gamma * k_norm_sq
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)[:, 1:, :]
        return self._ffn(tiles + self.out_norm(attn))


class RBFUnnormProdOutNorm(TileSABase):
    """Unnorm + SuSIE + register + post-attn out-norm."""
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.pos_weight = nn.Parameter(torch.full((TILE_EMBED_DIM // 2,), 0.5))
        self.lookaway = nn.Parameter(torch.zeros(TILE_EMBED_DIM))
        self.out_norm = RMSNorm(TILE_EMBED_DIM)
        self.register_buffer("susie", _make_tile_susie(TILE_GRID_N, TILE_EMBED_DIM), persistent=False)

    def forward(self, tiles):
        B_, N, D = tiles.shape
        la = self.lookaway.view(1, 1, D).expand(B_, 1, D)
        seq = torch.cat([la, tiles], dim=1)
        normed = self.norm_in(seq)
        q = self.q(normed); k = self.k(normed); v = self.v(normed)
        pos_real = self.susie * self.pos_weight.repeat(2)
        pos_emb = F.pad(pos_real, (0, 0, 1, 0))
        q = q + pos_emb; k = k + pos_emb
        gamma = self.gamma.exp()
        diff = q.unsqueeze(-2) - k.unsqueeze(-3)
        dist_sq = (diff * diff).sum(-1)
        weights = torch.exp(-gamma * dist_sq)
        attn = torch.matmul(weights, v)[:, 1:, :]
        return self._ffn(tiles + self.out_norm(attn))


def train_variant(model: nn.Module, obs: torch.Tensor, target: torch.Tensor, *, opt_name: str):
    model.train()
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    elif opt_name == "muon":
        opt = make_normuon_optimizer(model, lr=LR)
    else:
        raise ValueError(opt_name)
    traj = []
    with torch.no_grad():
        first = (model(obs) - target).pow(2).mean().item()
    traj.append(first)
    for _ in range(N_STEPS):
        opt.zero_grad()
        loss = (model(obs) - target).pow(2).mean()
        loss.backward()
        opt.step()
        traj.append(loss.item())
    return traj


def main():
    torch.manual_seed(SEED)
    obs = torch.randn(B, TILE_GRID_N, TILE_EMBED_DIM)
    target = torch.randn(B, TILE_GRID_N, TILE_EMBED_DIM)

    variants = [
        ("QK-norm+RoPE+softmax (legacy)", QKNormRoPESoftmax),
        ("RBF softmax (bare)", RBFSoftmaxBare),
        ("RBF softmax + SuSIE", RBFSoftmaxSuSIE),
        ("RBF softmax + SuSIE + reg (PROD, no out-norm)", RBFSoftmaxProdFull),
        ("RBF softmax + SuSIE + reg + out-norm", RBFSoftmaxProdOutNorm),
        ("RBF unnorm + SuSIE + reg (no out-norm)", RBFUnnormProdFull),
        ("RBF unnorm + SuSIE + reg + out-norm", RBFUnnormProdOutNorm),
    ]

    results = []
    for name, cls in variants:
        for opt_name in ("adam", "muon"):
            torch.manual_seed(SEED)
            model = cls()
            traj = train_variant(model, obs, target, opt_name=opt_name)
            results.append((f"{name} [{opt_name}]", traj))

    cps = [0, 100, 500, 1500, 5000, 10000, N_STEPS]
    header = ["Variant"] + [f"loss@{c}" for c in cps] + ["final/init"]
    print(f"{header[0]:<56} " + " ".join(f"{h:>9}" for h in header[1:]))
    print("-" * (56 + 10 * len(header[1:])))
    for name, traj in results:
        vals = [traj[c] for c in cps]
        ratio = vals[-1] / vals[0]
        row = f"{name:<56} " + " ".join(f"{v:>9.4f}" for v in vals) + f" {ratio:>9.3f}"
        print(row)


if __name__ == "__main__":
    main()
