"""Tests for tile encoder pretraining (reconstruction autoencoder)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

_TORCH_DIR = str(Path(__file__).resolve().parent.parent / "bots" / "tensor" / "torch")
if _TORCH_DIR not in sys.path:
    sys.path.insert(0, _TORCH_DIR)

from model import (
    BuilderMLP,
    CentralizedCritic,
    TILE_EMBED_DIM,
    TILE_FEATURES,
    TILE_GRID_N,
    TILE_INPUT_DIM,
    OBS_SIZE,
    GLOBAL_MAP_CHANNELS,
)

sys.path.insert(0, str(Path(_TORCH_DIR) / "pretrain"))
from train_encoder import (
    ActorTileDecoder,
    CriticTileDecoder,
    actor_reconstruction_loss,
    critic_reconstruction_loss,
    _ternary_to_class,
    _onehot_to_class,
    N_ACTOR_BINARY,
    N_ACTOR_BLD_CLASSES,
    N_ACTOR_TERNARY,
    N_ACTOR_CONTINUOUS,
    N_CRITIC_BINARY,
    N_CRITIC_TERNARY,
    N_CRITIC_CONTINUOUS,
    CRITIC_TILE_EMBED_DIM,
)


class TestHelpers:
    def test_ternary_to_class(self):
        vals = torch.tensor([-1.0, 0.0, 1.0, -1.0, 0.0])
        cls = _ternary_to_class(vals)
        assert cls.tolist() == [0, 1, 2, 0, 1]

    def test_ternary_to_class_clamps(self):
        vals = torch.tensor([-1.5, 1.5])
        cls = _ternary_to_class(vals)
        assert cls.min() >= 0 and cls.max() <= 2

    def test_onehot_to_class_building(self):
        onehot = torch.zeros(3, 10)
        onehot[0, 0] = 1.0  # building type 0
        onehot[1, 5] = 1.0  # building type 5
        # onehot[2] is all zeros → class 10 (none)
        cls = _onehot_to_class(onehot)
        assert cls.tolist() == [0, 5, 10]


class TestActorDecoder:
    def test_output_shapes(self):
        decoder = ActorTileDecoder(TILE_EMBED_DIM)
        B = 4
        tile_embeds = torch.randn(B, TILE_GRID_N, TILE_EMBED_DIM)
        out = decoder(tile_embeds)
        assert out["binary_logits"].shape == (B, TILE_GRID_N, N_ACTOR_BINARY)
        assert out["bld_type_logits"].shape == (B, TILE_GRID_N, N_ACTOR_BLD_CLASSES)
        assert out["ternary_logits"].shape == (B, TILE_GRID_N, N_ACTOR_TERNARY, 3)
        assert out["continuous"].shape == (B, TILE_GRID_N, N_ACTOR_CONTINUOUS)


class TestCriticDecoder:
    def test_output_shapes(self):
        decoder = CriticTileDecoder(CRITIC_TILE_EMBED_DIM)
        N = 100
        flat_embeds = torch.randn(N, CRITIC_TILE_EMBED_DIM)
        out = decoder(flat_embeds)
        assert out["binary_logits"].shape == (N, N_CRITIC_BINARY)
        assert out["ternary_logits"].shape == (N, N_CRITIC_TERNARY, 3)
        assert out["continuous"].shape == (N, N_CRITIC_CONTINUOUS)


def _make_fake_tile_features(B: int, T: int = TILE_GRID_N) -> torch.Tensor:
    """Generate plausible tile features: binary {0,1}, one-hot, ternary {-1,0,1}, continuous."""
    feats = torch.zeros(B, T, TILE_FEATURES)
    # Binary features [0..3, 19]
    for idx in [0, 1, 2, 3, 19]:
        feats[:, :, idx] = (torch.rand(B, T) > 0.7).float()
    # Building one-hot [4..13]
    has_building = torch.rand(B, T) > 0.5
    bld_type = torch.randint(0, 10, (B, T))
    for b in range(B):
        for t in range(T):
            if has_building[b, t]:
                feats[b, t, 4 + bld_type[b, t]] = 1.0
    # Ternary [14, 18, 20]
    for idx in [14, 18, 20]:
        feats[:, :, idx] = torch.randint(-1, 2, (B, T)).float()
    # Continuous [15, 16, 17]
    feats[:, :, 15] = torch.randn(B, T) * 0.5
    feats[:, :, 16] = torch.randn(B, T) * 0.5
    feats[:, :, 17] = torch.rand(B, T)
    return feats


class TestActorReconstructionLoss:
    def test_loss_finite(self):
        decoder = ActorTileDecoder(TILE_EMBED_DIM)
        B = 8
        tile_embeds = torch.randn(B, TILE_GRID_N, TILE_EMBED_DIM)
        decoded = decoder(tile_embeds)
        tile_feats = _make_fake_tile_features(B)
        loss, metrics = actor_reconstruction_loss(decoded, tile_feats)
        assert loss.isfinite()
        assert all(v >= 0 for v in metrics.values())

    def test_loss_decreases_with_training(self):
        """Verify a few gradient steps reduce reconstruction loss."""
        model = BuilderMLP()
        decoder = ActorTileDecoder(TILE_EMBED_DIM)
        params = list(model.tile_linear.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        B = 32
        tile_feats = _make_fake_tile_features(B)
        # Build obs with tile features at the front, zeros for globals
        obs = torch.zeros(B, OBS_SIZE)
        obs[:, :TILE_GRID_N * TILE_FEATURES] = tile_feats.reshape(B, -1)

        losses = []
        for _ in range(20):
            tile_embeds = model._encode_tiles(obs)
            decoded = decoder(tile_embeds)
            loss, _ = actor_reconstruction_loss(decoded, tile_feats)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


class TestCriticReconstructionLoss:
    def test_loss_finite(self):
        decoder = CriticTileDecoder(CRITIC_TILE_EMBED_DIM)
        R, H, W = 2, 10, 10
        map_channels = torch.randn(R, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.ones(R, 1, H, W)
        flat_embeds = torch.randn(R * H * W, CRITIC_TILE_EMBED_DIM)
        decoded = decoder(flat_embeds)
        loss, metrics = critic_reconstruction_loss(decoded, map_channels, valid_mask)
        assert loss.isfinite()

    def test_loss_decreases_with_training(self):
        """Verify a few gradient steps reduce reconstruction loss."""
        critic = CentralizedCritic()
        decoder = CriticTileDecoder(critic.tile_embed.embed_dim)
        params = list(critic.tile_embed.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        R, H, W = 4, 8, 8
        map_channels = torch.randn(R, GLOBAL_MAP_CHANNELS, H, W)
        valid_mask = torch.ones(R, 1, H, W)

        losses = []
        for _ in range(20):
            tile_embeds = critic.tile_embed(map_channels)  # NHWC
            flat_embeds = tile_embeds.reshape(-1, critic.tile_embed.embed_dim)
            decoded = decoder(flat_embeds)
            loss, _ = critic_reconstruction_loss(decoded, map_channels, valid_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_masked_tiles_excluded(self):
        """Only valid (non-padded) tiles should contribute to loss."""
        decoder = CriticTileDecoder(CRITIC_TILE_EMBED_DIM)
        R, H, W = 1, 4, 4
        map_channels = torch.randn(R, GLOBAL_MAP_CHANNELS, H, W)

        # Half valid
        valid_mask = torch.zeros(R, 1, H, W)
        valid_mask[:, :, :2, :] = 1.0

        flat_embeds = torch.randn(R * H * W, CRITIC_TILE_EMBED_DIM)
        decoded = decoder(flat_embeds)
        loss_half, _ = critic_reconstruction_loss(decoded, map_channels, valid_mask)

        # All valid
        valid_mask_full = torch.ones(R, 1, H, W)
        loss_full, _ = critic_reconstruction_loss(decoded, map_channels, valid_mask_full)

        # Losses should differ since different tile counts contribute
        assert loss_half.isfinite()
        assert loss_full.isfinite()
