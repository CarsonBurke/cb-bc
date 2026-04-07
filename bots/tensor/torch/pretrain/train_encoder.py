#!/usr/bin/env python3
"""Pretrain actor and critic tile encoders via reconstruction (autoencoder).

Uses CE loss on categorical/binary features and MSE on continuous features.
Data comes from replay files via ActorBCDataset (per-unit obs) and
CriticBCDataset (global map states).

Usage:
    python bots/tensor/torch/pretrain/train_encoder.py --replay-dir replay_data --epochs 10
    python bots/tensor/torch/pretrain/train_encoder.py --replay-dir replay_data --epochs 10 --critic-only
    python bots/tensor/torch/pretrain/train_encoder.py --replay-dir replay_data --epochs 10 --actor-only
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_TORCH_DIR = Path(__file__).resolve().parent.parent
_TENSOR_DIR = _TORCH_DIR.parent
sys.path.insert(0, str(_TORCH_DIR))
sys.path.insert(0, str(_TENSOR_DIR))

from model import (
    BuilderMLP,
    CentralizedCritic,
    DEVICE,
    GLOBAL_MAP_CHANNELS,
    GLOBAL_MAP_MAX_H,
    GLOBAL_MAP_MAX_W,
    OBS_SIZE,
    TILE_EMBED_DIM,
    TILE_FEATURES,
    TILE_GRID_N,
    TILE_INPUT_DIM,
    get_centralized_critic,
    load_centralized_critic_checkpoint,
    save_centralized_critic_checkpoint,
)

from dataset import ActorBCDataset, CriticBCDataset
from training_utils import jsonl_init, jsonl_append
from pretrain.gpu import (
    autocast_context,
    configure_cuda_runtime,
    resolve_cuda_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "replay_data"
SESSIONS_DIR = PROJECT_ROOT / "bots" / "tensor" / "sessions"
WEIGHT_PATH = PROJECT_ROOT / "bots" / "tensor_inference" / "weights.pt"

# ---- Actor tile feature layout (21 features per tile) ----
# 0: wall (binary), 1: ore_titanium (binary), 2: ore_axionite (binary),
# 3: not_passable (binary)
# 4-13: building_one_hot[10] (10-class → convert to 11-class with "none")
# 14: building_team (-1/0/+1 → 3-class)
# 15: building_dir_x (continuous), 16: building_dir_y (continuous)
# 17: building_cooldown (continuous)
# 18: building_stored_resource (-1/0/+1 → 3-class)
# 19: builder_bot_present (binary)
# 20: builder_bot_team (-1/0/+1 → 3-class)
_ACTOR_BINARY_IDX = [0, 1, 2, 3, 19]
_ACTOR_BLD_ONEHOT_SLICE = slice(4, 14)  # 10 channels
_ACTOR_TERNARY_IDX = [14, 18, 20]  # each in {-1, 0, +1}
_ACTOR_CONTINUOUS_IDX = [15, 16, 17]

N_ACTOR_BINARY = len(_ACTOR_BINARY_IDX)
N_ACTOR_BLD_CLASSES = 11  # 10 building types + "none"
N_ACTOR_TERNARY = len(_ACTOR_TERNARY_IDX)
N_ACTOR_CONTINUOUS = len(_ACTOR_CONTINUOUS_IDX)

# ---- Critic map channel layout (23 channels) ----
# 0: wall, 1: titanium_ore, 2: axionite_ore, 3: reserved, 4: reserved
# 5: harvester, 6: conveyor/armoured, 7: splitter, 8: road, 9: barrier,
# 10: bridge, 11: foundry, 12: weapons, 13: core
# 14: building_team_sign, 15: building_dir_x, 16: building_dir_y
# 17: builder_bot_present, 18: builder_bot_team_sign
# 19: stored_resource, 20: harvester_cooldown
# 21: own_flow_distance, 22: enemy_flow_distance
_CRITIC_BINARY_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17]
_CRITIC_TERNARY_IDX = [14, 18, 19]
_CRITIC_CONTINUOUS_IDX = [15, 16, 20, 21, 22]

N_CRITIC_BINARY = len(_CRITIC_BINARY_IDX)
N_CRITIC_TERNARY = len(_CRITIC_TERNARY_IDX)
N_CRITIC_CONTINUOUS = len(_CRITIC_CONTINUOUS_IDX)
CRITIC_TILE_EMBED_DIM = 8


def _ternary_to_class(vals: torch.Tensor) -> torch.Tensor:
    """Map {-1, 0, +1} float values to class indices {0, 1, 2}."""
    return (vals + 1.0).round().clamp(0, 2).long()


def _onehot_to_class(onehot: torch.Tensor) -> torch.Tensor:
    """Convert (N, 10) one-hot building channels to (N,) class in [0..10].

    Class 10 = no building (all zeros).
    """
    has_building = onehot.sum(dim=-1) > 0.5
    cls = onehot.argmax(dim=-1)
    cls[~has_building] = 10
    return cls


# ---- Actor decoder (temporary, for pretraining only) ----

class ActorTileDecoder(nn.Module):
    """Decode actor tile embeddings back to raw tile features."""

    def __init__(self, embed_dim: int = TILE_EMBED_DIM):
        super().__init__()
        self.binary_head = nn.Linear(embed_dim, N_ACTOR_BINARY)
        self.bld_type_head = nn.Linear(embed_dim, N_ACTOR_BLD_CLASSES)
        self.ternary_head = nn.Linear(embed_dim, N_ACTOR_TERNARY * 3)
        self.continuous_head = nn.Linear(embed_dim, N_ACTOR_CONTINUOUS)

    def forward(self, tile_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        """tile_embeds: (B, TILE_GRID_N, embed_dim)."""
        B, T, D = tile_embeds.shape
        flat = tile_embeds.reshape(B * T, D)
        return {
            "binary_logits": self.binary_head(flat).reshape(B, T, N_ACTOR_BINARY),
            "bld_type_logits": self.bld_type_head(flat).reshape(B, T, N_ACTOR_BLD_CLASSES),
            "ternary_logits": self.ternary_head(flat).reshape(B, T, N_ACTOR_TERNARY, 3),
            "continuous": self.continuous_head(flat).reshape(B, T, N_ACTOR_CONTINUOUS),
        }


def actor_reconstruction_loss(
    decoded: dict[str, torch.Tensor],
    tile_features: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute reconstruction loss from decoded outputs vs raw tile features.

    tile_features: (B, TILE_GRID_N, TILE_FEATURES=21).
    """
    B, T, _ = tile_features.shape

    # Binary CE
    binary_targets = tile_features[:, :, _ACTOR_BINARY_IDX]
    binary_loss = F.binary_cross_entropy_with_logits(
        decoded["binary_logits"], binary_targets, reduction="mean",
    )

    # Building type CE
    bld_onehot = tile_features[:, :, _ACTOR_BLD_ONEHOT_SLICE]
    bld_targets = _onehot_to_class(bld_onehot.reshape(-1, 10))
    bld_loss = F.cross_entropy(
        decoded["bld_type_logits"].reshape(-1, N_ACTOR_BLD_CLASSES),
        bld_targets,
        reduction="mean",
    )

    # Ternary CE
    ternary_targets = _ternary_to_class(
        tile_features[:, :, _ACTOR_TERNARY_IDX].reshape(-1, N_ACTOR_TERNARY),
    )
    ternary_logits = decoded["ternary_logits"].reshape(-1, N_ACTOR_TERNARY, 3)
    ternary_loss = sum(
        F.cross_entropy(ternary_logits[:, i, :], ternary_targets[:, i], reduction="mean")
        for i in range(N_ACTOR_TERNARY)
    ) / N_ACTOR_TERNARY

    # Continuous MSE
    continuous_targets = tile_features[:, :, _ACTOR_CONTINUOUS_IDX]
    continuous_loss = F.mse_loss(decoded["continuous"], continuous_targets, reduction="mean")

    total = binary_loss + bld_loss + ternary_loss + continuous_loss
    metrics = {
        "binary_ce": binary_loss.item(),
        "bld_type_ce": bld_loss.item(),
        "ternary_ce": ternary_loss.item(),
        "continuous_mse": continuous_loss.item(),
    }
    return total, metrics


# ---- Critic decoder (temporary, for pretraining only) ----

class CriticTileDecoder(nn.Module):
    """Decode critic tile embeddings back to raw map channels."""

    def __init__(self, embed_dim: int = CRITIC_TILE_EMBED_DIM):
        super().__init__()
        self.binary_head = nn.Linear(embed_dim, N_CRITIC_BINARY)
        self.ternary_head = nn.Linear(embed_dim, N_CRITIC_TERNARY * 3)
        self.continuous_head = nn.Linear(embed_dim, N_CRITIC_CONTINUOUS)

    def forward(self, tile_embeds: torch.Tensor) -> dict[str, torch.Tensor]:
        """tile_embeds: (N, embed_dim) where N = total spatial positions."""
        return {
            "binary_logits": self.binary_head(tile_embeds),
            "ternary_logits": self.ternary_head(tile_embeds).reshape(-1, N_CRITIC_TERNARY, 3),
            "continuous": self.continuous_head(tile_embeds),
        }


def critic_reconstruction_loss(
    decoded: dict[str, torch.Tensor],
    map_channels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute reconstruction loss from decoded outputs vs raw map channels.

    map_channels: (R, C, H, W).  valid_mask: (R, 1, H, W).
    We only compute loss over valid (non-padded) tiles.
    """
    R, C, H, W = map_channels.shape
    # Reshape to per-tile: (R*H*W, C)
    tiles = map_channels.permute(0, 2, 3, 1).reshape(-1, C)
    mask = valid_mask.squeeze(1).reshape(-1) > 0.5  # (R*H*W,)

    # Filter to valid tiles only
    tiles = tiles[mask]
    binary_logits = decoded["binary_logits"][mask]
    ternary_logits = decoded["ternary_logits"][mask]
    continuous_preds = decoded["continuous"][mask]

    if tiles.shape[0] == 0:
        zero = torch.tensor(0.0, device=map_channels.device)
        return zero, {"binary_ce": 0.0, "ternary_ce": 0.0, "continuous_mse": 0.0}

    # Binary CE
    binary_targets = tiles[:, _CRITIC_BINARY_IDX]
    binary_loss = F.binary_cross_entropy_with_logits(
        binary_logits, binary_targets, reduction="mean",
    )

    # Ternary CE
    ternary_targets = _ternary_to_class(tiles[:, _CRITIC_TERNARY_IDX])
    ternary_loss = sum(
        F.cross_entropy(ternary_logits[:, i, :], ternary_targets[:, i], reduction="mean")
        for i in range(N_CRITIC_TERNARY)
    ) / N_CRITIC_TERNARY

    # Continuous MSE
    continuous_targets = tiles[:, _CRITIC_CONTINUOUS_IDX]
    continuous_loss = F.mse_loss(continuous_preds, continuous_targets, reduction="mean")

    total = binary_loss + ternary_loss + continuous_loss
    metrics = {
        "binary_ce": binary_loss.item(),
        "ternary_ce": ternary_loss.item(),
        "continuous_mse": continuous_loss.item(),
    }
    return total, metrics


def _autocast(cuda_config):
    """Autocast context manager, falling back to nullcontext if no config."""
    from contextlib import nullcontext
    if cuda_config is None or not cuda_config.enabled:
        return nullcontext()
    return autocast_context(cuda_config)


# ---- Training loops ----

def pretrain_actor_encoder(
    model: BuilderMLP,
    dataset: ActorBCDataset,
    *,
    epochs: int = 10,
    lr: float = 3e-4,
    batch_size: int = 4096,
    games_per_batch: int = 8,
    cuda_config=None,
    log_path: Path | None = None,
) -> dict[str, float]:
    """Pretrain actor tile_linear via reconstruction."""
    decoder = ActorTileDecoder(TILE_EMBED_DIM).to(DEVICE)

    # Only train encoder + decoder, freeze everything else
    encoder_params = list(model.tile_linear.parameters())
    decoder_params = list(decoder.parameters())
    optimizer = torch.optim.AdamW(encoder_params + decoder_params, lr=lr, weight_decay=1e-4)

    model.train()
    decoder.train()
    # Freeze all non-encoder params
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.tile_linear.parameters():
        p.requires_grad_(True)

    n_games = len(dataset.games)
    if n_games == 0:
        print("No games found for actor encoder pretraining.")
        return {}

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_metrics: dict[str, float] = {}
        game_order = list(range(n_games))
        random.shuffle(game_order)

        for batch_start in range(0, n_games, games_per_batch):
            game_indices = game_order[batch_start:batch_start + games_per_batch]
            obs_list = []
            for gi in game_indices:
                game = dataset.games[gi]
                try:
                    obs, _actions, _bot_ids, _turns = dataset._get_parsed(
                        game["path"], game["team"],
                    )
                    if obs.shape[0] > 0:
                        obs_list.append(obs)
                except Exception:
                    continue

            if not obs_list:
                continue

            all_obs = torch.cat(obs_list, dim=0)
            n_samples = all_obs.shape[0]

            # Shuffle and iterate in mini-batches
            perm = torch.randperm(n_samples)
            for mb_start in range(0, n_samples, batch_size):
                idx = perm[mb_start:mb_start + batch_size]
                obs_mb = all_obs[idx].to(device=DEVICE, dtype=torch.float32)

                # Extract raw tile features: (B, 25, 21)
                tile_raw = obs_mb[:, :TILE_GRID_N * TILE_FEATURES].reshape(-1, TILE_GRID_N, TILE_FEATURES)

                with _autocast(cuda_config):
                    tile_embeds = model._encode_tiles(obs_mb)  # (B, 25, 8)
                    decoded = decoder(tile_embeds)
                    loss, metrics = actor_reconstruction_loss(decoded, tile_raw)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder_params + decoder_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item() * idx.shape[0]
                epoch_steps += idx.shape[0]
                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v * idx.shape[0]

        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
            if avg_loss < best_loss:
                best_loss = avg_loss
            elapsed = time.time() - t0
            avg_metrics = {k: v / epoch_steps for k, v in epoch_metrics.items()}
            print(
                f"  Actor encoder epoch {epoch + 1}/{epochs}: "
                f"loss={avg_loss:.4f} (best={best_loss:.4f}) "
                f"binary={avg_metrics.get('binary_ce', 0):.4f} "
                f"bld={avg_metrics.get('bld_type_ce', 0):.4f} "
                f"ternary={avg_metrics.get('ternary_ce', 0):.4f} "
                f"cont={avg_metrics.get('continuous_mse', 0):.4f} "
                f"[{elapsed:.0f}s]"
            )
            if log_path:
                jsonl_append(log_path, {
                    "phase": "actor_encoder",
                    "episode": epoch + 1,
                    "pi_loss": avg_loss,
                    "binary_ce": avg_metrics.get("binary_ce", 0),
                    "bld_type_ce": avg_metrics.get("bld_type_ce", 0),
                    "ternary_ce": avg_metrics.get("ternary_ce", 0),
                    "continuous_mse": avg_metrics.get("continuous_mse", 0),
                    "samples": epoch_steps,
                    "elapsed": elapsed,
                })

    # Restore requires_grad
    for p in model.parameters():
        p.requires_grad_(True)

    return {"actor_encoder_loss": best_loss, "actor_encoder_epochs": epochs}


def pretrain_critic_encoder(
    critic: CentralizedCritic,
    dataset: CriticBCDataset,
    *,
    epochs: int = 10,
    lr: float = 3e-4,
    batch_rounds: int = 64,
    games_per_batch: int = 4,
    cuda_config=None,
    log_path: Path | None = None,
) -> dict[str, float]:
    """Pretrain critic tile_embed via reconstruction."""
    embed_dim = critic.tile_embed.embed_dim
    decoder = CriticTileDecoder(embed_dim).to(DEVICE)

    encoder_params = list(critic.tile_embed.parameters())
    decoder_params = list(decoder.parameters())
    optimizer = torch.optim.AdamW(encoder_params + decoder_params, lr=lr, weight_decay=1e-4)

    critic.train()
    decoder.train()
    # Freeze all non-encoder params
    for p in critic.parameters():
        p.requires_grad_(False)
    for p in critic.tile_embed.parameters():
        p.requires_grad_(True)

    n_games = len(dataset.games)
    if n_games == 0:
        print("No games found for critic encoder pretraining.")
        return {}

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_tiles = 0
        epoch_metrics: dict[str, float] = {}
        game_order = list(range(n_games))
        random.shuffle(game_order)

        for batch_start in range(0, n_games, games_per_batch):
            game_indices = game_order[batch_start:batch_start + games_per_batch]
            states_list = []
            for gi in game_indices:
                game = dataset.games[gi]
                try:
                    buffers = dataset._get_buffers(game["path"], game["team"])
                    states = buffers["states"]  # (T, C, H, W) float16
                    if states.shape[0] > 0:
                        states_list.append(states)
                except Exception:
                    continue

            if not states_list:
                continue

            # Process each game's states separately (maps have different spatial dims)
            for states in states_list:
                n_rounds = states.shape[0]
                perm = torch.randperm(n_rounds)
                for mb_start in range(0, n_rounds, batch_rounds):
                    idx = perm[mb_start:mb_start + batch_rounds]
                    map_batch = states[idx].to(device=DEVICE, dtype=torch.float32)
                    R, C, H, W = map_batch.shape
                    valid_mask = torch.ones(R, 1, H, W, device=DEVICE)

                    with _autocast(cuda_config):
                        tile_embeds = critic.tile_embed(map_batch)
                        flat_embeds = tile_embeds.permute(0, 2, 3, 1).reshape(-1, embed_dim)
                        decoded = decoder(flat_embeds)
                        loss, metrics = critic_reconstruction_loss(decoded, map_batch, valid_mask)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder_params + decoder_params, 1.0)
                    optimizer.step()

                    n_valid = int(valid_mask.sum().item())
                    epoch_loss += loss.item() * n_valid
                    epoch_tiles += n_valid
                    for k, v in metrics.items():
                        epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v * n_valid

        if epoch_tiles > 0:
            avg_loss = epoch_loss / epoch_tiles
            if avg_loss < best_loss:
                best_loss = avg_loss
            elapsed = time.time() - t0
            avg_metrics = {k: v / epoch_tiles for k, v in epoch_metrics.items()}
            print(
                f"  Critic encoder epoch {epoch + 1}/{epochs}: "
                f"loss={avg_loss:.4f} (best={best_loss:.4f}) "
                f"binary={avg_metrics.get('binary_ce', 0):.4f} "
                f"ternary={avg_metrics.get('ternary_ce', 0):.4f} "
                f"cont={avg_metrics.get('continuous_mse', 0):.4f} "
                f"[{elapsed:.0f}s]"
            )
            if log_path:
                jsonl_append(log_path, {
                    "phase": "critic_encoder",
                    "episode": epoch + 1,
                    "vf_loss": avg_loss,
                    "binary_ce": avg_metrics.get("binary_ce", 0),
                    "ternary_ce": avg_metrics.get("ternary_ce", 0),
                    "continuous_mse": avg_metrics.get("continuous_mse", 0),
                    "tiles": epoch_tiles,
                    "elapsed": elapsed,
                })

    # Restore requires_grad
    for p in critic.parameters():
        p.requires_grad_(True)

    return {"critic_encoder_loss": best_loss, "critic_encoder_epochs": epochs}


def main():
    parser = argparse.ArgumentParser(description="Pretrain tile encoders via reconstruction")
    parser.add_argument("--replay-dir", type=str, default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--actor-batch-size", type=int, default=4096)
    parser.add_argument("--critic-batch-rounds", type=int, default=64)
    parser.add_argument("--max-games", type=int, default=0, help="Limit games loaded (0=all)")
    parser.add_argument("--actor-only", action="store_true")
    parser.add_argument("--critic-only", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="Skip saving weights after pretraining")
    parser.add_argument("--session", type=str, default="",
                        help="Session name for dashboard logging")
    args = parser.parse_args()

    configure_cuda_runtime(DEVICE)
    cuda_cfg = resolve_cuda_config(DEVICE, spatial_inputs=False, enable_compile=False)
    cuda_cfg_spatial = resolve_cuda_config(DEVICE, spatial_inputs=True, enable_compile=False)

    session_name = args.session or f"encoder_pretrain_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir = SESSIONS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "train_log.jsonl"
    jsonl_init(log_path, config={
        "mode": "encoder_pretrain",
        "lr": args.lr,
        "epochs": args.epochs,
        "max_games": args.max_games,
        "actor_only": args.actor_only,
        "critic_only": args.critic_only,
        "session": session_name,
    })

    print(f"Device: {DEVICE}, replay_dir: {args.replay_dir}")
    print(f"Session: {session_name}")
    t0 = time.time()

    # Use index.jsonl to avoid slow metadata scan of all replay files
    index_path = Path(args.replay_dir) / "index.jsonl"
    if not index_path.exists():
        index_path = None
        print("Warning: no index.jsonl found, dataset scan may be slow")

    if not args.critic_only:
        print("\n=== Actor tile encoder pretraining ===")
        actor_dataset = ActorBCDataset(
            args.replay_dir, index_path=str(index_path) if index_path else None,
            winner_only=True,
        )
        if args.max_games > 0 and len(actor_dataset.games) > args.max_games:
            random.shuffle(actor_dataset.games)
            actor_dataset.games = actor_dataset.games[:args.max_games]
        print(f"  {len(actor_dataset.games)} games, ~{actor_dataset.total_samples} samples")
        model = BuilderMLP()
        model.load_weights(str(WEIGHT_PATH))
        model.to(DEVICE)
        actor_stats = pretrain_actor_encoder(
            model,
            actor_dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.actor_batch_size,
            cuda_config=cuda_cfg,
            log_path=log_path,
        )
        if not args.no_save:
            model.save_weights(str(WEIGHT_PATH))
            print(f"  Saved actor weights to {WEIGHT_PATH}")
        print(f"  Actor stats: {actor_stats}")

    if not args.actor_only:
        print("\n=== Critic tile encoder pretraining ===")
        critic_dataset = CriticBCDataset(
            args.replay_dir,
            index_path=str(index_path) if index_path else None,
        )
        if args.max_games > 0:
            random.shuffle(critic_dataset.games)
            critic_dataset.games = critic_dataset.games[:args.max_games]
        print(f"  {len(critic_dataset.games)} games, ~{critic_dataset.total_turns} turns")
        load_centralized_critic_checkpoint(WEIGHT_PATH)
        critic, _critic_opt = get_centralized_critic()
        critic_stats = pretrain_critic_encoder(
            critic,
            critic_dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_rounds=args.critic_batch_rounds,
            cuda_config=cuda_cfg_spatial,
            log_path=log_path,
        )
        if not args.no_save:
            save_centralized_critic_checkpoint(str(WEIGHT_PATH))
            print(f"  Saved critic weights")
        print(f"  Critic stats: {critic_stats}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
