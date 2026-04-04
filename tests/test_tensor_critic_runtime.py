"""Regression coverage for PPO critic runtime configuration."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))

import model as model_mod


def test_resolve_critic_minibatch_size_is_n_over_batches():
    # Auto: n_steps / critic_batches_per_epoch (pure ratio, no floor)
    auto = model_mod._resolve_critic_minibatch_size(
        n_steps=16000,
        critic_minibatch_size=None,
        critic_batches_per_epoch=8,
    )
    assert auto == 2000

    # Explicit override wins
    explicit = model_mod._resolve_critic_minibatch_size(
        n_steps=16000,
        critic_minibatch_size=4096,
        critic_batches_per_epoch=8,
    )
    assert explicit == 4096

    # Small dataset: no floor, just n/8
    small = model_mod._resolve_critic_minibatch_size(
        n_steps=100,
        critic_minibatch_size=None,
        critic_batches_per_epoch=8,
    )
    assert small == 13  # ceil(100/8)


def test_configure_centralized_critic_runtime_forwards_compile_flag(monkeypatch):
    seen: dict[str, object] = {}

    def fake_resolve(device, *, spatial_inputs, enable_compile, **_kwargs):
        seen["device"] = device
        seen["spatial_inputs"] = spatial_inputs
        seen["enable_compile"] = enable_compile
        return model_mod.PretrainCudaConfig(
            enabled=False,
            amp_dtype=None,
            channels_last=False,
            compile_model=False,
        )

    monkeypatch.setattr(model_mod, "resolve_cuda_config", fake_resolve)
    monkeypatch.setattr(model_mod, "_centralized_critic_step", object())
    monkeypatch.setattr(model_mod, "_centralized_critic_loss_step", object())

    model_mod.configure_centralized_critic_runtime(enable_compile=False)

    assert seen["device"] == model_mod.DEVICE
    assert seen["spatial_inputs"] is True
    assert seen["enable_compile"] is False
    assert model_mod._centralized_critic_step is None
    assert model_mod._centralized_critic_loss_step is None
