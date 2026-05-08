"""Regression coverage for PPO critic runtime configuration."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))

import model as model_mod


def test_resolve_critic_minibatch_size_is_fixed_n_over_eight():
    assert model_mod._resolve_critic_minibatch_size(n_steps=16000) == 2000
    assert model_mod._resolve_critic_minibatch_size(n_steps=100) == 13


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


def test_load_centralized_critic_checkpoint_rejects_nonfinite_weights(monkeypatch, tmp_path):
    reset_calls: list[bool] = []
    load_calls: list[dict[str, torch.Tensor]] = []

    class DummyCritic:
        def load_state_dict(self, state):
            load_calls.append(state)

    class DummyOpt:
        def load_state_dict(self, state):
            raise AssertionError("optimizer state should not load after finite check fails")

    monkeypatch.setattr(model_mod, "reset_centralized_critic", lambda: reset_calls.append(True))
    monkeypatch.setattr(
        model_mod,
        "get_centralized_critic",
        lambda lr=3e-4: (DummyCritic(), DummyOpt()),
    )

    actor_weight_path = tmp_path / "weights.json"
    ckpt_path = actor_weight_path.with_name("weights.critic.pt")
    torch.save(
        {
            "critic": {"global_cls": torch.tensor([float("nan")])},
            "optimizer": {"state": {0: {"exp_avg": torch.tensor([1.0])}}},
        },
        ckpt_path,
    )

    assert not model_mod.load_centralized_critic_checkpoint(actor_weight_path)
    assert reset_calls == [True, True]
    assert load_calls == []
