from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFER_DIR = PROJECT_ROOT / "bots" / "tensor_inference"
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"

for path in (PROJECT_ROOT, INFER_DIR, TORCH_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tensor_lib import ActorModel
from tensor_lib.nn import MLP
from model import BuilderMLP, OBS_SIZE, H_DIM
import main as infer_main
from bench_obs import FakeController


def _make_valid_packed_obs():
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
    return obs, h_prev, packed


def test_infer_applies_tanh_to_h_cand_only():
    model = MLP(
        input_size=1,
        hidden_sizes=[],
        output_heads={"h_cand": 1, "other": 1},
    )
    model.heads["h_cand"].weight.data[:] = [1.0]
    model.heads["h_cand"].bias.data[:] = [0.0]
    model.heads["other"].weight.data[:] = [1.0]
    model.heads["other"].bias.data[:] = [0.0]

    out = model.infer([2.0])

    assert out["h_cand"][0] < 1.0
    assert out["h_cand"][0] == pytest.approx(torch.tanh(torch.tensor(1.0)).item(), abs=1e-6)
    assert out["other"] == pytest.approx([1.0], abs=1e-6)


def test_head_projection_keeps_heads_narrower_than_trunk():
    model = MLP(
        input_size=1,
        hidden_sizes=[2],
        head_input_size=1,
        output_heads={"out": 1},
    )
    model.trunk.layers[0].weight.data[:] = [1.0, 0.0]
    model.trunk.layers[0].bias.data[:] = [0.0, 0.0]
    model.head_projection.weight.data[:] = [1.0, 0.0]
    model.head_projection.bias.data[:] = [0.0]
    model.heads["out"].weight.data[:] = [1.0]
    model.heads["out"].bias.data[:] = [0.0]

    out = model.infer([3.0])

    assert out["out"] == pytest.approx([1.0], abs=1e-6)


def test_actor_model_matches_exported_torch_actor():
    torch_model = BuilderMLP().eval()
    state = torch_model.export_weights()
    infer_model = ActorModel(
        output_heads={
            "build_type": 13,
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": 28,
            "move_dir": 9,
            "temperature": 5,
            "h_gate": H_DIM,
            "h_cand": H_DIM,
        },
    )
    infer_model.load(state)

    obs, h_prev, packed = _make_valid_packed_obs()

    with torch.no_grad():
        bt_l, bd_l, cd_l, brt_l, md_l, _head_temperature, h_next = torch_model.infer_packed(packed)
        latent = torch_model.actor_latent(packed[:, :OBS_SIZE], packed[:, OBS_SIZE:])
        recurrent_in = torch.cat([latent, h_prev], dim=-1)
        gate = torch_model.head_h_gate(recurrent_in)
        cand = torch.tanh(torch_model.head_h_cand(recurrent_in))
    infer_out = infer_model.infer(obs + h_prev[0].tolist())

    assert infer_out["build_type"] == pytest.approx(bt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["build_dir"] == pytest.approx(bd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["conv_dir"] == pytest.approx(cd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["bridge_target"] == pytest.approx(brt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["move_dir"] == pytest.approx(md_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_gate"] == pytest.approx(gate[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_cand"] == pytest.approx(cand[0].tolist(), rel=1e-5, abs=1e-5)


def test_tile_rbf_gamma_gradient_matches_explicit_logits():
    model = BuilderMLP().eval()
    packed = torch.randn(2, OBS_SIZE + H_DIM)

    with torch.no_grad():
        tiles, _ = model._split_obs_pre(model._obs_encoder_pre(packed[:, :OBS_SIZE]))
        q = model.tile_sa_q_proj(tiles)
        k = model.tile_sa_k_proj(tiles)
        v = model.tile_sa_v_proj(tiles)

    gamma_param_actual = model.tile_sa_gamma.detach().clone().requires_grad_(True)
    gamma_actual = gamma_param_actual.exp()
    k_norm_sq = (k * k).sum(dim=-1, keepdim=True).transpose(-1, -2)
    gamma_scale = torch.sqrt(2.0 * gamma_actual)
    actual = F.scaled_dot_product_attention(
        (q * gamma_scale).unsqueeze(1),
        (k * gamma_scale).unsqueeze(1),
        v.unsqueeze(1),
        attn_mask=(-gamma_actual * k_norm_sq).unsqueeze(1),
        scale=1.0,
    ).squeeze(1)
    actual.sum().backward()

    gamma_param_expected = model.tile_sa_gamma.detach().clone().requires_grad_(True)
    gamma_expected = gamma_param_expected.exp()
    qk = torch.bmm(q, k.transpose(-1, -2))
    expected_scores = 2.0 * gamma_expected * qk - gamma_expected * k_norm_sq
    expected = torch.bmm(torch.softmax(expected_scores, dim=-1), v)
    expected.sum().backward()

    assert gamma_param_actual.grad is not None
    assert gamma_param_expected.grad is not None
    torch.testing.assert_close(
        gamma_param_actual.grad,
        gamma_param_expected.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_actor_load_accepts_legacy_state_schema():
    torch_model = BuilderMLP().eval()
    full_state = torch_model.export_weights()
    legacy_state = full_state[:-16]

    infer_model = ActorModel(
        output_heads={
            "build_type": 13,
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": 28,
            "move_dir": 9,
            "temperature": 5,
            "h_gate": H_DIM,
            "h_cand": H_DIM,
        },
    )
    infer_model.load(full_state)
    infer_model.load(legacy_state)

    compat_model = BuilderMLP().eval()
    compat_model.import_weights(full_state)
    compat_model.import_weights(legacy_state)

    obs, h_prev, packed = _make_valid_packed_obs()

    with torch.no_grad():
        bt_l, bd_l, cd_l, brt_l, md_l, _head_temperature, _h_next = compat_model.infer_packed(packed)
    infer_out = infer_model.infer(obs + h_prev[0].tolist())

    assert torch.isfinite(bt_l).all()
    assert infer_out["build_type"] == pytest.approx(bt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["build_dir"] == pytest.approx(bd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["conv_dir"] == pytest.approx(cd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["bridge_target"] == pytest.approx(brt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["move_dir"] == pytest.approx(md_l[0].tolist(), rel=1e-5, abs=1e-5)


def test_actor_load_rejects_partial_truncation():
    full_state = BuilderMLP().eval().export_weights()
    truncated_state = full_state[:-1]

    infer_model = ActorModel(
        output_heads={
            "build_type": 13,
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": 28,
            "move_dir": 9,
            "temperature": 5,
            "h_gate": H_DIM,
            "h_cand": H_DIM,
        },
    )

    with pytest.raises(AssertionError, match="expected"):
        infer_model.load(truncated_state)

    with pytest.raises(AssertionError, match="expected"):
        BuilderMLP().eval().import_weights(truncated_state)


def test_actor_model_matches_exported_torch_actor_on_symbolic_obs():
    torch_model = BuilderMLP().eval()
    state = torch_model.export_weights()
    infer_model = ActorModel(
        output_heads={
            "build_type": 13,
            "build_dir": 9,
            "conv_dir": 4,
            "bridge_target": 28,
            "move_dir": 9,
            "temperature": 5,
            "h_gate": H_DIM,
            "h_cand": H_DIM,
        },
    )
    infer_model.load(state)

    obs, _h_prev, packed = _make_valid_packed_obs()

    with torch.no_grad():
        bt_l, bd_l, cd_l, brt_l, md_l, _head_temperature, _h_next = torch_model.infer_packed(packed)
        latent = torch_model.actor_latent(packed[:, :OBS_SIZE], packed[:, OBS_SIZE:])
        recurrent_in = torch.cat([latent, packed[:, OBS_SIZE:]], dim=-1)
        gate = torch_model.head_h_gate(recurrent_in)
        cand = torch.tanh(torch_model.head_h_cand(recurrent_in))
    infer_out = infer_model.infer(obs + packed[0, OBS_SIZE:].tolist())

    assert infer_out["build_type"] == pytest.approx(bt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["build_dir"] == pytest.approx(bd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["conv_dir"] == pytest.approx(cd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["bridge_target"] == pytest.approx(brt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["move_dir"] == pytest.approx(md_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_gate"] == pytest.approx(gate[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_cand"] == pytest.approx(cand[0].tolist(), rel=1e-5, abs=1e-5)
