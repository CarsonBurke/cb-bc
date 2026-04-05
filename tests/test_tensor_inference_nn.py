from pathlib import Path
import sys

import pytest
import torch

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

    obs = torch.randn(1, OBS_SIZE)
    h_prev = torch.randn(1, H_DIM)
    packed = torch.cat([obs, h_prev], dim=-1)

    with torch.no_grad():
        bt_l, bd_l, cd_l, brt_l, md_l, _head_temperature, h_next = torch_model.infer_packed(packed)
        latent = torch_model.actor_latent(obs, h_prev)
        recurrent_in = torch.cat([latent, h_prev], dim=-1)
        gate = torch_model.head_h_gate(recurrent_in)
        cand = torch.tanh(torch_model.head_h_cand(recurrent_in))
    infer_out = infer_model.infer(obs[0].tolist() + h_prev[0].tolist())

    assert infer_out["build_type"] == pytest.approx(bt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["build_dir"] == pytest.approx(bd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["conv_dir"] == pytest.approx(cd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["bridge_target"] == pytest.approx(brt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["move_dir"] == pytest.approx(md_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_gate"] == pytest.approx(gate[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_cand"] == pytest.approx(cand[0].tolist(), rel=1e-5, abs=1e-5)


def test_actor_load_accepts_legacy_state_prefix():
    torch_model = BuilderMLP().eval()
    full_state = torch_model.export_weights()
    legacy_state = full_state[:-11]

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

    obs = torch.randn(1, OBS_SIZE)
    h_prev = torch.randn(1, H_DIM)
    packed = torch.cat([obs, h_prev], dim=-1)

    with torch.no_grad():
        bt_l, bd_l, cd_l, brt_l, md_l, _head_temperature, _h_next = compat_model.infer_packed(packed)
    infer_out = infer_model.infer(obs[0].tolist() + h_prev[0].tolist())

    assert torch.isfinite(bt_l).all()
    assert infer_out["build_type"] == pytest.approx(bt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["build_dir"] == pytest.approx(bd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["conv_dir"] == pytest.approx(cd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["bridge_target"] == pytest.approx(brt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["move_dir"] == pytest.approx(md_l[0].tolist(), rel=1e-5, abs=1e-5)


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
        assume_symbolic_tiles=True,
    )
    infer_model.load(state)

    ct = FakeController()
    player = infer_main.Player()
    player.my_team = ct.get_team()
    player.core_center = infer_main.Position(18, 20)
    player.enemy_core = infer_main.Position(25, 22)
    player._spawn_round = 100
    player._prev_resources = (160.0, 35.0)

    obs = player._encode_state(ct, ct.nearby_tiles, ct.nearby_entities)
    h_prev = [0.0] * H_DIM
    packed = torch.tensor([obs + h_prev], dtype=torch.float32)

    with torch.no_grad():
        bt_l, bd_l, cd_l, brt_l, md_l, _head_temperature, _h_next = torch_model.infer_packed(packed)
        latent = torch_model.actor_latent(packed[:, :OBS_SIZE], packed[:, OBS_SIZE:])
        recurrent_in = torch.cat([latent, packed[:, OBS_SIZE:]], dim=-1)
        gate = torch_model.head_h_gate(recurrent_in)
        cand = torch.tanh(torch_model.head_h_cand(recurrent_in))
    infer_out = infer_model.infer(obs + h_prev)

    assert infer_out["build_type"] == pytest.approx(bt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["build_dir"] == pytest.approx(bd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["conv_dir"] == pytest.approx(cd_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["bridge_target"] == pytest.approx(brt_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["move_dir"] == pytest.approx(md_l[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_gate"] == pytest.approx(gate[0].tolist(), rel=1e-5, abs=1e-5)
    assert infer_out["h_cand"] == pytest.approx(cand[0].tolist(), rel=1e-5, abs=1e-5)
