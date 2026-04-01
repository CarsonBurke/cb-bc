from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INFER_DIR = PROJECT_ROOT / "bots" / "tensor_inference"

if str(INFER_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_DIR))

from tensor_lib.nn import MLP


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
    assert out["h_cand"][0] > 0.96
    assert out["other"] == [2.0]


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

    assert out["out"] == [3.0]
