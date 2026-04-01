from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"
TENSOR_DIR = PROJECT_ROOT / "bots" / "tensor"
INFER_DIR = PROJECT_ROOT / "bots" / "tensor_inference"

for path in (TORCH_DIR, TENSOR_DIR, INFER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model as torch_model
import shared as shared_mod


def test_shared_actor_constants_match_torch_actor():
    assert shared_mod.HIDDEN_SIZES == torch_model.HIDDEN
    assert shared_mod.H_DIM == torch_model.H_DIM
    assert shared_mod.COMPRESS_DIM == torch_model.COMPRESS_DIM
    assert len(shared_mod.BUILD_ACTIONS) == torch_model.N_BUILD
    assert len(shared_mod.BRIDGE_OFFSETS) == torch_model.N_BRIDGE_TARGETS
