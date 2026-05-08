"""tensor_lib — Lightweight neural network library using only Python stdlib.

Provides:
- Tensor: N-dimensional tensor (forward-pass only)
- Neural network layers: Linear, ReLU, Tanh, Sigmoid, LayerNorm, Dropout, MLP
- Position embeddings: Sinusoidal

Training is done in PyTorch (bots/tensor/torch/).
"""

from .tensor import Tensor
from .nn import (
    Linear,
    ReLU,
    Tanh,
    SiLU,
    Sigmoid,
    LayerNorm,
    RMSNorm,
    Dropout,
    Sequential,
    MLP,
)
from .actor import ActorModel
from .embed import SinusoidalPositionEmbedding

__all__ = [
    "Tensor",
    "Linear",
    "ReLU",
    "Tanh",
    "SiLU",
    "Sigmoid",
    "LayerNorm",
    "RMSNorm",
    "Dropout",
    "Sequential",
    "MLP",
    "ActorModel",
    "SinusoidalPositionEmbedding",
]
