"""tensor_lib — Lightweight neural network library using only Python stdlib.

Provides:
- Tensor: N-dimensional tensor (forward-pass only)
- Neural network layers: Linear, Tanh, Sigmoid, SiLU, RMSNorm, Sequential, MLP
- Position embeddings: Sinusoidal

Training is done in PyTorch (bots/tensor/torch/).
"""

from .tensor import Tensor
from .nn import (
    Linear,
    Tanh,
    SiLU,
    Sigmoid,
    RMSNorm,
    Sequential,
    MLP,
)
from .embed import SinusoidalPositionEmbedding

__all__ = [
    "Tensor",
    "Linear",
    "Tanh",
    "SiLU",
    "Sigmoid",
    "RMSNorm",
    "Sequential",
    "MLP",
    "SinusoidalPositionEmbedding",
]
