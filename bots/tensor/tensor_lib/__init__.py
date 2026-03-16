"""tensor_lib — Lightweight neural network library using only Python stdlib.

Provides:
- Tensor: N-dimensional tensor with reverse-mode autograd
- Neural network layers: Linear, ReLU, Tanh, Sigmoid, LayerNorm, Dropout, MLP
- Optimizers: SGD, Adam, RMSProp
- Position embeddings: Sinusoidal, Learned, RelativeGrid
"""

from .tensor import Tensor
from .nn import (
    Linear,
    ReLU,
    Tanh,
    SiLU,
    Sigmoid,
    LayerNorm,
    Dropout,
    Sequential,
    MLP,
)
from .optim import SGD, Adam, RMSProp
from .embed import (
    SinusoidalPositionEmbedding,
    LearnedPositionEmbedding,
    RelativeGridEmbedding,
)

__all__ = [
    "Tensor",
    "Linear",
    "ReLU",
    "Tanh",
    "SiLU",
    "Sigmoid",
    "LayerNorm",
    "Dropout",
    "Sequential",
    "MLP",
    "SGD",
    "Adam",
    "RMSProp",
    "SinusoidalPositionEmbedding",
    "LearnedPositionEmbedding",
    "RelativeGridEmbedding",
]
