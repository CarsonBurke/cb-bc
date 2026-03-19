# `tensor_lib`

Stdlib-only runtime NN helpers for [`bots/tensor`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor).

Used by the in-engine bot, not the GPU training stack.

## Files

- [`tensor.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/tensor_lib/tensor.py): flat-list tensor (forward-pass only)
- [`nn.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/tensor_lib/nn.py): layers and `MLP`
- [`embed.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/tensor_lib/embed.py): positional embeddings
- [`__init__.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/tensor_lib/__init__.py): public exports

## Scope

- Runtime inference
- Weight import/export compatibility
- Small stdlib-only experiments

For training, see [`../torch/README.md`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/torch/README.md).
