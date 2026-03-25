# CLAUDE.md

You are an machine learning developer working on the `bots/tensor` bot and its training stack.

## Core Constraints

- The shipped bot must run in sandboxed CPython with stdlib only.
- Inference code under `bots/tensor_inference/` must stay deployment-safe: no torch, no third-party packages.
- Runtime budget is tight: about 2ms CPU per unit per round. Hot-path logic in the bot matters.
- Training code can use PyTorch and richer tooling, but exported behavior must still match the stdlib inference bot.

## Engineering Priorities

- Keep training and inference behavior aligned.
- Be ambitious, try to create the best model and perform the necessary refactors to get there.
- Test driven development to verify that changes improved performance or learning outcomes.

## Testing Discipline

- Prefer test-driven changes for bugs and training logic.
- When fixing a bug or making changes, add or update a focused regression test first when practical.
- Prefer a small number of integration tests over a mess of specific tests.

## Validation

Use the project venv tools when validating behavior:

```bash
# do training
python3 -u bots/tensor/torch/train_batched.py --session sp_[datetime] --episodes 500 --opponents 4 --games-per-opp 4 --dummy-episodes 500
# do tests
pytest -q <targeted tests> 
# do prod verification
.venv/bin/cambc run <bot_a> <bot_b> [map.map26]
.venv/bin/cambc test-run <bot_a> <bot_b> [map.map26]
```