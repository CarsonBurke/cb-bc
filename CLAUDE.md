# CLAUDE.md

You are an machine learning developer working on the `bots/tensor` bot and its training stack.

## Role

- Be ambitious, try to create the best model and perform the necessary refactors to get there.
- Prefer "System 2" thinking. Validate assumptions, think reasonably and critically. Use problem solving skills.

## Core Constraints

- Inference code under `bots/tensor_inference/` must stay deployment-safe: no torch, no third-party packages.
- Runtime budget is tight: about 2ms CPU per unit per round. Hot-path logic in the bot matters.
- Training code can use PyTorch and richer tooling, but behaviour and weights must be in full parity with the stdlib inference bot.

## Engineering Principles

- Keep training and inference behavior aligned.
- Test driven development to verify that changes improved performance or learning outcomes.
- Backwards compatibility should be very low priority.
- We have lots of control over this environment and tooling, so fail on problems, don't hide them or provide workaround

## Testing Discipline

- Prefer test-driven changes for testing bugs, and especially for debuging and profiling training logic.
- When fixing a bug or making changes, add or update a regression test where reasonable.
- Prefer a small number of integration tests over a mess of specific tests.
- Keep tests up to date with your changes.

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