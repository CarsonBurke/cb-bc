Test the tensor bot remotely with CPU enforcement against an opponent.

## Steps

1. Find the most recent session with weights matching the current `TILE_FEATURES` and `GLOBAL_FEATURES` layout by checking first-layer size in `weights.pkl` files under `bots/tensor/sessions/`.
2. Export those weights to `bots/tensor/weights.json` (pickle → JSON).
3. Create a temp copy of `bots/tensor/` excluding `sessions/`, `torch/`, `__pycache__/`, `league/`, and `*.pkl`.
4. Run `.venv/bin/cambc test-run <tmpdir>/tensor bots/$1` (default opponent: `starter`).
5. Poll `.venv/bin/cambc test-matches` until the match completes.
6. Report the score and clean up the temp directory.
