Test the tensor bot remotely with CPU enforcement against an opponent.

## Steps

1. Find the most recent session with weights matching the current `TILE_FEATURES` and `GLOBAL_FEATURES` layout by checking first-layer size in `weights.pkl` files under `bots/tensor/sessions/`.
2. Export those weights to `bots/tensor_inference/weights.json` (pickle → JSON).
3. Run `.venv/bin/cambc test-run tensor_inference $1` (default opponent: `starter`).
4. Poll `.venv/bin/cambc test-matches` until the match completes.
5. Report the score.
