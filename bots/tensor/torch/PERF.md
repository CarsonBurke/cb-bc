# Training Pipeline Performance Review

Bottlenecks identified across `model.py`, `train_selfplay.py`, `train_batched.py`, and `train_gpu.py`.

## Implemented

### 1. `model.py` — GAE vectorized on GPU (was: scalar loop with GPU syncs)

**Before:** Python loop per-timestep with `values[t+1]` scalar indexing on CUDA → ~48k GPU syncs for self-play.
**After:** Pad episodes to `(E, max_len)`, reverse-scan vectorized across all episodes per timestep.
**Result:** 24.3ms vs 579ms → **23.9x speedup** (32k steps, 30 episodes, RTX 5090)

### 2. `train_selfplay.py` — Single pool for all opponents

**Before:** Sequential per-opponent `collect_games` calls, each creating/destroying an `mp.Pool`.
**After:** `collect_all_games` submits all games across all opponents in one pool. `_run_game_with_opp` passes opponent weight path per-game via env var.

### 3. `train_gpu.py` — Position embedding vectorized

**Before:** Python loop over `_POS_QUARTER` with 4 tensor ops per iteration.
**After:** Single broadcast multiply + sin/cos + stack (interleaved to match main.py ordering).

### 4. `model.py` — Chunked value forward pass

**Before:** `model(obs, h)` on all N observations at once → OOM risk for large batches.
**After:** Chunked in 8192-sample batches.

### 5. `train_gpu.py` — Resource flow: gather-based + correctness fix

**Before:** 8-direction `shift_2d` loop per chain iteration (~120 kernel launches/step). Also had a **sign bug**: `shift_2d(connected, dy, dx)` checks `connected[y-dy, x-dx]` (opposite of where conveyor points), meaning conveyors near core were incorrectly marked as disconnected.
**After:** Precompute target indices via gather, one gather+mask per chain iteration. `max_pool2d` for adjacency check (replaces 9 shift ops). Connectivity direction is now correct.
**Result:** 3.0ms vs 3.6ms → **1.2x speedup**. Delivering harvesters doubled (0.26 → 0.50) due to bugfix.

### End-to-end `train_gpu.py` benchmark (2048 envs, 200 steps, RTX 5090)

~133k samples/sec, ~3.1s/episode (1.7s collect + 1.4s train).

---

## Not yet implemented

### JSON weight serialization

`json.dump` on 50k+ floats is 10-50× slower than binary. Use `torch.save`/pickle for league checkpoints, keep JSON only for final bot export.

### `observe()` redundant allocations

Each call does 3× `F.pad`, `torch.meshgrid`, multiple `gather`/`one_hot`/`cat`. Cache padded env grid (only changes on build), precompute meshgrid offsets in `__init__`.

### Collection/training overlap

All scripts run collection and training sequentially. Double-buffer with background thread for collection while GPU trains.
