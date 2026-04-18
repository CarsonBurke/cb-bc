# `torch`

Offline PyTorch training and tooling for the tensor bot.

## Architecture

```
tensor/shared.py          ← constants, obs encoding, masks, rewards (stdlib-only)
tensor/torch/             ← PyTorch training (this directory)
tensor_proxy/main.py      ← training bot (pipes obs to torch, imports shared.py)
tensor_proxy_opp/main.py  ← opponent proxy (same, no trajectory logging)
tensor_opp/main.py        ← self-play opponent (MLP inference via tensor_inference/tensor_lib)
tensor_inference/          ← deploy bot (stdlib-only MLP, bundled weights)
```

## Files

- [`model.py`](model.py): model, masks, PPO helpers
- [`train_batched.py`](train_batched.py): main batched trainer
- [`training_utils.py`](training_utils.py): sessions, league, trajectory I/O
- [`bench_episode.py`](bench_episode.py): one-episode benchmark
- [`bench_critic.py`](bench_critic.py): critic benchmarks
- [`distill.py`](distill.py): teacher -> student distillation
- [`game_worker.py`](game_worker.py): subprocess game runner

## Main Commands

```bash
python bots/tensor/torch/train_batched.py
python bots/tensor/torch/train_batched.py --dummy-episodes 10
python bots/tensor/torch/full_train.py --session <name>
python bots/tensor/torch/bench_episode.py --json
python bots/tensor/torch/bench_episode.py --dummy-opponent --json
python bots/tensor/torch/distill.py --session <session_name>
```

## Full Training

Use [`full_train.py`](full_train.py) for the normal end-to-end pipeline:

1. actor BC pretraining
2. critic pretraining using the actor-BC checkpoint for replay actor latents
3. PPO fine-tuning using both pretrained checkpoints as initialization

Example:

```bash
python bots/tensor/torch/full_train.py \
  --session full_run_a \
  --actor-epochs 2 \
  --critic-epochs 1 \
  --ppo-episodes 100
```

Dry-run the exact command handoff first:

```bash
python bots/tensor/torch/full_train.py --session full_run_a --dry-run
```

Notes:

- PPO still uses the pretrained actor and critic weights as initialization, exactly as before.
- Standalone critic pretraining now expects `--actor-weights`; the intended direction is to pretrain the actor first, then pretrain the critic against that actor's latent space.
- The pretrain scripts default to the fast path: pinned CPU staging, batched transfers, `torch.compile` on CUDA, and mixed precision where the codepath supports it. Only use `--actor-no-compile` or `--critic-no-compile` when debugging.

## Exporting Weights

Training auto-exports to `bots/tensor_inference/weights.json` on completion (unless `--teacher`).
To manually deploy from a session checkpoint:

```bash
python3.13 -c "
import json, pickle
w = pickle.load(open('bots/tensor/sessions/<session>/weights.pkl', 'rb'))
json.dump(w, open('bots/tensor_inference/weights.json', 'w'))
"
```

Only the policy weights are exported — the critic is excluded.
For running and submitting, use `bots/tensor_inference` (see its README).

## Notes

- `--opponents` and `--games-per-opp` control total games per episode.
- `--dummy-episodes` uses a scripted side-B policy for the first `N` episodes.
- Sessions are written under [`../sessions`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/sessions).
