# `torch`

Offline PyTorch training and tooling for `bots/tensor`.

## Files

- [`model.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/torch/model.py): model, masks, PPO helpers
- [`train_batched.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/torch/train_batched.py): main batched trainer
- [`training_utils.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/torch/training_utils.py): sessions, league, trajectory I/O
- [`replay_parser.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/torch/replay_parser.py): replay augmentation
- [`bench_episode.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/torch/bench_episode.py): one-episode benchmark
- [`distill.py`](/home/marvin/Documents/GitHub/cb-bc/bots/tensor/torch/distill.py): teacher -> student distillation

## Main Commands

```bash
python bots/tensor/torch/train_batched.py
python bots/tensor/torch/train_batched.py --dummy-episodes 10
python bots/tensor/torch/bench_episode.py --json
python bots/tensor/torch/bench_episode.py --dummy-opponent --json
python bots/tensor/torch/distill.py --session <session_name>
```

## Exporting Weights

Training auto-exports to `bots/tensor/weights.json` on completion (unless `--teacher`).
To deploy weights to `tensor_inference` from a session checkpoint:

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
