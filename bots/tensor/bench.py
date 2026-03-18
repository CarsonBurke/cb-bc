"""Benchmarks for tensor_lib — measures per-call latency of core operations.

Usage:
    python bots/tensor/bench.py              # Run all benchmarks
    python bots/tensor/bench.py matmul       # Run specific benchmark(s)

Reports median and p95 times. Sizes are chosen to reflect typical
in-game neural network dimensions (input ~30-60, hidden 32-64, output 5-10).
"""

import sys
import time
import random
import gc

gc.disable()

sys.path.insert(0, "bots/tensor")
from tensor_lib import Tensor, Linear, ReLU, SiLU, LayerNorm, Sequential, MLP, Adam, SGD
from tensor_lib.embed import SinusoidalPositionEmbedding, LearnedPositionEmbedding, RelativeGridEmbedding
from main import (
    INPUT_SIZE as PROD_INPUT_SIZE,
    HIDDEN_SIZES as PROD_HIDDEN_SIZES,
    H_DIM as PROD_H_DIM,
    BUILD_ACTIONS as PROD_BUILD_ACTIONS,
    POS_EMBED_DIM as PROD_POS_EMBED_DIM,
    RMSNORM_LAYERS as PROD_RMSNORM_LAYERS,
)


def bench(fn, n=2000, warmup=10, label=""):
    """Run fn n times, report median and p95 in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    med = times[n // 2]
    p95 = times[int(n * 0.95)]
    print(f"  {label:40s}  median {med:8.1f} us   p95 {p95:8.1f} us")
    return med


# ── Tensor creation ──────────────────────────────────────────────

def bench_creation():
    print("\n== Tensor Creation ==")
    bench(lambda: Tensor.zeros((64, 32)), label="zeros (64, 32)")
    bench(lambda: Tensor.randn((64, 32)), label="randn (64, 32)")
    bench(lambda: Tensor.xavier(32, 16), label="xavier (32, 16)")
    bench(lambda: Tensor([0.0] * 2048, (64, 32)), label="from flat list (2048)")


# ── Elementwise ops ──────────────────────────────────────────────

def bench_elementwise():
    print("\n== Elementwise Ops ==")
    a = Tensor.randn((8, 32), requires_grad=True)
    b = Tensor.randn((8, 32), requires_grad=True)
    bias = Tensor.randn((32,), requires_grad=True)

    bench(lambda: a + b, label="add (8,32) + (8,32)")
    bench(lambda: a * b, label="mul (8,32) * (8,32)")
    bench(lambda: a + bias, label="add broadcast (8,32) + (32,)")
    bench(lambda: a * 0.5, label="scalar mul (8,32) * 0.5")
    bench(lambda: a - b, label="sub (8,32) - (8,32)")


# ── Matmul ───────────────────────────────────────────────────────

def bench_matmul():
    print("\n== Matmul ==")
    sizes = [
        ((1, 32), (32, 16), "1x32 @ 32x16 (single input)"),
        ((4, 32), (32, 16), "4x32 @ 32x16 (small batch)"),
        ((8, 32), (32, 32), "8x32 @ 32x32 (hidden layer)"),
        ((8, 64), (64, 32), "8x64 @ 64x32 (larger hidden)"),
        ((1, 64), (64, 64), "1x64 @ 64x64 (single, wide)"),
        ((16, 32), (32, 8), "16x32 @ 32x8 (output head)"),
    ]
    for (m, k), (_, n), label in sizes:
        a = Tensor.randn((m, k), requires_grad=True)
        b = Tensor.randn((k, n), requires_grad=True)
        bench(lambda a=a, b=b: a @ b, label=label)


# ── Transpose ────────────────────────────────────────────────────

def bench_transpose():
    print("\n== Transpose ==")
    a = Tensor.randn((32, 64), requires_grad=True)
    bench(lambda: a.T(), label="transpose (32, 64)")
    b = Tensor.randn((64, 64))
    bench(lambda: b.T(), label="transpose (64, 64)")


# ── Activations ──────────────────────────────────────────────────

def bench_activations():
    print("\n== Activations ==")
    x = Tensor.randn((8, 32), requires_grad=True)
    bench(lambda: x.relu(), label="relu (8, 32)")
    bench(lambda: x.tanh(), label="tanh (8, 32)")
    bench(lambda: x.silu(), label="silu (8, 32)")
    bench(lambda: x.sigmoid(), label="sigmoid (8, 32)")
    bench(lambda: x.softmax(), label="softmax (8, 32)")

    x1 = Tensor.randn((1, 32), requires_grad=True)
    bench(lambda: x1.softmax(), label="softmax (1, 32)")


# ── Reductions ───────────────────────────────────────────────────

def bench_reductions():
    print("\n== Reductions ==")
    x = Tensor.randn((8, 32), requires_grad=True)
    bench(lambda: x.sum(), label="sum (8, 32)")
    bench(lambda: x.mean(), label="mean (8, 32)")
    bench(lambda: x.log(), label="log (8, 32)")
    bench(lambda: x.clamp(-1.0, 1.0), label="clamp (8, 32)")


# ── Layers ───────────────────────────────────────────────────────

def bench_layers():
    print("\n== Layers ==")
    x1 = Tensor.randn((1, 32))
    x8 = Tensor.randn((8, 32))

    lin = Linear(32, 16)
    bench(lambda: lin(x1), label="Linear(32->16) batch=1")
    bench(lambda: lin(x8), label="Linear(32->16) batch=8")

    lin_big = Linear(64, 32)
    x64 = Tensor.randn((1, 64))
    bench(lambda: lin_big(x64), label="Linear(64->32) batch=1")

    ln = LayerNorm(32)
    bench(lambda: ln(x8), label="LayerNorm(32) batch=8")

    relu = ReLU()
    bench(lambda: relu(x8), label="ReLU layer (8, 32)")

    silu = SiLU()
    bench(lambda: silu(x8), label="SiLU layer (8, 32)")


# ── Sequential / MLP ────────────────────────────────────────────

def bench_mlp():
    print("\n== MLP Forward ==")
    x = Tensor.randn((1, 32))

    net_small = Sequential(Linear(32, 16), ReLU(), Linear(16, 8))
    bench(lambda: net_small(x), label="Sequential 32->16->8 batch=1")

    mlp = MLP(32, [32], {"action": 5, "value": 1}, activation="relu")
    bench(lambda: mlp(x), label="MLP 32->[32]->{5,1} batch=1")

    mlp_ln = MLP(32, [32], {"action": 5}, activation="silu", use_layernorm=True)
    bench(lambda: mlp_ln(x), label="MLP+LN 32->[32]->{5} batch=1")

    mlp_big = MLP(64, [64, 32], {"action": 8, "value": 1}, activation="relu")
    x64 = Tensor.randn((1, 64))
    bench(lambda: mlp_big(x64), label="MLP 64->[64,32]->{8,1} batch=1")


# ── Backward pass ────────────────────────────────────────────────

def bench_backward():
    print("\n== Backward Pass ==")

    def forward_backward_linear():
        x = Tensor.randn((1, 32), requires_grad=True)
        lin = Linear(32, 16)
        out = lin(x)
        loss = out.sum()
        loss.backward()

    bench(forward_backward_linear, label="Linear(32->16) fwd+bwd")

    def forward_backward_mlp():
        x = Tensor.randn((1, 32), requires_grad=True)
        net = Sequential(Linear(32, 32), ReLU(), Linear(32, 5))
        out = net(x)
        loss = out.sum()
        loss.backward()

    bench(forward_backward_mlp, label="Seq 32->32->5 fwd+bwd")

    def forward_backward_mlp_heads():
        x = Tensor.randn((1, 32), requires_grad=True)
        mlp = MLP(32, [32], {"action": 5, "value": 1}, activation="relu")
        heads = mlp(x)
        loss = heads["action"].sum() + heads["value"].sum()
        loss.backward()

    bench(forward_backward_mlp_heads, label="MLP 32->[32]->{5,1} fwd+bwd")


# ── Optimizer step ───────────────────────────────────────────────

def bench_optimizers():
    print("\n== Optimizer Step ==")

    def make_params():
        return [Tensor.randn((32, 32), requires_grad=True),
                Tensor.randn((32,), requires_grad=True),
                Tensor.randn((32, 8), requires_grad=True),
                Tensor.randn((8,), requires_grad=True)]

    params = make_params()
    for p in params:
        p.grad = Tensor.randn(p.shape)
    sgd = SGD(params, lr=0.01)
    bench(lambda: sgd.step(), label="SGD step (1280 params)")

    params = make_params()
    for p in params:
        p.grad = Tensor.randn(p.shape)
    adam = Adam(params, lr=0.001)
    bench(lambda: adam.step(), label="Adam step (1280 params)")


# ── Embeddings ───────────────────────────────────────────────────

def bench_embeddings():
    print("\n== Embeddings ==")
    sin_emb = SinusoidalPositionEmbedding(16)
    bench(lambda: sin_emb.encode(10, 15), label="Sinusoidal encode (d=16)")
    bench(lambda: sin_emb.encode_batch([(i, j) for i in range(3) for j in range(3)]),
          label="Sinusoidal batch 9 pos (d=16)")

    learned = LearnedPositionEmbedding(25, 16)
    bench(lambda: learned(list(range(25))), label="Learned embed 25 indices (d=16)")

    grid = RelativeGridEmbedding(5, 16)
    bench(lambda: grid.encode(10, 10), label="RelativeGrid 5x5 (d=16)")


# ── Full inference simulation ────────────────────────────────────

def bench_inference():
    print("\n== Full Inference (game-like) ==")

    mlp = MLP(32, [32], {"action": 5, "value": 1}, activation="relu")
    mlp.eval()
    sin_emb = SinusoidalPositionEmbedding(16)

    def single_unit_inference():
        # Simulate: encode position + features, run MLP, pick action
        pos_enc = sin_emb.encode(15, 20)
        features = [random.random() for _ in range(16)]
        x = Tensor(pos_enc + features, (1, 32))
        heads = mlp(x)
        action_probs = heads["action"].softmax()
        _ = max(range(5), key=lambda i: action_probs.data[i])

    bench(single_unit_inference, label="Single unit inference (enc+mlp+argmax)")

    mlp_big = MLP(48, [48, 32], {"action": 8, "dir": 4, "value": 1},
                  activation="silu", use_layernorm=True)
    mlp_big.eval()
    sin_emb2 = SinusoidalPositionEmbedding(24)

    def single_unit_inference_big():
        pos_enc = sin_emb2.encode(15, 20)
        features = [random.random() for _ in range(24)]
        x = Tensor(pos_enc + features, (1, 48))
        heads = mlp_big(x)
        _ = max(range(8), key=lambda i: heads["action"].data[i])

    bench(single_unit_inference_big, label="Big unit inference (enc+mlp_big)")

    # MLP.infer() fast path
    mlp_infer = MLP(32, [32], {"action": 5, "value": 1}, activation="relu")
    mlp_infer.eval()

    def single_unit_infer_fast():
        pos_enc = sin_emb.encode(15, 20)
        features = [random.random() for _ in range(16)]
        heads = mlp_infer.infer(pos_enc + features)
        _ = max(range(5), key=lambda i: heads["action"][i])

    bench(single_unit_infer_fast, label="Single unit MLP.infer() fast path")

    mlp_infer_big = MLP(48, [48, 32], {"action": 8, "dir": 4, "value": 1},
                        activation="silu", use_layernorm=True)
    mlp_infer_big.eval()

    def single_unit_infer_fast_big():
        pos_enc = sin_emb2.encode(15, 20)
        features = [random.random() for _ in range(24)]
        heads = mlp_infer_big.infer(pos_enc + features)
        _ = max(range(8), key=lambda i: heads["action"][i])

    bench(single_unit_infer_fast_big, label="Big unit MLP.infer() fast path")

    # Actual production bot config
    prod_heads = {
        "build_type": len(PROD_BUILD_ACTIONS),
        "build_dir": 9,
        "move_dir": 9,
        "h_next": PROD_H_DIM,
    }
    mlp_bot = MLP(
        PROD_INPUT_SIZE + PROD_H_DIM,
        PROD_HIDDEN_SIZES,
        prod_heads,
        activation="silu",
        rmsnorm_layers=PROD_RMSNORM_LAYERS,
    )
    mlp_bot.eval()
    sin_bot = SinusoidalPositionEmbedding(PROD_POS_EMBED_DIM)

    def make_prod_features():
        base = [random.random() for _ in range(PROD_INPUT_SIZE - PROD_POS_EMBED_DIM)]
        base.extend(sin_bot.encode(15, 20))
        base.extend([0.0] * PROD_H_DIM)
        return base

    def bot_tensor_path():
        features = make_prod_features()
        x = Tensor(features, (1, PROD_INPUT_SIZE + PROD_H_DIM))
        outputs = mlp_bot(x)
        _ = max(range(len(PROD_BUILD_ACTIONS)), key=lambda i: outputs["build_type"].data[i])

    def bot_infer_path():
        features = make_prod_features()
        outputs = mlp_bot.infer(features)
        _ = max(range(len(PROD_BUILD_ACTIONS)), key=lambda i: outputs["build_type"][i])

    bench(
        bot_tensor_path,
        label=f"Production bot Tensor path ({PROD_INPUT_SIZE + PROD_H_DIM}->{PROD_HIDDEN_SIZES})",
    )
    bench(
        bot_infer_path,
        label=f"Production bot infer() path ({PROD_INPUT_SIZE + PROD_H_DIM}->{PROD_HIDDEN_SIZES})",
    )


# ── Training step simulation ────────────────────────────────────

def bench_training_step():
    print("\n== Training Step ==")

    def full_train_step():
        mlp = MLP(32, [32], {"action": 5}, activation="relu")
        opt = Adam(mlp.parameters(), lr=0.001)
        x = Tensor.randn((4, 32), requires_grad=True)
        target = [random.randint(0, 4) for _ in range(4)]

        opt.zero_grad()
        heads = mlp(x)
        logits = heads["action"]
        probs = logits.softmax()
        log_probs = probs.log()
        # Cross-entropy-ish loss
        loss_val = 0.0
        for i in range(4):
            loss_val -= log_probs.data[i * 5 + target[i]]
        # Use sum as differentiable proxy
        loss = (probs * -1.0).sum()
        loss.backward()
        opt.step()

    bench(full_train_step, n=50, warmup=5, label="Full train step (batch=4)")


# ── Multi-unit scaling ───────────────────────────────────────────

def bench_scaling():
    """Simulate N units each calling inference in one round.

    Each unit has its own Player instance with its own MLP, so we
    measure total wall time for N sequential inference calls — the
    actual cost the engine charges against the 2ms budget per unit.
    """
    print("\n== Multi-Unit Scaling ==")
    print("  (Each unit gets 2000 us. Times shown are per-unit avg + total.)")

    sin_emb = SinusoidalPositionEmbedding(16)

    # Small MLP — typical lightweight bot
    mlp_small = MLP(32, [32], {"action": 5, "value": 1}, activation="relu")
    mlp_small.eval()

    def run_n_units_small(n):
        for _ in range(n):
            pos_enc = sin_emb.encode(random.randint(0, 30), random.randint(0, 30))
            features = [random.random() for _ in range(16)]
            x = Tensor(pos_enc + features, (1, 32))
            heads = mlp_small(x)
            action_probs = heads["action"].softmax()
            _ = max(range(5), key=lambda i: action_probs.data[i])

    # Big MLP — heavier bot with layernorm
    mlp_big = MLP(48, [48, 32], {"action": 8, "dir": 4, "value": 1},
                  activation="silu", use_layernorm=True)
    mlp_big.eval()
    sin_emb2 = SinusoidalPositionEmbedding(24)

    def run_n_units_big(n):
        for _ in range(n):
            pos_enc = sin_emb2.encode(random.randint(0, 30), random.randint(0, 30))
            features = [random.random() for _ in range(24)]
            x = Tensor(pos_enc + features, (1, 48))
            heads = mlp_big(x)
            _ = max(range(8), key=lambda i: heads["action"].data[i])

    for n_units in [1, 4, 8, 12, 16, 20, 25]:
        # Small MLP
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            run_n_units_small(n_units)
            times.append((time.perf_counter() - t0) * 1e6)
        times.sort()
        total = times[len(times) // 2]
        per_unit = total / n_units
        ok = "OK" if per_unit < 2000 else "OVER"
        print(f"  small MLP x{n_units:2d} units: "
              f"per-unit {per_unit:7.0f} us   total {total:8.0f} us  [{ok}]")

    print()
    for n_units in [1, 4, 8, 12, 16, 20, 25]:
        # Big MLP
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            run_n_units_big(n_units)
            times.append((time.perf_counter() - t0) * 1e6)
        times.sort()
        total = times[len(times) // 2]
        per_unit = total / n_units
        ok = "OK" if per_unit < 2000 else "OVER"
        print(f"  big   MLP x{n_units:2d} units: "
              f"per-unit {per_unit:7.0f} us   total {total:8.0f} us  [{ok}]")

    # Shared-model optimization: reuse same MLP instance (realistic — weights are global)
    print("\n  --- With batched matmul (N inputs concatenated) ---")
    lin1 = Linear(32, 32)
    lin2 = Linear(32, 5)
    relu = ReLU()

    def batched_inference(n):
        # Encode all units, stack into one (N, 32) tensor, single forward pass
        data = []
        for _ in range(n):
            pos_enc = sin_emb.encode(random.randint(0, 30), random.randint(0, 30))
            features = [random.random() for _ in range(16)]
            data.extend(pos_enc + features)
        x = Tensor(data, (n, 32))
        h = relu(lin1(x))
        out = lin2(h)
        # Per-unit argmax
        for i in range(n):
            row = out[i]
            _ = max(range(5), key=lambda j: row.data[j])

    for n_units in [1, 4, 8, 12, 16, 20, 25]:
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            batched_inference(n_units)
            times.append((time.perf_counter() - t0) * 1e6)
        times.sort()
        total = times[len(times) // 2]
        per_unit = total / n_units
        ok = "OK" if per_unit < 2000 else "OVER"
        print(f"  batched   x{n_units:2d} units: "
              f"per-unit {per_unit:7.0f} us   total {total:8.0f} us  [{ok}]")


# ── Runner ───────────────────────────────────────────────────────

ALL_BENCHMARKS = {
    "creation": bench_creation,
    "elementwise": bench_elementwise,
    "matmul": bench_matmul,
    "transpose": bench_transpose,
    "activations": bench_activations,
    "reductions": bench_reductions,
    "layers": bench_layers,
    "mlp": bench_mlp,
    "backward": bench_backward,
    "optimizers": bench_optimizers,
    "embeddings": bench_embeddings,
    "inference": bench_inference,
    "scaling": bench_scaling,
    "training": bench_training_step,
}

if __name__ == "__main__":
    print("tensor_lib benchmarks")
    print("=" * 70)

    selected = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_BENCHMARKS)
    for name in selected:
        if name in ALL_BENCHMARKS:
            ALL_BENCHMARKS[name]()
        else:
            print(f"\nUnknown benchmark: {name}")
            print(f"Available: {', '.join(ALL_BENCHMARKS)}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("Done. Times are per-call. Game budget: 2000 us/unit/round.")
