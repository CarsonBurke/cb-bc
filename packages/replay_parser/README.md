# replay_parser

Rust + PyO3 parser for `.replay26` protobuf files. Returns per-turn map state and scalars as contiguous buffers for `torch.frombuffer()`.

## Build

```bash
pip install maturin
cd packages/replay_parser
maturin develop --release
```
