"""Shared helpers for replay buffer loading and copy semantics."""

from __future__ import annotations

from dataclasses import dataclass
import mmap
from pathlib import Path


ReplayBuffer = bytes | bytearray | memoryview | mmap.mmap


@dataclass(frozen=True, slots=True)
class ReplayBufferSpec:
    path_key: str
    buf_key: str
    writable: bool = False


STATE_BUFFER = ReplayBufferSpec("states_path", "states_buf", writable=True)
SCALAR_BUFFER = ReplayBufferSpec("scalars_path", "scalars_buf", writable=True)
TRANSFER_EVENTS_BUFFER = ReplayBufferSpec("transfer_events_path", "transfer_events_buf")
TI_DELIVERED_BUFFER = ReplayBufferSpec("ti_delivered_path", "ti_delivered_buf")
AX_DELIVERED_BUFFER = ReplayBufferSpec("ax_delivered_path", "ax_delivered_buf")

ALL_REPLAY_BUFFERS = (
    STATE_BUFFER,
    SCALAR_BUFFER,
    TRANSFER_EVENTS_BUFFER,
    TI_DELIVERED_BUFFER,
    AX_DELIVERED_BUFFER,
)


def coerce_writable_replay_buffer(buf: ReplayBuffer | None) -> ReplayBuffer | None:
    """Return a writable buffer for zero-copy torch.frombuffer views."""
    if buf is None or isinstance(buf, (bytearray, mmap.mmap)):
        return buf
    if isinstance(buf, memoryview):
        if not buf.readonly:
            return buf
        return bytearray(buf)
    if isinstance(buf, bytes):
        return bytearray(buf)
    return buf


def load_replay_buffer(global_states: dict, spec: ReplayBufferSpec) -> ReplayBuffer | None:
    """Load a replay buffer from the dict, hydrating from disk on demand."""
    buf = global_states.get(spec.buf_key)
    if buf is None:
        path = global_states.get(spec.path_key)
        if not path:
            return None
        try:
            buf = Path(path).read_bytes()
        except OSError:
            return None
        global_states[spec.buf_key] = buf
    return buf


def load_torch_replay_buffer(global_states: dict, spec: ReplayBufferSpec) -> ReplayBuffer | None:
    """Load a replay buffer and coerce it writable only when torch needs that."""
    buf = load_replay_buffer(global_states, spec)
    if buf is None or not spec.writable:
        return buf
    writable_buf = coerce_writable_replay_buffer(buf)
    if writable_buf is not buf:
        global_states[spec.buf_key] = writable_buf
    return writable_buf


def preload_replay_buffers(
    global_states_list: list[dict | None],
    specs: tuple[ReplayBufferSpec, ...] = ALL_REPLAY_BUFFERS,
) -> None:
    """Hydrate path-backed replay buffers before temporary files are removed."""
    for global_states in global_states_list:
        if global_states is None:
            continue
        for spec in specs:
            load_replay_buffer(global_states, spec)
