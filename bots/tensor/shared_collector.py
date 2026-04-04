"""Shared-memory ring-buffer collector protocol for tensor training proxies.

Each worker process shares one mmap-backed collector block with the training
server. Every side owns two SPSC rings:
- request ring: proxy -> server
- response ring: server -> proxy

Notify pipes are still used for wakeups, but bulk payloads stay in shared
memory and the rings can drain multiple queued items per wakeup.
"""

from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
import struct

try:
    from shared import INPUT_SIZE
except ImportError:
    from bots.tensor.shared import INPUT_SIZE

COLLECTOR_MAGIC = b"cbcoll01"
COLLECTOR_VERSION = 1
COLLECTOR_RING_SLOTS_DEFAULT = 8

FILE_HEADER = struct.Struct("=8sII")
COUNTER = struct.Struct("=Q")
REQUEST_META = struct.Struct("=ii")
RESPONSE_ACTIONS = struct.Struct("=iiiii")

OBS_FLOATS = INPUT_SIZE
OBS_BYTES = OBS_FLOATS * 4
REQUEST_SLOT_BYTES = REQUEST_META.size + OBS_BYTES
RESPONSE_SLOT_BYTES = RESPONSE_ACTIONS.size
RING_COUNTER_BYTES = COUNTER.size * 2  # head, tail

_CLIENT_ENV = {
    "a": ("TENSOR_REQ_NOTIFY_WFD_A", "TENSOR_RESP_NOTIFY_RFD_A"),
    "b": ("TENSOR_REQ_NOTIFY_WFD_B", "TENSOR_RESP_NOTIFY_RFD_B"),
}


@dataclass(frozen=True, slots=True)
class CollectorRequest:
    slot_index: int
    unit_id: int
    round_idx: int
    obs_buffer: memoryview


def _side_index(side: str) -> int:
    if side == "a":
        return 0
    if side == "b":
        return 1
    raise ValueError(f"unsupported collector side: {side!r}")


def _side_bytes(slot_count: int) -> int:
    return (
        RING_COUNTER_BYTES
        + slot_count * REQUEST_SLOT_BYTES
        + RING_COUNTER_BYTES
        + slot_count * RESPONSE_SLOT_BYTES
    )


def collector_bytes(slot_count: int) -> int:
    if slot_count <= 0:
        raise ValueError(f"collector slot count must be > 0, got {slot_count}")
    return FILE_HEADER.size + 2 * _side_bytes(slot_count)


def initialize_collector(collector_mm: mmap.mmap, slot_count: int) -> None:
    FILE_HEADER.pack_into(
        collector_mm,
        0,
        COLLECTOR_MAGIC,
        COLLECTOR_VERSION,
        int(slot_count),
    )


class _SideLayout:
    def __init__(self, slot_count: int, side: str):
        side_base = FILE_HEADER.size + _side_index(side) * _side_bytes(slot_count)
        req_counters_base = side_base
        req_slots_base = req_counters_base + RING_COUNTER_BYTES
        resp_counters_base = req_slots_base + slot_count * REQUEST_SLOT_BYTES
        resp_slots_base = resp_counters_base + RING_COUNTER_BYTES

        self.slot_count = slot_count
        self.req_head_offset = req_counters_base
        self.req_tail_offset = req_counters_base + COUNTER.size
        self.req_slots_base = req_slots_base
        self.resp_head_offset = resp_counters_base
        self.resp_tail_offset = resp_counters_base + COUNTER.size
        self.resp_slots_base = resp_slots_base

    def request_meta_offset(self, slot_index: int) -> int:
        return self.req_slots_base + slot_index * REQUEST_SLOT_BYTES

    def request_obs_offset(self, slot_index: int) -> int:
        return self.request_meta_offset(slot_index) + REQUEST_META.size

    def response_offset(self, slot_index: int) -> int:
        return self.resp_slots_base + slot_index * RESPONSE_SLOT_BYTES


class _RingAccessor:
    def __init__(self, collector_mm: mmap.mmap, layout: _SideLayout):
        self._mm = collector_mm
        self._layout = layout

    def _read_counter(self, offset: int) -> int:
        return COUNTER.unpack_from(self._mm, offset)[0]

    def _write_counter(self, offset: int, value: int) -> None:
        COUNTER.pack_into(self._mm, offset, int(value))

    def _req_head(self) -> int:
        return self._read_counter(self._layout.req_head_offset)

    def _req_tail(self) -> int:
        return self._read_counter(self._layout.req_tail_offset)

    def _resp_head(self) -> int:
        return self._read_counter(self._layout.resp_head_offset)

    def _resp_tail(self) -> int:
        return self._read_counter(self._layout.resp_tail_offset)

    def _set_req_head(self, value: int) -> None:
        self._write_counter(self._layout.req_head_offset, value)

    def _set_req_tail(self, value: int) -> None:
        self._write_counter(self._layout.req_tail_offset, value)

    def _set_resp_head(self, value: int) -> None:
        self._write_counter(self._layout.resp_head_offset, value)

    def _set_resp_tail(self, value: int) -> None:
        self._write_counter(self._layout.resp_tail_offset, value)


class CollectorServerSide(_RingAccessor):
    """Server-side view over one side of a shared collector block."""

    def __init__(self, collector_mm: mmap.mmap, side: str):
        magic, version, slot_count = FILE_HEADER.unpack_from(collector_mm, 0)
        if magic != COLLECTOR_MAGIC:
            raise ValueError("collector mmap has invalid magic")
        if version != COLLECTOR_VERSION:
            raise ValueError(f"unsupported collector version: {version}")
        layout = _SideLayout(slot_count, side)
        super().__init__(collector_mm, layout)
        self.side = side
        self.slot_count = slot_count
        self._req_obs_views = tuple(
            memoryview(collector_mm)[
                layout.request_obs_offset(slot_index):
                layout.request_obs_offset(slot_index) + OBS_BYTES
            ]
            for slot_index in range(slot_count)
        )

    def request_obs_buffer(self, slot_index: int) -> memoryview:
        return self._req_obs_views[slot_index]

    def pop_requests(self, limit: int | None = None) -> list[CollectorRequest]:
        head = self._req_head()
        tail = self._req_tail()
        pending = tail - head
        if pending < 0 or pending > self.slot_count:
            raise RuntimeError(
                f"collector request ring corrupt for side {self.side}: head={head} tail={tail}"
            )
        if pending == 0:
            return []
        take = pending if limit is None else min(pending, int(limit))
        out: list[CollectorRequest] = []
        for seq in range(head, head + take):
            slot_index = seq % self.slot_count
            unit_id, round_idx = REQUEST_META.unpack_from(
                self._mm,
                self._layout.request_meta_offset(slot_index),
            )
            out.append(
                CollectorRequest(
                    slot_index=slot_index,
                    unit_id=unit_id,
                    round_idx=round_idx,
                    obs_buffer=self._req_obs_views[slot_index],
                )
            )
        self._set_req_head(head + take)
        return out

    def write_response(self, bt: int, bd: int, cd: int, brt: int, md: int) -> None:
        head = self._resp_head()
        tail = self._resp_tail()
        if tail - head >= self.slot_count:
            raise RuntimeError(f"collector response ring full for side {self.side}")
        slot_index = tail % self.slot_count
        RESPONSE_ACTIONS.pack_into(
            self._mm,
            self._layout.response_offset(slot_index),
            int(bt),
            int(bd),
            int(cd),
            int(brt),
            int(md),
        )
        self._set_resp_tail(tail + 1)

    def close(self) -> None:
        for view in self._req_obs_views:
            view.release()


class CollectorClient(_RingAccessor):
    """Client-side shared-memory transport for one proxy side."""

    def __init__(self, collector_fd: int, request_notify_wfd: int, response_notify_rfd: int, side: str):
        self.side = side
        self._mm = mmap.mmap(collector_fd, 0, access=mmap.ACCESS_WRITE)
        magic, version, slot_count = FILE_HEADER.unpack_from(self._mm, 0)
        if magic != COLLECTOR_MAGIC:
            raise ValueError("collector mmap has invalid magic")
        if version != COLLECTOR_VERSION:
            raise ValueError(f"unsupported collector version: {version}")
        layout = _SideLayout(slot_count, side)
        super().__init__(self._mm, layout)
        self.slot_count = slot_count
        self._request_notify_wfd = request_notify_wfd
        self._response_notify_rfd = response_notify_rfd
        self._req_obs_views = tuple(
            memoryview(self._mm)[
                layout.request_obs_offset(slot_index):
                layout.request_obs_offset(slot_index) + OBS_BYTES
            ]
            for slot_index in range(slot_count)
        )

    @classmethod
    def from_env(cls, side: str) -> "CollectorClient | None":
        collector_fd = int(os.environ.get("TENSOR_COLLECTOR_FD", "-1"))
        req_key, resp_key = _CLIENT_ENV[side]
        request_notify_wfd = int(os.environ.get(req_key, "-1"))
        response_notify_rfd = int(os.environ.get(resp_key, "-1"))
        if collector_fd < 0 or request_notify_wfd < 0 or response_notify_rfd < 0:
            return None
        return cls(collector_fd, request_notify_wfd, response_notify_rfd, side)

    def _push_request(self, unit_id: int, round_idx: int, obs) -> None:
        head = self._req_head()
        tail = self._req_tail()
        if tail - head >= self.slot_count:
            raise RuntimeError(f"collector request ring full for side {self.side}")
        slot_index = tail % self.slot_count
        REQUEST_META.pack_into(
            self._mm,
            self._layout.request_meta_offset(slot_index),
            int(unit_id),
            int(round_idx),
        )
        self._req_obs_views[slot_index][:] = memoryview(obs).cast("B")
        self._set_req_tail(tail + 1)

    def _pop_response(self) -> tuple[int, int, int, int, int]:
        head = self._resp_head()
        tail = self._resp_tail()
        if tail <= head:
            raise RuntimeError(f"collector response ring empty for side {self.side}")
        slot_index = head % self.slot_count
        result = RESPONSE_ACTIONS.unpack_from(self._mm, self._layout.response_offset(slot_index))
        self._set_resp_head(head + 1)
        return result

    def request(self, unit_id: int, round_idx: int, obs) -> tuple[int, int, int, int, int] | None:
        self._push_request(unit_id, round_idx, obs)
        os.write(self._request_notify_wfd, b"\x01")
        token = os.read(self._response_notify_rfd, 1)
        if not token:
            return None
        return self._pop_response()

    def close(self) -> None:
        for view in self._req_obs_views:
            view.release()
        self._mm.close()
