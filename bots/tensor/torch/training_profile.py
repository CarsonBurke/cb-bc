"""Optional end-to-end profiling helpers for train_batched."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import time

import torch


def parse_profile_episode_spec(spec: str) -> frozenset[int]:
    """Parse a 1-based episode selection like ``1,3-5`` or ``all``."""
    spec = spec.strip()
    if not spec:
        return frozenset()
    if spec.lower() in {"all", "*"}:
        return frozenset({-1})

    episodes: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid episode range: {part}")
            episodes.update(range(start, end + 1))
            continue
        episode = int(part)
        if episode <= 0:
            raise ValueError(f"Episode numbers must be >= 1: {part}")
        episodes.add(episode)
    return frozenset(episodes)


@dataclass(frozen=True, slots=True)
class EpisodeProfilerConfig:
    episodes: frozenset[int]
    output_dir: Path
    profile_ops: bool = False
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    row_limit: int = 40

    def enabled_for(self, episode: int) -> bool:
        return -1 in self.episodes or episode in self.episodes

    def create(self, episode: int) -> "EpisodeProfiler | None":
        if not self.enabled_for(episode):
            return None
        return EpisodeProfiler(config=self, episode=episode)


@dataclass
class EpisodeProfiler:
    config: EpisodeProfilerConfig
    episode: int

    def __post_init__(self) -> None:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "trace_path", output_dir / f"episode_{self.episode:04d}_trace.json")
        object.__setattr__(self, "table_path", output_dir / f"episode_{self.episode:04d}_ops.txt")
        object.__setattr__(self, "summary_path", output_dir / f"episode_{self.episode:04d}_summary.json")
        object.__setattr__(self, "_started_at", time.perf_counter())
        object.__setattr__(self, "_events", [])
        profiler = None
        if self.config.profile_ops:
            profiler = torch.profiler.profile(
                activities=_profiler_activities(),
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
                acc_events=True,
            )
            profiler.start()
        object.__setattr__(self, "_profiler", profiler)

    @contextmanager
    def scope(self, name: str):
        start = time.perf_counter()
        profiler = self._profiler
        pushed_nvtx = False
        try:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)
                pushed_nvtx = True
            if profiler is None:
                with nullcontext():
                    yield
            else:
                with torch.profiler.record_function(name):
                    yield
        finally:
            if pushed_nvtx:
                torch.cuda.nvtx.range_pop()
            end = time.perf_counter()
            self._events.append({
                "name": name,
                "ts_us": (start - self._started_at) * 1_000_000.0,
                "dur_us": (end - start) * 1_000_000.0,
            })

    def finish(self, *, coarse_metrics: dict[str, float] | None = None, metadata: dict | None = None) -> dict[str, str]:
        profiler = self._profiler
        if profiler is not None:
            profiler.stop()
        self.trace_path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": event["name"],
                            "cat": "scope",
                            "ph": "X",
                            "ts": round(event["ts_us"], 3),
                            "dur": round(event["dur_us"], 3),
                            "pid": 1,
                            "tid": 1,
                        }
                        for event in self._events
                    ],
                    "displayTimeUnit": "ms",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.table_path.write_text(
            _format_scope_table(
                self._events,
                row_limit=self.config.row_limit,
                profiler=profiler,
            ),
            encoding="utf-8",
        )
        event_totals = {}
        for event in self._events:
            bucket = event_totals.setdefault(event["name"], {"count": 0, "total_us": 0.0})
            bucket["count"] += 1
            bucket["total_us"] += event["dur_us"]
        payload = {
            "episode": self.episode,
            "trace_path": str(self.trace_path),
            "table_path": str(self.table_path),
            "elapsed_s": time.perf_counter() - self._started_at,
            "profile_ops": self.config.profile_ops,
            "events": {
                name: {
                    "count": values["count"],
                    "total_ms": round(values["total_us"] / 1000.0, 3),
                }
                for name, values in sorted(
                    event_totals.items(),
                    key=lambda item: item[1]["total_us"],
                    reverse=True,
                )
            },
            "coarse_metrics": coarse_metrics or {},
            "metadata": metadata or {},
        }
        self.summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "trace_path": str(self.trace_path),
            "table_path": str(self.table_path),
            "summary_path": str(self.summary_path),
        }


def _profiler_activities() -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def _format_scope_table(
    events: list[dict[str, float | str]],
    *,
    row_limit: int,
    profiler,
) -> str:
    lines = ["Scope                          Count   Total ms   Avg ms"]
    totals: dict[str, dict[str, float]] = {}
    for event in events:
        name = str(event["name"])
        bucket = totals.setdefault(name, {"count": 0.0, "total_us": 0.0})
        bucket["count"] += 1.0
        bucket["total_us"] += float(event["dur_us"])
    for name, values in sorted(
        totals.items(),
        key=lambda item: item[1]["total_us"],
        reverse=True,
    )[:row_limit]:
        count = int(values["count"])
        total_ms = values["total_us"] / 1000.0
        avg_ms = total_ms / max(count, 1)
        lines.append(f"{name:<30} {count:>5d} {total_ms:>10.3f} {avg_ms:>8.3f}")
    if profiler is not None:
        lines.append("")
        lines.append("Torch op summary")
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        lines.append(profiler.key_averages().table(sort_by=sort_key, row_limit=row_limit))
    return "\n".join(lines) + "\n"


@contextmanager
def trace_scope(trace: EpisodeProfiler | None, name: str):
    """Enter a torch profiler record_function scope only when profiling is active."""
    if trace is None:
        with nullcontext():
            yield
        return
    with trace.scope(name):
        yield
