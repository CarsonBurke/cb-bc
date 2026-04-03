"""Regression coverage for optional train_batched profiling helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORCH_DIR = PROJECT_ROOT / "bots" / "tensor" / "torch"

if str(TORCH_DIR) not in sys.path:
    sys.path.insert(0, str(TORCH_DIR))

import training_profile as profile_mod


def test_parse_profile_episode_spec_supports_ranges_and_all():
    assert profile_mod.parse_profile_episode_spec("1,3-5,8") == frozenset({1, 3, 4, 5, 8})
    assert profile_mod.parse_profile_episode_spec("all") == frozenset({-1})


def test_parse_profile_episode_spec_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        profile_mod.parse_profile_episode_spec("0")
    with pytest.raises(ValueError):
        profile_mod.parse_profile_episode_spec("5-3")


def test_episode_profiler_config_returns_none_when_disabled(tmp_path):
    cfg = profile_mod.EpisodeProfilerConfig(episodes=frozenset({2}), output_dir=tmp_path)
    assert cfg.create(1) is None


def test_episode_profiler_writes_trace_table_and_summary_without_op_profiling(tmp_path):
    cfg = profile_mod.EpisodeProfilerConfig(
        episodes=frozenset({1}),
        output_dir=tmp_path,
        row_limit=12,
    )

    profiler = cfg.create(1)
    assert profiler is not None

    with profiler.scope("episode.collect"):
        pass
    with profiler.scope("episode.train"):
        pass

    paths = profiler.finish(
        coarse_metrics={"collection_total_s": 1.25},
        metadata={"episode": 1, "n_steps": 1234},
    )

    trace = json.loads(Path(paths["trace_path"]).read_text(encoding="utf-8"))
    assert trace["traceEvents"][0]["name"] == "episode.collect"
    table_text = Path(paths["table_path"]).read_text(encoding="utf-8")
    assert "episode.collect" in table_text
    assert "Scope" in table_text

    summary = json.loads(Path(paths["summary_path"]).read_text(encoding="utf-8"))
    assert summary["profile_ops"] is False
    assert summary["coarse_metrics"]["collection_total_s"] == 1.25
    assert summary["metadata"]["n_steps"] == 1234
    assert "episode.collect" in summary["events"]


def test_episode_profiler_writes_trace_table_and_summary_with_op_profiling(monkeypatch, tmp_path):
    class FakeAverages:
        def table(self, sort_by: str, row_limit: int) -> str:
            return f"{sort_by}:{row_limit}"

    class FakeProfiler:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.export_path = None

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def export_chrome_trace(self, path: str) -> None:
            self.export_path = path
            Path(path).write_text("trace", encoding="utf-8")

        def key_averages(self) -> FakeAverages:
            return FakeAverages()

    fake_profiler = FakeProfiler()
    monkeypatch.setattr(profile_mod.torch.profiler, "profile", lambda **_: fake_profiler)

    cfg = profile_mod.EpisodeProfilerConfig(
        episodes=frozenset({1}),
        output_dir=tmp_path,
        profile_ops=True,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        row_limit=12,
    )

    profiler = cfg.create(1)
    assert profiler is not None

    with profiler.scope("episode.collect"):
        pass

    paths = profiler.finish(
        coarse_metrics={"collection_total_s": 1.25},
        metadata={"episode": 1, "n_steps": 1234},
    )

    assert fake_profiler.started
    assert fake_profiler.stopped
    trace = json.loads(Path(paths["trace_path"]).read_text(encoding="utf-8"))
    assert trace["traceEvents"][0]["name"] == "episode.collect"
    expected_sort_key = "self_cuda_time_total" if profile_mod.torch.cuda.is_available() else "self_cpu_time_total"
    assert f"{expected_sort_key}:12" in Path(paths["table_path"]).read_text(encoding="utf-8")

    summary = json.loads(Path(paths["summary_path"]).read_text(encoding="utf-8"))
    assert summary["profile_ops"] is True
    assert summary["coarse_metrics"]["collection_total_s"] == 1.25
    assert summary["metadata"]["n_steps"] == 1234
