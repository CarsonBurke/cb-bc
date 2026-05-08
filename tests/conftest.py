"""Mitata-inspired pytest reporter — grouped tables with timing bars."""

from __future__ import annotations

import time
from collections import defaultdict

import pytest

# ── block-bar rendering ──────────────────────────────────────────────

_BLOCKS = " ▏▎▍▌▋▊▉█"


def _bar(value: float, max_val: float, width: int = 20) -> str:
    if max_val <= 0:
        return " " * width
    ratio = min(value / max_val, 1.0)
    full = ratio * width
    whole = int(full)
    frac = full - whole
    idx = int(frac * (len(_BLOCKS) - 1))
    bar = _BLOCKS[-1] * whole
    if whole < width:
        bar += _BLOCKS[idx]
        bar += " " * (width - whole - 1)
    return bar


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:6.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:6.1f} ms"
    return f"{seconds:6.2f}  s"


# ── plugin ────────────────────────────────────────────────────────────

class MitataReporter:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config
        self.results: list[dict] = []
        self._suite_start: float = 0.0

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self._suite_start = time.perf_counter()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        # group by file::class or file
        parts = report.nodeid.split("::")
        group = parts[0]
        if len(parts) >= 2:
            group = f"{parts[0]}::{parts[1]}" if parts[1][0].isupper() else parts[0]
        name = parts[-1]
        self.results.append({
            "group": group,
            "name": name,
            "duration": report.duration,
            "passed": report.passed,
            "failed": report.failed,
            "skipped": report.skipped,
        })

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config) -> None:
        if not self.results:
            return
        tw = terminalreporter._tw

        total_elapsed = time.perf_counter() - self._suite_start
        passed = sum(1 for r in self.results if r["passed"])
        failed = sum(1 for r in self.results if r["failed"])
        skipped = sum(1 for r in self.results if r["skipped"])

        # ── group results ──
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in self.results:
            groups[r["group"]].append(r)

        global_max = max(r["duration"] for r in self.results) if self.results else 1.0

        tw.sep("─", "benchmark-style summary", cyan=True)
        tw.line()

        for group_name, items in groups.items():
            short_group = group_name.replace("tests/", "").replace(".py", "")
            tw.line(f"  \033[1m{short_group}\033[0m", cyan=True)

            max_name = max(len(r["name"]) for r in items)
            for r in items:
                icon = "✓" if r["passed"] else ("✗" if r["failed"] else "○")
                color = "\033[32m" if r["passed"] else ("\033[31m" if r["failed"] else "\033[33m")
                bar = _bar(r["duration"], global_max)
                timing = _fmt_time(r["duration"])
                name_pad = r["name"].ljust(max_name)
                tw.line(f"    {color}{icon}\033[0m {name_pad}  {timing}  \033[36m{bar}\033[0m")

            group_total = sum(r["duration"] for r in items)
            tw.line(f"    {'':>{max_name}}  {_fmt_time(group_total)}  ({len(items)} tests)")
            tw.line()

        # ── totals ──
        tw.line(f"  \033[1m{passed} passed\033[0m", green=True)
        if failed:
            tw.line(f"  \033[1m{failed} failed\033[0m", red=True)
        if skipped:
            tw.line(f"  {skipped} skipped", yellow=True)
        tw.line(f"  {_fmt_time(total_elapsed)} total")
        tw.line()


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(MitataReporter(config), "mitata_reporter")
