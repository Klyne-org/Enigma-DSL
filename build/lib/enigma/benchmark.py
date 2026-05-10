# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Benchmark utilities for Enigma kernels.

Mirrors tilelang's MPS benchmark utilities (PR #1547). Provides:

  * :func:`bench`              — wall-clock timing of any Python callable.
  * :func:`bench_gpu`          — GPU-side timing via Metal timestamps
    (uses :meth:`enigma.PreparedKernel.dispatch_timed`).
  * :func:`format_bench_result` — single-line ``min/median/max`` formatter.
"""

import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "BenchResult",
    "bench",
    "bench_gpu",
    "format_bench_result",
]


@dataclass
class BenchResult:
    """Aggregated benchmark measurements (microseconds)."""

    label: str
    samples_us: Sequence[float]
    warmup: int

    @property
    def n(self) -> int:
        return len(self.samples_us)

    @property
    def min_us(self) -> float:
        return min(self.samples_us) if self.samples_us else 0.0

    @property
    def max_us(self) -> float:
        return max(self.samples_us) if self.samples_us else 0.0

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples_us) if self.samples_us else 0.0

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.samples_us) if self.samples_us else 0.0

    @property
    def stdev_us(self) -> float:
        return statistics.stdev(self.samples_us) if self.n >= 2 else 0.0

    def __repr__(self) -> str:
        return (
            f"BenchResult({self.label!r}, n={self.n}, "
            f"median={self.median_us:.2f}us, "
            f"min={self.min_us:.2f}us, max={self.max_us:.2f}us)"
        )


def _validate(repeat: int, warmup: int) -> None:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")


def _run_samples(measure_one: Callable[[], float], repeat: int, warmup: int) -> List[float]:
    for _ in range(warmup):
        measure_one()
    return [measure_one() for _ in range(repeat)]


def bench(
    fn: Callable,
    *args,
    repeat: int = 50,
    warmup: int = 5,
    label: Optional[str] = None,
    **kwargs,
) -> BenchResult:
    """Wall-clock time ``fn(*args, **kwargs)`` over ``repeat`` runs.

    The first ``warmup`` runs are discarded. Uses :func:`time.perf_counter_ns`.
    """
    _validate(repeat, warmup)

    def _one() -> float:
        t0 = time.perf_counter_ns()
        fn(*args, **kwargs)
        return (time.perf_counter_ns() - t0) / 1000.0

    return BenchResult(
        label=label or getattr(fn, "__name__", "fn"),
        samples_us=_run_samples(_one, repeat, warmup),
        warmup=warmup,
    )


def bench_gpu(
    prepared,
    grid: Tuple[int, int, int],
    threads: Tuple[int, int, int],
    *,
    repeat: int = 50,
    warmup: int = 5,
    label: Optional[str] = None,
) -> BenchResult:
    """Time a :class:`enigma.PreparedKernel` using GPU timestamps."""
    _validate(repeat, warmup)
    return BenchResult(
        label=label or getattr(prepared, "_kernel_name", "prepared"),
        samples_us=_run_samples(
            lambda: float(prepared.dispatch_timed(grid, threads)), repeat, warmup,
        ),
        warmup=warmup,
    )


def format_bench_result(r: BenchResult, *, throughput_gbps: Optional[float] = None) -> str:
    """One-line string for printing benchmark tables."""
    parts = [
        r.label,
        f"n={r.n}",
        f"median={r.median_us:.2f}us",
        f"min={r.min_us:.2f}us",
        f"max={r.max_us:.2f}us",
    ]
    if throughput_gbps is not None:
        parts.append(f"BW={throughput_gbps:.2f}GB/s")
    return "  ".join(parts)
