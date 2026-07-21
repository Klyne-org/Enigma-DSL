# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Low-overhead, Proton-style profiling for Enigma runtime dispatch.

Design follows the Proton profiler for Triton (Zhou et al., CGO 2026):
hierarchical scopes form a call-path tree, user scopes can attach custom
metrics (``flops``, ``bytes``), kernel hooks derive metrics per dispatch,
and results aggregate by name or by call path with derived throughput
(GFLOP/s, GB/s). Output formats: text table, call tree, Hatchet JSON,
Chrome trace.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional


@dataclass(frozen=True)
class ProfilerEvent:
    """One timed profiler event."""

    name: str
    category: str
    start_ns: int
    end_ns: int
    kernel_name: Optional[str] = None
    grid: Optional[tuple[int, int, int]] = None
    threads: Optional[tuple[int, int, int]] = None
    input_bytes: int = 0
    output_bytes: int = 0
    buffer_count: int = 0
    gpu_time_us: float = 0.0
    call_path: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_us(self) -> float:
        return max(0.0, (self.end_ns - self.start_ns) / 1000.0)

    @property
    def label(self) -> str:
        return self.name if self.kernel_name is None else f"{self.name}:{self.kernel_name}"

    @property
    def node_path(self) -> tuple[str, ...]:
        """Full call-tree path including this event's own node."""
        if self.call_path and self.call_path[-1] == self.name:
            return self.call_path
        return self.call_path + (self.label,)


@dataclass(frozen=True)
class ProfilerRow:
    """Aggregated profiler row for a name/kernel pair."""

    name: str
    kernel_name: Optional[str]
    calls: int
    cpu_total_us: float
    gpu_total_us: float
    input_bytes: int
    output_bytes: int
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def cpu_avg_us(self) -> float:
        return self.cpu_total_us / self.calls if self.calls else 0.0

    @property
    def gpu_avg_us(self) -> float:
        return self.gpu_total_us / self.calls if self.calls else 0.0

    @property
    def _throughput_time_us(self) -> float:
        return self.gpu_total_us if self.gpu_total_us > 0.0 else self.cpu_total_us

    @property
    def gflops_per_s(self) -> float:
        """Derived from the ``flops`` metric over GPU time (CPU fallback)."""
        flops = self.metrics.get("flops", 0.0)
        t = self._throughput_time_us
        return flops / (t * 1000.0) if flops and t else 0.0

    @property
    def gbytes_per_s(self) -> float:
        """Derived from the ``bytes`` metric over GPU time (CPU fallback)."""
        nbytes = self.metrics.get("bytes", 0.0)
        t = self._throughput_time_us
        return nbytes / (t * 1000.0) if nbytes and t else 0.0


class ProfilerResult:
    """Aggregated view over profiler events."""

    def __init__(self, events: Iterable[ProfilerEvent], *, group_by: str = "name"):
        if group_by not in ("name", "call_path"):
            raise ValueError("group_by must be 'name' or 'call_path'")
        rows: dict[tuple[str, Optional[str]], list[ProfilerEvent]] = {}
        for event in events:
            if group_by == "call_path":
                key = ("/".join(event.node_path), event.kernel_name)
            else:
                key = (event.name, event.kernel_name)
            rows.setdefault(key, []).append(event)
        self._rows = [
            ProfilerRow(
                name=name,
                kernel_name=kernel_name,
                calls=len(group),
                cpu_total_us=sum(e.duration_us for e in group),
                gpu_total_us=sum(e.gpu_time_us for e in group),
                input_bytes=sum(e.input_bytes for e in group),
                output_bytes=sum(e.output_bytes for e in group),
                metrics=_sum_metrics(e.metrics for e in group),
            )
            for (name, kernel_name), group in rows.items()
        ]

    def rows(self, *, sort_by: str = "cpu_total_us") -> list[ProfilerRow]:
        return sorted(self._rows, key=lambda row: getattr(row, sort_by), reverse=True)

    def by_name(self, name: str) -> ProfilerRow:
        matches = [row for row in self._rows if row.name == name]
        if not matches:
            raise KeyError(name)
        if len(matches) == 1:
            return matches[0]
        return ProfilerRow(
            name=name,
            kernel_name=None,
            calls=sum(r.calls for r in matches),
            cpu_total_us=sum(r.cpu_total_us for r in matches),
            gpu_total_us=sum(r.gpu_total_us for r in matches),
            input_bytes=sum(r.input_bytes for r in matches),
            output_bytes=sum(r.output_bytes for r in matches),
            metrics=_sum_metrics(r.metrics for r in matches),
        )

    def table(self, *, sort_by: str = "cpu_total_us") -> str:
        rows = self.rows(sort_by=sort_by)
        has_metrics = any(row.metrics for row in rows)
        headers = [
            "Name",
            "Calls",
            "CPU total",
            "CPU avg",
            "GPU total",
            "GPU avg",
            "Input bytes",
            "Output bytes",
        ]
        if has_metrics:
            headers += ["GFLOP/s", "GB/s"]
        body = []
        for row in rows:
            if row.kernel_name is None or row.name.endswith(f":{row.kernel_name}"):
                label = row.name
            else:
                label = f"{row.name}:{row.kernel_name}"
            cells = [
                label,
                str(row.calls),
                f"{row.cpu_total_us:.2f}us",
                f"{row.cpu_avg_us:.2f}us",
                f"{row.gpu_total_us:.2f}us",
                f"{row.gpu_avg_us:.2f}us",
                str(row.input_bytes),
                str(row.output_bytes),
            ]
            if has_metrics:
                cells += [
                    f"{row.gflops_per_s:.2f}" if row.gflops_per_s else "-",
                    f"{row.gbytes_per_s:.2f}" if row.gbytes_per_s else "-",
                ]
            body.append(cells)
        widths = [
            max(len(value) for value in column)
            for column in zip(headers, *body, strict=False)
        ]

        def fmt(values: list[str]) -> str:
            cells = [values[0].ljust(widths[0])]
            cells.extend(value.rjust(width) for value, width in zip(values[1:], widths[1:]))
            return "  ".join(cells).rstrip()

        lines = [fmt(headers)]
        lines.extend(fmt(row) for row in body)
        return "\n".join(lines)


def _sum_metrics(metric_dicts: Iterable[dict[str, float]]) -> dict[str, float]:
    total: dict[str, float] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            total[key] = total.get(key, 0.0) + value
    return total


_ACTIVE_PROFILER: contextvars.ContextVar[Optional["Profiler"]] = contextvars.ContextVar(
    "enigma_active_profiler", default=None
)
_SCOPE_STACK: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "enigma_profiler_scope_stack", default=()
)

_KERNEL_HOOKS: dict[str, Callable[..., Optional[dict[str, float]]]] = {}

# Registered by MetalRuntime at construction.
_CAPTURE_BACKEND: Optional[tuple[Callable[[str], int], Callable[[], None]]] = None


def set_capture_backend(start: Callable[[str], int], stop: Callable[[], None]) -> None:
    """Register the runtime's GPU capture entry points (internal API)."""
    global _CAPTURE_BACKEND
    _CAPTURE_BACKEND = (start, stop)


def register_kernel_hook(
    kernel_name: str, hook: Callable[..., Optional[dict[str, float]]]
) -> None:
    """Attach a metrics hook to every profiled dispatch of ``kernel_name``.

    The hook is called as ``hook(kernel_name=..., grid=..., threads=...)`` and
    should return a metrics dict such as ``{"flops": ..., "bytes": ...}``.
    """
    _KERNEL_HOOKS[kernel_name] = hook


def unregister_kernel_hook(kernel_name: str) -> None:
    _KERNEL_HOOKS.pop(kernel_name, None)


class _TreeNode:
    __slots__ = ("name", "calls", "cpu_us", "gpu_us", "metrics", "children")

    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.cpu_us = 0.0
        self.gpu_us = 0.0
        self.metrics: dict[str, float] = {}
        self.children: dict[str, "_TreeNode"] = {}

    def child(self, name: str) -> "_TreeNode":
        node = self.children.get(name)
        if node is None:
            node = _TreeNode(name)
            self.children[name] = node
        return node


def _build_tree(events: Iterable[ProfilerEvent]) -> _TreeNode:
    root = _TreeNode("enigma")
    for event in events:
        node = root
        for part in event.node_path:
            node = node.child(part)
        node.calls += 1
        node.cpu_us += event.duration_us
        node.gpu_us += event.gpu_time_us
        for key, value in event.metrics.items():
            node.metrics[key] = node.metrics.get(key, 0.0) + value
    return root


class Profiler:
    """Context manager that records Enigma profiler events."""

    def __init__(
        self,
        *,
        record_shapes: bool = False,
        profile_memory: bool = False,
        capture: Optional[str | Path] = None,
    ):
        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        if capture is not None and not str(capture).endswith(".gputrace"):
            raise ValueError("capture path must end with '.gputrace'")
        self.capture = str(capture) if capture is not None else None
        self._events: list[ProfilerEvent] = []
        self._token: Optional[contextvars.Token] = None
        self._capturing = False

    def __enter__(self) -> "Profiler":
        if self.capture is not None:
            if _CAPTURE_BACKEND is None:
                raise RuntimeError(
                    "GPU capture requires a Metal runtime; create enigma.MetalRuntime() first"
                )
            rc = _CAPTURE_BACKEND[0](self.capture)
            if rc != 0:
                raise RuntimeError(
                    f"Metal GPU capture failed to start (rc={rc}). "
                    "Run with MTL_CAPTURE_ENABLED=1 in the environment."
                )
            self._capturing = True
        self._token = _ACTIVE_PROFILER.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _ACTIVE_PROFILER.reset(self._token)
            self._token = None
        if self._capturing:
            self._capturing = False
            _CAPTURE_BACKEND[1]()

    def add_event(self, event: ProfilerEvent) -> None:
        updates: dict[str, Any] = {}
        stack = _SCOPE_STACK.get()
        if stack and not event.call_path:
            updates["call_path"] = stack
        if event.category == "metal" and event.kernel_name and not event.metrics:
            hook = _KERNEL_HOOKS.get(event.kernel_name)
            if hook is not None:
                derived = hook(
                    kernel_name=event.kernel_name, grid=event.grid, threads=event.threads
                )
                if derived:
                    updates["metrics"] = {k: float(v) for k, v in derived.items()}
        if updates:
            event = dataclasses.replace(event, **updates)
        self._events.append(event)

    def events(self) -> list[ProfilerEvent]:
        return list(self._events)

    def key_averages(self, *, group_by: str = "name") -> ProfilerResult:
        return ProfilerResult(self._events, group_by=group_by)

    def tree(self) -> str:
        """Render the call-path tree with inclusive CPU/GPU times."""
        lines = ["Name" + " " * 36 + "Calls    CPU total    GPU total"]

        def emit(node: _TreeNode, depth: int) -> None:
            label = ("  " * depth + node.name).ljust(38)[:38]
            extra = ""
            if node.metrics.get("flops") and node.gpu_us:
                extra += f"  {node.metrics['flops'] / (node.gpu_us * 1000.0):.2f} GFLOP/s"
            if node.metrics.get("bytes") and node.gpu_us:
                extra += f"  {node.metrics['bytes'] / (node.gpu_us * 1000.0):.2f} GB/s"
            lines.append(
                f"{label}  {node.calls:5d}  {node.cpu_us:9.2f}us  {node.gpu_us:9.2f}us{extra}"
            )
            for child in node.children.values():
                emit(child, depth + 1)

        for child in _build_tree(self._events).children.values():
            emit(child, 0)
        return "\n".join(lines)

    def export_hatchet(self, path: str | Path) -> None:
        """Write the call-path tree as Hatchet-compatible literal JSON."""

        def to_dict(node: _TreeNode) -> dict[str, Any]:
            metrics: dict[str, float] = {
                "count": float(node.calls),
                "cpu_time_us": node.cpu_us,
                "gpu_time_us": node.gpu_us,
            }
            metrics.update(node.metrics)
            out: dict[str, Any] = {"frame": {"name": node.name}, "metrics": metrics}
            if node.children:
                out["children"] = [to_dict(c) for c in node.children.values()]
            return out

        root = _build_tree(self._events)
        Path(path).write_text(json.dumps([to_dict(root)], indent=2))

    def export_chrome_trace(self, path: str | Path) -> None:
        trace_events = []
        categories = ["python", "compile", "runtime", "metal", "memory"]
        cat_to_tid = {cat: i for i, cat in enumerate(categories, 1)}

        for cat, tid in cat_to_tid.items():
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 1,
                    "tid": tid,
                    "args": {"name": cat},
                }
            )

        for event in self._events:
            args = dict(event.metadata)
            if event.kernel_name is not None:
                args["kernel_name"] = event.kernel_name
            if event.grid is not None:
                args["grid"] = event.grid
            if event.threads is not None:
                args["threads"] = event.threads
            if event.input_bytes:
                args["input_bytes"] = event.input_bytes
            if event.output_bytes:
                args["output_bytes"] = event.output_bytes
            if event.gpu_time_us:
                args["gpu_time_us"] = event.gpu_time_us
            if event.call_path:
                args["call_path"] = "/".join(event.call_path)
            if event.metrics:
                args.update(event.metrics)
            trace_events.append(
                {
                    "name": event.name,
                    "cat": event.category,
                    "ph": "X",
                    "ts": event.start_ns / 1000.0,
                    "dur": event.duration_us,
                    "pid": 1,
                    "tid": cat_to_tid.get(event.category, 0),
                    "args": args,
                }
            )
        Path(path).write_text(json.dumps({"traceEvents": trace_events}, indent=2))


def profile(
    *,
    record_shapes: bool = False,
    profile_memory: bool = False,
    capture: Optional[str | Path] = None,
) -> Profiler:
    """Create an Enigma profiler context manager.

    ``capture`` writes a Xcode ``.gputrace`` document for the profiled region
    (requires ``MTL_CAPTURE_ENABLED=1`` and an existing ``MetalRuntime``).
    """

    return Profiler(
        record_shapes=record_shapes, profile_memory=profile_memory, capture=capture
    )


def get_active_profiler() -> Optional[Profiler]:
    """Return the current profiler, if one is active in this context."""

    return _ACTIVE_PROFILER.get()


@contextlib.contextmanager
def scope(
    name: str,
    *,
    metrics: Optional[dict[str, float]] = None,
    category: str = "scope",
    **metadata: Any,
) -> Iterator[None]:
    """Profile a named region and nest events recorded inside it.

    Custom metrics (e.g. ``{"flops": 2 * M * N * K, "bytes": ...}``) attach to
    the scope and surface as derived GFLOP/s / GB/s in tables and trees.
    No-op when no profiler is active.
    """
    profiler = get_active_profiler()
    if profiler is None:
        yield
        return

    parent = _SCOPE_STACK.get()
    path = parent + (name,)
    token = _SCOPE_STACK.set(path)
    start_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        end_ns = time.perf_counter_ns()
        _SCOPE_STACK.reset(token)
        profiler.add_event(
            ProfilerEvent(
                name=name,
                category=category,
                start_ns=start_ns,
                end_ns=end_ns,
                call_path=path,
                metrics={k: float(v) for k, v in (metrics or {}).items()},
                metadata=dict(metadata),
            )
        )


def record_function(name: str, *, category: str = "python", **metadata: Any):
    """Record a named Python scope when profiling is active (nests like scope)."""

    return scope(name, category=category, **metadata)


def record_event(event: ProfilerEvent) -> None:
    """Append an already measured event to the active profiler."""

    profiler = get_active_profiler()
    if profiler is not None:
        profiler.add_event(event)


def profile_kernel(
    prepared,
    *,
    grid: tuple[int, int, int],
    threads: tuple[int, int, int],
    repeat: int = 50,
    warmup: int = 5,
    profile_memory: bool = True,
) -> ProfilerResult:
    """Profile repeated dispatches of a prepared kernel."""

    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        prepared.dispatch(grid, threads)

    with profile(profile_memory=profile_memory) as prof:
        for _ in range(repeat):
            prepared.dispatch(grid, threads)
    return prof.key_averages()


@dataclass(frozen=True)
class KernelBenchmark:
    """Unbiased steady-state GPU timing for one prepared kernel.

    Times come from Metal command-buffer GPU timestamps, never CPU wall
    clock, so Python/ctypes overhead and unified-memory page warmup during
    the warmup phase do not pollute the numbers.
    """

    kernel_name: str
    calls: int
    gpu_times_us: tuple[float, ...]
    flops: Optional[float] = None
    bytes_moved: Optional[float] = None

    @property
    def min_us(self) -> float:
        return min(self.gpu_times_us)

    @property
    def max_us(self) -> float:
        return max(self.gpu_times_us)

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.gpu_times_us)

    @property
    def median_us(self) -> float:
        return statistics.median(self.gpu_times_us)

    @property
    def p90_us(self) -> float:
        ordered = sorted(self.gpu_times_us)
        return ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]

    @property
    def gflops_per_s(self) -> float:
        return self.flops / (self.median_us * 1000.0) if self.flops and self.median_us else 0.0

    @property
    def gbytes_per_s(self) -> float:
        return (
            self.bytes_moved / (self.median_us * 1000.0)
            if self.bytes_moved and self.median_us
            else 0.0
        )

    def summary(self) -> str:
        parts = [
            f"{self.kernel_name}: {self.calls} calls",
            f"min {self.min_us:.2f}us",
            f"p50 {self.median_us:.2f}us",
            f"mean {self.mean_us:.2f}us",
            f"p90 {self.p90_us:.2f}us",
            f"max {self.max_us:.2f}us",
        ]
        if self.gflops_per_s:
            parts.append(f"{self.gflops_per_s:.2f} GFLOP/s (p50)")
        if self.gbytes_per_s:
            parts.append(f"{self.gbytes_per_s:.2f} GB/s (p50)")
        return "  ".join(parts)


def benchmark_kernel(
    prepared,
    *,
    grid: tuple[int, int, int],
    threads: tuple[int, int, int],
    repeat: int = 100,
    warmup: int = 10,
    flops: Optional[float] = None,
    bytes_moved: Optional[float] = None,
) -> KernelBenchmark:
    """Benchmark a prepared kernel using GPU timestamps only.

    Warmup dispatches absorb pipeline state, driver, and unified-memory page
    residency effects so the measured distribution reflects steady-state
    kernel time. Report ``median_us`` for comparisons; ``min_us`` is the
    best-case and ``p90_us`` exposes tail variance (thermals, scheduling).
    """
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        prepared.dispatch(grid, threads)

    times = tuple(prepared.dispatch_timed(grid, threads) for _ in range(repeat))
    return KernelBenchmark(
        kernel_name=getattr(prepared, "_kernel_name", "") or "kernel",
        calls=repeat,
        gpu_times_us=times,
        flops=flops,
        bytes_moved=bytes_moved,
    )
