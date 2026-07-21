# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

import json
import time

import numpy as np
import pytest

import enigma
from enigma.compiler.compiler import CompiledKernel
from enigma.runtime_dispatch import runtime as runtime_mod
from enigma.runtime_dispatch.runtime import MetalRuntime
from enigma.runtime_dispatch.runtime import PreparedKernel


def test_record_function_collects_events_and_key_averages():
    with enigma.profile() as prof:
        with enigma.record_function("outer"):
            time.sleep(0.00001)
            with enigma.record_function("inner"):
                time.sleep(0.00001)

    events = prof.events()
    assert [e.name for e in events] == ["inner", "outer"]
    assert all(e.category == "python" for e in events)
    assert all(e.duration_us >= 0.0 for e in events)

    averages = prof.key_averages()
    assert averages.by_name("outer").calls == 1
    assert averages.by_name("inner").calls == 1
    assert "outer" in averages.table()
    assert "CPU total" in averages.table()


def test_record_function_is_noop_without_active_profiler():
    with enigma.record_function("outside"):
        time.sleep(0.00001)


def test_key_averages_groups_repeated_events():
    with enigma.profile() as prof:
        for _ in range(3):
            with enigma.record_function("repeat"):
                time.sleep(0.00001)

    row = prof.key_averages().by_name("repeat")
    assert row.calls == 3
    assert row.cpu_total_us >= row.cpu_avg_us
    assert row.gpu_total_us == 0.0


def test_table_renders_aligned_columns():
    with enigma.profile() as prof:
        for name in ("short", "much_longer_scope"):
            with enigma.record_function(name):
                time.sleep(0.00001)

    table = prof.key_averages().table()
    lines = table.splitlines()
    assert lines[0].startswith("Name")
    assert "  Calls  " in lines[0]
    assert "  CPU total  " in lines[0]
    assert "short" in table
    assert "much_longer_scope" in table

    calls_index = lines[0].index("Calls")
    calls_right_edge = calls_index + len("Calls") - 1
    assert lines[1].index("1") == calls_right_edge
    assert lines[2].index("1") == calls_right_edge


def test_export_chrome_trace_writes_valid_trace(tmp_path):
    trace_path = tmp_path / "enigma_trace.json"

    with enigma.profile() as prof:
        with enigma.record_function("scope"):
            time.sleep(0.00001)

    prof.export_chrome_trace(trace_path)

    payload = json.loads(trace_path.read_text())
    assert "traceEvents" in payload
    thread_names = [e for e in payload["traceEvents"] if e["ph"] == "M"]
    assert all(isinstance(e["tid"], int) for e in thread_names)

    scope_event = next(e for e in payload["traceEvents"] if e["ph"] == "X")
    assert scope_event["name"] == "scope"
    assert scope_event["cat"] == "python"
    assert isinstance(scope_event["tid"], int)


class _FakeRuntimeLib:
    def __init__(self):
        self.dispatch_calls = 0
        self.dispatch_timed_calls = 0
        self.dispatch_profiled_calls = 0

    def enigma_dispatch(self, *args):
        self.dispatch_calls += 1
        return 0

    def enigma_dispatch_timed(self, *args):
        self.dispatch_timed_calls += 1
        out_gpu_time_us = args[-1]
        out_gpu_time_us._obj.value = 12.5
        return 0

    def enigma_dispatch_profiled(self, *args):
        self.dispatch_profiled_calls += 1
        timings = args[-1]
        timings[0] = 12.5  # gpu_us
        timings[1] = 3.0   # scheduling_us
        timings[2] = 1.0   # queue_wait_us
        timings[3] = 11.0  # encoder_gpu_us
        timings[4] = 1.0   # counters supported
        return 0

    def enigma_pipeline_stats(self, _pso, out_vals):
        out_vals[0] = 256  # maxTotalThreadsPerThreadgroup
        out_vals[1] = 32   # threadExecutionWidth
        out_vals[2] = 1024  # staticThreadgroupMemoryLength

    def enigma_buffer_length(self, _buf):
        return 4


class _FakeRuntime:
    _pipeline_stats_metadata = MetalRuntime._pipeline_stats_metadata

    def __init__(self):
        self._lib = _FakeRuntimeLib()
        self._device = object()
        self._queue = object()


def _fake_prepared_kernel():
    rt = _FakeRuntime()
    prepared = PreparedKernel(
        rt,
        pso=object(),
        mtl_lib=object(),
        gpu_bufs=[object(), object(), object()],
        buf_arr=object(),
        out_buf=object(),
        output_size=16,
        kernel_name="fake_kernel",
    )
    return rt, prepared


def test_prepared_dispatch_uses_existing_path_when_profiler_disabled():
    rt, prepared = _fake_prepared_kernel()

    prepared.dispatch(grid=(16, 1, 1), threads=(8, 1, 1))

    assert rt._lib.dispatch_calls == 1
    assert rt._lib.dispatch_timed_calls == 0


def test_prepared_dispatch_records_gpu_event_when_profiler_enabled(monkeypatch):
    rt, prepared = _fake_prepared_kernel()
    ticks = iter([1_000, 2_000, 102_000, 111_000])
    monkeypatch.setattr(runtime_mod.time, "perf_counter_ns", lambda: next(ticks))

    with enigma.profile() as prof:
        prepared.dispatch(grid=(16, 1, 1), threads=(8, 1, 1))

    events = prof.events()
    names = [e.name for e in events]
    assert names == ["gpu_dispatch", "prepared_dispatch"]
    assert rt._lib.dispatch_calls == 0
    assert rt._lib.dispatch_profiled_calls == 1

    gpu_event = events[0]
    assert gpu_event.category == "metal"
    assert gpu_event.kernel_name == "fake_kernel"
    assert gpu_event.grid == (16, 1, 1)
    assert gpu_event.threads == (8, 1, 1)
    assert gpu_event.gpu_time_us == 12.5
    assert gpu_event.duration_us == 100.0
    assert gpu_event.buffer_count == 3
    assert gpu_event.metadata["scheduling_us"] == 3.0
    assert gpu_event.metadata["queue_wait_us"] == 1.0
    assert gpu_event.metadata["encoder_gpu_us"] == 11.0
    assert gpu_event.metadata["max_threads_per_threadgroup"] == 256
    assert gpu_event.metadata["thread_execution_width"] == 32
    assert gpu_event.metadata["static_threadgroup_memory_bytes"] == 1024
    assert gpu_event.metadata["threadgroup_occupancy"] == 8 / 256


class _FakeExecuteLib:
    def __init__(self):
        self.dispatch_calls = 0
        self.dispatch_timed_calls = 0
        self._next_handle = 10
        self._buffers = {}

    def _handle(self):
        self._next_handle += 1
        return self._next_handle

    def enigma_load_library(self, *args):
        return self._handle()

    def enigma_create_pipeline(self, *args):
        return self._handle()

    def enigma_create_buffer(self, *args):
        return self._handle()

    def enigma_create_buffer_empty(self, _device, length):
        handle = self._handle()
        buf = b"out!" if length == 4 else bytes(int(length))
        self._buffers[handle] = bytearray(buf)
        return handle

    def enigma_buffer_contents(self, handle):
        import ctypes

        data = self._buffers[handle]
        cbuf = (ctypes.c_char * len(data)).from_buffer(data)
        self._buffers[(handle, "ctypes")] = cbuf
        return ctypes.addressof(cbuf)

    def enigma_dispatch(self, *args):
        self.dispatch_calls += 1
        return 0

    def enigma_dispatch_timed(self, *args):
        self.dispatch_timed_calls += 1
        out_gpu_time_us = args[-1]
        out_gpu_time_us._obj.value = 33.0
        return 0

    def enigma_dispatch_profiled(self, *args):
        self.dispatch_timed_calls += 1
        timings = args[-1]
        timings[0] = 33.0
        timings[1] = 2.0
        timings[2] = 0.5
        timings[3] = 30.0
        timings[4] = 1.0
        return 0

    def enigma_pipeline_stats(self, _pso, out_vals):
        out_vals[0] = 1024
        out_vals[1] = 32
        out_vals[2] = 0

    def enigma_release(self, *args):
        return None


def _fake_metal_runtime():
    rt = object.__new__(MetalRuntime)
    rt._lib = _FakeExecuteLib()
    rt._device = object()
    rt._queue = object()
    return rt


def test_execute_records_runtime_stages_when_profiler_enabled():
    rt = _fake_metal_runtime()
    compiled = CompiledKernel(
        kernel_name="fake_execute",
        metallib_path="/tmp/fake.metallib",
        metallib_bytes=b"",
        metal_source="",
    )
    arr = np.array([1.0], dtype=np.float32)

    with enigma.profile(profile_memory=True) as prof:
        out = rt.execute(compiled, [arr], output_size=4, grid=(1, 1, 1), threads=(1, 1, 1))

    assert out == b"out!"
    names = [e.name for e in prof.events()]
    assert "load_library" in names
    assert "create_pipeline" in names
    assert "create_input_buffers" in names
    assert "create_output_buffers" in names
    assert "gpu_dispatch" in names
    assert "read_output" in names
    assert "execute_total" in names
    assert rt._lib.dispatch_calls == 0
    assert rt._lib.dispatch_timed_calls == 1

    execute_total = prof.key_averages().by_name("execute_total")
    gpu_dispatch = prof.key_averages().by_name("gpu_dispatch")
    assert execute_total.input_bytes == 4
    assert execute_total.output_bytes == 4
    assert gpu_dispatch.gpu_total_us == 33.0


def test_profile_kernel_runs_warmup_unprofiled_and_repeats_profiled():
    rt, prepared = _fake_prepared_kernel()

    result = enigma.profile_kernel(
        prepared,
        grid=(16, 1, 1),
        threads=(8, 1, 1),
        repeat=2,
        warmup=1,
    )

    assert rt._lib.dispatch_calls == 1
    assert rt._lib.dispatch_profiled_calls == 2
    assert result.by_name("gpu_dispatch").calls == 2
    assert result.by_name("prepared_dispatch").calls == 2


# ---------------------------------------------------------------------------
# Proton-style features: scopes, call paths, metrics, hooks, tree, exports.
# ---------------------------------------------------------------------------


def test_scope_nesting_builds_call_paths():
    with enigma.profile() as prof:
        with enigma.scope("outer"):
            with enigma.scope("inner"):
                pass

    by_name = {e.name: e for e in prof.events()}
    assert by_name["inner"].call_path == ("outer", "inner")
    assert by_name["outer"].call_path == ("outer",)
    assert by_name["inner"].node_path == ("outer", "inner")


def test_scope_is_noop_without_active_profiler():
    with enigma.scope("ignored", metrics={"flops": 1.0}):
        pass
    assert enigma.profiler.get_active_profiler() is None


def test_runtime_events_inherit_enclosing_scope_path(monkeypatch):
    rt, prepared = _fake_prepared_kernel()

    with enigma.profile() as prof:
        with enigma.scope("layer0"):
            prepared.dispatch(grid=(16, 1, 1), threads=(8, 1, 1))

    gpu_event = next(e for e in prof.events() if e.name == "gpu_dispatch")
    assert gpu_event.call_path == ("layer0",)
    assert gpu_event.node_path == ("layer0", "gpu_dispatch:fake_kernel")


def test_scope_metrics_produce_derived_throughput():
    with enigma.profile() as prof:
        with enigma.scope("matmul", metrics={"flops": 2.0e9, "bytes": 1.0e9}):
            time.sleep(0.001)

    row = prof.key_averages().by_name("matmul")
    assert row.metrics["flops"] == 2.0e9
    assert row.gflops_per_s > 0.0
    assert row.gbytes_per_s > 0.0
    table = prof.key_averages().table()
    assert "GFLOP/s" in table
    assert "GB/s" in table


def test_table_has_no_metric_columns_without_metrics():
    with enigma.profile() as prof:
        with enigma.scope("plain"):
            pass
    assert "GFLOP/s" not in prof.key_averages().table()


def test_kernel_hook_attaches_metrics_to_gpu_dispatch():
    rt, prepared = _fake_prepared_kernel()

    def hook(*, kernel_name, grid, threads):
        assert kernel_name == "fake_kernel"
        return {"flops": float(grid[0] * threads[0])}

    enigma.register_kernel_hook("fake_kernel", hook)
    try:
        with enigma.profile() as prof:
            prepared.dispatch(grid=(16, 1, 1), threads=(8, 1, 1))
    finally:
        enigma.unregister_kernel_hook("fake_kernel")

    gpu_event = next(e for e in prof.events() if e.name == "gpu_dispatch")
    assert gpu_event.metrics == {"flops": 128.0}
    row = prof.key_averages().by_name("gpu_dispatch")
    assert row.gflops_per_s > 0.0


def test_key_averages_group_by_call_path_separates_contexts():
    with enigma.profile() as prof:
        for parent in ("attention", "mlp"):
            with enigma.scope(parent):
                with enigma.scope("gemm"):
                    pass

    flat = prof.key_averages().by_name("gemm")
    assert flat.calls == 2
    paths = {row.name for row in prof.key_averages(group_by="call_path").rows()}
    assert "attention/gemm" in paths
    assert "mlp/gemm" in paths


def test_tree_renders_hierarchy():
    with enigma.profile() as prof:
        with enigma.scope("step"):
            with enigma.scope("gemm", metrics={"flops": 1.0e6}):
                pass

    tree = prof.tree()
    lines = tree.splitlines()
    assert any(line.startswith("step") for line in lines)
    assert any(line.startswith("  gemm") for line in lines)


def test_export_hatchet_writes_nested_json(tmp_path):
    with enigma.profile() as prof:
        with enigma.scope("outer"):
            with enigma.scope("inner", metrics={"bytes": 64.0}):
                pass

    out = tmp_path / "profile.json"
    prof.export_hatchet(out)
    payload = json.loads(out.read_text())
    root = payload[0]
    assert root["frame"]["name"] == "enigma"
    outer = root["children"][0]
    assert outer["frame"]["name"] == "outer"
    inner = outer["children"][0]
    assert inner["frame"]["name"] == "inner"
    assert inner["metrics"]["bytes"] == 64.0
    assert inner["metrics"]["count"] == 1.0


def test_chrome_trace_includes_call_path_and_metrics(tmp_path):
    with enigma.profile() as prof:
        with enigma.scope("outer"):
            with enigma.scope("inner", metrics={"flops": 2.0}):
                pass

    out = tmp_path / "trace.json"
    prof.export_chrome_trace(out)
    events = json.loads(out.read_text())["traceEvents"]
    inner = next(e for e in events if e["name"] == "inner")
    assert inner["args"]["call_path"] == "outer/inner"
    assert inner["args"]["flops"] == 2.0


def test_benchmark_kernel_uses_gpu_timestamps_only():
    rt, prepared = _fake_prepared_kernel()

    bench = enigma.benchmark_kernel(
        prepared,
        grid=(16, 1, 1),
        threads=(8, 1, 1),
        repeat=4,
        warmup=2,
        flops=1.0e6,
        bytes_moved=2.0e6,
    )

    # warmup uses the fast untimed path; measured calls use GPU timestamps
    assert rt._lib.dispatch_calls == 2
    assert rt._lib.dispatch_timed_calls == 4
    assert bench.calls == 4
    assert bench.gpu_times_us == (12.5, 12.5, 12.5, 12.5)
    assert bench.median_us == 12.5
    assert bench.gflops_per_s == 1.0e6 / (12.5 * 1000.0)
    assert bench.gbytes_per_s == 2.0e6 / (12.5 * 1000.0)
    assert "fake_kernel" in bench.summary()


def test_benchmark_kernel_validates_args():
    _, prepared = _fake_prepared_kernel()
    with pytest.raises(ValueError):
        enigma.benchmark_kernel(prepared, grid=(1, 1, 1), threads=(1, 1, 1), repeat=0)


def test_profile_capture_rejects_non_gputrace_path():
    with pytest.raises(ValueError):
        enigma.profile(capture="trace.json")


def test_profile_capture_requires_runtime_backend(monkeypatch):
    from enigma import profiler as profiler_mod

    monkeypatch.setattr(profiler_mod, "_CAPTURE_BACKEND", None)
    with pytest.raises(RuntimeError, match="MetalRuntime"):
        with enigma.profile(capture="/tmp/t.gputrace"):
            pass


def test_profile_capture_starts_and_stops_backend(monkeypatch):
    from enigma import profiler as profiler_mod

    calls = []
    monkeypatch.setattr(
        profiler_mod,
        "_CAPTURE_BACKEND",
        (lambda p: calls.append(("start", p)) or 0, lambda: calls.append(("stop",))),
    )
    with enigma.profile(capture="/tmp/t.gputrace"):
        assert calls == [("start", "/tmp/t.gputrace")]
    assert calls == [("start", "/tmp/t.gputrace"), ("stop",)]


def test_profile_capture_start_failure_raises(monkeypatch):
    from enigma import profiler as profiler_mod

    stops = []
    monkeypatch.setattr(
        profiler_mod, "_CAPTURE_BACKEND", (lambda p: -1, lambda: stops.append(1))
    )
    with pytest.raises(RuntimeError, match="MTL_CAPTURE_ENABLED"):
        with enigma.profile(capture="/tmp/t.gputrace"):
            pass
    assert stops == []  # never started, so never stopped


def test_execute_gpu_dispatch_carries_phase_and_pipeline_metadata():
    rt = _fake_metal_runtime()
    compiled = CompiledKernel(
        kernel_name="fake_execute",
        metallib_path="/tmp/fake.metallib",
        metallib_bytes=b"",
        metal_source="",
    )
    arr = np.array([1.0], dtype=np.float32)

    with enigma.profile() as prof:
        rt.execute(compiled, [arr], output_size=4, grid=(4, 1, 1), threads=(2, 1, 1))

    gpu = next(e for e in prof.events() if e.name == "gpu_dispatch")
    assert gpu.metadata["scheduling_us"] == 2.0
    assert gpu.metadata["queue_wait_us"] == 0.5
    assert gpu.metadata["encoder_gpu_us"] == 30.0
    assert gpu.metadata["max_threads_per_threadgroup"] == 1024
    assert gpu.metadata["thread_execution_width"] == 32
    assert gpu.metadata["threadgroup_occupancy"] == 2 / 1024


def test_profiled_timings_metadata_omits_encoder_when_unsupported():
    supported = runtime_mod._profiled_timings_metadata([7.0, 2.0, 0.5, 30.0, 1.0])
    assert supported["scheduling_us"] == 2.0
    assert supported["queue_wait_us"] == 0.5
    assert supported["encoder_gpu_us"] == 30.0

    unsupported = runtime_mod._profiled_timings_metadata([7.0, 2.0, 0.5, 30.0, 0.0])
    assert unsupported["scheduling_us"] == 2.0
    assert unsupported["queue_wait_us"] == 0.5
    assert "encoder_gpu_us" not in unsupported


def test_kernel_benchmark_statistics_over_varied_samples():
    from enigma.profiler import KernelBenchmark

    times = (30.0, 10.0, 50.0, 20.0, 40.0)  # deliberately unordered
    b = KernelBenchmark(
        kernel_name="k",
        calls=5,
        gpu_times_us=times,
        flops=1.0e9,
        bytes_moved=2.0e9,
    )
    assert b.min_us == 10.0
    assert b.max_us == 50.0
    assert b.mean_us == 30.0
    assert b.median_us == 30.0
    assert b.p90_us == 50.0  # int(0.9*5)=4 -> sorted[4]
    assert b.gflops_per_s == 1.0e9 / (30.0 * 1000.0)
    assert b.gbytes_per_s == 2.0e9 / (30.0 * 1000.0)
    summary = b.summary()
    assert "min 10.00us" in summary
    assert "p50 30.00us" in summary
    assert "p90 50.00us" in summary
    assert "max 50.00us" in summary
    assert "GFLOP/s (p50)" in summary
    assert "GB/s (p50)" in summary


def test_scope_exception_records_event_and_restores_stack():
    from enigma.profiler import _SCOPE_STACK

    with enigma.profile() as prof:
        with pytest.raises(ValueError):
            with enigma.scope("boom", metrics={"flops": 10.0}):
                raise ValueError("kernel blew up")
        # Stack must unwind even though the scope body raised.
        assert _SCOPE_STACK.get() == ()

    event = next(e for e in prof.events() if e.name == "boom")
    assert event.metrics["flops"] == 10.0
    assert _SCOPE_STACK.get() == ()


def test_deep_scope_nesting_builds_full_call_path():
    with enigma.profile() as prof:
        with enigma.scope("a"):
            with enigma.scope("b"):
                with enigma.scope("c"):
                    with enigma.scope("d"):
                        pass

    paths = {e.name: e.call_path for e in prof.events()}
    assert paths["d"] == ("a", "b", "c", "d")
    assert paths["c"] == ("a", "b", "c")
    assert paths["a"] == ("a",)


def test_by_name_raises_keyerror_for_missing_name():
    with enigma.profile() as prof:
        with enigma.scope("present"):
            pass
    with pytest.raises(KeyError):
        prof.key_averages().by_name("absent")


def test_key_averages_rejects_unknown_group_by():
    with enigma.profile() as prof:
        with enigma.scope("present"):
            pass
    with pytest.raises(ValueError):
        prof.key_averages(group_by="invalid")


def test_profile_kernel_validates_args():
    _, prepared = _fake_prepared_kernel()
    with pytest.raises(ValueError):
        enigma.profile_kernel(prepared, grid=(1, 1, 1), threads=(1, 1, 1), repeat=0)
    with pytest.raises(ValueError):
        enigma.profile_kernel(prepared, grid=(1, 1, 1), threads=(1, 1, 1), warmup=-1)
