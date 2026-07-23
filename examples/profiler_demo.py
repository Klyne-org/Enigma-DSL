# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Klyne Research

"""Demo of the Enigma profiler on real Metal hardware."""

import json

import numpy as np

import enigma


@enigma.kernel
def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]


@enigma.kernel
def vector_scale(A: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] * 2.0


def main() -> None:
    rt = enigma.MetalRuntime()
    n = 1 << 20
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)

    add = enigma.compile(vector_add)
    scale = enigma.compile(vector_scale)

    # Hooks: metrics attach automatically to every profiled dispatch.
    enigma.register_kernel_hook(
        "vector_add",
        lambda *, kernel_name, grid, threads: {"flops": float(grid[0]), "bytes": 12.0 * grid[0]},
    )
    enigma.register_kernel_hook(
        "vector_scale",
        lambda *, kernel_name, grid, threads: {"flops": float(grid[0]), "bytes": 8.0 * grid[0]},
    )

    with enigma.profile() as prof:
        with enigma.scope("step0"):
            with enigma.scope("add_layer"):
                out = rt.execute(add, [a, b], n * 4, grid=(n, 1, 1), threads=(256, 1, 1))
            with enigma.scope("scale_layer"):
                rt.execute(scale, [a], n * 4, grid=(n, 1, 1), threads=(256, 1, 1))

    res = np.frombuffer(out, dtype=np.float32)
    print("max error:", float(np.abs(res - (a + b)).max()))

    print("\n=== key_averages (by name) ===")
    print(prof.key_averages().table(sort_by="gpu_total_us"))

    print("\n=== key_averages (by call path) ===")
    print(prof.key_averages(group_by="call_path").table(sort_by="gpu_total_us"))

    print("\n=== call tree ===")
    print(prof.tree())

    prof.export_hatchet("/tmp/enigma_profile.json")
    prof.export_chrome_trace("/tmp/enigma_trace.json")
    print("\nwrote /tmp/enigma_profile.json (Hatchet) and /tmp/enigma_trace.json (Chrome)")

    print("\n=== unbiased benchmark (GPU timestamps, steady state) ===")
    prepared = rt.prepare(add, [a, b], n * 4)
    try:
        bench = enigma.benchmark_kernel(
            prepared,
            grid=(n, 1, 1),
            threads=(256, 1, 1),
            repeat=100,
            warmup=10,
            bytes_moved=12.0 * n,
        )
    finally:
        prepared.release()
    print(bench.summary())

    def row_dict(row):
        return {
            "name": row.name,
            "kernel_name": row.kernel_name,
            "calls": row.calls,
            "cpu_total_us": row.cpu_total_us,
            "cpu_avg_us": row.cpu_avg_us,
            "gpu_total_us": row.gpu_total_us,
            "gpu_avg_us": row.gpu_avg_us,
            "input_bytes": row.input_bytes,
            "output_bytes": row.output_bytes,
            "metrics": row.metrics,
            "gflops_per_s": row.gflops_per_s,
            "gbytes_per_s": row.gbytes_per_s,
        }

    results = {
        "key_averages_by_name": [
            row_dict(r) for r in prof.key_averages().rows(sort_by="gpu_total_us")
        ],
        "key_averages_by_call_path": [
            row_dict(r)
            for r in prof.key_averages(group_by="call_path").rows(sort_by="gpu_total_us")
        ],
        "tree": prof.tree(),
        "benchmark": {
            "kernel_name": bench.kernel_name,
            "calls": bench.calls,
            "min_us": bench.min_us,
            "p50_us": bench.median_us,
            "mean_us": bench.mean_us,
            "p90_us": bench.p90_us,
            "max_us": bench.max_us,
            "gbytes_per_s_p50": bench.gbytes_per_s,
            "gpu_times_us": list(bench.gpu_times_us),
        },
    }
    with open("/tmp/enigma_profiler_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote /tmp/enigma_profiler_results.json (all outputs)")


if __name__ == "__main__":
    main()
