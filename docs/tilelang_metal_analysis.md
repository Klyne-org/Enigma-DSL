# TileLang Metal Backend Analysis vs Enigma DSL

A side-by-side analysis of the Metal/Apple Silicon support in `tile-ai/tilelang`
and what we currently have in **Enigma DSL**, with a concrete list of changes /
features we should consider porting or building.

> Sources:
> - https://github.com/tile-ai/tilelang (main, Nov 2025)
> - PRs: #799, #1051, #1021, #1289, #1547, #1857, #1869, #2118, #2130, #2110, #2114
> - Local Enigma tree: `/Users/tanmay/Desktop/ENIGMA WORK/Enigma-DSL`

---

## 1. TileLang at a Glance (Metal-Focused)

TileLang is a TVM-based DSL that compiles a tile-level Python IR (`T.Kernel`,
`T.Pipelined`, `T.copy`, `T.gemm`, `T.alloc_shared`, `T.alloc_fragment`, …) to
multiple GPU backends. The Metal path is one of several backends; CUDA / HIP
are still primary, but Apple Silicon support has been actively ramped up
through 2025.

### 1.1 Key Metal-related repository surface (verified on `main`)

Dirs / files exclusive to Metal that exist today:

```
src/backend/metal/CMakeLists.txt
testing/python/metal/test_metal_codegen.py
testing/python/metal/test_metal_codegen_linux.py
tilelang/carver/arch/metal.py             # METAL TileDevice arch class
tilelang/jit/adapter/torch/metal.py       # MetalKernelAdapter (uses torch.mps.compile_shader)
tilelang/transform/metal/__init__.py
tilelang/transform/metal/mark_host_metal_context.py
requirements-test-metal.txt
```

The user-visible target name is just `"metal"` (see `docs/get_started/targets.md`).
TileLang relies on **TVM's existing Metal codegen** (TVM has had MSL emission for
years) plus tilelang-specific lowering passes; on top of that PR #1869 / #2130
introduce a *forked* `codegen_metal.cc` inside tilelang itself, decoupled from
TVM, so simdgroup intrinsics can be emitted without forking TVM.

### 1.2 Metal-specific PR timeline

| PR    | Date    | State    | What it adds                                                                                                                |
| ----- | ------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| #799  | 10/2025 | merged   | First Metal backend. Targets `torch.mps.compile_shader`. Adds a new `torch` execution backend. GEMM via `T.copy + T.Parallel + T.Serial` (no simdgroup MMA yet). 2-3x slower than MPSGraph, but works. |
| #1021 | 2025    | merged   | Bugfix: `torch.mps.is_available` checks for older PyTorch builds.                                                          |
| #1051 | 2025    | merged   | More MPS availability shims across torch versions.                                                                          |
| #1289 | 2025    | merged   | `tvm-ffi` for Metal: passes `torch::mps::get_command_buffer()` into TVM so kernels share command buffers with PyTorch.       |
| #1547 | 2025    | merged   | `improve benchmark on mps`. Adds repeatable MPS benchmarking utilities.                                                      |
| #1857 | 2025    | merged   | **`[Codegen] Metal codegen on Linux`** — Metal source generation works on non-Apple hosts (no GPU runtime, source only). Adds float32/float16/int32 codegen tests. Lets CI run on Linux. |
| #1869 | open    | open     | **`[Metal] Add Metal GEMM support with simdgroup_matrix MMA`**. Forks TVM `codegen_metal.cc` into tilelang, adds 8x8 `simdgroup_load/store/multiply_accumulate`, multi-warp partitioning, 128-bit vectorized copies, and a `metal_fragment_to_simdgroup` pass so users keep writing target-agnostic `T.gemm` + `alloc_fragment` and it lowers to simdgroup MMA on Metal. |
| #2118 | open    | open     | **`Add Metal scalar fallback for T.gemm`**. Correctness-first scalar fallback so HD128 paged-attention kernels lower on macOS even without simdgroup MMA. Adds replicated fragments, local reductions, `tl.infinity`, target-scoped passes, and shared-mem codegen compat. |
| #2130 | open    | open     | **`Rebase Metal simdgroup GEMM support and runtime coverage`**. Rebase of #1869 onto current backend-local CMake layout (#2114), adds scalar `local.var` lowering, MPS as JIT fallback when CUDA is missing, register-tile / packed uint8 quant probes, GDN/attention-style runtime tests. 39 tests pass, 3 skipped. |
| #2114 | merged  | refactor | Backend-local CMake split (`src/backend/{nvidia,amd,apple,cpu,webgpu}`).                                                     |
| #2110 | open    | refactor | Python-side mirror split (`tilelang/backend/{nvidia,amd,apple,…}`), explicit registry with backend specs (FFI builders, exec backend, pass hooks, cache metadata).                                                |

### 1.3 What Metal currently supports in TileLang (`main` + open PRs)

- **Codegen**: TVM's MSL emitter + tilelang fork `codegen_metal.cc` (#1869) for
  simdgroup intrinsics and `device __packed_*` 128-bit vectorized copies.
- **Compile flow**: tilelang lowering → TVM Metal module → MSL string →
  `torch.mps.compile_shader(source)` to obtain a callable shader.
- **Dispatch**: PyTorch's MPS command buffer is reused via `tvm-ffi` (#1289),
  so a tilelang kernel sits inside the same MPS stream as the surrounding torch
  workload — no extra synchronisation cost.
- **Tile primitives that lower to Metal**:
  - `T.Kernel((gx, gy), threads=N)` → `dispatchThreadgroups`
  - `T.alloc_shared` → `threadgroup` MSL address space
  - `T.alloc_fragment` → `thread`-local arrays (and, with #1869, `metal.simdgroup` storage scope)
  - `T.copy(global → shared, coalesced_width=…)` → emits 128-bit
    `device __packed_float4` loads
  - `T.Pipelined` → unrolled tiles (no async copy on Metal yet)
  - `T.gemm(A_s, B_s, C_l)` → simdgroup 8x8 MMA chain (#1869) or scalar
    fallback (#2118).
  - `T.Parallel` / `T.Serial` over fragments
- **Datatypes verified in tests**: `float32`, `float16`, `int32`. `bf16` and
  `fp8/fp4` are explicitly **not** native on Metal yet — #2130 keeps them
  fail-closed and uses packed `uint8` probes.
- **CI**: tests live in `testing/python/metal/`; codegen tests run on Linux
  (#1857), runtime tests require Apple Silicon (`tilelang.testing.requires_metal`).

### 1.4 What TileLang does **not** have on Metal yet

- No async copy / `cp.async` equivalent (Apple has no real DMA, but
  threadgroup `metal::async_copy` would be a fit and is unused).
- No tensor cores beyond 8x8 simdgroup MMA (Apple's M3+ MPSGraph tensor
  cores / matmul ops are not exposed).
- No `bf16`, `fp8`, `fp4` storage in shaders (only via packed uint8 probes).
- No autotuner pass-set tuned for Metal (`carver/arch/metal.py` is a stub).
- No tile-of-textures path (Apple GPUs prefer image blocks for some loads, not used).
- No native MPSGraph fallback for `T.gemm` (#1869 chose simdgroup MMA path
  over MPSGraph for portability).

---

## 2. Enigma DSL at a Glance

Local layout:

```
enigma/
├── _tracing.py              # SSA IR + KernelBuilder + Metal intrinsics tracing
├── core.py                  # Layout algebra (CuTe-style)
├── tensor.py                # Tracing / runtime Tensor with TV layout
├── tuple.py                 # IntTuple helpers
├── typing.py                # f32/f16/bf16/i*/u*/Bool scalar types
├── compiler/
│   ├── compiler.py          # Trace -> CompiledKernel
│   ├── kernel.py            # @enigma.kernel / @enigma.jit / KernelHandle
│   └── mlir_emitter.py      # IR -> Metal C++ source (currently emits MSL directly, name is legacy)
└── runtime_dispatch/
    ├── runtime.py           # MetalRuntime (ctypes -> Swift dylib)
    ├── mlx_interop.py       # mlx.core.array zero-copy on unified memory
    └── swift/libenigma_runtime.{swift,dylib}

Enigma-Dialect/                # Sister MLIR project (C++ / TableGen)
├── include/enigma/Dialect/Enigma/IR/EnigmaOps.td
├── lib/Target/MSL/            # MSL emitter
└── tools/{enigma-opt,enigma-translate,enigma-runner}
```

### 2.1 What Enigma already has

- **Layout algebra** (`enigma.core`): full CuTe-style algebra
  (`composition`, `complement`, `coalesce`, `zipped_divide`, `make_layout_tv`,
  `Swizzle`, `recast_layout`, …).
- **Tracing IR** (`_tracing.py`): SSA values, constant folding, thread index
  decomposition, control flow (`for_range`, `if_`, `while_`, `Carry`),
  register tensors, predicated `load_if/store_if`, async-copy stubs,
  pipeline scoping.
- **Metal intrinsics** (re-exported from `enigma/__init__.py`):
  - SIMD: `simd_sum/product/min/max/and/or/xor`, `simd_shuffle{,_up,_down,_xor}`,
    `simd_broadcast`, `simd_prefix_*_sum/product`, `simd_barrier`.
  - Quad: `quad_*` analogous set.
  - Atomics: `atomic_load/store/exchange/compare_exchange_weak/fetch_{add,sub,min,max,and,or,xor}`.
  - Threadgroup: `barrier`, `threadgroup_alloc`, async copy stubs.
  - Simdgroup matrix: `simdgroup_matrix_load/store`, `simdgroup_multiply_accumulate`,
    `make_filled_simdgroup_matrix`, plus `matmul`, `transpose`, `determinant` helpers.
  - Scalar math (full MSL math.h): `sqrt/abs/ceil/floor/round/sin/cos/...` etc.
  - Vector helpers: `make_float{2,3,4}`, `vec_extract`, geometry (`dot`, `length`, `cross`, `normalize`, `reflect`, `refract`, `faceforward`).
  - Pack/unpack: `pack_float_to_{snorm,unorm}{4x8,2x16}`, srgb, `unorm10a2`.
  - Bit ops: `popcount`, `clz`, `ctz`, `reverse_bits`, `extract_bits`, `insert_bits`.
  - Saturating arith: `add_sat`, `sub_sat`, `mul_hi`, `mad_sat`, `abs_diff`.
- **Function constants**: `function_constant(name, dtype)` (Metal FC pipeline).
- **Decorators**: `@enigma.kernel`, `@enigma.jit` with TV-layout host code
  driving grid/block.
- **Runtime**:
  - Swift dylib (`libenigma_runtime.swift`) loaded via ctypes; manages device,
    queues, pipelines, dispatches.
  - Auto-build of dylib on stale source.
  - Function-constant pipeline with packed scalar FCs.
  - GPU timestamp benchmarking.
  - `mlx.core.array` zero-copy in / out (Triton-style API).
- **Companion MLIR project (`Enigma-Dialect/`)**: independent MLIR dialect
  with `enigma.kernel`, `thread_position_in_grid`, threadgroup barrier ops,
  MSL emitter, and a `enigma-runner` end-to-end on `.metallib`. Not yet
  integrated with the Python tracing pipeline (the Python emitter prints
  Metal C++ directly).
- **Examples**: vector_add (naive + TV + float4 + mlx), matmul, GEMM, FA forward
  showcase, atomic counter, threadgroup shared memory test, RMSNorm benchmark,
  control-flow tests, SIMD/quad/atomic/pack/vector/geom/binary-ternary/sqrt/int
  smoke tests, `mlir_ops_smoke.py`.
- **Tests**: ~30 passing tests covering layout algebra, GPU execution, MSL
  source export, IR tracing.

### 2.2 What Enigma does **not** have today

- No autotuning / pass-set search.
- No `T.Pipelined` style multistage pipeline lowering with ring buffers; we
  have async-copy stubs but no commit/wait scheduling and no ring rotation.
- No `T.gemm` tile op — users still hand-roll MMA via `simdgroup_matrix_*`.
- No native bf16, fp8, fp4 storage / packing.
- No PyTorch MPS shared-command-buffer integration. Our runtime owns its own
  command queue; if a user has torch tensors we copy via MLX. Sharing the same
  MPS command buffer with `torch.mps.compile_shader` (#1289 pattern) would
  remove sync overhead.
- No `T.copy(... coalesced_width=k)` sugar — users hand-build `make_float4`.
- No dialect-driven lowering: `Enigma-Dialect` exists in isolation; the
  Python pipeline goes Tracing IR → Metal C++ string directly.
- No `requires_metal` testing helper; tests assume Apple Silicon.
- No autodispatch fallback to MPSGraph for `T.gemm`-equivalent.

---

## 3. Feature Comparison Matrix

| Capability                                  | TileLang (Metal)                        | Enigma DSL                                | Notes                                                                                                                |
| ------------------------------------------- | --------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Target string `"metal"`                     | yes                                     | implicit (only Metal target)              | We don't expose a multi-target string; Enigma is Metal-only by design.                                              |
| MSL emission on Linux (CI)                  | yes (#1857)                             | partial (Python emitter is host-agnostic) | Our `xcrun metal` step only runs on macOS; we should split *emit* (any host) from *compile* (mac).                  |
| `torch.mps.compile_shader` integration      | yes (#799)                              | no                                        | Big interop win: tilelang can be embedded inside torch MPS programs without copies. We embed via MLX only.          |
| Shared MPS command buffer (tvm-ffi)         | yes (#1289)                             | no                                        | Required for Triton-like fusion with PyTorch.                                                                       |
| Simdgroup 8x8 MMA codegen                   | open (#1869, #2130)                     | partial — intrinsics exist, no tile op    | We expose `simdgroup_multiply_accumulate` ops but no `T.gemm`-style tile op that lowers to them automatically.       |
| Scalar `T.gemm` fallback                    | open (#2118)                            | no                                        | A correctness-first fallback would let attention kernels run before we ship MMA tiling.                              |
| Vectorized 128-bit copies                   | yes (`coalesced_width=k` → `__packed_float4`) | partial (manual `make_float4`)        | Auto-vectorising `tensor.load()/store()` based on alignment + stride is a low-hanging fruit.                         |
| `T.Pipelined` multistage shared buffers     | yes (no real async copy on Metal yet)   | no (only `pipeline` scope, stubs)         | Even unrolled multistage helps; full async needs `metal::simdgroup_async_copy`.                                      |
| Layout algebra (CuTe style)                 | TVM `Layout`, simpler                   | full CuTe TV layouts                      | Enigma is *richer* here — this is one of our differentiators.                                                        |
| Autotuner / Carver                          | yes (CUDA), Metal stub `carver/arch/metal.py` | no                                  | Long-term task.                                                                                                      |
| MLIR-native pipeline                        | no (TVM TIR → MSL string)               | partial (`Enigma-Dialect/` separate)      | We could integrate the dialect into the Python tracer; tilelang has chosen *not* to do MLIR.                         |
| Backend registry / multi-backend Python     | open (#2110)                            | no                                        | Enigma is Metal-only; a registry is overkill until we add CPU/WebGPU.                                                |
| `bf16` / `fp8` / `fp4` storage              | no (probes only)                        | bf16 type exists (typing), no packing      | Metal HW gap; both are limited.                                                                                      |
| Packed-uint8 quantization probes            | yes (#2130)                             | no                                        | Useful for low-bit kernels (dequant GEMM).                                                                           |
| `requires_metal` test marker                | yes                                     | no                                        | Trivial port.                                                                                                        |
| MPS benchmark utilities                     | yes (#1547)                             | partial (`benchmark_*.py` scripts)         | Standardise into `enigma.benchmark` module.                                                                          |

---

## 4. Recommended Changes for Enigma DSL

Below is a concrete, prioritised checklist. Each item is sized so it can be
landed in 1-3 PRs.

### 4.1 Tier 1 — High-impact, low-risk (do first)

- [ ] **Add `requires_metal` testing helper.**
  Mirror `tilelang.testing.requires_metal`: skip the test if
  `mlx.core.metal.is_available()` (or our own probe) returns False. Today
  every test silently assumes Apple Silicon; this blocks CI on Linux.

- [ ] **Split MSL *emit* from *compile*.**
  Make `enigma.compile(kernel)` always produce the MSL string regardless of
  host OS, and only invoke `xcrun metal` / `metallib` if `sys.platform ==
  "darwin"`. This is exactly what tilelang #1857 did and immediately gives us
  Linux CI for the compiler.

- [ ] **PyTorch MPS adapter (Triton-style).**
  Add `enigma/runtime_dispatch/torch_mps.py` that wraps a compiled kernel
  source with `torch.mps.compile_shader` and dispatches torch tensors
  directly (no copy, unified memory). Implementation literally mirrors
  `tilelang/jit/adapter/torch/metal.py` (~50 lines). Lets users do:
  ```python
  out = enigma.compile(my_kernel).torch_launch(a, b, threads=(...), groups=(...))
  ```

- [ ] **Shared MPS command buffer interop.**
  Optional `command_buffer=...` parameter on `MetalRuntime.execute(...)` so
  the kernel can be appended to an existing torch MPS command buffer
  (`torch.mps.current_command_buffer()` if exposed, otherwise via the
  `torch._C._mps_*` interfaces tilelang #1289 uses). Avoids sync stalls when
  composing with torch ops.

- [ ] **`coalesced_width` argument on `Tensor.load/store`.**
  Auto-emit `device __packed_float4*` reinterpret + 128-bit ld/st when
  `coalesced_width=4` and alignment permits. Currently users handcraft this
  via `make_float4`. This already exists on tilelang (`T.copy(..., coalesced_width=2|4)`).

- [ ] **Standardise benchmark utilities.**
  Move the patterns in `examples/benchmark_*.py` into `enigma.benchmark`
  with a `bench(fn, repeat=…, warmup=…, gpu_timestamps=True)` helper.
  Mirrors tilelang #1547.

### 4.2 Tier 2 — Medium-effort feature parity

- [ ] **`enigma.gemm(A_s, B_s, C_l)` tile op.**
  A small tile operator that, given two shared-memory tiles and a fragment
  accumulator, expands to a chain of `simdgroup_load → simdgroup_multiply_accumulate
  → simdgroup_store` over an 8x8 grid, with multi-warp partitioning chosen
  from the layout algebra. This is the Enigma equivalent of tilelang #1869's
  `T.gemm` lowering, but built on top of our existing intrinsics — we don't
  need a TVM fork.

  Concretely:
  - New module `enigma/ops/gemm.py` with `gemm(A_s, B_s, C_l, transpose_A=False, transpose_B=False)`.
  - Layout-algebra-driven warp partition: `make_layout_tv` already gives us
    the per-warp tile coordinates; reuse it instead of inventing a new
    partitioner.
  - Fall back to scalar accumulator when shapes are not multiples of 8x8 (this
    is what tilelang #2118 ships).

- [ ] **Scalar GEMM fallback (correctness-first).**
  Even before MMA, ensure that any attention / norm kernel using the new
  `enigma.gemm` op compiles and runs correctly via a triple `for_range`
  scalar path. Mirrors tilelang #2118 exactly. Lets us claim "any kernel
  written today still runs on Metal even if MMA path is incomplete."

- [ ] **Multistage pipelined copy.**
  Implement a real `enigma.pipeline(num_stages=k)` that allocates `k`
  shadow shared buffers and rotates them across iterations. Even without
  Metal `simdgroup_async_copy`, double-buffering reduces register pressure
  and improves overlap on Apple GPUs.

- [ ] **Async copy via `metal::simdgroup_async_copy`.**
  We already expose `async_copy_to_threadgroup`/`async_copy_commit`/`async_copy_wait`
  in `_tracing.py` but the emitter currently lowers them to plain copies.
  Wire them to `simdgroup_async_copy` + `simdgroup_async_copy_wait_group(0)` and
  validate against MPSGraph reference.

- [ ] **Packed-uint8 / int4 quant probes.**
  Add `enigma.pack_uint8`, `enigma.unpack_int4`, etc. Used for dequant GEMM
  (BitNet-style) and is what tilelang #2130 uses to work around Apple's
  lack of native fp8/fp4 storage.

### 4.3 Tier 3 — Architectural / longer-term

- [ ] **Wire `Enigma-Dialect/` into the Python tracer.**
  Today the MLIR project lives in isolation. Add a `--mlir` flag to
  `enigma.compile` that, instead of emitting MSL strings directly, emits
  `enigma` dialect IR and runs `enigma-translate --enigma-to-msl`. Two
  benefits:
  1. We can write peephole passes (constant folding, layout propagation,
     MMA fusion) as MLIR passes instead of inside Python.
  2. The IR becomes inspectable / testable with lit + FileCheck, which
     scales much better than diff-asserting Metal source.

- [ ] **MPSGraph fallback for `enigma.gemm`.**
  When the user's kernel is a pure GEMM (no fused epilogue), short-circuit
  to `MPSMatrixMultiplication` from MPS. This is what most production
  Metal stacks do (PyTorch's default MPS path) and is 2-3x faster than
  hand-rolled Metal at large sizes per tilelang #799's own numbers.

- [ ] **Carver/autotuner stub.**
  Port `tilelang/carver/arch/metal.py` shape: a thin `MetalArch` class that
  carries `simd_size=32`, `max_threads_per_threadgroup`, `shared_mem_per_block`,
  used later by an autotuner search over block / warp partitions.

- [ ] **Backend registry (only if we add CPU / WebGPU).**
  Mirror tilelang #2110's pattern. Defer until there's a second backend.

- [ ] **bf16 storage support.**
  Apple GPUs *do* support bf16 in MSL since macOS 14 / M-series; tilelang
  hasn't exposed this yet. We could lead here: emit `bfloat` types, gate
  with a function-constant check, fall back to `half` on older devices.

### 4.4 Smaller polish items

- [ ] Consistent target string. Today `enigma.compile` doesn't take one.
  Adding `target="metal"` (and erroring on anything else) makes the API
  shape match tilelang and primes us for adding more backends later.
- [ ] Document the IR ops in `docs/ir.md` similar to
  `Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaOps.td` so the
  Python and MLIR projects share a vocabulary.
- [ ] Add `.kernel_source` accessor on the compiled kernel object — tilelang
  exposes `jit_kernel.kernel_source`, very useful for debugging / blogging.
- [ ] Rename `enigma/compiler/mlir_emitter.py` since it currently emits
  Metal C++, not MLIR. (`metal_emitter.py` is honest.)

---

## 5. Quick Mapping Cheatsheet (TileLang ↔ Enigma)

| TileLang construct                         | Enigma equivalent                                                  | Notes                                                |
| ------------------------------------------ | ------------------------------------------------------------------ | ---------------------------------------------------- |
| `@tilelang.jit(target="metal")`            | `@enigma.kernel` + `enigma.compile`                                | We don't have target dispatch yet.                   |
| `T.Kernel(grid_x, grid_y, threads=N)`      | `kernel(...).launch(grid=(…), block=(…))` inside `@enigma.jit`      | Same intent, different surface.                      |
| `T.alloc_shared((M,N), dtype)`             | `enigma.threadgroup_alloc((M,N), dtype)`                           |                                                      |
| `T.alloc_fragment((M,N), dtype)`           | `enigma.register_tensor((M,N), dtype)` / `RegisterTensor`           |                                                      |
| `T.copy(A_g, A_s, coalesced_width=4)`      | manual `make_float4` + cast pointer (today)                         | **add `coalesced_width` sugar**                      |
| `T.gemm(A_s, B_s, C_l)`                    | manual `simdgroup_matrix_load/store/multiply_accumulate`           | **add `enigma.gemm` tile op**                        |
| `T.Pipelined(num_stages=k)`                | `enigma.pipeline(...)` (stub)                                      | **needs ring buffer codegen**                        |
| `T.Parallel(M, N) {...}`                   | `for_range` + thread index decomposition                            |                                                      |
| `T.use_swizzle(panel_size=10)`             | `enigma.Swizzle(...)` in layout algebra                             | We're stronger here.                                 |
| `simdgroup_*` (CUDA/HIP)                   | `enigma.simd_*`                                                    |                                                      |
| `tilelang.testing.requires_metal`          | (none)                                                              | **add `enigma.testing.requires_metal`**              |
| `kernel_source` attribute                  | `CompiledKernel.metal_source`                                       | We have it; expose / document.                       |
| `torch.mps.compile_shader` adapter          | (none)                                                              | **add MPS torch adapter**                            |

---

## 6. References (verified live PRs)

- #799 [Backend] Add metal backend — https://github.com/tile-ai/tilelang/pull/799 (merged)
- #1289 [tvm-ffi] Enable tvm-ffi for metal backend — https://github.com/tile-ai/tilelang/pull/1289 (merged)
- #1547 improve benchmark on mps — https://github.com/tile-ai/tilelang/pull/1547 (merged)
- #1857 [Codegen] Metal codegen on Linux — https://github.com/tile-ai/tilelang/pull/1857 (merged)
- #1869 [Metal] Add Metal GEMM support with simdgroup_matrix MMA — https://github.com/tile-ai/tilelang/pull/1869 (open)
- #2118 Add Metal scalar fallback for T.gemm — https://github.com/tile-ai/tilelang/pull/2118 (open)
- #2130 Rebase Metal simdgroup GEMM support and runtime coverage — https://github.com/tile-ai/tilelang/pull/2130 (open)
- #2110 Refactor TileLang backend ownership and registry dispatch — https://github.com/tile-ai/tilelang/pull/2110 (open)
- #2114 [Refactor][Build] Separate CMakeLists into different backends — https://github.com/tile-ai/tilelang/pull/2114 (merged)

Search query used: `repo:tile-ai/tilelang is:pr metal` (306 results) and
`repo:tile-ai/tilelang is:pr apple` (21 results).
