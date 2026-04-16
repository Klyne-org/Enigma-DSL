# Dialect & Runtime Improvements Required

Status after systematic testing of every DSL surface against the
`enigma-dialect 0.1.0` wheel (commit `44b4f1c`, April 2026).

---

## Bugs (must fix)

### Bug 1 — Simdgroup matrix type declaration

**Component**: Dialect MSL emitter (`lib/Target/MSL/`)

The emitter does not map `vector<8x8xf32>` to `simdgroup_float8x8` when
declaring local variables. The function calls themselves emit correctly.

```cpp
// Current output (broken):
/* unsupported matrix dims 8x8 */ float v4;
simdgroup_load(v4, v0, v3);
float v5 = 0.0;
/* unsupported matrix dims 8x8 */ float v6 = make_filled_simdgroup_matrix<float, 8, 8>(v5);
/* unsupported matrix dims 8x8 */ float v7;
simdgroup_multiply_accumulate(v7, v4, v4, v6);
simdgroup_store(v7, v1, v3);

// Expected output:
simdgroup_float8x8 v4;
simdgroup_load(v4, v0, v3);
float v5 = 0.0;
simdgroup_float8x8 v6 = make_filled_simdgroup_matrix<float, 8, 8>(v5);
simdgroup_float8x8 v7;
simdgroup_multiply_accumulate(v7, v4, v4, v6);
simdgroup_store(v7, v1, v3);
```

**Fix location**: The function that converts an MLIR type to an MSL type
string. When the type is `vector<RxCxT>` where R and C are both > 1, emit
`simdgroup_T{R}x{C}` (e.g., `simdgroup_float8x8`, `simdgroup_half8x8`).

**Reproducer**:

```python
@enigma.kernel
def simd_gemm(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    a = enigma.simdgroup_matrix_load(A, 8)
    b = enigma.simdgroup_matrix_load(B, 8)
    zero = enigma.metal_cast(0, "float")
    c = enigma.make_filled_simdgroup_matrix(zero)
    r = enigma.simdgroup_multiply_accumulate(a, b, c)
    enigma.simdgroup_matrix_store(r, C, 8)

enigma.compile(simd_gemm)  # xcrun metal fails
```

**Error**: `xcrun metal` rejects the `.metal` because a `float` variable
is passed to `simdgroup_load` which expects `simdgroup_float8x8`.

**Blocks**: All simdgroup matrix operations end-to-end (GEMM, matmul via
hardware matrix units).

---

### Bug 2 — Threadgroup atomics emit `device` address space

**Component**: Dialect MSL emitter (`lib/Target/MSL/`)

When emitting atomic operations on a `threadgroup`-address-space memref,
the emitter always casts the pointer to `device atomic_int*`. Metal requires
`threadgroup atomic_int*` for threadgroup buffers.

```cpp
// Current output (broken):
atomic_fetch_add_explicit((device atomic_int*)&shared[idx], val, memory_order_relaxed);

// Expected output:
atomic_fetch_add_explicit((threadgroup atomic_int*)&shared[idx], val, memory_order_relaxed);
```

**Fix location**: The atomic emission functions in `MSLEmitterCore.cpp` (or
wherever the `(device atomic_T*)` cast is constructed). Check the memref's
memory space attribute — if memory space is 2 (threadgroup), emit
`threadgroup atomic_T*` instead of `device atomic_T*`.

**Reproducer**:

```python
@enigma.kernel
def tg_atomic(A: enigma.u32, Out: enigma.u32):
    tid = enigma.thread_position_in_grid
    shared = enigma.threadgroup_alloc("uint", 1)
    shared[enigma.metal_cast(0, "uint")] = enigma.metal_cast(0, "uint")
    enigma.barrier()
    _ = shared.atomic_fetch_add(0, enigma.metal_cast(1, "uint"))
    enigma.barrier()
    Out[tid] = shared[enigma.metal_cast(0, "uint")]

enigma.compile(tg_atomic)  # xcrun metal fails
```

**Error**: 10 instances of `"C-style cast from 'threadgroup int *' to
'device metal::atomic_int *' converts between mismatching address spaces"`.

**Affects**: Every atomic op on threadgroup memory — `atomic_load`,
`atomic_store`, `atomic_exchange`, `atomic_fetch_add`, `atomic_fetch_sub`,
`atomic_fetch_min`, `atomic_fetch_max`, `atomic_fetch_and`,
`atomic_fetch_or`, `atomic_fetch_xor`, `atomic_compare_exchange_weak`.

**Blocks**: Local reductions, histogram building, spinlocks, and any
pattern that uses atomics on shared memory.

---

### Bug 3 — `function_constant` runtime dispatch crashes

**Component**: Swift runtime dylib + `MetalRuntime` Python API

The dialect's MSL emission for `function_constant` is now correct (the new
wheel hoists the declaration to file scope). However, Metal requires
specialization constants to be set via `MTLFunctionConstantValues` before
creating the pipeline state.

```
validateWithDevice:1437: failed assertion
  'function fc_run cannot be used to build a pipeline state.
   Use newFunctionWithName:constantValues:... to get the specialized function'
```

**Fix location**: Two changes needed:

1. **Swift runtime** (`libenigma_runtime.swift`): Add a new C-exported
   function `enigma_create_pipeline_with_constants` that accepts a
   serialized list of `(index, type, value)` triples, builds an
   `MTLFunctionConstantValues` object, and calls
   `newFunction(name:constantValues:)` followed by
   `makeComputePipelineState(function:)`.

2. **Python runtime** (`enigma/runtime_dispatch/runtime.py`): Extend
   `MetalRuntime.execute()` and `MetalRuntime.prepare()` to accept an
   optional `constants: dict[int, (str, Any)]` parameter, serialize it,
   and call the new Swift function.

**Reproducer**:

```python
@enigma.kernel
def fc_run(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    alpha = enigma.function_constant("float", 0)
    B[tid] = A[tid] * alpha

compiled = enigma.compile(fc_run)    # OK — MSL is correct
rt = enigma.MetalRuntime()
rt.execute(compiled, [A], N * 4, ...)  # ABORT — Metal validation trap
```

**Blocks**: Any kernel using `enigma.function_constant()` at runtime.

---

### Bug 4 — No validation when TV tile exceeds tensor dimensions

**Component**: DSL (`enigma/tensor.py` or `enigma/core.py`)

When the tiler shape is larger than the tensor in any dimension, the grid
computes to 0 in that axis. The kernel dispatches zero threads and silently
returns a zeroed output buffer with no error.

**Reproducer**:

```python
thr = enigma.make_ordered_layout((4, 64), order=(1, 0))
val = enigma.make_ordered_layout((4, 4), order=(1, 0))
tiler_mn, tv = enigma.make_layout_tv(thr, val)  # tiler = (16, 256)

# N=64 < tiler[1]=256 — tile doesn't fit
mA = Tensor("A", 0, "float", enigma.Layout((16, 64), (64, 1)))
compiled = enigma.compile(launch, mA, mB, mC)
# compiled.grid = (0, 1, 1) — zero work, silent wrong result
```

**Fix**: In `tensor_zipped_divide` or `enigma.compile`, check that
`product(tensor.shape[i]) >= product(tiler[i])` for all modes. Raise
`EnigmaError` with a clear message if violated.

---

## Missing features (by priority)

### Priority 1 — Control flow (`if` / `for` / `while`)

**Why critical**: Without loops, the DSL can only express straight-line
kernels. Every kernel that iterates over a dimension (matmul inner loop,
attention score accumulation, prefix scan across tiles, any reduction
larger than a SIMD group) must be manually unrolled at Python trace time
with a fixed trip count. This is the single biggest limitation.

**Dialect work**:

1. Link the `scf` dialect in the wheel:
   ```cmake
   # python/CMakeLists.txt
   MLIRCAPISCF
   ```
   ```cpp
   // EnigmaModule.cpp
   #include "mlir-c/Dialect/SCF.h"
   mlirDialectHandleRegisterDialect(mlirGetDialectHandle__scf__(), context);
   ```

2. The MSL emitter already handles `scf.if` / `scf.for` per
   `MSLEmitterControlFlow.cpp`. Verify it works with the new bindings.

**DSL work**:

Add context-manager APIs to `_tracing.py`:

```python
# for loop
with enigma.for_range(0, K, step=1) as i:
    acc = acc + A[row * K + i] * B[i * N + col]

# conditional
with enigma.if_(condition) as (then_block, else_block):
    with then_block:
        Out[tid] = a
    with else_block:
        Out[tid] = b
```

This requires restructuring `KernelBuilder` to support nested op regions
instead of a flat list.

**Estimated effort**: 3-5 days (dialect linking: half day, DSL tracer
rewrite: 2-3 days, tests: 1 day).

---

### Priority 2 — Textures

**Dialect status**: `texture_read`, `texture_write`, `texture_sample`,
`texture_get_width`, `texture_get_height` ops and MSL emission all exist.
Python bindings do not expose the texture MLIR type.

**DSL work**:

1. Add `enigma.Texture2D(dtype, access)` annotation type for kernel params.
2. In `_tracing.py`, create `TracingTexture` with `.read(uv)`,
   `.write(val, uv)`, `.sample(uv)` methods.
3. In `mlir_emitter.py`, map the param type to `!enigma.texture2d<T, access>`.

**Estimated effort**: 1-2 days (pure DSL, no dialect changes).

---

### Priority 3 — Relational ops (`all` / `any` / vector `select`)

**Dialect status**: `EnigmaRelationalOps.td` defines the ops.

**Blocked by**: Nothing now — `vec_make` / `vec_extract` landed, so vector
operands can be constructed. Just needs MLIR emitter dispatch and DSL
surface wiring.

**Estimated effort**: Half a day.

---

### Priority 4 — Vertex / fragment shaders

**Dialect status**: `vertex_return`, `fragment_return`, `vertex_id`,
`instance_id` all exist with MSL emission.

**DSL work**: Add `@enigma.vertex_kernel` / `@enigma.fragment_kernel`
decorators. Low priority — GPGPU-first DSL.

**Estimated effort**: 1 day.

---

## What works (verified end-to-end on GPU)

All of the following compile to valid MSL, dispatch on the GPU, and produce
numerically correct results:

| Category | Features tested |
|---|---|
| **Data types** | f32, f16, bf16, i8, u8, i16, u16, i32, u32, i64, u64 |
| **Arithmetic** | `+`, `-`, `*`, `/`, `//`, `%`, unary `-` (float and int) |
| **Unary float math** | sqrt, rsqrt, abs, ceil, floor, round, trunc, sign, saturate, fract, exp, exp2, log, log2, log10, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh |
| **Binary float math** | fmin, fmax, pow, fmod, atan2, step, copysign |
| **Ternary float math** | clamp, fma, mix, smoothstep |
| **Float predicates** | isnan, isinf, isfinite, signbit, isnormal |
| **Integer math** | imin, imax, iclamp, abs_diff, add_sat, sub_sat, mul_hi, rotate, mad_sat |
| **Bit ops** | popcount, clz, ctz, reverse_bits, extract_bits, insert_bits |
| **Comparisons** | cmp_eq/ne/lt/le/gt/ge (signed), cmp_ult/ule/ugt/uge (unsigned) |
| **Select** | `enigma.where(false_val, true_val, cond)` with comparison chains |
| **Casting** | metal_cast (across all type pairs), as_type (bitwise reinterpret) |
| **Vectors** | make_float2/3/4, make_vec, vec_extract, `.x`/`.y`/`.z`/`.w`, vec4 arithmetic (`+`, `*`) |
| **Geometry** | dot, length, distance, cross, normalize, reflect, refract, faceforward |
| **Pack/Unpack** | pack_float_to_unorm4x8, pack_float_to_snorm4x8, unpack round-trips (all 12 ops compile) |
| **SIMD group** | simd_sum, simd_product, simd_min, simd_max, simd_and/or/xor, simd_prefix_exclusive/inclusive_sum/product, simd_shuffle/shuffle_up/shuffle_down/shuffle_xor, simd_broadcast |
| **Quad group** | quad_sum/product/min/max/and/or/xor, quad_prefix_exclusive/inclusive_sum, quad_shuffle/shuffle_up/shuffle_down/shuffle_xor, quad_broadcast |
| **Barriers** | threadgroup_barrier (all mem_flags), simdgroup_barrier |
| **Shared memory** | threadgroup_alloc (single and multiple allocations), load/store on threadgroup buffers |
| **Device atomics** | atomic_load, atomic_store, atomic_exchange, atomic_fetch_add/sub/min/max/and/or/xor, atomic_compare_exchange_weak |
| **Grid queries** | All 12 query ops with x/y/z dimensions, 1D/2D/3D grid dispatch |
| **arch namespace** | arch.thread_idx(), arch.block_idx(), arch.block_dim() |
| **vec_width** | vec_width=2 and vec_width=4 buffer promotion |
| **TV layout** | tv_load/tv_add/tv_store per-element lowering, tensor_zipped_divide, tensor_composition, make_layout_tv |
| **function_constant** | MSL emission correct (hoisted to file scope); runtime dispatch needs API extension |
| **Simdgroup matrix** | MLIR emission correct (traces to `vector<8x8xf32>` ops); MSL type declaration needs dialect fix |
| **Compilation** | dump_ir, dump_mlir, keep_metal_source, export_metal, work_dir |
| **Stress tests** | Deep expression chains (14+ ops), 8-buffer kernels, large kernels (165 MSL lines / 50+ traced ops), Python-level unrolled loops |

---

## Roadmap — what the DSL needs to become a real CuTe-for-Metal

The features above fix what's broken. This section covers what's **missing**
— the gap between "working toy DSL" and "a system you can write a
production GEMM or FlashAttention in." Ordered by dependency chain: each
item unlocks the ones below it.

### R1 — Control flow (`for` / `if` / `while`)

**The single blocker that makes everything else academic.**

Without loops, every kernel is a fixed-size straight-line program. You
cannot write:
- A matmul for arbitrary K (the inner accumulation loop)
- A reduction over more than 32 elements (one SIMD group)
- Attention for variable sequence length
- Any tiled algorithm that iterates over tiles
- A prefix scan across threadgroups

Today, the only workaround is Python-level unrolling at trace time with a
hardcoded trip count. This means separate compilations for every problem
size, and code that explodes in length (the showcase attention kernel
manually unrolls 8 chunks of float4 dot — 32 multiply-accumulates written
line by line, and it still only handles K=32).

CuTe itself doesn't provide control flow (it relies on CUDA `for`/`if`),
but Enigma traces Python — so it must capture Python control flow. The
context-manager approach fits a tracing DSL:

```python
with enigma.for_range(0, K, step=1) as i:
    acc = acc + A[row * K + i] * B[i * N + col]

with enigma.if_(cond) as (then_b, else_b):
    with then_b:
        Out[tid] = a
    with else_b:
        Out[tid] = b
```

**Requires**: `scf` dialect linked in the wheel, `KernelBuilder`
restructured to support nested op regions instead of a flat list.

**Effort**: 3-5 days. **Unlocks**: everything below.

---

### R2 — Scalar kernel arguments

Every parameter today is a `device T*` buffer. There is no way to pass a
scalar like `N`, `K`, `alpha`, or `epsilon` as a kernel argument. Users
either hardcode constants (not general) or waste a 1-element buffer
(ugly and slow — burns a buffer binding slot + allocation).

Metal supports scalar arguments directly:

```metal
kernel void gemm(device float* A [[buffer(0)]],
                 constant uint& N [[buffer(3)]],   // scalar
                 constant float& alpha [[buffer(4)]]) { ... }
```

**DSL surface**:

```python
@enigma.kernel
def gemm(A: enigma.f32, B: enigma.f32, C: enigma.f32,
         M: enigma.Scalar(enigma.u32),
         K: enigma.Scalar(enigma.u32),
         alpha: enigma.Scalar(enigma.f32)):
    ...
```

**Requires**: New annotation type, tracing support, MLIR emission
(`constant T& [[buffer(N)]]`), runtime API to pass scalar values.

**Effort**: 1-2 days. **Unlocks**: general-purpose kernels without
recompilation per problem size.

---

### R3 — Tiled copy primitive (`enigma.copy`)

TV layout gets data from global memory to per-thread register values, but
there is no first-class "copy a tile from buffer A into shared memory
buffer S" primitive. CuTe's `cute::copy(src_tensor, dst_tensor)` handles
the TV-layout-to-memref mapping, optional vectorization, and boundary
predication in one call.

```python
# Copy a tile of A from device to shared, respecting TV layout
enigma.copy(src=global_tile_A, dst=shared_tile_A)
enigma.barrier()

# Copy from shared to per-thread registers
enigma.copy(src=shared_tile_A, dst=reg_A)
```

**Requires**: Control flow (R1) for the loop that iterates over tiles,
plus a `copy` op that lowers to the appropriate load/store sequence.

**Effort**: 2-3 days after R1. **Unlocks**: clean tiled algorithms.

---

### R4 — Register-level tensor abstraction

CuTe has register-backed tensors (the rmem level). Enigma has device
buffers and threadgroup buffers but no concept of "this tensor lives in
registers." For GEMM, the pattern is:

1. Load A tile to shared
2. Load B tile to shared
3. Each thread accumulates into a register-resident C fragment
4. Store C fragment to global

The register fragment is just local variables, but the DSL needs a way
to declare a small fixed-size tensor that lowers to locals and supports
the same `.load()` / `.store()` / slicing API as `Tensor`:

```python
acc = enigma.register_tensor(shape=(4, 4), dtype="float", fill=0.0)
# ... inside the K loop:
acc[vi, vj] = enigma.fma(a_reg[vi], b_reg[vj], acc[vi, vj])
```

**Requires**: A new `RegisterTensor` class in `_tracing.py` that lowers
to local variable declarations and scalar load/store.

**Effort**: 2 days. **Unlocks**: register tiling for GEMM/attention.

---

### R5 — Predicated loads/stores for boundary tiles

When tensor dimensions are not divisible by the tile size, the boundary
tile has fewer valid elements. CuTe handles this with predication — "load
only if this coordinate is in-bounds, else return 0."

Without this, every dimension must be a multiple of the tile size, which
is unacceptable for production use.

**Two possible approaches**:

1. **Masked load/store** (simpler, works without `if`):
   ```python
   val = enigma.load_if(buf, idx, mask=in_bounds, default=0.0)
   enigma.store_if(buf, idx, val, mask=in_bounds)
   ```

2. **Control flow** (more general, needs R1):
   ```python
   with enigma.if_(in_bounds):
       val = buf[idx]
   ```

**Requires**: Either a new `select`-based load op in the dialect, or
control flow (R1).

**Effort**: 1 day for masked load, or free with R1. **Unlocks**: arbitrary
problem sizes without padding.

---

### R6 — Async copy (`simdgroup_async_copy`)

Metal 3.1+ (Apple Silicon M3/A17+) has
`simdgroup_async_copy_from_device_to_threadgroup` for overlapping memory
transfers with compute. Earlier hardware (M1/M2/A15/A16) does not support
this — on those devices, shared memory loads are always synchronous.

The DSL should gate this behind a capability check or a user opt-in so
kernels remain portable across Apple Silicon generations:

```python
# Explicit opt-in for Metal 3.1+ features
enigma.async_copy(src=device_ptr, dst=shared_ptr, count=tile_size)
enigma.commit_async_copy()
# ... compute on previous tile ...
enigma.wait_async_copy()
```

**Requires**: New dialect ops (`enigma.async_copy_to_threadgroup`,
`enigma.async_copy_commit`, `enigma.async_copy_wait`) + MSL emission +
Metal GPU family capability gating.

**Effort**: 1-2 days (dialect + DSL). **Unlocks**: double-buffered
pipelines on M3+ hardware.

---

### R7 — Pipeline / double-buffering abstraction

The classic tiled GEMM ping-pongs between two shared memory buffers:
while computing on tile k from buffer 0, prefetch tile k+1 into buffer 1.
Next iteration, swap.

On M1/M2 (no async copy), double-buffering still helps by structuring
the load-compute overlap at the threadgroup level via barriers. On M3+
with async copy (R6), it becomes a proper async pipeline.

```python
with enigma.pipeline(stages=2) as pipe:
    pipe.prefetch(src=global_A[k+1], dst=shared_A[pipe.next_stage])
    # compute on shared_A[pipe.current_stage]
    pipe.advance()
```

**Requires**: Control flow (R1), and optionally async copy (R6) for M3+.
Synchronous double-buffering (M1/M2) only needs R1 + barriers.

**Effort**: 2-3 days after R1. **Unlocks**: production-grade bandwidth
utilization.

---

### R8 — Layout swizzling for bank-conflict avoidance

CuTe has `Swizzle<B, M, S>` to XOR-remap shared memory addresses and
eliminate bank conflicts. Apple Silicon threadgroup memory has 32 banks
with 4-byte granularity — the same conflict patterns as CUDA shared
memory apply. A float4x4 tile loaded column-major will serialize across
banks without swizzling.

```python
swizzled = enigma.swizzle(layout, bits=3, base=3, shift=0)
# Composes with existing Layout: shared_layout = composition(base_layout, swizzled)
```

**Requires**: A `Swizzle` class in `enigma/core.py` that composes with
`Layout` and modifies the offset calculation with an XOR.

**Effort**: 1-2 days (pure Python layout algebra). **Unlocks**: optimal
shared memory throughput for tiled GEMM on Apple Silicon.

---

### R9 — Multiple output buffers

`MetalRuntime.execute()` assumes the last buffer is the single output.
Real kernels often write to multiple outputs:

- Softmax backward: writes both `dX` and `dscale`
- Fused attention: writes both `O` and `L` (log-sum-exp)
- LayerNorm: writes `Y`, `mean`, and `rstd`

**Requires**: Extend `execute()` to accept `output_sizes: list[int]`
or `output_indices: list[int]` and return multiple byte buffers.

**Effort**: Half a day (Swift runtime + Python API). **Unlocks**: fused
multi-output kernels.

---

### R10 — Metal GPU family capability queries

Apple Silicon has significant feature variation across generations:

| Feature | M1/A14 | M2/A15 | M3/A17 | M4 |
|---|---|---|---|---|
| Simdgroup matrix (8x8) | Yes | Yes | Yes | Yes |
| Async copy to threadgroup | No | No | Yes | Yes |
| Simdgroup size | 32 | 32 | 32 | 32 |
| Max threadgroup memory | 32 KB | 32 KB | 32 KB | 32 KB |
| Max threads per threadgroup | 1024 | 1024 | 1024 | 1024 |

The DSL should expose device capabilities so kernels can be tuned
without rewriting:

```python
rt = enigma.MetalRuntime()
caps = rt.device_capabilities()
# caps.gpu_family -> "apple8" / "apple9" / ...
# caps.supports_async_copy -> bool
# caps.max_threadgroup_memory -> int
# caps.simdgroup_size -> int
```

And at compile time, the DSL could select code paths based on these
(once control flow lands).

**Requires**: Query `MTLDevice.supportsFamily()` in the Swift runtime,
expose via ctypes.

**Effort**: Half a day. **Unlocks**: portable kernels across Apple
Silicon generations.

---

### Dependency chain for production GEMM on Apple Silicon

```
R1 (control flow)                          ← FOUNDATION
 ├── R2 (scalar args) ──── general kernel launch without recompilation
 ├── R3 (tiled copy)
 │    └── R4 (register tensors) ── per-thread accumulator tiles
 │         └── R8 (swizzle) ── bank-conflict-free shared memory
 ├── R5 (predicated loads) ── arbitrary M/N/K without padding
 └── R7 (double buffering) ── overlap load/compute via barriers
      └── R6 (async copy, M3+ only) ── hardware async pipeline
```

**Minimum viable tiled GEMM** (works on all Apple Silicon):
R1 + R2 + R4 + R5 (~8 days).

**Production-grade with optimal bandwidth** (M1/M2):
add R7 + R8 (~4 more days).

**Peak performance on M3+**:
add R6 (~2 more days).

Total: ~2 weeks from current state for a complete, portable, tiled GEMM
that runs across all Apple Silicon generations.

---

## Suggested fix order

1. **Bug 2** (threadgroup atomics address space) — smallest fix, highest
   impact. One-line change in the atomic emission path: check memref memory
   space, emit `threadgroup` instead of `device`. Unblocks shared-memory
   reductions.

2. **Bug 1** (simdgroup matrix type) — small fix in the type-to-string
   function. Unblocks hardware GEMM.

3. **Bug 3** (function_constant runtime) — needs Swift + Python changes.
   Unblocks specialization constants.

4. **Bug 4** (TV tile validation) — pure Python fix in the DSL. Prevents
   silent wrong results.

5. **Priority 1** (control flow) — largest effort but transforms the DSL
   from "elementwise-only" to "general-purpose GPU programming".
