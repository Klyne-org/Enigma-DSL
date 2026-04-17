# Dialect & Runtime Improvements Required

Status after systematic testing of every DSL surface against the
`enigma-dialect 0.1.0` wheel (commit `6b6628b`, April 2026).

**Last updated**: 2026-04-17 — dialect wheel upgrade (scf bindings +
Bug 1/2 fix) and DSL-side push for R1 iter_args, R2 scalar args, R3
copy, R4 register tensors, R5 load_if/store_if, R6 async_copy surface
with M3+ gate, R7 pipeline, R9 multi-output, R10 capability queries,
and Bug 3 specialization constants.

---

## Bugs (must fix)

### ~~Bug 1 — Simdgroup matrix type declaration~~ FIXED

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

### ~~Bug 2 — Threadgroup atomics emit `device` address space~~ FIXED

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

### ~~Bug 3 — `function_constant` runtime dispatch crashes~~ FIXED

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

**Status** (2026-04-17): Fixed end-to-end. Swift side adds
`enigma_create_pipeline_with_constants` which builds an
`MTLFunctionConstantValues` from parallel arrays of (index, type_tag,
value) and calls `library.makeFunction(name:constantValues:)`.
Python side `MetalRuntime.execute(..., constants={idx: (type_name,
value), ...})` packs and dispatches. Verified on M4 with a
scale-by-alpha kernel.

**Blocks**: None — live.

---

### ~~Bug 4 — No validation when TV tile exceeds tensor dimensions~~ FIXED

**Component**: DSL (`enigma/tensor.py`)

**Status**: Fixed. `tensor_zipped_divide()` now validates that
`product(tensor.shape[i]) >= product(tiler[i])` for all modes. Raises
`EnigmaError` with a clear message if violated, instead of silently
producing a zero-element grid.

**Fix location**: `enigma/tensor.py:tensor_zipped_divide()`

---

## Missing features (by priority)

### ~~Priority 1 — Control flow (`if` / `for` / `while`)~~ DSL DONE

**Status**: DSL-side implementation complete.

**What was implemented**:

1. **`KernelBuilder` restructured** (`_tracing.py`): Added `_region_stack`
   for nested op regions. `IROp` extended with `regions` field for child
   op lists. Ops are recorded into the top of the stack; control flow
   context managers push/pop regions.

2. **Three context managers** (`_tracing.py`):

   ```python
   # for loop — traces to scf_for with 1 body region
   with enigma.for_range(0, K, step=1) as i:
       acc = acc + A[row * K + i] * B[i * N + col]

   # conditional (if-only) — traces to scf_if with 1 region
   with enigma.if_(condition):
       Out[tid] = a

   # conditional (if/else) — traces to scf_if with 2 regions
   with enigma.if_(condition) as (then_block, else_block):
       with then_block:
           Out[tid] = a
       with else_block:
           Out[tid] = b

   # while loop — traces to scf_while with 2 regions (before/after)
   with enigma.while_(lambda: enigma.cmp_lt(i, n)):
       # body
   ```

3. **MLIR emitter updated** (`mlir_emitter.py`): Op-processing loop
   refactored into recursive `_emit_ops()`. Handlers for `scf_for` →
   `scf.ForOp`, `scf_if` → `scf.IfOp`, `scf_while` → `scf.WhileOp`.
   Graceful error if SCF Python bindings aren't available yet.

4. **Exports**: `enigma.for_range`, `enigma.if_`, `enigma.while_`

5. **Tests**: 15 tests in `tests/test_control_flow.py` covering basic
   tracing, nesting (for+if, for+for, while+if), sequencing, region
   stack balance, and op isolation.

6. **Examples**: `examples/control_flow_test.py` with 6 verified kernels:
   - Array sum (`for_range`)
   - Clamp with nested if/else (`if_`)
   - Sum positive elements (`for_range` + `if_`)
   - Matmul inner loop (`for_range` with 2 loads + multiply-accumulate)
   - Linear search (`while_`)
   - IR tree dump (visualization)

**Dialect work remaining**:

Register `scf` dialect in the Python wheel (`EnigmaModule.cpp`):

```cpp
#include "mlir-c/Dialect/SCF.h"

// Inside register_dialect():
MlirDialectHandle scfHandle = mlirGetDialectHandle__scf__();
mlirDialectHandleRegisterDialect(scfHandle, context);
if (load) {
    mlirDialectHandleLoadDialect(scfHandle, context);
}
```

Note: `scf` is already linked in `InitAll.cpp` and the MSL emitter
already handles `scf.for`/`scf.if` in `MSLEmitterControlFlow.cpp`.
Only the Python binding registration is missing.

After this one change, the DSL-side control flow will work end-to-end:
DSL → traced IR → MLIR (scf ops) → MSL → Metal.

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

### DSL-side only (tracing verified, awaiting dialect for end-to-end)

| Category | Features tested |
|---|---|
| **Control flow** | `for_range` (basic, IRValue bounds, custom dtype), `if_` (if-only, if/else), `while_`, nested (for+if, for+for, while+if), sequential loops, region stack balance |
| **Swizzle** | `Swizzle(bits, base, shift)`, `SwizzledLayout`, `swizzle()` function, self-inverse property, unique offsets, bank-conflict reduction verified |
| **TV tile validation** | `tensor_zipped_divide` raises `EnigmaError` when tiler exceeds tensor dims |

---

## Roadmap — what the DSL needs to become a real CuTe-for-Metal

The features above fix what's broken. This section covers what's **missing**
— the gap between "working toy DSL" and "a system you can write a
production GEMM or FlashAttention in." Ordered by dependency chain: each
item unlocks the ones below it.

### ~~R1 — Control flow (`for` / `if` / `while`)~~ DONE

**Status**: End-to-end. Dialect wheel now registers the `scf` Python
bindings (commit `6b6628b`). `KernelBuilder` threads nested regions
through `IROp.regions`; context managers trace `scf_for`, `scf_if`,
`scf_while`. MLIR emitter builds `scf.ForOp` / `scf.IfOp` /
`scf.WhileOp` recursively.

**Loop-carried values (iter_args)**: `enigma.for_range(lo, hi,
init=[x0, ...])` yields `(i, carry)` where `carry` is a
`Carry` slot list. Writes to `carry[i]` become `scf.yield` operands,
and the op's results are rebound to the carry list on exit —
accumulators now lower correctly.

**Tests**: 15 in `tests/test_control_flow.py` + 5 iter_args/carry
tests in `tests/test_dsl_extensions.py`.

---

### ~~R2 — Scalar kernel arguments~~ DONE

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

**Implemented**: `enigma.Scalar(dtype)` annotation type in
`enigma/typing.py`. Tracer records `(name, buffer_index, metal_dtype)`
in `KernelBuilder.scalar_params` and emits a `load(buffer[0])` at
function entry so the kernel body sees an IRValue. `MetalRuntime.execute`
accepts `scalars=[v, ...]` (one per Scalar param) and packs each as a
1-element numpy buffer at the correct slot.

**Convention**: Scalars must appear BEFORE output buffers in the kernel
signature — outputs are always appended last in the GPU binding layout.

**Verified end-to-end** on M4:

```python
@enigma.kernel
def scale(A: enigma.f32, alpha: enigma.Scalar(enigma.f32), B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = A[tid] * alpha

rt.execute(compiled, [A], output_size=..., scalars=[3.0], ...)
```

---

### ~~R3 — Tiled copy primitive (`enigma.copy`)~~ DONE

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

**Implemented**: `enigma.copy(src, dst, count, src_offset=0,
dst_offset=0, mask_fn=None)` lowers to a `for_range` loop of scalar
load+store. Optional `mask_fn(i) -> bool` integrates with R5 for
boundary predication.

---

### ~~R4 — Register-level tensor abstraction~~ DONE

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

**Implemented**: `enigma.register_tensor(shape, dtype, fill=0.0)`
returns a `RegisterTensor` whose `__getitem__`/`__setitem__` accept
static integer tuples and route through SSA IRValues backed by
per-thread locals. Row-major strides, no device load/store.

---

### ~~R5 — Predicated loads/stores for boundary tiles~~ DONE

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

2. **Control flow** (more general, now available via R1):
   ```python
   with enigma.if_(in_bounds):
       val = buf[idx]
   ```

**Implemented**:

- `enigma.load_if(buf, index, mask, default=0.0)` — unconditional load
  followed by `select(default, val, mask)`. Caller must ensure the load
  is a safe read (e.g. clamp the index before calling).
- `enigma.store_if(buf, index, value, mask)` — wraps the store inside
  an `scf.if` so nothing writes when `mask` is false.

---

### R6 — Async copy (`simdgroup_async_copy`) — DSL SURFACE + M3+ GATE DONE

**Status**: DSL surface implemented; dialect ops (`async_copy_to_threadgroup`,
`async_copy_commit`, `async_copy_wait`) still pending in the dialect
wheel, so emitting these will raise until the dialect ships them.

**M3+ runtime gate**: Each of `enigma.async_copy_to_threadgroup`,
`enigma.async_copy_commit`, `enigma.async_copy_wait` calls
`_require_m3_runtime()` during tracing. That helper queries the active
`MetalRuntime` instance's `device_capabilities()` and invokes
`caps.require_m3(feature)`. On M1/M2 this raises
`RuntimeError: async_copy_* requires Apple GPU family 9 (M3/A17) or
newer`. On M3/M4, tracing proceeds.

**Capability source**: `MetalRuntime._lib.enigma_device_supports_family`
probes `MTLDevice.supportsFamily(MTLGPUFamily)` for codes 1010→1001,
and `DeviceCapabilities.is_m3_or_newer` returns true when the highest
supported family is ≥ 1009 (`MTLGPUFamilyApple9`).

---

### ~~R7 — Pipeline / double-buffering abstraction~~ DONE

**Implemented**: `enigma.pipeline(dtype, size, stages=2)` returns a
`Pipeline` holding two `threadgroup_alloc` buffers and a phase counter.
Methods: `front()`, `back()`, `swap()`. Stages > 2 currently rejected.
Designed for the M1/M2 barrier-driven case — M3+ can layer async_copy
on top of the same object once the dialect ops land.

---

### ~~R8 — Layout swizzling for bank-conflict avoidance~~ DONE

**Status**: Implemented in DSL. Apple Silicon threadgroup memory has 32
banks with 4-byte granularity — the same conflict patterns as CUDA shared
memory. The `Swizzle` and `SwizzledLayout` classes in `enigma/core.py`
implement CuTe-style `Swizzle<B, M, S>` XOR-based address remapping.

```python
# Swizzle a 16x16 float tile to avoid bank conflicts on column access
tile = enigma.Layout((16, 16), (16, 1))
swizzled = enigma.swizzle(tile, bits=3, base=0, shift=4)
offset = swizzled((row, col))  # bank-conflict-free offset
```

Properties verified:
- Self-inverse (XOR): `swizzle(swizzle(x)) == x`
- Unique offsets: no collisions across all coordinates
- Bank distribution: 2 unique banks → 8 unique banks for column access
  on a 16x16 float tile

**Classes**: `Swizzle(bits, base, shift)`, `SwizzledLayout(layout, swizzle)`
**Function**: `swizzle(layout, bits, base, shift) -> SwizzledLayout`
**Exports**: `enigma.Swizzle`, `enigma.SwizzledLayout`, `enigma.swizzle`

---

### ~~R9 — Multiple output buffers~~ DONE

`MetalRuntime.execute()` assumes the last buffer is the single output.
Real kernels often write to multiple outputs:

- Softmax backward: writes both `dX` and `dscale`
- Fused attention: writes both `O` and `L` (log-sum-exp)
- LayerNorm: writes `Y`, `mean`, and `rstd`

**Implemented**: `MetalRuntime.execute(..., output_sizes=[n1, n2, ...])`
returns `list[bytes]` of the same length. The existing `output_size=n`
scalar path is preserved for back-compat and returns a single `bytes`.
Outputs are appended after inputs + scalar buffers in the binding list.

---

### ~~R10 — Metal GPU family capability queries~~ DONE

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
(now possible with control flow).

**Implemented**: Swift side adds
`enigma_device_supports_family`, `enigma_device_name`,
`enigma_device_max_threadgroup_memory`,
`enigma_device_max_threads_per_threadgroup`. Python side exposes
`rt.device_capabilities() -> DeviceCapabilities` with fields
`gpu_family`, `gpu_family_raw`, `supports_async_copy`,
`supports_simdgroup_matrix`, `simdgroup_size`, `max_threadgroup_memory`,
`max_threads_per_threadgroup`, `device_name`, and
`is_m3_or_newer` / `require_m3(feature)` helpers. Used internally to
gate R6 async_copy.

**Known quirk**: On M4, the linked Metal SDK may not export
`MTLGPUFamily.apple10` raw value 1010; the probe then reports
`apple9`. `is_m3_or_newer` still returns `True`.

---

### Dependency chain for production GEMM on Apple Silicon

```
R1 (control flow)     ← DONE (scf bindings shipped 2026-04-17)
 ├── R2 (scalar args)  ← DONE
 ├── R3 (tiled copy)   ← DONE
 │    └── R4 (register tensors) ← DONE
 │         └── R8 (swizzle)      ← DONE
 ├── R5 (predicated loads/stores) ← DONE
 └── R7 (double buffering)        ← DONE
      └── R6 (async copy, M3+ only) ← DSL surface + M3+ gate DONE;
                                       dialect ops pending
```

Everything needed for a minimum-viable tiled GEMM on M1/M2 is now in
place. On M3/M4 the async_copy DSL surface is ready but will error at
emit time until the dialect ships `async_copy_*` ops.

---

## Suggested fix order

All Bug 1-4 and R1-R10 (except R6 dialect ops) are now resolved. The
only remaining dialect-side work is shipping the three async_copy ops
(`async_copy_to_threadgroup`, `async_copy_commit`, `async_copy_wait`) +
their MSL emission. The DSL surface and M3+ runtime gate are already
in place, so the wheel upgrade is a drop-in for M3/M4 devices.

---

## Changes log

| Date | Item | Side | Description |
|---|---|---|---|
| Apr 2026 | Bug 4 | DSL | Fixed: `tensor_zipped_divide` validates tiler fits tensor |
| Apr 2026 | R1 | DSL | Implemented: `for_range`, `if_`, `while_` context managers, `KernelBuilder` region stack, recursive MLIR emitter, 15 tests, 6 example kernels |
| Apr 2026 | R8 | DSL | Implemented: `Swizzle`, `SwizzledLayout`, `swizzle()` for bank-conflict avoidance |
| 2026-04-17 | dialect wheel | Dialect | Upgraded to `6b6628b`: Bug 1 (simdgroup type), Bug 2 (threadgroup atomic addr space), `scf` Python bindings |
| 2026-04-17 | R1 iter_args | DSL | `Carry` + `init=[...]` on `for_range`, scf.for result rebinding; emitter threads iter_args through `scf.ForOp` |
| 2026-04-17 | R2 | DSL+RT | `enigma.Scalar(dtype)` param type; `MetalRuntime.execute(scalars=...)` packs values at correct slot |
| 2026-04-17 | R3 | DSL | `enigma.copy(src, dst, count, mask_fn=...)` — loop-lowered tiled copy |
| 2026-04-17 | R4 | DSL | `register_tensor` / `RegisterTensor` — per-thread SSA-backed small tensors |
| 2026-04-17 | R5 | DSL | `load_if` (select-based) and `store_if` (scf.if-wrapped) |
| 2026-04-17 | R6 | DSL+RT | `async_copy_*` DSL surface with `_require_m3_runtime()` M3+ gate |
| 2026-04-17 | R7 | DSL | `enigma.pipeline(...)` / `Pipeline` — 2-stage double-buffered shared allocs |
| 2026-04-17 | R9 | RT | `MetalRuntime.execute(output_sizes=[...])` returns `list[bytes]` |
| 2026-04-17 | R10 | RT | `DeviceCapabilities` + `rt.device_capabilities()` via `supportsFamily()` |
| 2026-04-17 | Bug 3 | RT | `enigma_create_pipeline_with_constants` + `execute(constants={idx: (type, val)})` — verified on M4 |
