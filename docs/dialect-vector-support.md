# Dialect changes: link `vector` dialect + MSL emission

Target repo: **Enigma-Dialect**.
Goal: let the DSL lower its TV-layout ops (`tv_load` / `tv_store` / `tv_add`)
into upstream `vector.*` ops so `enigma-DSL/compiler/metal_emitter.py` can be
deleted without any TV-layout perf regression.

We need **zero new ops** in the dialect. Everything below is linking and
emission support for the upstream `vector` dialect — the same pattern you
already used for `arith`, `memref`, and `func`.

---

## Why

Today the DSL does tracing on the Python side and directly emits Metal text
for TV-layout kernels, bypassing MLIR entirely. That Metal text uses the
`float4` reinterpret-cast pattern which is responsible for our
memory-bandwidth-saturating TV perf (~96 GB/s on `benchmark_naive_vs_tv`).

If instead the DSL emits `vector.load` / `vector.store` / `arith.addf` on
vector types, the MSLEmitter can translate those back to the same `float4`
pattern. Same machine code, same perf, but now the entire pipeline is MLIR.

**Non-goal**: modeling TV algebra or layout in the dialect. TV stays a
DSL-side abstraction. The dialect just needs to understand vectorized
load/store/arith on `memref`.

---

## Change 1 — Link MLIRCAPIVector into the wheel

**File**: `python/CMakeLists.txt`

Find the CAPI link list (around line 26–29) and append `MLIRCAPIVector`:

```cmake
    MLIRCAPITransforms
    MLIRCAPIArith
    MLIRCAPIMemRef
    MLIRCAPIFunc
    MLIRCAPIVector          # <-- add
```

## Change 2 — Register + load the dialect at context creation

**File**: `python/EnigmaModule.cpp`

Inside the `register_dialect` binding (the function that already registers
enigma/arith/memref/func — around line 27–45), add `vector`:

```cpp
#include "mlir-c/Dialect/Vector.h"   // top of file alongside the other includes
```

```cpp
// in register_dialect, alongside the arith/memref/func registrations:
MlirDialectHandle vectorHandle = mlirGetDialectHandle__vector__();
mlirDialectHandleRegisterDialect(vectorHandle, context);

// and inside the `if (load)` block:
mlirDialectHandleLoadDialect(vectorHandle, context);
```

## Change 3 — MSLEmitter support for vector ops

**File**: `lib/Target/MSL/MSLEmitter.cpp` (and any header that registers
op-handler cases).

Add emission for four things. All Metal-native, all one-liners.

### 3a. Vector type printing

When emitting a `VectorType`, map `vector<N x {elem}>` to Metal's
short-vector name. Widths Metal supports natively: **1, 2, 3, 4**.

| MLIR type          | MSL type |
|--------------------|----------|
| `vector<1xf32>`    | `float`  |
| `vector<2xf32>`    | `float2` |
| `vector<3xf32>`    | `float3` |
| `vector<4xf32>`    | `float4` |
| `vector<2xf16>`    | `half2`  |
| `vector<4xf16>`    | `half4`  |
| `vector<4xi32>`    | `int4`   |
| `vector<4xui32>`   | `uint4`  |

Reject (or emit an `// UNHANDLED` comment for) widths ∉ {1,2,3,4} and
multi-dim vector types — the DSL won't produce them.

### 3b. `vector.load`

Input:
```mlir
%v = vector.load %buf[%i] : memref<?xf32>, vector<4xf32>
```

Emit:
```cpp
float4 v = *reinterpret_cast<device const float4*>(&buf[i]);
```

Notes:
- The `const` vs non-const in the cast should follow the memref's mutability
  (same logic you already use for scalar `memref.load`). In TV kernels, loads
  come from non-written buffers.
- **No element-by-element splat.** The whole point is one 128-bit transaction;
  lowering to `float4 v; v.x = buf[i]; v.y = buf[i+1]; ...` would defeat the
  purpose. The `reinterpret_cast` form is the required one.

### 3c. `vector.store`

Input:
```mlir
vector.store %v, %buf[%i] : memref<?xf32>, vector<4xf32>
```

Emit:
```cpp
*reinterpret_cast<device float4*>(&buf[i]) = v;
```

### 3d. Arith ops on vector operands

`arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.remf` on
`vector<Nxf32>` / `vector<Nxf16>` — and their integer counterparts
(`arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.remsi`) on
`vector<Nxi32>` etc.

These need **no special handling** beyond type printing: Metal natively
supports `float4 + float4`, `float4 * float4`, etc. If your existing arith
emitter just does `"{T} {res} = {a} {op} {b};"`, it already works once the
vector type prints as `float4`. Double-check that's true — this is the
single most likely place for a regression.

### 3e. (Optional) `arith.constant` on vector types

```mlir
%c = arith.constant dense<0.0> : vector<4xf32>
```
→
```cpp
float4 c = float4(0.0);
```

Not required for the current TV kernels. Add if easy.

### 3f. (Optional, NOT needed for TV) `vector.extract` / `vector.insert`

Tv-layout kernels don't generate these — they load full vectors, do
elementwise ops on full vectors, store full vectors. Skip unless a future
use case needs them.

---

## Change 4 — Test

Add an integration test at `test/Target/MSL/vector_load_store.mlir`:

```mlir
// RUN: enigma-translate --enigma-to-msl %s | FileCheck %s

module {
  enigma.kernel @vec_add(
      %arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
    %idx = enigma.thread_position_in_grid x
    %off = arith.muli %idx, %idx : index   // or any base offset
    %a = vector.load %arg0[%off] : memref<?xf32>, vector<4xf32>
    %b = vector.load %arg1[%off] : memref<?xf32>, vector<4xf32>
    %c = arith.addf %a, %b : vector<4xf32>
    vector.store %c, %arg2[%off] : memref<?xf32>, vector<4xf32>
    enigma.return
  }
}

// CHECK: kernel void vec_add(
// CHECK: float4 {{.*}} = *reinterpret_cast<device const float4*>(&{{.*}}[{{.*}}]);
// CHECK: float4 {{.*}} = *reinterpret_cast<device const float4*>(&{{.*}}[{{.*}}]);
// CHECK: float4 {{.*}} = {{.*}} + {{.*}};
// CHECK: *reinterpret_cast<device float4*>(&{{.*}}[{{.*}}]) = {{.*}};
```

Also add a variant with `vector<2xf32>` → `float2` to cover tail cases.

---

## Change 5 — Rebuild & ship

```bash
source ~/.local/enigma-llvm/activate.sh
./scripts/build-wheel.sh
```

New wheel drops in `dist/`. The DSL side will pick it up via
`pip install --force-reinstall`.

---

## Acceptance criteria

A DSL-side smoke test must pass after the new wheel is installed:

```python
from mlir import ir
from mlir.dialects import enigma
with ir.Context() as ctx, ir.Location.unknown():
    enigma.register_dialect(ctx)
    ctx.load_all_available_dialects()

    assert ctx.is_registered_operation("vector.load")
    assert ctx.is_registered_operation("vector.store")

    mod = ir.Module.parse("""
      module {
        enigma.kernel @t(%a: memref<?xf32>, %b: memref<?xf32>) {
          %i = enigma.thread_position_in_grid x
          %v = vector.load %a[%i] : memref<?xf32>, vector<4xf32>
          vector.store %v, %b[%i] : memref<?xf32>, vector<4xf32>
          enigma.return
        }
      }
    """)
    msl = enigma.translate_to_msl(mod.operation)
    assert "float4" in msl
    assert "reinterpret_cast<device const float4*>" in msl
    assert "reinterpret_cast<device float4*>" in msl
```

When that passes, the DSL team will:

1. Delete `enigma/compiler/metal_emitter.py` entirely.
2. Rewrite `tv_load` / `tv_store` / `tv_add` lowering in
   `enigma/compiler/mlir_emitter.py` (~30 lines) using `vector.load`,
   `vector.store`, and `arith.addf`-on-vectors.
3. Re-run `examples/benchmark_naive_vs_tv.py`. TV number must match the
   pre-change baseline (±5%). If it drops significantly, the MSLEmitter is
   scalarizing vector ops instead of using `reinterpret_cast` — fix point 3b.

---

## Effort estimate

- Changes 1 + 2: ~5 min
- Change 3: ~1–2 hours (emitter cases + vector type printing)
- Change 4: ~30 min (one FileCheck test)
- Change 5: mechanical rebuild

Total: **half a day** of dialect work. Zero new ops, zero new types, zero
TableGen regeneration beyond what the vector dialect already provides.
