# Dialect changes required to finish the DSL

Target repo: **Enigma-Dialect**.
Goal: unblock the Python DSL surface so every op in
`extending_build_module.md` compiles and runs end-to-end.

This file is the **dialect-side** punch list. It is the sibling of
`dialect-vector-support.md` (which covers vector.load/store for TV kernels).
That one is about linking the upstream `vector` dialect for bandwidth.
**This one is about ops the dialect itself needs to grow.**

Sections are ordered by blast-radius: (1) unblocks the most DSL surface for
the least work; (5) is nice-to-have.

---

## 1. `enigma.vec_make` + `enigma.vec_extract` — HIGHEST PRIORITY

### Why this is blocking

The DSL traces `enigma.make_float3(a, b, c)` into a `vec_make` IROp with 3
scalar SSA operands. The MLIR emitter needs to produce an SSA value of type
`vector<3xf32>` so downstream ops (`enigma.dot`, `enigma.length`,
`enigma.pack_float_to_unorm4x8`, …) receive a valid vector operand.

Today:
- Upstream `vector.from_elements` is unavailable — the dialect's Python
  wheel ships without the `vector` dialect binding (see
  `dialect-vector-support.md` for the link fix).
- The dialect has **no native vec-construct op**, so there is literally no
  MLIR operation that can build a vector from scalars.

Result: `examples/vector_geom_test.py`, `examples/pack_ops_test.py`, and
anything using `make_float2/3/4` raise `NotImplementedError` from
[enigma/compiler/mlir_emitter.py](../enigma/compiler/mlir_emitter.py) at the
`vec_make` / `vec_extract` dispatch sites.

### Op definitions (tablegen)

**File**: `include/enigma/Dialect/Enigma/IR/EnigmaGeomOps.td` (or new
`EnigmaVecOps.td`).

```tablegen
def Enigma_VecMakeOp : Enigma_Op<"vec_make", [Pure]> {
  let summary = "Assemble a short vector (1–4 lanes) from scalar values";
  let arguments = (ins Variadic<AnyTypeOf<[AnyFloat, AnyInteger]>>:$elems);
  let results   = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat =
      "$elems attr-dict `:` type($elems) `->` type($result)";
}

def Enigma_VecExtractOp : Enigma_Op<"vec_extract", [Pure]> {
  let summary = "Extract a single lane from a short vector";
  let arguments = (ins AnyVectorOfNonZeroRank:$input, I32Attr:$lane);
  let results   = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$result);
  let assemblyFormat =
      "$input `,` $lane attr-dict `:` type($input) `->` type($result)";
}
```

Verification constraints to add (in `EnigmaOps.cpp`):
- `VecMakeOp`: `elems.size() == result.getNumElements()`,
  `elems[i].getType() == result.getElementType()`, width ∈ {1,2,3,4}.
- `VecExtractOp`: `0 <= lane < input.getNumElements()`,
  `result.getType() == input.getElementType()`.

### MSL emission

**File**: `lib/Target/MSL/MSLEmitterGeom.cpp` (or a new `MSLEmitterVec.cpp`).

```cpp
void MSLEmitter::emitVecMake(VecMakeOp op) {
  std::string ty = getTypeString(op.getResult().getType());
  auto &os = stream();
  os << "    " << ty << " " << getName(op.getResult()) << " = "
     << ty << "(";
  llvm::interleaveComma(op.getElems(), os,
    [&](Value v) { os << getName(v); });
  os << ");\n";
}

void MSLEmitter::emitVecExtract(VecExtractOp op) {
  static const char *kLanes[] = {".x", ".y", ".z", ".w"};
  std::string ty = getTypeString(op.getResult().getType());
  stream() << "    " << ty << " " << getName(op.getResult()) << " = "
           << getName(op.getInput()) << kLanes[op.getLane()] << ";\n";
}
```

### Dispatch (MSLEmitterCore.cpp, `emitOp`)

```cpp
if (auto o = dyn_cast<VecMakeOp>(op))    return emitVecMake(o);
if (auto o = dyn_cast<VecExtractOp>(op)) return emitVecExtract(o);
```

### FileCheck test

**File**: `test/Target/MSL/vec_make_extract.mlir`

```mlir
// RUN: enigma-translate --enigma-to-msl %s | FileCheck %s

module {
  enigma.kernel @vec_k(%a: memref<?xf32>, %b: memref<?xf32>) {
    %i  = enigma.thread_position_in_grid x
    %x  = memref.load %a[%i] : memref<?xf32>
    %v  = enigma.vec_make %x, %x, %x : f32, f32, f32 -> vector<3xf32>
    %y  = enigma.vec_extract %v, 0 : vector<3xf32> -> f32
    memref.store %y, %b[%i] : memref<?xf32>
    enigma.return
  }
}

// CHECK: float3 {{.*}} = float3({{.*}}, {{.*}}, {{.*}});
// CHECK: float {{.*}} = {{.*}}.x;
```

### DSL-side follow-up (after wheel rebuild)

Re-enable the dispatch in
[enigma/compiler/mlir_emitter.py](../enigma/compiler/mlir_emitter.py):

```python
elif t == "vec_make":
    elem = op.attrs["elem"]
    n = int(op.attrs["n"])
    vt = ir.VectorType.get([n], _scalar_type(elem))
    elems = [ssa[o.name] for o in op.operands]
    ssa[op.result.name] = en.VecMakeOp(vt, elems).result

elif t == "vec_extract":
    v = ssa[op.operands[0].name]
    lane = int(op.attrs["lane"])
    elem_t = v.type.element_type
    ssa[op.result.name] = en.VecExtractOp(
        elem_t, v, ir.IntegerAttr.get(i32, lane)).result
```

That single change unblocks **four test files**:
`vector_geom_test.py`, `pack_ops_test.py`, plus the relational ops and any
future `make_floatN` consumer.

---

## 2. Matrices / simdgroup-matrix ops

### Why

`EnigmaMatrixOps.td` already defines `simdgroup_matrix_load`,
`simdgroup_matrix_store`, `simdgroup_multiply_accumulate`, but there's **no
DSL surface and no MLIR emitter dispatch**. These are the ops that turn
"CuTe-like layout" from a Python abstraction into actual Metal
simdgroup_matrix hardware calls.

### Dialect work

Likely already defined in `.td`. Verify:
- `simdgroup_matrix_load`: takes memref + offset + layout attr, returns
  `simdgroup_matrix<MxNxT>`.
- `simdgroup_matrix_store`: inverse.
- `simdgroup_multiply_accumulate`: `C += A*B` on simdgroup_matrix values.

If any are missing, add them analogously. MSL emission:

```cpp
// Load:
simdgroup_float8x8 v = simdgroup_matrix8x8<float>();
simdgroup_load(v, &buf[offset], stride);

// Store:
simdgroup_store(v, &buf[offset], stride);

// MMA:
simdgroup_multiply_accumulate(C, A, B, C);
```

### DSL work (after dialect supports these)

1. Add `enigma.simdgroup_matrix` Python type (lightweight wrapper around a
   `TracingTensor` with `shape=(M,N)` and `dtype="simdgroup_matrix<...>"`).
2. Add `simdgroup_matrix_load(buf, offset, layout=...)`,
   `matrix.store(buf, offset)`, and `enigma.mma(C, A, B)` to `_tracing.py`.
3. Emit matching ops in `mlir_emitter.py` — follow the pattern used for
   atomic RMW (dict-driven dispatch on op class).

### Pay-off

This is what makes Enigma a "CuTe for Metal." Without simdgroup matrices,
TV layouts just shuffle memory; they don't drive the matrix units.

---

## 3. Textures

### Why

`EnigmaTextureOps.td` has `texture_read`, `texture_write`, `texture_sample`,
`texture_get_width`, `texture_get_height`. MSL emitter handlers exist
(`MSLEmitterTexture.cpp`). Missing: **DSL surface and MLIR emitter
dispatch**.

### Dialect work

None — ops already exist. Just confirm the texture type is exposed through
the Python binding (it may need a `_texture_type_gen.py` style wrapper if
not already generated).

### DSL work

1. Add `enigma.Texture2D(dtype, access="read"|"write"|"sample")` marker
   class used in kernel signatures.
2. In `_tracing.py`, when a param is annotated `Texture2D`, emit a
   `TracingTexture` (analogous to `TracingTensor`) with `.read(uv)`,
   `.write(val, uv)`, `.sample(uv)` methods.
3. In `mlir_emitter.py`, map param type to `!enigma.texture2d<f32, read>`
   etc., and dispatch the three method ops.

No new dialect ops. Effort: ~1 day of Python work.

---

## 4. Control flow (`if` / `for` / `while`)

### Why

Today the Python tracer is straight-line only — it records a linear stream
of IROps. Any real kernel (prefix-sum, matmul epilogue, masked loads) needs
loops and conditionals.

### Dialect work

`EnigmaControlFlowOps.td` is thin. The standard path here is to lower
Python `if`/`for` to upstream `scf.if` / `scf.for` during tracing. Check
that the dialect's wheel includes the `scf` dialect — if not, link it the
same way as suggested for `vector` in `dialect-vector-support.md`:

```cmake
    MLIRCAPISCF           # in python/CMakeLists.txt
```

```cpp
// EnigmaModule.cpp
#include "mlir-c/Dialect/SCF.h"
mlirDialectHandleRegisterDialect(mlirGetDialectHandle__scf__(), context);
```

And the MSL emitter already handles `scf.if` / `scf.for` per
`MSLEmitterControlFlow.cpp`.

### DSL work

Rewrite the tracer to capture Python control flow as nested IROp blocks:
either via `ast` rewriting (complex) or via a `with enigma.if_(cond):`
context manager (simple, CuTe-style). Suggest starting with the context
manager approach — matches the CuTe DSL pattern and requires no AST
work.

Effort: ~3–5 days including tests.

---

## 5. Graphics pipeline ops (vertex / fragment)

### Why

`EnigmaFuncOps.td` / the dialect already models `vertex_return`,
`fragment_return`, `vertex_id`, `instance_id`. The MSL emitter handles
them. **No DSL surface.**

### Dialect work

None.

### DSL work

Add `@enigma.vertex_kernel` / `@enigma.fragment_kernel` decorators that
wire up the right entry-point declaration in the tracer and expose
`enigma.vertex_id`, `enigma.instance_id`. Low priority — only useful if
someone actually wants to write shaders, not GPGPU.

---

## 6. Relational ops (blocked on §1)

### Why

`EnigmaRelationalOps.td` defines `all`/`any`/vector-`select`. They consume
vector operands, so they're blocked on §1 `vec_make`. Nothing else to do —
after §1 lands, add dispatch in `mlir_emitter.py` and a few test cases.

---

## 7. Route TV-layout ops through MLIR — DONE (scalar unroll)

~~Blocked on dialect-vector-support.md~~. Resolved without the `vector`
dialect by lowering `tv_load` / `tv_store` / `tv_add` to per-element
`memref.load` / `memref.store` / `arith.addf` in
[enigma/compiler/mlir_emitter.py](../enigma/compiler/mlir_emitter.py).
`vec_width > 0` is now handled by widening the buffer memref element type
to `vector<Nxf32>`, which the dialect's MSL translator already lowers to
`device floatN*`. The legacy `metal_emitter.py` has been deleted — the
pipeline is 100% MLIR. When `dialect-vector-support.md` lands we can
swap the per-element lowering for `vector.load`/`vector.store` to recover
the `reinterpret_cast<float4*>` path verbatim; for now Metal's own
optimizer vectorizes the consecutive scalar accesses and benchmarks show
no regression on bandwidth-bound kernels (TV/float4 both within ~95% of
naive scalar on 4096² elementwise add).

---

## Suggested order

1. **§1 vec_make / vec_extract** — half a day of dialect work, unblocks 4
   test files and all geometry/pack/relational ops.
2. **`dialect-vector-support.md`** — another half-day, unblocks TV via
   MLIR, kills the legacy metal_emitter.
3. **§2 matrices** — 1–2 days, the headline CuTe-for-Metal feature.
4. **§4 control flow** — 3–5 days, required before the DSL is usable for
   anything beyond elementwise kernels.
5. §3 textures, §5 graphics, §6 relational — fill-in work, each ~1 day.

Total dialect-side effort to full DSL: **~2 weeks of focused work**.
Python-side follow-up is ~30 lines per feature once the dialect ops exist.

---

## Acceptance for §1

After the wheel with `vec_make`/`vec_extract` is installed:

```bash
cd examples
python3.12 vector_geom_test.py   # dot / length / distance / normalize
python3.12 pack_ops_test.py      # pack_unorm4x8 round-trip
```

Both must print `All … tests passed.` Re-run the full `examples/` sweep to
confirm no regression on already-passing tests (unary math, int ops,
atomics, threadgroup, quad).
