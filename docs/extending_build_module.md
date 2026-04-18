# Extending `_build_module` — What You Can Add

This guide lists the ops the Enigma dialect already defines (in `Enigma-Dialect/include/enigma/Dialect/Enigma/IR/*.td`) but that the Python MLIR emitter [enigma/compiler/mlir_emitter.py](../enigma/compiler/mlir_emitter.py) does **not** yet lower to from the traced IR. Each item is a bite-sized learning task: add one op end-to-end, then test it.

---

## How the pipeline fits together

```
@enigma.kernel (Python)
        │
        ▼
  _tracing.py          ← records IROp objects into KernelBuilder
        │
        ▼
  mlir_emitter._build_module()   ← THIS is what you're extending
        │  (matches op.op_type strings → enigma dialect ops)
        ▼
  MLIR module (enigma dialect)
        │
        ▼
  en.translate_to_msl(...)        ← dialect's C++ MSL emitter
        │
        ▼
  Metal Shading Language source
```

To add a new op you touch **three** places:

1. **Tracing layer** — [enigma/_tracing.py](../enigma/_tracing.py)
   Add a method / dunder / helper that records an `IROp` with a new `op_type` string.
2. **Emitter** — [enigma/compiler/mlir_emitter.py](../enigma/compiler/mlir_emitter.py)
   Add an `elif t == "your_op":` branch that calls the matching `en.XxxOp(...)` binding.
3. **User-facing surface** (optional) — [enigma/__init__.py](../enigma/__init__.py) or [enigma/core.py](../enigma/core.py)
   Expose a friendly name (`enigma.sqrt`, `enigma.barrier`, etc).

Then write a test under `tests/` that traces a kernel using it and asserts on either the emitted MLIR string or the final MSL.

---

## Currently wired in the emitter

Only these `op_type` strings are handled today:

- `thread_position_in_grid`, `thread_position_in_threadgroup`
- `threadgroup_position_in_grid`, `threads_per_threadgroup`
- `const`
- `load`, `store`
- `neg`
- `add`, `sub`, `mul`, `div`, `mod`

Everything below is **available in the dialect** (Python binding auto-generated via tablegen) but **not yet lowered** from tracing.

---

## 1. Unary math ops (easy — start here)

Source: [EnigmaMathOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaMathOps.td)

Dialect classes: `en.AbsOp`, `en.CeilOp`, `en.FloorOp`, `en.RoundOp`, `en.TruncOp`, `en.SignOp`, `en.SaturateOp`, `en.FractOp`, `en.SqrtOp`, `en.RsqrtOp`, `en.ExpOp`, `en.Exp2Op`, `en.LogOp`, `en.Log2Op`, `en.Log10Op`, `en.SinOp`, `en.CosOp`, `en.TanOp`, `en.AsinOp`, `en.AcosOp`, `en.AtanOp`, `en.SinhOp`, `en.CoshOp`, `en.TanhOp`.

**How to add** (example: `sqrt`):

1. In `_tracing.py` add a helper:
   ```python
   def sqrt(x: IRValue) -> IRValue:
       builder = get_builder()
       result = builder.new_value(x.dtype)
       builder.record(IROp("sqrt", result, [x]))
       return result
   ```
2. Expose it in `enigma/__init__.py` (`from ._tracing import sqrt`).
3. In `mlir_emitter.py` add inside the op loop:
   ```python
   elif t == "sqrt":
       a = ssa[op.operands[0].name]
       ssa[op.result.name] = en.SqrtOp(a).result
   ```
   Tip: the 20+ unary math ops share a shape. Use a dict lookup:
   ```python
   _UNARY_MATH = {"sqrt": en.SqrtOp, "sin": en.SinOp, ...}
   ```

**Test** (`tests/test_math_ops.py`):
```python
@enigma.kernel
def k(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.sqrt(A[tid])

compiled = enigma.compile(k)
assert "sqrt" in compiled.metal_source
# run on MetalRuntime and compare to np.sqrt(A)
```

## 2. Binary math ops

Classes: `en.FminOp`, `en.FmaxOp`, `en.PowOp`, `en.FmodOp`, `en.Atan2Op`, `en.StepOp`, `en.CopysignOp`.

Same recipe as above but with two operands. Good practice task: wire all seven with a single dict-driven branch.

## 3. Ternary math ops

Classes: `en.ClampOp`, `en.FmaOp`, `en.MixOp`, `en.SmoothstepOp`.

Expose as `enigma.clamp(x, lo, hi)`, `enigma.fma(a, b, c)`, `enigma.mix(a, b, t)`, `enigma.smoothstep(e0, e1, x)`.

## 4. Float predicates (return bool/i1)

Classes: `en.IsNanOp`, `en.IsInfOp`, `en.IsFiniteOp`, `en.SignbitOp`, `en.IsNormalOp`.

Note: result dtype is i1 — plan how you represent bools in `IRValue.dtype` (e.g. `"bool"`). Test by storing the result into a `uint8` buffer after cast.

## 5. Select / int min-max-clamp

Classes: `en.SelectOp`, `en.IMinOp`, `en.IMaxOp`, `en.IClampOp`.

`select(cond, a, b)` is a fundamental building block — add once predicates exist.

## 6. Integer bit ops

Source: [EnigmaIntOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaIntOps.td)

Classes: `en.PopcountOp`, `en.ClzOp`, `en.CtzOp`, `en.ReverseBitsOp`, `en.AbsDiffUnaryOp`, `en.AbsDiffBinOp`, `en.AddSatOp`, `en.SubSatOp`, `en.MulHiOp`, `en.RotateOp`, `en.ExtractBitsOp`, `en.InsertBitsOp`, `en.MadSatOp`.

## 7. Synchronization ops (high value — enables shared memory algorithms)

Source: [EnigmaSyncOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaSyncOps.td)

Classes: `en.ThreadgroupBarrierOp`, `en.SimdgroupBarrierOp`, `en.ThreadgroupAllocOp`.

- Expose as `enigma.barrier()` / `enigma.simd_barrier()`.
- `ThreadgroupBarrierOp` takes a `MemFlags` attribute — read `EnigmaEnums.td` for the values (`mem_device`, `mem_threadgroup`, …).
- `ThreadgroupAllocOp` returns a memref you can load/store into — makes tile-based kernels possible.

**Test idea**: a reduction kernel that writes partial sums into threadgroup memory, barriers, then reduces.

## 8. SIMD group ops (warp-level primitives)

Source: [EnigmaSimdOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaSimdOps.td)

Classes: `en.SimdSumOp`, `en.SimdProductOp`, `en.SimdMinOp`, `en.SimdMaxOp`, `en.SimdAndOp`, `en.SimdOrOp`, `en.SimdXorOp`, `en.SimdShuffleOp`, `en.SimdShuffleUpOp`, `en.SimdShuffleDownOp`, `en.SimdShuffleXorOp`, `en.SimdBroadcastOp`.

Unlocks fast warp-reductions. Test: sum 32 elements per simdgroup, compare to `np.sum`.

## 9. Relational ops

Source: [EnigmaRelationalOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaRelationalOps.td). Needed for `==`, `!=`, `<`, `<=`, `>`, `>=` returning i1 — pair with `SelectOp`.

## 10. Atomics

Source: [EnigmaAtomicOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaAtomicOps.td)

Classes: `en.AtomicLoadOp`, `en.AtomicStoreOp`, `en.AtomicExchangeOp`, `en.AtomicCompareExchangeWeakOp`, `en.AtomicFetchAddOp`, `en.AtomicFetchSubOp`, `en.AtomicFetchMinOp`, `en.AtomicFetchMaxOp`, `en.AtomicFetchAndOp`, `en.AtomicFetchOrOp`, `en.AtomicFetchXorOp`.

Each takes a `MemoryOrder` attribute (`relaxed`, `acquire`, `release`, `acq_rel`). Test with a global-counter kernel.

## 11. Cast ops

Source: [EnigmaCastOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaCastOps.td)

Classes: `en.MetalCastOp`, `en.AsTypeOp`, `en.FunctionConstantOp`. Needed once you mix dtypes (`f32 ↔ f16`, `i32 ↔ f32`).

## 12. Geometry ops (graphics math)

Source: [EnigmaGeomOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaGeomOps.td)

Classes: `en.DotOp`, `en.CrossOp`, `en.LengthOp`, `en.DistanceOp`, `en.NormalizeOp`, `en.ReflectOp`, `en.RefractOp`, `en.FaceforwardOp`. Requires vector types — a larger task.

## 13. Control flow

Source: [EnigmaControlFlowOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaControlFlowOps.td). Needed for `if`/`for` in traced kernels. Biggest design task — tracing has to capture region bodies.

## 14. Other families (for later)

- [EnigmaQuadOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaQuadOps.td) — quadgroup ops
- [EnigmaMatrixOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaMatrixOps.td) — simdgroup matrices (tensor cores)
- [EnigmaTextureOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaTextureOps.td) — textures + samplers
- [EnigmaPackOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaPackOps.td) — packed-vector pack/unpack
- [EnigmaFuncOps.td](../Enigma-Dialect/include/enigma/Dialect/Enigma/IR/EnigmaFuncOps.td) — vertex / fragment entry points

---

## Recommended learning order

1. One unary math op end-to-end (`sqrt`) — learn the 3-layer flow.
2. All remaining unary math (dict-driven) — practice refactoring.
3. Binary + ternary math.
4. `select` + relational — unlocks branches.
5. Barriers + threadgroup alloc — unlocks shared memory.
6. SIMD reductions.
7. Atomics.
8. Casts, then vectors/geometry.
9. Control flow (requires tracing rework).

---

## Testing template

```python
import numpy as np, enigma

@enigma.kernel
def k(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.<your_op>(A[tid])

compiled = enigma.compile(k)

# 1. MLIR-level check
mlir = compiled.mlir_source  # if exposed; else call emit_mlir(builder)
assert "enigma.<your_op>" in mlir

# 2. MSL-level check
assert "<expected_metal_builtin>" in compiled.metal_source

# 3. Numerical check
rt = enigma.MetalRuntime()
A = np.random.randn(1024).astype(np.float32)
out = np.frombuffer(
    rt.execute(compiled, [A], 1024 * 4, grid=(1024,1,1), threads=(256,1,1)),
    dtype=np.float32,
)
np.testing.assert_allclose(out, np.<your_op>(A), rtol=1e-5)
```

Look at [tests/test_vector_add.py](../tests/test_vector_add.py) for the established pattern.
