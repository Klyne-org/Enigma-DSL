# Blocked features — DSL surface vs. dialect gaps

Features exposed in the Python DSL but that cannot currently produce valid
Metal because of a dialect-side gap. The Python API is kept stable so these
Just Work once the dialect ships a fix.

For the full dialect punch list (with tablegen, MSL emission examples, and
FileCheck tests), see [dsl-completion-dialect-changes.md](./dsl-completion-dialect-changes.md).

---

## 1. `make_float2/3/4`, `vec_extract`, geometry & pack/unpack ops

- **DSL surface**: `enigma.make_float2/3/4`, `enigma.make_vec`,
  `enigma.vec_extract`, `enigma.dot`, `enigma.length`, `enigma.distance`,
  `enigma.cross`, `enigma.normalize`, `enigma.reflect`, `enigma.refract`,
  `enigma.faceforward`, all `pack_float_to_*` / `unpack_*_to_float`.
- **Blocked by**: the dialect's Python wheel does not bind the upstream
  `vector` dialect, and the dialect has no native `vec_make` op.
  Building `vector<Nxf32>` from Python scalars is impossible.
- **Status**: emitter raises `NotImplementedError` at `vec_make`.
- **Fix**: implement `enigma.vec_make` + `enigma.vec_extract` (see §1 of
  the dialect punch list) — then re-enable the dispatch in
  [enigma/compiler/mlir_emitter.py](../enigma/compiler/mlir_emitter.py).

## 2. `matmul`, `transpose`, `determinant`

- **DSL surface**: `enigma.matmul`, `enigma.transpose`,
  `enigma.determinant`.
- **Blocked by**: no way to *construct* a `float4x4` (or any matrix type)
  from Python scalars. Metal's `transpose`/`determinant` only accept
  matrix types, not vectors; `matmul` needs `vector<CxRxT>` operands.
- **Status**: DSL traces the op, emitter dispatches to the dialect op,
  but there's no producer op for the input. End-to-end compilation
  fails at the first use.
- **Fix**: add a matrix-constructor op to the dialect (e.g.
  `enigma.mat_make` or `enigma.mat_from_cols`).

## 3. `function_constant`

- **DSL surface**: `enigma.function_constant(index=N, dtype=...)`.
- **Blocked by**: dialect's MSL emitter places the `[[function_constant(N)]]`
  attribute on a **local** declaration inside the kernel body. MSL requires
  it at **file scope**:
  ```c++
  constant float alpha [[function_constant(0)]];
  kernel void k(...) { ... }
  ```
- **Status**: `xcrun metal -c` fails with a compile error on the generated
  `.metal`.
- **Fix**: dialect MSL emitter must hoist `function_constant` declarations
  to file scope.

## 4. Simdgroup matrix ops

- **DSL surface**: not yet exposed (no `enigma.simdgroup_matrix_load`
  etc.). Intentionally gated — these are the headline "CuTe-for-Metal"
  ops.
- **Blocked by**: dialect defines `simdgroup_matrix_load/store`,
  `simdgroup_multiply_accumulate` in `EnigmaMatrixOps.td` but has no
  MSL emitter dispatch and no DSL surface.
- **Fix**: §2 of the dialect punch list.

## 5. Textures with runtime bindings

- **DSL surface**: none (no `enigma.Texture2D`).
- **Blocked by**: dialect has `texture_read/write/sample` ops and MSL
  emission for them, but the Python bindings don't expose an MLIR
  texture type.
- **Fix**: §3 of the dialect punch list. Pure DSL work, ~1 day.

## 6. Control flow (`if` / `for` / `while`)

- **DSL surface**: none. The tracer is straight-line only.
- **Blocked by**: no `scf.if` / `scf.for` lowering in the dialect's
  Python wheel; no Python-side tracer support for Python-level control
  flow.
- **Fix**: §4 of the dialect punch list. Non-trivial (~3–5 days).

## 7. Vertex / fragment shaders

- **DSL surface**: none.
- **Blocked by**: DSL has no `@enigma.vertex_kernel` /
  `@enigma.fragment_kernel`. Dialect already models the ops and MSL
  emission works.
- **Fix**: §5 of the dialect punch list. Low priority — GPGPU-first DSL.

## 8. Relational ops (`all` / `any` / vector-`select`)

- **Blocked by**: transitively on §1 (need vector operands).

---

## Not blocked — recently landed

- **TV-layout (tv_load/tv_add/tv_store)** — lowered to per-element
  `memref.load`/`memref.store`/`arith.addf` in the MLIR emitter. Legacy
  `enigma/compiler/metal_emitter.py` has been deleted.
- **`vec_width > 0`** — the MLIR emitter widens the buffer element type
  to `vector<Nxf32>`, which the dialect's MSL translator lowers to
  `device floatN*`. Same MLIR path as scalar kernels.
- **Comparisons** — `enigma.cmp_eq/ne/lt/le/gt/ge` for both integer and
  float operands, routed through `arith.cmpi` / `arith.cmpf`.
- **Grid/thread queries with x/y/z dimension** — all 12 query ops
  (`thread_position_in_grid`, `threads_per_simdgroup`,
  `simdgroup_index_in_threadgroup`, …) accept an optional `dim="x"|"y"|"z"`.
- **Extended dtypes** — `bf16`, `i8`, `u8`, `i16`, `u16`, `i64`, `u64`
  are now accepted as kernel argument and element types.
