# Enigma DSL — API Reference

Enigma is a Python DSL for writing Apple Metal GPU compute kernels. You write
Python functions decorated with `@enigma.kernel`, and Enigma traces, compiles,
and dispatches them on the GPU through MLIR and the Metal Shading Language.

**Current release: v0.1.1**  ·  PyPI: <https://pypi.org/project/enigma-dsl/>

---

## 0. Install

Requirements: Apple Silicon Mac (M1 through M5), macOS 14 / 15 (or any newer
release), Python 3.11 / 3.12 / 3.13.

```bash
pip install enigma-dsl
```

The wheel bundles the Python DSL, the native MLIR `enigma` dialect, and the
LLVM runtime libraries — no separate steps, no toolchain on your machine.
`pip` picks the right wheel for your Python version and macOS version
automatically; the release ships six wheels (3 Pythons × 2 macOS deployment
targets, 14.0 and 15.0).

For building from source (only needed if you are hacking on the dialect),
see [the project README](../README.md#building-from-source).

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Data Types](#2-data-types)
3. [Kernel Definition](#3-kernel-definition)
4. [Compilation](#4-compilation)
5. [Runtime Dispatch](#5-runtime-dispatch)
6. [Thread & Grid Queries](#6-thread--grid-queries)
7. [Arithmetic Operators](#7-arithmetic-operators)
8. [Unary Float Math](#8-unary-float-math)
9. [Binary Float Math](#9-binary-float-math)
10. [Ternary Float Math](#10-ternary-float-math)
11. [Float Predicates](#11-float-predicates)
12. [Integer Math](#12-integer-math)
13. [Integer Bit Operations](#13-integer-bit-operations)
14. [Comparison Operations](#14-comparison-operations)
15. [Select / Conditional](#15-select--conditional)
16. [Type Casting](#16-type-casting)
17. [Vector Construction & Extraction](#17-vector-construction--extraction)
18. [Geometry Operations](#18-geometry-operations)
19. [Pack / Unpack Operations](#19-pack--unpack-operations)
20. [SIMD Group Operations](#20-simd-group-operations)
21. [Quad Group Operations](#21-quad-group-operations)
22. [Barriers & Synchronization](#22-barriers--synchronization)
23. [Threadgroup Shared Memory](#23-threadgroup-shared-memory)
24. [Atomic Operations](#24-atomic-operations)
25. [Simdgroup Matrix Operations](#25-simdgroup-matrix-operations)
26. [Matrix Operations](#26-matrix-operations)
27. [Function Constants](#27-function-constants)
28. [Layout Algebra (CuTe-style)](#28-layout-algebra-cute-style)
29. [Tensor & TV Layout](#29-tensor--tv-layout)
30. [JIT Path (@enigma.jit)](#30-jit-path-enigmajit)
31. [arch Namespace](#31-arch-namespace)
32. [Tuple Utilities](#32-tuple-utilities)
33. [Errors](#33-errors)
34. [Async Copy (Experimental)](#34-async-copy-experimental)
35. [Control Flow](#35-control-flow)  — `for_range`, `if_`, `while_`, `range_constexpr`
36. [Masked Load / Store](#36-masked-load--store)  — `load_if`, `store_if`, `register_tensor`
37. [High-Level Tile Helpers](#37-high-level-tile-helpers)  — `gemm`, `copy`, `pipeline`
38. [Quantization Helpers](#38-quantization-helpers)  — `dequantize_int8` (+ pack/unpack in §19)
39. [Tooling: `enigma.testing` and `enigma.benchmark`](#39-tooling-enigmatesting-and-enigmabenchmark)
40. [Compiler: emit-only mode](#40-compiler-emit-only-mode)

---

## 1. Quick Start

```python
import numpy as np
import enigma

# 1. Define a kernel
@enigma.kernel
def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]

# 2. Compile to .metallib
compiled = enigma.compile(vector_add)

# 3. Run on GPU
runtime = enigma.MetalRuntime()
N = 1024
A = np.random.randn(N).astype(np.float32)
B = np.random.randn(N).astype(np.float32)
result_bytes = runtime.execute(
    compiled,
    inputs=[A, B],
    output_size=N * 4,
    grid=(N, 1, 1),
    threads=(min(N, 256), 1, 1),
)
C = np.frombuffer(result_bytes, dtype=np.float32)
```

---

## 2. Data Types

Type annotations on `@enigma.kernel` parameters tell the compiler what Metal
buffer element type to use. Each type has a class form and a singleton shorthand.

| Shorthand      | Class        | Metal type | Bits | Description             |
|----------------|--------------|------------|------|-------------------------|
| `enigma.f32`   | `Float32`    | `float`    | 32   | Single-precision float  |
| `enigma.f16`   | `Float16`    | `half`     | 16   | Half-precision float    |
| `enigma.bf16`  | `BFloat16`   | `bfloat`   | 16   | Brain float 16          |
| `enigma.i8`    | `Int8`       | `char`     | 8    | Signed 8-bit integer    |
| `enigma.u8`    | `UInt8`      | `uchar`    | 8    | Unsigned 8-bit integer  |
| `enigma.i16`   | `Int16`      | `short`    | 16   | Signed 16-bit integer   |
| `enigma.u16`   | `UInt16`     | `ushort`   | 16   | Unsigned 16-bit integer |
| `enigma.i32`   | `Int32`      | `int`      | 32   | Signed 32-bit integer   |
| `enigma.u32`   | `UInt32`     | `uint`     | 32   | Unsigned 32-bit integer |
| `enigma.i64`   | `Int64`      | `long`     | 64   | Signed 64-bit integer   |
| `enigma.u64`   | `UInt64`     | `ulong`    | 64   | Unsigned 64-bit integer |
| `enigma.b1`    | `Bool`       | `bool`     | 1    | Boolean                 |

**Usage in kernel signatures:**

```python
@enigma.kernel
def my_kernel(A: enigma.f32, B: enigma.i32, C: enigma.f16):
    ...
```

Each parameter becomes a `device T*` buffer argument in the generated Metal
kernel, bound to sequential `[[buffer(N)]]` indices starting at 0.

---

## 3. Kernel Definition

### `@enigma.kernel`

```python
@enigma.kernel
def my_kernel(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]
```

- **Parameters**: Each parameter must have a type annotation from
  [Data Types](#2-data-types). Parameters map to Metal `device T*` buffers in
  declaration order.
- **Returns**: A `KernelDef` object (not called directly — pass to `enigma.compile()`).
- **Inside the body**: Use `enigma.*` functions to express GPU operations. The
  body is *traced*, not executed — Python control flow (`if`, `for`) is not
  captured (the tracer records a straight-line sequence of ops).

**Buffer indexing:**

```python
val = A[index]     # load: reads A[index]
A[index] = val     # store: writes val to A[index]
```

`index` must be an `IRValue` (returned by grid queries, arithmetic, etc.) or a
Python `int` (auto-wrapped as a constant).

---

## 4. Compilation

### `enigma.compile(fn, *args, **kwargs) -> CompiledKernel`

Compiles a `@enigma.kernel` or `@enigma.jit` function to a `.metallib`.

**Parameters:**

| Parameter            | Type           | Default | Description                                                  |
|----------------------|----------------|---------|--------------------------------------------------------------|
| `fn`                 | KernelDef/JitDef | —     | The decorated kernel or jit function                         |
| `*args`              | Tensor...      | —       | Required for `@enigma.jit` (tensor arguments with layouts)   |
| `keep_metal_source`  | bool           | False   | Keep the `.metal` source file on disk                        |
| `dump_ir`            | bool           | False   | Print the traced IR and generated Metal source               |
| `dump_mlir`          | bool           | False   | Print the MLIR (enigma dialect) intermediate representation  |
| `work_dir`           | str \| None    | None    | Directory for build artifacts (default: temp dir)            |
| `vec_width`          | int            | 0       | Vectorize buffer element types to `vector<N>` (0 = scalar)   |

**Returns:** `CompiledKernel`

**Example:**

```python
compiled = enigma.compile(my_kernel, dump_ir=True, vec_width=4)
```

### `CompiledKernel`

| Attribute         | Type          | Description                                      |
|-------------------|---------------|--------------------------------------------------|
| `kernel_name`     | str           | Name of the compiled kernel function              |
| `metallib_path`   | str           | Path to the compiled `.metallib` file             |
| `metallib_bytes`  | bytes         | Raw bytes of the `.metallib`                      |
| `metal_source`    | str           | Generated MSL source code                         |
| `mlir_source`     | str \| None   | MLIR intermediate representation (always present) |
| `grid`            | tuple \| None | Grid dimensions (set by `@jit` path only)         |
| `block`           | tuple \| None | Block dimensions (set by `@jit` path only)        |

**Methods:**

#### `compiled.export_metal(path=None) -> str`

Write the Metal source to a file.

- **path** (str, optional): Output file path. Defaults to `"{kernel_name}.metal"`.
- **Returns**: The path written to.

```python
compiled.export_metal("my_kernel.metal")
```

---

## 5. Runtime Dispatch

### `enigma.MetalRuntime(dylib_path=None)`

Creates a Metal device and command queue for GPU dispatch.

- **dylib_path** (str, optional): Path to a custom Swift runtime dylib. If
  omitted, the bundled runtime is auto-compiled on first use.

### `runtime.execute(compiled, inputs, output_size, grid, threads) -> bytes`

One-shot dispatch: allocates GPU buffers, dispatches the kernel, reads back
the output buffer, and cleans up.

| Parameter      | Type                  | Description                                        |
|----------------|-----------------------|----------------------------------------------------|
| `compiled`     | CompiledKernel        | The compiled kernel to dispatch                    |
| `inputs`       | list[np.ndarray]      | Input arrays (one per kernel buffer param, except the last which is output) |
| `output_size`  | int                   | Size in bytes of the output buffer                 |
| `grid`         | tuple(int, int, int)  | Grid dimensions `(x, y, z)`                       |
| `threads`      | tuple(int, int, int)  | Threads per threadgroup `(x, y, z)`                |

**Returns:** `bytes` — Raw output buffer contents. Use `np.frombuffer()` to decode.

```python
rt = enigma.MetalRuntime()
out_bytes = rt.execute(compiled, [A, B], N * 4, grid=(N, 1, 1), threads=(256, 1, 1))
result = np.frombuffer(out_bytes, dtype=np.float32)
```

### `runtime.prepare(compiled, inputs, output_size) -> PreparedKernel`

Pre-allocate GPU resources for repeated dispatch (benchmarking, iterative
algorithms).

**Returns:** `PreparedKernel`

### `PreparedKernel`

| Method                                       | Returns | Description                                          |
|----------------------------------------------|---------|------------------------------------------------------|
| `pk.dispatch(grid, threads)`                 | None    | Dispatch the kernel (blocking)                       |
| `pk.dispatch_timed(grid, threads)`           | float   | Dispatch and return GPU time in microseconds         |
| `pk.read_output()`                           | bytes   | Read the output buffer contents                      |
| `pk.release()`                               | None    | Free all GPU resources                               |

```python
pk = rt.prepare(compiled, [A, B], N * 4)
pk.dispatch(grid=(N, 1, 1), threads=(256, 1, 1))
result = np.frombuffer(pk.read_output(), dtype=np.float32)
gpu_us = pk.dispatch_timed(grid=(N, 1, 1), threads=(256, 1, 1))
pk.release()
```

---

## 6. Thread & Grid Queries

These functions return an `IRValue` of dtype `uint` representing the
thread/group index in the specified dimension.

All functions below must be called **inside** a `@enigma.kernel` body.

### 1D shorthand (x-only)

```python
tid = enigma.thread_position_in_grid  # property-style, returns x dimension
```

### Explicit dimension queries

Every query function takes an optional `dim` parameter: `"x"` (default),
`"y"`, or `"z"`.

| Function                                          | Metal equivalent                             | Description                                      |
|---------------------------------------------------|----------------------------------------------|--------------------------------------------------|
| `enigma.thread_position_in_grid_xyz(dim="x")`     | `thread_position_in_grid.{x\|y\|z}`          | Global thread index                              |
| `enigma.thread_position_in_threadgroup(dim="x")`  | `thread_position_in_threadgroup.{x\|y\|z}`   | Thread index within its threadgroup              |
| `enigma.threadgroup_position_in_grid(dim="x")`    | `threadgroup_position_in_grid.{x\|y\|z}`     | Threadgroup index in the grid                    |
| `enigma.threads_per_threadgroup(dim="x")`         | `threads_per_threadgroup.{x\|y\|z}`          | Number of threads per threadgroup                |
| `enigma.threads_per_grid(dim="x")`                | `threads_per_grid.{x\|y\|z}`                 | Total threads in the grid                        |
| `enigma.threadgroups_per_grid(dim="x")`            | `threadgroups_per_grid.{x\|y\|z}`            | Number of threadgroups in the grid               |
| `enigma.grid_size(dim="x")`                        | `grid_size.{x\|y\|z}`                        | Grid size (alias for threadgroups_per_grid)       |
| `enigma.thread_index_in_threadgroup()`              | `thread_index_in_threadgroup`                 | Flattened 1D index within threadgroup            |
| `enigma.thread_index_in_simdgroup()`                | `thread_index_in_simdgroup`                   | Lane index within the SIMD group (0-31)          |
| `enigma.simdgroup_index_in_threadgroup()`           | `simdgroup_index_in_threadgroup`              | SIMD group index within the threadgroup          |
| `enigma.threads_per_simdgroup()`                    | `threads_per_simdgroup`                       | Threads per SIMD group (typically 32)            |
| `enigma.simdgroups_per_threadgroup()`               | `simdgroups_per_threadgroup`                  | Number of SIMD groups in the threadgroup         |

**Returns:** `IRValue` (dtype `"uint"`)

**Example (2D grid):**

```python
@enigma.kernel
def kernel_2d(A: enigma.f32, Out: enigma.f32):
    row = enigma.thread_position_in_grid_xyz("y")
    col = enigma.thread_position_in_grid_xyz("x")
    Out[row * 64 + col] = A[row * 64 + col]
```

---

## 7. Arithmetic Operators

`IRValue` supports Python arithmetic operators. These work on both scalar
and vector values.

| Operator   | Operation        | Float lowering | Int lowering |
|------------|------------------|----------------|--------------|
| `a + b`    | Addition         | `arith.addf`   | `arith.addi` |
| `a - b`    | Subtraction      | `arith.subf`   | `arith.subi` |
| `a * b`    | Multiplication   | `arith.mulf`   | `arith.muli` |
| `a / b`    | Division         | `arith.divf`   | `arith.divsi`|
| `a // b`   | Floor division   | `arith.divf`   | `arith.divsi`|
| `a % b`    | Modulo           | `arith.remf`   | `arith.remsi`|
| `-a`       | Negation         | `arith.negf`   | `0 - a`      |

Python `int` literals are auto-wrapped as `uint` constants:

```python
tid = enigma.thread_position_in_grid
idx = tid * 4 + 1    # int literals auto-promote
```

---

## 8. Unary Float Math

All functions take an `IRValue` and return an `IRValue` of the same dtype.
They map directly to Metal Standard Library functions.

| Function              | Metal equivalent | Description                    |
|-----------------------|------------------|--------------------------------|
| `enigma.sqrt(x)`     | `sqrt(x)`        | Square root                    |
| `enigma.rsqrt(x)`    | `rsqrt(x)`       | Reciprocal square root (1/sqrt)|
| `enigma.abs(x)`      | `abs(x)`         | Absolute value                 |
| `enigma.ceil(x)`     | `ceil(x)`        | Round up to nearest integer    |
| `enigma.floor(x)`    | `floor(x)`       | Round down to nearest integer  |
| `enigma.round(x)`    | `round(x)`       | Round to nearest integer       |
| `enigma.trunc(x)`    | `trunc(x)`       | Truncate toward zero           |
| `enigma.sign(x)`     | `sign(x)`        | Sign (-1, 0, or 1)            |
| `enigma.saturate(x)` | `saturate(x)`    | Clamp to [0, 1]               |
| `enigma.fract(x)`    | `fract(x)`       | Fractional part (x - floor(x))|
| `enigma.exp(x)`      | `exp(x)`         | e^x                           |
| `enigma.exp2(x)`     | `exp2(x)`        | 2^x                           |
| `enigma.log(x)`      | `log(x)`         | Natural logarithm              |
| `enigma.log2(x)`     | `log2(x)`        | Base-2 logarithm               |
| `enigma.log10(x)`    | `log10(x)`       | Base-10 logarithm              |
| `enigma.sin(x)`      | `sin(x)`         | Sine                           |
| `enigma.cos(x)`      | `cos(x)`         | Cosine                         |
| `enigma.tan(x)`      | `tan(x)`         | Tangent                        |
| `enigma.asin(x)`     | `asin(x)`        | Arc sine                       |
| `enigma.acos(x)`     | `acos(x)`        | Arc cosine                     |
| `enigma.atan(x)`     | `atan(x)`        | Arc tangent                    |
| `enigma.sinh(x)`     | `sinh(x)`        | Hyperbolic sine                |
| `enigma.cosh(x)`     | `cosh(x)`        | Hyperbolic cosine              |
| `enigma.tanh(x)`     | `tanh(x)`        | Hyperbolic tangent             |

**Example:**

```python
@enigma.kernel
def apply_sqrt(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    B[tid] = enigma.sqrt(A[tid])
```

---

## 9. Binary Float Math

All take two `IRValue` arguments and return an `IRValue`.

| Function                    | Metal equivalent     | Description                              |
|-----------------------------|----------------------|------------------------------------------|
| `enigma.fmin(a, b)`        | `fmin(a, b)`         | Minimum (NaN-safe)                       |
| `enigma.fmax(a, b)`        | `fmax(a, b)`         | Maximum (NaN-safe)                       |
| `enigma.pow(a, b)`         | `pow(a, b)`          | a raised to the power b                  |
| `enigma.fmod(a, b)`        | `fmod(a, b)`         | Floating-point remainder                 |
| `enigma.atan2(a, b)`       | `atan2(a, b)`        | Two-argument arc tangent                 |
| `enigma.step(edge, x)`     | `step(edge, x)`      | 0.0 if x < edge, else 1.0               |
| `enigma.copysign(a, b)`    | `copysign(a, b)`     | Magnitude of a with sign of b            |

---

## 10. Ternary Float Math

All take three `IRValue` arguments and return an `IRValue`.

| Function                           | Metal equivalent          | Description                                    |
|------------------------------------|---------------------------|------------------------------------------------|
| `enigma.clamp(x, lo, hi)`         | `clamp(x, lo, hi)`       | Clamp x to range [lo, hi]                     |
| `enigma.fma(a, b, c)`             | `fma(a, b, c)`           | Fused multiply-add: a*b + c                    |
| `enigma.mix(a, b, t)`             | `mix(a, b, t)`           | Linear interpolation: a + t*(b-a)              |
| `enigma.smoothstep(e0, e1, x)`    | `smoothstep(e0, e1, x)`  | Hermite interpolation between e0 and e1        |

---

## 11. Float Predicates

Return an `IRValue` of dtype `i1` (boolean).

| Function                  | Metal equivalent    | Description                    |
|---------------------------|---------------------|--------------------------------|
| `enigma.isnan(x)`        | `isnan(x)`          | True if x is NaN               |
| `enigma.isinf(x)`        | `isinf(x)`          | True if x is infinity          |
| `enigma.isfinite(x)`     | `isfinite(x)`       | True if x is finite            |
| `enigma.signbit(x)`      | `signbit(x)`        | True if sign bit is set        |
| `enigma.isnormal(x)`     | `isnormal(x)`       | True if x is a normal number   |

---

## 12. Integer Math

| Function                     | Description                                       |
|------------------------------|---------------------------------------------------|
| `enigma.imin(a, b)`         | Integer minimum (signed)                          |
| `enigma.imax(a, b)`         | Integer maximum (signed)                          |
| `enigma.iclamp(x, lo, hi)`  | Integer clamp to [lo, hi]                         |
| `enigma.abs_diff(a, b)`     | Absolute difference \|a - b\|                     |
| `enigma.abs_diff_unary(x)`  | Absolute value (integer)                          |
| `enigma.add_sat(a, b)`      | Saturating addition (clamps to type range)        |
| `enigma.sub_sat(a, b)`      | Saturating subtraction                            |
| `enigma.mul_hi(a, b)`       | High bits of full multiply                        |
| `enigma.rotate(a, b)`       | Bitwise rotate left by b bits                     |
| `enigma.mad_sat(a, b, c)`   | Saturating multiply-add: clamp(a*b + c)           |

---

## 13. Integer Bit Operations

| Function                                       | Description                                             |
|------------------------------------------------|---------------------------------------------------------|
| `enigma.popcount(x)`                          | Count number of set bits                                |
| `enigma.clz(x)`                               | Count leading zeros                                     |
| `enigma.ctz(x)`                               | Count trailing zeros                                    |
| `enigma.reverse_bits(x)`                      | Reverse bit order                                       |
| `enigma.extract_bits(value, offset, bits)`     | Extract `bits` bits starting at `offset`                |
| `enigma.insert_bits(base, insert, offset, bits)` | Insert `bits` bits of `insert` into `base` at `offset` |

**`extract_bits` and `insert_bits` parameters:**

- **value** / **base** / **insert**: `IRValue` — the integer value(s)
- **offset**: `int` — bit position to start at
- **bits**: `int` — number of bits to extract/insert
- **Returns**: `IRValue` with same dtype as input

---

## 14. Comparison Operations

All comparison functions take two `IRValue` arguments and return an `IRValue`
of dtype `i1` (boolean). They work on both integer and float operands.

**Signed comparisons:**

| Function               | Predicate           | Description          |
|------------------------|---------------------|----------------------|
| `enigma.cmp_eq(a, b)` | `a == b`            | Equal                |
| `enigma.cmp_ne(a, b)` | `a != b`            | Not equal            |
| `enigma.cmp_lt(a, b)` | `a < b` (signed)    | Less than            |
| `enigma.cmp_le(a, b)` | `a <= b` (signed)   | Less or equal        |
| `enigma.cmp_gt(a, b)` | `a > b` (signed)    | Greater than         |
| `enigma.cmp_ge(a, b)` | `a >= b` (signed)   | Greater or equal     |

**Unsigned comparisons (integers only):**

| Function                | Predicate            | Description               |
|-------------------------|----------------------|---------------------------|
| `enigma.cmp_ult(a, b)` | `a < b` (unsigned)   | Unsigned less than        |
| `enigma.cmp_ule(a, b)` | `a <= b` (unsigned)  | Unsigned less or equal    |
| `enigma.cmp_ugt(a, b)` | `a > b` (unsigned)   | Unsigned greater than     |
| `enigma.cmp_uge(a, b)` | `a >= b` (unsigned)  | Unsigned greater or equal |

---

## 15. Select / Conditional

### `enigma.where(false_val, true_val, condition) -> IRValue`

Conditional select (ternary operator). Returns `true_val` where `condition` is
true, `false_val` where it is false.

| Parameter    | Type    | Description                           |
|--------------|---------|---------------------------------------|
| `false_val`  | IRValue | Value when condition is false         |
| `true_val`   | IRValue | Value when condition is true          |
| `condition`  | IRValue | Boolean condition (dtype `i1`)        |

**Returns:** `IRValue` with same dtype as `true_val`.

**Example — elementwise max:**

```python
@enigma.kernel
def elem_max(A: enigma.f32, B: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    a, b = A[tid], B[tid]
    Out[tid] = enigma.where(b, a, enigma.cmp_gt(a, b))
```

---

## 16. Type Casting

### `enigma.metal_cast(x, dtype) -> IRValue`

Static cast (Metal `static_cast<T>(x)`). Changes the numeric type of a value.

| Parameter | Type         | Description                                     |
|-----------|--------------|-------------------------------------------------|
| `x`       | IRValue/int  | Value to cast                                   |
| `dtype`   | str          | Target type: `"float"`, `"int"`, `"uint"`, `"half"`, etc. |

**Returns:** `IRValue` with the target dtype.

### `enigma.as_type(x, dtype) -> IRValue`

Bitwise reinterpret cast (Metal `as_type<T>(x)`). Reinterprets the bits of
a value as a different type of the same bit-width.

| Parameter | Type    | Description                               |
|-----------|---------|-------------------------------------------|
| `x`       | IRValue | Value to reinterpret                      |
| `dtype`   | str     | Target type (must have same bit-width)    |

**Returns:** `IRValue` with the target dtype.

**Example:**

```python
i = enigma.metal_cast(3.14, "int")       # float -> int
f = enigma.metal_cast(tid, "float")      # uint -> float
bits = enigma.as_type(float_val, "uint") # reinterpret float bits as uint
```

---

## 17. Vector Construction & Extraction

Enigma represents short vectors (float2/3/4, etc.) as `IRValue` with dtype
`"vec<N,elem>"`. These map to Metal `floatN`, `halfN`, `intN`, `uintN` types.

### `enigma.make_vec(*components) -> IRValue`

Assemble a vector from 2, 3, or 4 scalar `IRValue`s. All components must
have the same dtype.

| Parameter      | Type       | Description                     |
|----------------|------------|---------------------------------|
| `*components`  | IRValue... | 2, 3, or 4 scalar values       |

**Returns:** `IRValue` with dtype `"vec<N,elem>"`.

### Convenience constructors

| Function                             | Equivalent                       |
|--------------------------------------|----------------------------------|
| `enigma.make_float2(x, y)`          | `enigma.make_vec(x, y)`         |
| `enigma.make_float3(x, y, z)`       | `enigma.make_vec(x, y, z)`      |
| `enigma.make_float4(x, y, z, w)`    | `enigma.make_vec(x, y, z, w)`   |

### `enigma.vec_extract(v, lane) -> IRValue`

Extract a single scalar element from a vector.

| Parameter | Type    | Description                             |
|-----------|---------|-----------------------------------------|
| `v`       | IRValue | Vector value (dtype `"vec<N,elem>"`)    |
| `lane`    | int     | Lane index (0-based, must be < N)       |

**Returns:** `IRValue` with the scalar element dtype.

### Property-style access

Vector values also support `.x`, `.y`, `.z`, `.w` accessors:

```python
v = enigma.make_float3(a, b, c)
x_component = v.x  # same as enigma.vec_extract(v, 0)
y_component = v.y  # same as enigma.vec_extract(v, 1)
z_component = v.z  # same as enigma.vec_extract(v, 2)
```

**Example:**

```python
@enigma.kernel
def vec_example(Ax: enigma.f32, Ay: enigma.f32, Az: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    v = enigma.make_float3(Ax[tid], Ay[tid], Az[tid])
    Out[tid] = v.x + v.y + v.z
```

---

## 18. Geometry Operations

These operate on vector `IRValue`s (dtype `"vec<N,elem>"`). They correspond
to Metal's geometry functions.

### Scalar-returning

| Function                      | Signature                  | Returns | Description                     |
|-------------------------------|----------------------------|---------|---------------------------------|
| `enigma.dot(a, b)`           | `(vec, vec) -> scalar`     | scalar  | Dot product                     |
| `enigma.length(v)`           | `(vec) -> scalar`          | scalar  | Euclidean length                |
| `enigma.distance(a, b)`      | `(vec, vec) -> scalar`     | scalar  | Euclidean distance              |

### Vector-returning

| Function                              | Signature                        | Returns | Description                              |
|---------------------------------------|----------------------------------|---------|------------------------------------------|
| `enigma.normalize(v)`                | `(vec) -> vec`                   | vec     | Unit vector (v / length(v))              |
| `enigma.cross(a, b)`                 | `(vec3, vec3) -> vec3`           | vec3    | Cross product (3D only)                  |
| `enigma.reflect(incident, normal)`   | `(vec, vec) -> vec`              | vec     | Reflect incident about normal            |
| `enigma.refract(incident, normal, eta)` | `(vec, vec, scalar) -> vec`   | vec     | Refract with index of refraction eta     |
| `enigma.faceforward(n, i, nref)`     | `(vec, vec, vec) -> vec`         | vec     | Flip n if dot(i, nref) < 0              |

**Example — dot product kernel:**

```python
@enigma.kernel
def dot_kernel(Ax: enigma.f32, Ay: enigma.f32, Az: enigma.f32,
               Bx: enigma.f32, By: enigma.f32, Bz: enigma.f32,
               Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    a = enigma.make_float3(Ax[tid], Ay[tid], Az[tid])
    b = enigma.make_float3(Bx[tid], By[tid], Bz[tid])
    Out[tid] = enigma.dot(a, b)
```

---

## 19. Pack / Unpack Operations

Convert between vector float values and packed integer representations.
Used for color encoding, texture packing, etc.

### Pack (vector -> uint)

All pack functions take a vector `IRValue` and return an `IRValue` of dtype
`"uint"`.

| Function                                     | Input   | Description                            |
|----------------------------------------------|---------|----------------------------------------|
| `enigma.pack_float_to_snorm4x8(v)`          | float4  | Pack to signed normalized 4x8-bit      |
| `enigma.pack_float_to_unorm4x8(v)`          | float4  | Pack to unsigned normalized 4x8-bit    |
| `enigma.pack_float_to_snorm2x16(v)`         | float2  | Pack to signed normalized 2x16-bit     |
| `enigma.pack_float_to_unorm2x16(v)`         | float2  | Pack to unsigned normalized 2x16-bit   |
| `enigma.pack_float_to_srgb_unorm4x8(v)`     | float4  | Pack to sRGB unsigned normalized 4x8   |
| `enigma.pack_float_to_unorm10a2(v)`         | float4  | Pack to 10-10-10-2 unsigned normalized |

### Unpack (uint -> vector)

All unpack functions take an `IRValue` (uint) and return a vector `IRValue`.

| Function                                      | Output  | Description                            |
|-----------------------------------------------|---------|----------------------------------------|
| `enigma.unpack_snorm4x8_to_float(x)`         | float4  | Unpack signed normalized 4x8-bit       |
| `enigma.unpack_unorm4x8_to_float(x)`         | float4  | Unpack unsigned normalized 4x8-bit     |
| `enigma.unpack_snorm2x16_to_float(x)`        | float2  | Unpack signed normalized 2x16-bit      |
| `enigma.unpack_unorm2x16_to_float(x)`        | float2  | Unpack unsigned normalized 2x16-bit    |
| `enigma.unpack_srgb_unorm4x8_to_float(x)`    | float4  | Unpack sRGB unsigned normalized 4x8    |
| `enigma.unpack_unorm10a2_to_float(x)`        | float4  | Unpack 10-10-10-2 unsigned normalized  |

**Example — pack/unpack round-trip:**

```python
v = enigma.make_float4(r, g, b, a)
packed = enigma.pack_float_to_unorm4x8(v)
unpacked = enigma.unpack_unorm4x8_to_float(packed)
r_back = unpacked.x  # quantized to 8-bit precision
```

---

## 20. SIMD Group Operations

SIMD group (simdgroup, warp) operations perform reductions and communication
across the 32 threads in a SIMD group.

### Reductions

All take an `IRValue` and return an `IRValue` of the same dtype.

| Function                                          | Description                              |
|---------------------------------------------------|------------------------------------------|
| `enigma.simd_sum(x)`                             | Sum across SIMD group                    |
| `enigma.simd_product(x)`                         | Product across SIMD group                |
| `enigma.simd_min(x)`                             | Minimum across SIMD group                |
| `enigma.simd_max(x)`                             | Maximum across SIMD group                |
| `enigma.simd_and(x)`                             | Bitwise AND across SIMD group            |
| `enigma.simd_or(x)`                              | Bitwise OR across SIMD group             |
| `enigma.simd_xor(x)`                             | Bitwise XOR across SIMD group            |

### Prefix scans

| Function                                               | Description                              |
|--------------------------------------------------------|------------------------------------------|
| `enigma.simd_prefix_exclusive_sum(x)`                 | Exclusive prefix sum                     |
| `enigma.simd_prefix_inclusive_sum(x)`                  | Inclusive prefix sum                     |
| `enigma.simd_prefix_exclusive_product(x)`              | Exclusive prefix product                 |
| `enigma.simd_prefix_inclusive_product(x)`              | Inclusive prefix product                 |

### Shuffle

All shuffles take `(value, index_or_delta)` and return an `IRValue`.

| Function                                    | Description                                          |
|---------------------------------------------|------------------------------------------------------|
| `enigma.simd_shuffle(value, lane)`          | Read value from thread at absolute `lane`            |
| `enigma.simd_shuffle_up(value, delta)`      | Read from thread `(current_lane - delta)`            |
| `enigma.simd_shuffle_down(value, delta)`    | Read from thread `(current_lane + delta)`            |
| `enigma.simd_shuffle_xor(value, mask)`      | Read from thread `(current_lane XOR mask)`           |
| `enigma.simd_broadcast(value, lane)`        | Broadcast value from `lane` to all threads           |

---

## 21. Quad Group Operations

Quad groups are 4-thread groups (pixel quads). Same API pattern as SIMD
group ops but operate within quads of 4 threads.

### Reductions

| Function                                            | Description                        |
|-----------------------------------------------------|------------------------------------|
| `enigma.quad_sum(x)`                               | Sum across quad                    |
| `enigma.quad_product(x)`                           | Product across quad                |
| `enigma.quad_min(x)`                               | Minimum across quad                |
| `enigma.quad_max(x)`                               | Maximum across quad                |
| `enigma.quad_and(x)`                               | Bitwise AND across quad            |
| `enigma.quad_or(x)`                                | Bitwise OR across quad             |
| `enigma.quad_xor(x)`                               | Bitwise XOR across quad            |

### Prefix scans

| Function                                            | Description                        |
|-----------------------------------------------------|------------------------------------|
| `enigma.quad_prefix_exclusive_sum(x)`              | Exclusive prefix sum within quad   |
| `enigma.quad_prefix_inclusive_sum(x)`               | Inclusive prefix sum within quad   |

### Shuffle

| Function                                      | Description                                        |
|-----------------------------------------------|----------------------------------------------------|
| `enigma.quad_shuffle(value, lane)`            | Read from thread at absolute lane within quad      |
| `enigma.quad_shuffle_up(value, delta)`        | Read from thread (lane - delta) within quad        |
| `enigma.quad_shuffle_down(value, delta)`      | Read from thread (lane + delta) within quad        |
| `enigma.quad_shuffle_xor(value, mask)`        | Read from thread (lane XOR mask) within quad       |
| `enigma.quad_broadcast(value, lane)`          | Broadcast from lane to all quad threads            |

---

## 22. Barriers & Synchronization

### `enigma.barrier(mem_flags="mem_threadgroup") -> None`

Threadgroup memory barrier. All threads in the threadgroup must reach this
point before any can proceed.

| Parameter    | Type | Default              | Description                                      |
|--------------|------|----------------------|--------------------------------------------------|
| `mem_flags`  | str  | `"mem_threadgroup"`  | Memory fence scope. Options: `"mem_none"`, `"mem_device"`, `"mem_threadgroup"`, `"mem_device_and_threadgroup"`, `"mem_texture"` |

### `enigma.simd_barrier(mem_flags="mem_threadgroup") -> None`

SIMD group barrier. Synchronizes threads within a SIMD group.

Same `mem_flags` parameter as `enigma.barrier()`.

---

## 23. Threadgroup Shared Memory

### `enigma.threadgroup_alloc(dtype, size) -> TracingTensor`

Allocate threadgroup-shared memory (Metal `threadgroup T[size]`).

| Parameter | Type | Description                                    |
|-----------|------|------------------------------------------------|
| `dtype`   | str  | Element type: `"float"`, `"int"`, `"uint"`, etc. |
| `size`    | int  | Number of elements                             |

**Returns:** A `TracingTensor` that supports `[]` indexing, `.atomic_*` methods.

**Example — reverse array using shared memory:**

```python
@enigma.kernel
def reverse(A: enigma.f32, B: enigma.f32):
    tid = enigma.thread_position_in_grid
    local_id = enigma.thread_position_in_threadgroup()
    block_size = enigma.threads_per_threadgroup()

    shared = enigma.threadgroup_alloc("float", 256)
    shared[local_id] = A[tid]
    enigma.barrier()
    B[tid] = shared[block_size - local_id - 1]
```

---

## 24. Atomic Operations

Atomics work on both device buffers (kernel params) and threadgroup shared
memory (from `threadgroup_alloc`). Available as both free functions and
methods on buffer/shared-memory objects.

### Free-function style

| Function                                                               | Returns  | Description                                |
|------------------------------------------------------------------------|----------|--------------------------------------------|
| `enigma.atomic_load(buf, index, order="relaxed")`                     | IRValue  | Atomically load value at index             |
| `enigma.atomic_store(buf, index, value, order="relaxed")`             | None     | Atomically store value at index            |
| `enigma.atomic_exchange(buf, index, value, order="relaxed")`          | IRValue  | Atomically swap, return old value          |
| `enigma.atomic_fetch_add(buf, index, value, order="relaxed")`         | IRValue  | Atomic add, return old value               |
| `enigma.atomic_fetch_sub(buf, index, value, order="relaxed")`         | IRValue  | Atomic subtract, return old value          |
| `enigma.atomic_fetch_min(buf, index, value, order="relaxed")`         | IRValue  | Atomic min, return old value               |
| `enigma.atomic_fetch_max(buf, index, value, order="relaxed")`         | IRValue  | Atomic max, return old value               |
| `enigma.atomic_fetch_and(buf, index, value, order="relaxed")`         | IRValue  | Atomic AND, return old value               |
| `enigma.atomic_fetch_or(buf, index, value, order="relaxed")`          | IRValue  | Atomic OR, return old value                |
| `enigma.atomic_fetch_xor(buf, index, value, order="relaxed")`         | IRValue  | Atomic XOR, return old value               |
| `enigma.atomic_compare_exchange_weak(buf, index, expected, desired, ...)` | IRValue (i1) | CAS: returns true if exchange succeeded |

### Method style

All atomics are also available as methods on buffer objects:

```python
old = A.atomic_fetch_add(index, value)
A.atomic_store(index, value)
```

### Memory ordering

The `order` parameter accepts: `"relaxed"`, `"acquire"`, `"release"`, `"acq_rel"`.

For `atomic_compare_exchange_weak`, there are two order params:
`success_order` and `failure_order`.

**Example — atomic counter:**

```python
@enigma.kernel
def atomic_counter(Input: enigma.i32, Counter: enigma.i32):
    tid = enigma.thread_position_in_grid
    val = Input[tid]
    enigma.atomic_fetch_add(Counter, 0, val)
```

---

## 25. Simdgroup Matrix Operations

Hardware-accelerated 8x8 matrix operations using Metal's simdgroup matrix
units. These operate on `simdgroup_float8x8` matrices distributed across
the 32 threads of a SIMD group.

> **Note**: The MLIR path for these ops is fully functional. End-to-end Metal
> compilation requires a dialect MSL emitter update for simdgroup matrix type
> declarations.

### `enigma.simdgroup_matrix_load(buf, elements_per_row, elem="float", rows=8, cols=8) -> IRValue`

Load an 8x8 matrix tile from a device buffer into a simdgroup matrix register.

| Parameter          | Type          | Default    | Description                           |
|--------------------|---------------|------------|---------------------------------------|
| `buf`              | TracingTensor | —          | Source buffer (kernel parameter)      |
| `elements_per_row` | int           | —          | Stride between rows in the buffer     |
| `elem`             | str           | `"float"`  | Element type                          |
| `rows`             | int           | 8          | Number of rows                        |
| `cols`             | int           | 8          | Number of columns                     |

**Returns:** `IRValue` with simdgroup matrix dtype.

### `enigma.simdgroup_matrix_store(matrix, buf, elements_per_row) -> None`

Store a simdgroup matrix back to a device buffer.

| Parameter          | Type          | Description                           |
|--------------------|---------------|---------------------------------------|
| `matrix`           | IRValue       | Simdgroup matrix value to store       |
| `buf`              | TracingTensor | Destination buffer                    |
| `elements_per_row` | int           | Stride between rows in the buffer     |

### `enigma.simdgroup_multiply_accumulate(a, b, c) -> IRValue`

Matrix multiply-accumulate: `result = a * b + c`. All operands and result
are simdgroup matrices.

| Parameter | Type    | Description        |
|-----------|---------|--------------------|
| `a`       | IRValue | Left matrix (8x8)  |
| `b`       | IRValue | Right matrix (8x8) |
| `c`       | IRValue | Accumulator (8x8)  |

**Returns:** `IRValue` — result matrix (8x8).

### `enigma.make_filled_simdgroup_matrix(value, elem="float", rows=8, cols=8) -> IRValue`

Create a simdgroup matrix filled with a scalar value.

| Parameter | Type        | Default   | Description              |
|-----------|-------------|-----------|--------------------------|
| `value`   | IRValue/int | —         | Fill value               |
| `elem`    | str         | `"float"` | Element type             |
| `rows`    | int         | 8         | Number of rows           |
| `cols`    | int         | 8         | Number of columns        |

**Returns:** `IRValue` — filled simdgroup matrix.

**Example — simdgroup GEMM tile:**

```python
@enigma.kernel
def simd_gemm(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    a_mat = enigma.simdgroup_matrix_load(A, 8)
    b_mat = enigma.simdgroup_matrix_load(B, 8)
    zero = enigma.metal_cast(0, "float")
    c_mat = enigma.make_filled_simdgroup_matrix(zero)
    result = enigma.simdgroup_multiply_accumulate(a_mat, b_mat, c_mat)
    enigma.simdgroup_matrix_store(result, C, 8)
```

---

## 26. Matrix Operations

Regular matrix operations on Metal matrix types (float4x4, etc.). These are
modeled as multi-dimensional vector types in MLIR.

> **Note**: These ops require a matrix constructor (blocked on dialect-side
> `mat_make` op). See `docs/blocked-features.md` for status.

| Function                                   | Description                                |
|--------------------------------------------|--------------------------------------------|
| `enigma.matmul(a, b, result_dtype=None)`   | Matrix multiply: `a * b`                   |
| `enigma.transpose(m, result_dtype=None)`   | Matrix transpose                           |
| `enigma.determinant(m, scalar_dtype=None)` | Matrix determinant (returns scalar)        |

---

## 27. Function Constants

Metal specialization constants bound at pipeline creation time.

### `enigma.function_constant(dtype, index) -> IRValue`

| Parameter | Type | Description                                       |
|-----------|------|---------------------------------------------------|
| `dtype`   | str  | Value type: `"float"`, `"int"`, `"uint"`, `"bool"` |
| `index`   | int  | Function constant index (matches pipeline config) |

**Returns:** `IRValue` with the specified dtype.

> **Note**: The dialect's MSL emitter currently places the declaration inside
> the kernel body instead of file scope. See `docs/blocked-features.md`.

---

## 28. Layout Algebra (CuTe-style)

Enigma includes a CuTe-style layout algebra for tiling and memory layout
transformations. A `Layout` is a `(Shape, Stride)` pair that maps
multi-dimensional coordinates to linear memory offsets.

### `enigma.Layout(shape, stride=None)`

Create a layout. If stride is omitted, a column-major compact stride is used.

```python
L = enigma.Layout((4, 8), (1, 4))   # 4x8, column-major
offset = L((2, 3))                    # -> 2*1 + 3*4 = 14
```

**Properties and methods:**

| Method / Property | Returns | Description                          |
|-------------------|---------|--------------------------------------|
| `L(coord)`        | int     | Map coordinate to linear offset      |
| `L.shape`         | tuple   | Shape of the layout                  |
| `L.stride`        | tuple   | Stride of the layout                 |
| `L.size(mode=None)` | int  | Total number of elements (or per-mode) |
| `L.rank()`        | int     | Number of modes (dimensions)         |
| `L.depth()`       | int     | Nesting depth of the shape           |
| `L.cosize()`      | int     | Maximum offset + 1                   |

### Layout constructors

| Function                                          | Description                                        |
|---------------------------------------------------|----------------------------------------------------|
| `enigma.make_layout(shape, stride=None)`          | Create a layout (same as `Layout(shape, stride)`)  |
| `enigma.make_ordered_layout(shape, order)`        | Layout with custom dimension ordering              |
| `enigma.make_identity_layout(shape)`              | Identity (column-major) layout                     |

**`make_ordered_layout` parameters:**

- **shape**: Tuple of dimension sizes
- **order**: Tuple of ints giving priority per dim (`0` = innermost/fastest)

```python
# Thread layout: 4 threads in dim-0, 64 threads in dim-1.
# order=(1, 0): dim-1 varies fastest (row-major).
thr = enigma.make_ordered_layout((4, 64), order=(1, 0))
```

### Layout operations

| Function                                    | Returns  | Description                                               |
|---------------------------------------------|----------|-----------------------------------------------------------|
| `enigma.size(x, mode=None)`                | int      | Size of a layout, tensor, or shape                        |
| `enigma.coalesce(layout)`                  | Layout   | Flatten and merge modes with compatible strides            |
| `enigma.complement(layout, cosize=None)`   | Layout   | Complementary layout covering elements not in layout      |
| `enigma.composition(a, b)`                 | Layout   | Compose layouts: `(a . b)(c) = a(b(c))`                  |
| `enigma.logical_divide(layout, tiler)`     | Layout   | Split layout into (tile, rest)                            |
| `enigma.zipped_divide(layout, tiler)`      | Layout   | Per-mode divide into ((tile_modes), (rest_modes))         |
| `enigma.blocked_product(a, b)`             | Layout   | Blocked product: each element of b gets a full copy of a  |
| `enigma.recast_layout(new_bits, old_bits, layout)` | Layout | Rescale layout for different element bit-widths    |

### Thread-Value (TV) Layout

| Function                                            | Returns              | Description                                      |
|-----------------------------------------------------|----------------------|--------------------------------------------------|
| `enigma.make_layout_tv(thr_layout, val_layout)`    | (tiler, tv_layout)   | Build TV layout from thread and value layouts    |

**Returns a tuple:**
- `tiler` — Shape tuple for tiling the global data
- `tv_layout` — Layout mapping `(thread_id, value_id)` to tile offsets

```python
thr_layout = enigma.make_ordered_layout((4, 64), order=(1, 0))
val_layout = enigma.make_ordered_layout((4, 4), order=(1, 0))
tiler, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)
```

---

## 29. Tensor & TV Layout

### `enigma.Tensor(name, buffer_index, metal_dtype, layout, base_offset=0)`

A tensor binds a buffer name to a layout. Used in the `@enigma.jit` path
for TV-layout tiled kernels.

| Parameter      | Type     | Description                                |
|----------------|----------|--------------------------------------------|
| `name`         | str      | Buffer name (e.g. `"A"`)                   |
| `buffer_index` | int      | Buffer binding index                       |
| `metal_dtype`  | str      | Element type (e.g. `"float"`)              |
| `layout`       | Layout   | Layout describing shape and stride         |
| `base_offset`  | int      | Starting offset into the buffer            |

**Methods:**

| Method                  | Returns  | Description                                      |
|-------------------------|----------|--------------------------------------------------|
| `tensor[coord]`         | Tensor   | Slice: `None` keeps a mode, int/IRValue fixes it |
| `tensor.load()`         | IRValue  | Vectorized load of all elements in the view      |
| `tensor.store(value)`   | None     | Vectorized store of all elements                 |
| `tensor.size(mode=None)` | int    | Number of elements                               |

### Tensor operations

| Function                                         | Returns | Description                                    |
|--------------------------------------------------|---------|------------------------------------------------|
| `enigma.tensor_composition(tensor, tv_layout, tiler)` | Tensor | Compose tensor with a TV layout            |
| `enigma.tensor_zipped_divide(tensor, tiler)`     | Tensor  | Tile a tensor using zipped divide              |
| `enigma.make_identity_tensor(shape)`             | Tensor  | Identity tensor for debugging                  |

---

## 30. JIT Path (@enigma.jit)

The `@enigma.jit` decorator is for host-side functions that perform layout
algebra and launch tiled kernels. It works with `Tensor` objects and the
TV-layout system.

### `@enigma.jit`

```python
@enigma.jit
def launch(mA, mB, mC):
    thr_layout = enigma.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = enigma.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)

    gA = enigma.tensor_zipped_divide(mA, tiler_mn)
    gB = enigma.tensor_zipped_divide(mB, tiler_mn)
    gC = enigma.tensor_zipped_divide(mC, tiler_mn)

    num_blocks = enigma.size(gA, mode=[1])
    threads = enigma.size(tv_layout, mode=[0])

    add_tv(gA, gB, gC, tv_layout, tiler_mn).launch(
        grid=(num_blocks * threads, 1, 1),
        block=(threads, 1, 1),
    )
```

**Calling `@kernel` inside `@jit`** returns a `KernelHandle` with a
`.launch(grid, block)` method. The kernel function receives `Tensor` objects
and uses `.load()` / `.store()` for vectorized access.

**Compiling:**

```python
M, N = 256, 512
mA = enigma.Tensor("A", 0, "float", enigma.Layout((M, N), (N, 1)))
mB = enigma.Tensor("B", 1, "float", enigma.Layout((M, N), (N, 1)))
mC = enigma.Tensor("C", 2, "float", enigma.Layout((M, N), (N, 1)))
compiled = enigma.compile(launch, mA, mB, mC)
```

---

## 31. arch Namespace

The `enigma.arch` namespace provides CUDA-style convenience accessors.

| Method                   | Returns           | Metal equivalent                        |
|--------------------------|-------------------|-----------------------------------------|
| `enigma.arch.thread_idx()` | (IRValue, 0, 0) | `thread_position_in_threadgroup.x`      |
| `enigma.arch.block_idx()`  | (IRValue, 0, 0) | `threadgroup_position_in_grid.x`        |
| `enigma.arch.block_dim()`  | (IRValue, 0, 0) | `threads_per_threadgroup.x`             |

Returns a 3-tuple where only the first element (x dimension) is an `IRValue`;
y and z are always `0`.

```python
@enigma.kernel
def k(A: enigma.f32, B: enigma.f32):
    tidx, _, _ = enigma.arch.thread_idx()
    bidx, _, _ = enigma.arch.block_idx()
    bdim, _, _ = enigma.arch.block_dim()
    gid = bidx * bdim + tidx
    B[gid] = A[gid]
```

---

## 32. Tuple Utilities

Helper functions re-exported from `enigma.tuple`.

| Function                       | Returns | Description                                           |
|--------------------------------|---------|-------------------------------------------------------|
| `enigma.product(x)`           | int     | Product of all elements in a (possibly nested) tuple  |
| `enigma.repeat_like(ref, val)` | tuple  | Create a tuple with the same structure as `ref`, filled with `val` |
| `enigma.select(x, idx)`       | any     | Select element(s) from a nested tuple by index        |

> **Note**: `enigma.where` (the conditional select op from [Section 15](#15-select--conditional))
> is internally named `select` in `_tracing.py` but re-exported as `enigma.where` to avoid
> colliding with `enigma.select` (the tuple utility).

---

## 33. Errors

### `enigma.EnigmaError`

Structured error raised by the DSL tracer and emitter. Subclass of
`Exception`.

Raised when:
- An op is used outside a `@enigma.kernel` / `@enigma.jit` context
- Invalid dimension string (not `"x"`, `"y"`, or `"z"`)
- Type mismatches in vector construction

```python
try:
    enigma.compile(my_kernel)
except enigma.EnigmaError as e:
    print(f"DSL error: {e}")
```

`RuntimeError` is raised for:
- Metal compilation failures (`xcrun metal` / `xcrun metallib`)
- GPU dispatch failures (no Metal device, command buffer errors)
- Missing Xcode Command Line Tools

---

## 34. Async Copy (Experimental)

Async device-threadgroup copy via AIR simdgroup intrinsics. Requires M3+ hardware.

> **Status**: Experimental. Uses `__asm("air.simdgroup_async_copy_*")` extern declarations to bind to undocumented AIR intrinsics (same technique as [percisely.xyz/gemm](https://percisely.xyz/gemm)). Apple may change or remove access to these intrinsics at any time.

### Functions

| Function | Description |
|---|---|
| `enigma.async_copy_1d_d2t(dst, dst_offset, src, src_offset, count)` | 1D async copy: device → threadgroup. Returns event. |
| `enigma.async_copy_1d_t2d(dst, dst_offset, src, src_offset, count)` | 1D async copy: threadgroup → device. Returns event. |
| `enigma.async_copy_2d_d2t(dst, dst_off, dst_epr, src, src_off, src_epr, tile_cols, tile_rows)` | 2D tile async copy: device → threadgroup. Returns event. |
| `enigma.async_copy_2d_t2d(dst, dst_off, dst_epr, src, src_off, src_epr, tile_cols, tile_rows)` | 2D tile async copy: threadgroup → device. Returns event. |
| `enigma.async_copy_wait(*events)` | Block until listed async-copy events complete. |

### Parameters

- **dst, src**: `Tensor`, `RegisterTensor`, or buffer name string
- **dst_offset, src_offset**: Element offset into the buffer
- **count**: Number of elements (1D)
- **dst_epr, src_epr**: Elements per row (2D stride)
- **tile_cols, tile_rows**: Tile dimensions (2D)
- **events**: Event handles returned by the copy functions

### Example

```python
@enigma.kernel
def tiled_load(A: enigma.f32, B: enigma.f32):
    tile = enigma.threadgroup_alloc("float", 64)
    c0 = enigma.metal_cast(0, "uint")
    cnt = enigma.metal_cast(64, "uint")

    ev = enigma.async_copy_1d_d2t(tile, c0, A, c0, cnt)
    enigma.async_copy_wait(ev)
    enigma.barrier()

    # 2D variant
    epr = enigma.metal_cast(8, "uint")
    ev2 = enigma.async_copy_2d_d2t(tile, c0, epr, A, c0, cnt,
                                    enigma.metal_cast(8, "uint"),
                                    enigma.metal_cast(8, "uint"))
    enigma.async_copy_wait(ev2)
```

### Legacy aliases

| Function | Maps to |
|---|---|
| `enigma.async_copy_to_threadgroup(src, dst, count, src_offset, dst_offset)` | `async_copy_1d_d2t` |
| `enigma.async_copy_commit(token)` | No-op (commit is implicit) |

---

## 35. Control Flow

Structured control flow inside `@enigma.kernel` bodies. Each helper is a
**context manager** that traces a region of MLIR ops rather than running
Python control flow — the body executes once during tracing to record the
loop/branch structure, and the recorded structure is what reaches the
GPU.

### `enigma.for_range(lo, hi, step=1, *, dtype="int", init=None)`

`scf.for`-style counted loop. Use `with ... as i` to bind the induction
variable. `lo`, `hi`, `step` may be Python ints or `IRValue`s produced
by `enigma.metal_cast(...)`; mixed types are coerced to `dtype`. The
`init` argument turns the loop into a reduction with carried values
(see "Loop-carried values" below).

```python
@enigma.kernel
def stencil(A: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    with enigma.for_range(0, 4) as i:
        Out[tid + i] = A[tid + i]
```

Variable bounds:

```python
lo   = enigma.metal_cast(0, "int")
hi   = enigma.metal_cast(10, "int")
step = enigma.metal_cast(2, "int")
with enigma.for_range(lo, hi, step) as i:
    ...
```

### `enigma.range_constexpr(*args)`

Python-side `range()` for **compile-time unrolling**. Iterates eagerly
during tracing — every iteration emits a fresh copy of the body ops with
the loop counter inlined as a constant. Use when the iteration count is
known at compile time and you want the loop fully unrolled. Same call
signature as `range()`:

```python
@enigma.kernel
def unrolled(A: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    for h in enigma.range_constexpr(4):   # 4 copies of the body emitted
        Out[tid + h] = enigma.metal_cast(float(h), "float")
```

Compare to `enigma.for_range`, which produces a real `scf.for` loop the
hardware steps through.

### `enigma.if_(condition)`

Single-armed or two-armed conditional, emitting `scf.if`. `condition`
must be a boolean `IRValue` (produced by `cmp_*` or any of the float
predicate ops in §11).

Single-arm:

```python
cond = enigma.cmp_gt(tid, 5)
with enigma.if_(cond):
    Out[tid] = A[tid]
```

Two-arm (then / else):

```python
val = A[tid]
cond = enigma.cmp_gt(val, enigma.metal_cast(0, "float"))
with enigma.if_(cond) as (then_b, else_b):
    with then_b:
        Out[tid] = val
    with else_b:
        Out[tid] = enigma.metal_cast(0, "float")
```

### `enigma.while_(cond_fn)`

Unbounded loop emitting `scf.while`. `cond_fn` is a **callable**
(typically a lambda) that re-evaluates the predicate at the top of every
iteration during tracing — this lets the condition reference IR values
defined inside the loop.

```python
i = enigma.metal_cast(0, "int")
n = enigma.metal_cast(8, "int")
with enigma.while_(lambda: enigma.cmp_lt(i, n)):
    _ = A[tid]
```

Nesting `if_` and `for_range` inside `while_` is supported.

### Loop-carried values

`enigma.for_range(lo, hi, step, init=initial_value)` turns the loop into
a reduction. The init value is bound to the body region's last argument,
and the value yielded at the end of the body becomes the next iteration's
carry. See `tests/portable/test_control_flow.py::TestCarry` for the full
trace shape.

---

## 36. Masked Load / Store

Predicated memory ops for tail-iteration handling. Both helpers accept
a `Tensor`, an index, and a boolean mask `IRValue`.

### `enigma.load_if(buf, index, mask, default=0) -> IRValue`

Conditional load — equivalent to `mask ? buf[index] : default`, but
emitted as a single `select` op (not an `scf.if`) so it stays in a
straight-line region and benefits from compiler vectorization.

```python
ten  = enigma.metal_cast(10, "int")
mask = enigma.cmp_lt(tid, ten)
val  = enigma.load_if(A, tid, mask, default=0.0)
```

### `enigma.store_if(buf, index, value, mask) -> None`

Conditional store — emits the store inside an `scf.if` region so
out-of-bounds threads truly skip the write (important when the buffer
extends past the valid range, where a `select`-style approach would
still read past the end).

```python
mask = enigma.cmp_lt(tid, ten)
val  = A[tid]
enigma.store_if(B, tid, val, mask)
```

Use `load_if` for vectorizable reads, `store_if` for safe writes.

### `enigma.register_tensor(shape, dtype="float", fill=0) -> RegisterTensor`

A small statically-shaped tensor that lives in registers (not threadgroup
memory). Indexing with **constant** tuples is fully unrolled at trace
time — no `scf.for` is emitted, no threadgroup memory is allocated.

```python
reg = enigma.register_tensor((2, 2), dtype="float", fill=0.0)
reg[0, 0] = A[tid]
v = reg[0, 0]
```

Useful for per-thread accumulators in tiled kernels (e.g. the `C`
fragment of a `gemm` tile).

---

## 37. High-Level Tile Helpers

These emit larger chunks of IR than the primitive ops — they're recipes
expressed in the DSL, useful when you don't want to write the tile-level
choreography by hand.

### `enigma.gemm(A, B, C, *, M, N, K, transpose_A=False, transpose_B=False, accum_dtype="float", use_simdgroup=None) -> None`

GEMM tile: `C += A @ B` over the threadgroup-resident tiles `A`, `B`,
`C`. Picks one of two emission paths:

| Path | When picked | What it emits |
|---|---|---|
| **Simdgroup** | `M==N==K==8` and `use_simdgroup` is `True`/`None` | `simdgroup_matrix_load`/`multiply_accumulate`/`store` ops + a `threadgroup_barrier` |
| **Scalar** | Any other tile size, or `use_simdgroup=False` | A doubly-nested `scf.for` over (m, n) with per-cell scalar accumulation |

Force a specific path with `use_simdgroup=True` (raises `EnigmaError`
on invalid shape) or `use_simdgroup=False`.

```python
@enigma.kernel
def matmul_tile(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    # A, B come from threadgroup memory; C is a register tile
    reg_C = enigma.register_tensor((3, 4), dtype="float", fill=0.0)
    enigma.gemm(A, B, reg_C, M=3, N=4, K=2)
```

### `enigma.copy(src, dst, count, *, src_offset=0, dst_offset=0, mask_fn=None, coalesced_width=1) -> None`

Bulk copy from one tensor to another over `count` elements. Wraps a
single `scf.for` loop.

- `coalesced_width=k` unrolls `k` consecutive lane copies inside each
  iteration so the inner body has `k` loads followed by `k` stores —
  the Metal compiler then vectorizes them into a coalesced transaction.
  `count` must be divisible by `k`.
- `mask_fn(i) -> mask_IRValue` is called for each lane to gate the
  per-lane copy (uses `load_if`/`store_if` internally).

```python
# Plain copy, one element per iteration:
enigma.copy(A, B, count=16)

# Coalesced 4-wide copy, count must be a multiple of 4:
enigma.copy(A, B, count=16, coalesced_width=4)
```

### `enigma.pipeline(dtype, size, stages=2) -> Pipeline`

Multi-stage threadgroup-memory pipeline for software-pipelining double-
or triple-buffered loads.

```python
pipe = enigma.pipeline("float", 8, stages=3)

front = pipe.front()    # current consumer buffer
mid   = pipe.stage(1)
back  = pipe.back()     # current producer buffer

pipe.swap()    # 2-stage only: rotates front <-> back
pipe.rotate()  # N-stage: rotate all buffers by one position
```

Calling `swap()` on a `stages != 2` pipeline raises
`enigma.EnigmaError`. `stages < 2` raises at construction.

---

## 38. Quantization Helpers

In addition to the `pack_*` / `unpack_*` ops in §19, the DSL ships a
small set of integer-quantization helpers.

### `enigma.dequantize_int8(x, scale, zero_point=0) -> IRValue`

Computes `(x - zero_point) * scale` and casts the result to `float`.
`x` is an int IRValue, `scale` is a float IRValue, `zero_point` may be
an int IRValue or a Python int (default 0).

```python
x          = enigma.metal_cast(64, "int")
scale      = enigma.metal_cast(0.0625, "float")
zero_point = enigma.metal_cast(0, "int")
y_float    = enigma.dequantize_int8(x, scale, zero_point=zero_point)
```

Emitted as a sequence of `sub` / `metal_cast` / `mul` ops.

The pack/unpack ops in §19 (`pack_uint8x4`, `unpack_uint8x4`,
`pack_int4x2`, `unpack_int4x2`) combine with `dequantize_int8` to
implement weight unpacking in quantized GEMM kernels.

---

## 39. Tooling: `enigma.testing` and `enigma.benchmark`

### `enigma.testing` — environment-gated test decorators

Helpers for writing tests that may or may not run on the current host
(e.g. CI sandboxes without Metal, non-Apple-Silicon developer boxes).

| Symbol | Returns | Purpose |
|---|---|---|
| `enigma.testing.is_darwin()` | `bool` | True if running on macOS |
| `enigma.testing.is_apple_silicon()` | `bool` | True if running on arm64 macOS |
| `enigma.testing.is_metal_available()` | `bool` | True if a Metal device can be opened |
| `enigma.testing.requires_metal(fn)` | decorator | Skips the wrapped test if Metal isn't available |
| `enigma.testing.requires_apple_silicon(fn)` | decorator | Skips the wrapped test on non-Apple-Silicon hosts |
| `enigma.testing.skip_if_no_metal` | alias of `requires_metal` | Convenience alias |

```python
import unittest, enigma

class TestKernel(unittest.TestCase):
    @enigma.testing.requires_metal
    def test_dispatch(self):
        ...   # only runs if Metal is available
```

### `enigma.benchmark` — micro-benchmarks for kernels

A lightweight timing harness. CPU-side benchmarks use `time.perf_counter`;
GPU-side benchmarks measure end-to-end dispatch wall-clock.

#### `enigma.benchmark.bench(fn, *args, repeat=50, warmup=5, label=None, **kwargs) -> BenchResult`

Measures a Python callable. Runs `warmup` discarded iterations, then
`repeat` timed iterations.

```python
from enigma import benchmark

def workload():
    ...

r = benchmark.bench(workload, repeat=10, warmup=2, label="my-op")
print(benchmark.format_bench_result(r))
# my-op: median=12.3us min=10.1us max=18.4us n=10
```

#### `enigma.benchmark.bench_gpu(prepared, grid, threads, *, repeat=50, warmup=5, label=None) -> BenchResult`

Specialized form for kernels: takes a `PreparedKernel` (see §5) and a
`grid`/`threads` shape, dispatches `repeat+warmup` times, returns timing
statistics.

```python
prepared = runtime.prepare(compiled, inputs=[A, B], output_size=N * 4)
r = benchmark.bench_gpu(prepared, grid=(N, 1, 1), threads=(256, 1, 1),
                        repeat=50, warmup=5, label="vector_add")
```

#### `BenchResult` dataclass

| Field | Type | Notes |
|---|---|---|
| `label` | `str` | The label you passed (or `""`) |
| `samples_us` | `list[float]` | Per-iteration wall-clock in microseconds |
| `warmup` | `int` | How many warmup runs were discarded |
| `n` (property) | `int` | `len(samples_us)` |
| `median_us` (property) | `float` | Median of `samples_us` |
| `min_us`, `max_us` (properties) | `float` | Extremes |

#### `enigma.benchmark.format_bench_result(r, *, throughput_gbps=None) -> str`

Pretty-prints a `BenchResult` as a single line. Pass `throughput_gbps`
to also include a computed `GB/s` figure.

---

## 40. Compiler: emit-only mode

For trace-and-emit workflows without going all the way to GPU dispatch
(useful in CI, for documentation, or when you want to inspect the MSL
the DSL generates):

```python
compiled = enigma.compile(my_kernel, emit_only=True)

assert compiled.kernel_source == compiled.metal_source   # alias
assert compiled.mlir_source is not None
assert not compiled.has_metallib
assert compiled.metallib_path is None
assert compiled.metallib_bytes is None
```

`emit_only=True` skips the `xcrun metal` / `xcrun metallib` invocations,
so it works on any host (no Xcode Command Line Tools required). The
returned `CompiledKernel` has `has_metallib == False` and the metallib
fields are `None`; everything else (MLIR source, generated MSL,
emit-only attributes) is populated as usual.

The environment variable `ENIGMA_EMIT_ONLY=1` toggles the same behavior
project-wide without changing call sites — handy for CI matrices that
include non-macOS runners.

---

## Complete Example — Matrix Multiply

A full matmul kernel using 2D grid indexing and float4 dot products:

```python
import numpy as np
import enigma

N_DIM = 64
K_DIM = 4

@enigma.kernel
def matmul(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    row = enigma.thread_position_in_grid_xyz("y")
    col = enigma.thread_position_in_grid_xyz("x")
    n = enigma.metal_cast(N_DIM, "uint")
    k = enigma.metal_cast(K_DIM, "uint")
    base_a = row * k
    base_b = col

    a0, a1 = A[base_a], A[base_a + 1]
    a2, a3 = A[base_a + 2], A[base_a + 3]
    b0, b1 = B[base_b], B[base_b + n]
    b2, b3 = B[base_b + n * 2], B[base_b + n * 3]

    avec = enigma.make_float4(a0, a1, a2, a3)
    bvec = enigma.make_float4(b0, b1, b2, b3)
    C[row * n + col] = enigma.dot(avec, bvec)

M = 32
compiled = enigma.compile(matmul)
rt = enigma.MetalRuntime()
A = np.random.randn(M, K_DIM).astype(np.float32)
B = np.random.randn(K_DIM, N_DIM).astype(np.float32)
out = np.frombuffer(
    rt.execute(compiled, [A.ravel(), B.ravel()], M * N_DIM * 4,
               grid=(N_DIM, M, 1), threads=(16, 16, 1)),
    dtype=np.float32,
).reshape(M, N_DIM)
np.testing.assert_allclose(out, A @ B, rtol=1e-4)
```
