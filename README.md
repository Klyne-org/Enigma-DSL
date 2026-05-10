<p align="center">
  <img src="https://cdn8.futura-sciences.com/a1280/images/actu/enigma-mer-baltique.jpeg" width="600" />
</p>

<h1 align="center"><code>E N I G M A</code></h1>

<p align="center">
  <sub>where Python meets Metal, and layouts become algebra</sub>
</p>

---

Enigma is a Python DSL for writing Apple Metal GPU compute kernels. You write
a Python function decorated with `@enigma.kernel`; Enigma traces it into MLIR,
emits Metal Shading Language, compiles through `xcrun metal`, and dispatches
on the GPU through a Swift runtime. The DSL surface is small. The generated
kernels run at hand-written-Metal bandwidth.

Inspired by NVIDIA's [CuTe DSL](https://github.com/NVIDIA/cutlass) — Enigma
brings the same layout-algebra and tiling calculus to Apple Silicon. CuTe
targets warps and tensor cores; Enigma targets simdgroups and threadgroups.

## Pipeline

```
Python @enigma.kernel  →  traced IR  →  MLIR (enigma dialect)  →  MSL
                                                                    ↓
                                                   xcrun metal → AIR
                                                                    ↓
                                              xcrun metallib → .metallib
                                                                    ↓
                                              Swift runtime (ctypes) → GPU
```

For tiled kernels, `@enigma.jit` runs CuTe-style layout algebra host-side
before launching: composition, complement, coalesce, zipped divide, and the
`make_layout_tv` constructor that maps thread × value indices to tile
coordinates with correct coalescing order. The entire vectorisation strategy
is decided at trace time; only the resulting offsets and memory transactions
reach the GPU.

## Install

Requirements: Apple Silicon Mac, Xcode Command Line Tools, Python 3.11 / 3.12 / 3.13.

Pre-built wheels for the dialect (Python ABI-specific) and the DSL (pure
Python) are shipped in the `wheelhouse/` directory. The fastest path:

```bash
git clone https://github.com/Klyne-Research/Enigma-DSL.git
cd Enigma-DSL
python3.12 -m venv .venv && source .venv/bin/activate
pip install wheelhouse/enigma_dsl-*-py3-none-any.whl \
            wheelhouse/enigma_dialect-*-cp312-cp312-*.whl
pip install numpy
python examples/vector_add.py
```

Pick the `enigma_dialect-*-cpXY-*` wheel that matches your Python ABI
(`cp311`, `cp312`, or `cp313`). The DSL wheel is the same `py3-none-any.whl`
for every Python version.

## Hello, Metal

```python
import numpy as np
import enigma

@enigma.kernel
def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]

compiled = enigma.compile(vector_add)
print(compiled.metal_source)   # generated MSL — readable, debuggable

N = 1024
a, b = np.random.randn(N).astype(np.float32), np.random.randn(N).astype(np.float32)

runtime = enigma.MetalRuntime()
raw = runtime.execute(compiled, inputs=[a, b], output_size=N * 4,
                      grid=(N, 1, 1), threads=(256, 1, 1))
c = np.frombuffer(raw, dtype=np.float32)
assert np.allclose(c, a + b)
```

## Showcase kernels

The `examples/` directory has four end-to-end showcase kernels. Each ships
in three forms: the Enigma DSL version, the equivalent handwritten Metal
shader for comparison, and (where applicable) a benchmark harness that
times both.

| Kernel | DSL | Handwritten | What it shows |
|---|---|---|---|
| **Vector add** | `vector_add.py`, `vector_add_tv.py` | `vector_add_naive.metal`, `vector_add_float4.metal`, `add_kernel_tv.metal` | The hello-world. The TV variant uses `@enigma.jit` + layout algebra to choose vectorisation. |
| **RMSNorm** | `benchmark_rmsnorm.py` | `rmsnorm_handwritten.metal` | Reduction across a row, threadgroup shared memory, simd_sum. The benchmark times Enigma vs handwritten side-by-side. |
| **SDPA / FlashAttention** | `benchmark_sdpa.py`, `showcase_attention.py` | `sdpa_handwritten.metal` | Fused attention forward — online softmax, multi-simdgroup tiling, threadgroup reductions. |
| **1D Laplacian** | `conv1d_laplacian.py` | `conv1d_laplacian_handwritten.metal` | Finite-difference stencil for PDE solvers (heat eq., diffusion). Boundary handling via `enigma.if_`. |

Run any of them directly:

```bash
python examples/conv1d_laplacian.py     # prints generated MSL + numpy diff
python examples/benchmark_rmsnorm.py    # benchmarks Enigma vs handwritten
python examples/showcase_attention.py   # FlashAttention forward
```

## What's in the box

- **Kernel surface**: arithmetic, unary/binary/ternary float math, integer
  intrinsics (popcount, clz, mulhi, …), vector ops (`make_float4`, `dot`,
  `cross`, `length`, …), pack/unpack ops, comparisons, `select`/`where`,
  `if_`, `for_range`, casts, atomics with explicit memory order.
- **GPU-specific**: thread/threadgroup/grid queries, simdgroup reductions
  and shuffles, quad-group ops, threadgroup shared memory and barriers,
  simdgroup matrix ops on Apple's hardware matrix units.
- **Layout algebra (CuTe-style)**: `Layout`, `Shape`, `Stride` with
  composition, complement, coalesce, recast, zipped divide. The
  `make_layout_tv` constructor builds Thread × Value layouts.
- **Two compilation paths**: `@enigma.kernel` (raw, you set the grid) and
  `@enigma.jit` (layout-driven, the engine sets the grid).
- **Compatibility**: tested on Apple M-series GPUs through
  `xcrun metal` / `xcrun metallib`. Float32, float16, bfloat16, integer
  widths 8–64.

## Documentation

- [`docs/api-reference.md`](docs/api-reference.md) — exhaustive op-by-op
  reference (30 sections, every primitive).
- [`Enigma-Dialect/`](Enigma-Dialect/) — submodule with the C++/MLIR dialect
  definition, MSL emitter, and dialect-level lit tests.

## Testing

```bash
python -m pytest tests/                              # full Python suite
bash Enigma-Dialect/test/run_tests.sh                # MLIR/lit suite
bash Enigma-Dialect/test/run_tests.sh --gpu          # plus GPU dispatch
```

## License

MIT. See [`LICENSE`](LICENSE).

## Versions

**v0.1.0** — initial release. Layout algebra engine (composition,
complement, coalesce, zipped divide, recast, TV layout construction).
Tracing IR with SSA values, constant folding, thread index decomposition.
Metal emitter supporting scalar, float4 vector pointer, and TV-layout
vectorised codegen. Swift runtime with device management, buffer
allocation, synchronous dispatch, GPU timestamp measurement. Dialect
TableGen definitions covering thread indexing, synchronisation, math,
atomics, simdgroup, quad, geometry, pack/unpack, and matrix ops.
