<p align="center">
  <img src="https://cdn8.futura-sciences.com/a1280/images/actu/enigma-mer-baltique.jpeg" width="600" />
</p>

<h1 align="center"><code>E N I G M A</code></h1>

<p align="center">
  <sub>where Python meets Metal, and layouts become algebra</sub>
</p>

<p align="center">
  <a href="https://klyne-research.mintlify.app/">📖 Documentation</a> &nbsp;·&nbsp;
  <a href="https://pypi.org/project/enigma-dsl/">📦 PyPI</a> &nbsp;·&nbsp;
  <a href="https://github.com/Klyne-org/Enigma-DSL">⭐ GitHub</a>
</p>

---

In 1945, an Enigma machine sank to the floor of the Baltic Sea. For decades it sat there, its rotors locked, its wiring intact, waiting. When divers finally pulled it from the silt, the mechanism still worked. The genius was never in the shell. It was in the rotors, the wiring, the algebra of permutations hidden inside.

Enigma DSL is built on the same principle. Inspired by NVIDIA's CuTe DSL, which brought layout algebra and tiling calculus to CUDA, Enigma brings the same mathematical framework to Apple Metal. Where CuTe targets tensor cores and warps on NVIDIA GPUs, Enigma targets simdgroups and threadgroups on Apple Silicon. The layout algebra is the same. The target is different. You write a Python function. Underneath, the algebra computes how threads map to memory, how tiles partition a tensor, how values flow through a simdgroup. The Python traces into an IR. The IR emits Metal C++. The Metal compiles to GPU machine code. Your function runs on Apple Silicon at hardware bandwidth limits. The surface is clean. The machinery is exact.

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

Requirements: Apple Silicon Mac (M1 through M5), macOS 14 / 15 (or any
newer release), Python 3.11 / 3.12 / 3.13.

```bash
pip install enigma-dsl
```

That single command pulls a self-contained wheel that bundles the Python
DSL **and** the native MLIR dialect (libLLVM, libMLIRPythonCAPI, the
Enigma dialect `.so`) — no separate steps, no LLVM toolchain on your
machine. `pip` picks the right wheel for your Python version and macOS
version automatically.

The release ships **six** wheels: 3 Python versions × 2 macOS deployment
targets (14.0 and 15.0). macOS 14-tagged wheels run on macOS 14, 15, 26
and every future version; macOS 15-tagged wheels are picked first on
macOS 15+ hosts because pip prefers the most specific match.

Quick verification:

```bash
python -c "import enigma; print(enigma.__version__)"
# 0.1.1
```

### 30-second example

Save as `add.py` and run with `python add.py` — no setup beyond
`pip install enigma-dsl numpy`. This is the shortest end-to-end
trace → MSL codegen → GPU dispatch you can write.

```python
import numpy as np
import enigma

@enigma.kernel
def add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    i = enigma.thread_position_in_grid
    C[i] = A[i] + B[i]

compiled = enigma.compile(add)

N = 1024
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)

raw = enigma.MetalRuntime().execute(
    compiled, inputs=[a, b], output_size=N * 4,
    grid=(N, 1, 1), threads=(256, 1, 1),
)
c = np.frombuffer(raw, dtype=np.float32)
print("max |error| =", float(np.max(np.abs(c - (a + b)))))   # 0.0
```

What ran:

1. `@enigma.kernel` traced the Python function into MLIR (the `enigma`
   dialect).
2. `enigma.compile` lowered it to Metal Shading Language, then ran it
   through `xcrun metal` → AIR → `xcrun metallib` → `.metallib`.
3. `MetalRuntime().execute(...)` mmap'd the `.metallib`, allocated GPU
   buffers, dispatched the compute pass, and returned the result bytes.

For richer examples — RMSNorm, FlashAttention, 1D Laplacian — see the
[Showcase kernels](#showcase-kernels) section below.

### Building from source

You only need this path if you are hacking on the dialect itself, or
porting to a future LLVM. The pipeline is **two stages** — dialect
(native, C++/MLIR) first, then the Python DSL on top of it.

```bash
git clone https://github.com/Klyne-Research/Enigma-DSL.git
cd Enigma-DSL
git submodule update --init --recursive   # pulls Enigma-Dialect

# Stage 1: build LLVM 22.x + MLIR (one-time, ~30-90 min).
# Produces ~/.local/enigma-llvm/ — isolated from any Homebrew LLVM.
bash Enigma-Dialect/scripts/build_llvm.sh

# Stage 2: build the merged wheel (DSL + dialect) for the Python
# version of your choice. Repeat the --python flag for each version.
./build_all.sh --python 3.12

# The wheel lands in wheelhouse/. The script also creates
# .venv-py3.12, installs into it, and runs pytest by default.
```

What `build_all.sh` actually does, in order:

1. Sources `~/.local/enigma-llvm/activate.sh` to put the local MLIR
   on `MLIR_DIR` / `LLVM_DIR`.
2. Builds `enigma-dsl` as a pure-Python wheel (`py3-none-any`).
3. Builds `enigma-dialect` as a native wheel (`cpXY-cpXY-macosx_*_arm64`)
   — one per Python version. This invokes `scikit-build-core` which in
   turn drives CMake against the local LLVM build.
4. Fixes Mach-O rpaths and re-codesigns the bundled dylibs so they
   load from `@loader_path`.
5. Merges the two wheels into a single `enigma_dsl-*-cpXY-cpXY-*.whl`
   containing both `enigma/` and `mlir/` packages — what users
   eventually `pip install` from PyPI.

Common variations:

```bash
# Multi-version build (publish-ready):
MACOSX_DEPLOYMENT_TARGET=14.0 \
  ./build_all.sh --python 3.11 --python 3.12 --python 3.13 \
                 --no-test --no-install --clean

# Build the dialect against an existing LLVM in a non-default location:
MLIR_DIR=/path/to/lib/cmake/mlir \
  ./build_all.sh --python 3.12

# Build only the dialect (skip Python DSL):
./build_all.sh --python 3.12 --skip-dsl

# Build only the DSL (reuse a previously built dialect wheel):
./build_all.sh --python 3.12 --skip-dialect
```

The LLVM step is the expensive one. After the first `build_llvm.sh`
finishes, every subsequent `build_all.sh` reuses it — incremental
dialect builds are ~3-5 min per Python.

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

## Benchmarks

Measured on a **MacBook Air M4** (8-core GPU, 16 GB unified memory,
120 GB/s memory bandwidth, ~3.6 TFLOPS FP32 theoretical peak). All
kernels pass correctness against a NumPy reference.

### SDPA — `examples/benchmark_sdpa.py`

128 tokens × 512 KV × 64 Q-heads × 8 KV-heads, head_dim=64, 4 simdgroups,
1.09 GFLOPs per dispatch.

| Implementation | Latency | Throughput | % of FP32 peak |
|---|---|---|---|
| **Enigma DSL** (gpt-oss layout, 4 sg) | **1831 µs** | **0.598 TFLOPS** | ~16.6% |
| gpt-oss handwritten Metal (4 sg) | 1987 µs | 0.551 TFLOPS | ~15.3% |

Enigma is **1.09× faster** than the handwritten Metal baseline. The ~17%
of FP32 peak is expected for online-softmax attention on Apple GPUs —
moving to fp16 + simdgroup matrix would unlock more.

### Qwen3-0.6B fused decode — `examples/qwen_megakernel.py`

Single-dispatch megakernel: RMSNorm → QKV proj → head-norm + RoPE → SDPA
→ O-proj → SwiGLU → down-proj, all in one threadgroup.

| Metric | Value |
|---|---|
| Throughput | **92.6 tok/s** |
| Latency | 10.79 ms/tok |
| Compile time | 0.08 s (14 KB of generated MSL) |
| Correctness | `max|err| = 3.87e-06` vs NumPy |

Competitive with `llama.cpp` on the same hardware for a 0.6B model — and
the entire decode step is one kernel launch, no host round-trips between
phases.

Reproduce:

```bash
python examples/benchmark_sdpa.py
python examples/qwen_megakernel.py
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

**v0.1.1** — first PyPI release. Six merged wheels published
(`enigma_dsl-0.1.1-cp{311,312,313}-cp{311,312,313}-macosx_{14_0,15_0}_arm64`).
Installable via `pip install enigma-dsl` on any Apple Silicon Mac
running macOS 14+. Project page: <https://pypi.org/project/enigma-dsl/>.

**v0.1.0** — initial release. Layout algebra engine (composition,
complement, coalesce, zipped divide, recast, TV layout construction).
Tracing IR with SSA values, constant folding, thread index decomposition.
Metal emitter supporting scalar, float4 vector pointer, and TV-layout
vectorised codegen. Swift runtime with device management, buffer
allocation, synchronous dispatch, GPU timestamp measurement. Dialect
TableGen definitions covering thread indexing, synchronisation, math,
atomics, simdgroup, quad, geometry, pack/unpack, and matrix ops.
