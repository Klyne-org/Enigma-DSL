<p align="center">
  <img src="https://cdn8.futura-sciences.com/a1280/images/actu/enigma-mer-baltique.jpeg" width="600" />
</p>

<h1 align="center">Enigma</h1>

<p align="center">
  <em>Decode the GPU. Write Python. Run Metal.</em>
</p>

<p align="center">
  A CuTe-style Python DSL that compiles to Apple Metal GPU kernels on Apple Silicon.
</p>

---

Like the cipher machine resting on the Baltic seabed — powerful machinery hidden beneath a clean surface — Enigma lets you write high-level Python while generating tight Metal C++ that runs at hardware bandwidth limits on Apple Silicon.

```python
import enigma

@enigma.kernel
def vector_add(A: enigma.f32, B: enigma.f32, C: enigma.f32):
    tid = enigma.thread_position_in_grid
    C[tid] = A[tid] + B[tid]

compiled = enigma.compile(vector_add)
```

This generates, compiles, and dispatches a Metal compute kernel — no Xcode, no Objective-C, no boilerplate.

## What Enigma does

**You write Python. Enigma generates Metal.**

```
@enigma.kernel / @enigma.jit
        |
   IR tracing (proxy tensors, IRValues)
        |
   Metal C++ emission (scalar, float4, TV-layout vectorized)
        |
   xcrun metal -> .air -> .metallib
        |
   Swift runtime -> MTLDevice -> GPU dispatch
        |
   numpy result
```

## Layout algebra

Enigma implements CuTe-style layout algebra for tiled, vectorized GPU kernels:

```python
@enigma.kernel
def add_kernel(gA, gB, gC, tv_layout, tiler):
    tidx, _, _ = enigma.arch.thread_idx()
    bidx, _, _ = enigma.arch.block_idx()

    blkA = gA[((None, None), bidx)]
    tidfrgA = tensor_composition(blkA, tv_layout, tiler)
    thrA = tidfrgA[(tidx, None)]

    # ...
    thrC.store(thrA.load() + thrB.load())  # float4 vectorized

@enigma.jit
def elementwise_add(mA, mB, mC):
    thr_layout = enigma.make_ordered_layout((4, 64), order=(1, 0))
    val_layout = enigma.make_ordered_layout((4, 4), order=(1, 0))
    tiler_mn, tv_layout = enigma.make_layout_tv(thr_layout, val_layout)

    gA = tensor_zipped_divide(mA, tiler_mn)
    # ...
    add_kernel(gA, gB, gC, tv_layout, tiler_mn).launch(grid=..., block=...)
```

The layout algebra (`make_layout`, `zipped_divide`, `composition`, `complement`, `make_layout_tv`) computes tiling and memory access patterns in pure Python. The `@kernel` body is traced with IRValues and emitted as vectorized Metal C++.

## Performance

GPU timestamps on Apple M4, 4096x4096 float32:

```
float  (scalar)    ~200 us    ~100 GB/s
float4 (vec)       ~200 us    ~100 GB/s
```

Generated kernels run at the same bandwidth as hand-written Metal — the DSL adds zero overhead to GPU execution.

## Quick start

```bash
pip install numpy
cd Enigma-DSL

# Naive vector add
python examples/vector_add.py

# TV-layout tiled vector add (@jit + @kernel)
python examples/vector_add_tv.py

# Benchmark with GPU timestamps
python examples/benchmark_naive_vs_tv.py

# Tests
python -m unittest discover tests
```

Requirements: macOS with Apple Silicon, Xcode command line tools (`xcrun metal`).

## Project structure

```
enigma/
    __init__.py             public API
    tuple.py                hierarchical tuple math
    core.py                 Layout algebra engine
    tensor.py               Tensor = ptr + Layout
    typing.py               f32, f16, bf16, ...
    _tracing.py             IR: IRValue, KernelBuilder
    compiler/
        kernel.py           @kernel, @jit decorators
        compiler.py          trace -> emit -> xcrun -> .metallib
        metal_emitter.py    IR -> Metal C++ source
    runtime_dispatch/
        runtime.py          MetalRuntime (ctypes -> Swift)
        swift/
            libenigma_runtime.swift

Enigma-Dialect/             MLIR dialect (submodule)
    dialect/include/EnigmaDialect/
        EnigmaDialect.td
        EnigmaOps.td        16 thread/sync ops
        EnigmaTypes.td      address spaces, layout attr
```

## License

MIT
