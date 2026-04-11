<p align="center">
  <img src="https://cdn8.futura-sciences.com/a1280/images/actu/enigma-mer-baltique.jpeg" width="600" />
</p>

<h1 align="center"><code>E N I G M A</code></h1>

<p align="center">
  <sub>where python meets metal, and layouts become algebra</sub>
</p>

In 1945, an Enigma machine sank to the floor of the Baltic Sea. For decades it sat there, its rotors locked, its wiring intact, waiting. When divers finally pulled it from the silt, the mechanism still worked. The genius was never in the shell. It was in the rotors, the wiring, the algebra of permutations hidden inside.

Enigma DSL is built on the same principle. Inspired by NVIDIA's CuTe DSL, which brought layout algebra and tiling calculus to CUDA, Enigma brings the same mathematical framework to Apple Metal. Where CuTe targets tensor cores and warps on NVIDIA GPUs, Enigma targets simdgroups and threadgroups on Apple Silicon. The layout algebra is the same. The target is different. You write a Python function. Underneath, the algebra computes how threads map to memory, how tiles partition a tensor, how values flow through a simdgroup. The Python traces into an IR. The IR emits Metal C++. The Metal compiles to GPU machine code. Your function runs on Apple Silicon at hardware bandwidth limits. The surface is clean. The machinery is exact.

The compilation pipeline has four stages. A Python kernel function decorated with `@enigma.kernel` is executed with proxy tensors that record every operation into an SSA intermediate representation. The IR captures loads, stores, arithmetic, and thread index decompositions. A Metal emitter walks the IR and produces Metal Shading Language source, choosing between scalar access, float4 vector pointer types, or TV layout vectorized codegen depending on the kernel pattern. The Metal source is compiled through Apple's toolchain, `xcrun metal` to AIR bitcode, then `xcrun metallib` to a Metal library binary. A Swift runtime loaded via ctypes creates the Metal device, command queue, and pipeline state, dispatches the compute kernel, and returns results to Python through unified memory.

For tiled kernels, the `@enigma.jit` decorator runs host side layout algebra before launching the GPU kernel. The layout algebra is a pure Python implementation of the CuTe tiling calculus adapted for Metal's memory hierarchy. A Layout is a pair of Shape and Stride that defines a function from logical coordinates to memory offsets. The engine provides composition, complement, coalesce, zipped divide, and the make_layout_tv constructor that builds a Thread Value layout mapping thread indices and value indices to tile coordinates with correct coalescing order. The `@jit` function tiles tensors, constructs the TV layout, and calls `@kernel(...).launch(grid, block)` which traces the kernel body with runtime IRValues for block and thread indices. Tensor slicing with IRValues generates offset arithmetic in the IR. Vectorized `.load()` and `.store()` on per thread tensor fragments emit grouped float4 reads and writes. The entire tiling, decomposition, and vectorization strategy is determined by the layout algebra at compile time. Only the final offset arithmetic and memory transactions reach the GPU.

The generated kernels run at the same bandwidth as hand written Metal. On Apple M4, both the DSL generated scalar kernel and the float4 kernel saturate DRAM bandwidth at approximately 100 GB/s for large tensors, matching native Metal benchmarks measured with GPU hardware timestamps.

### Versions

**v0.1.0** Initial release. Layout algebra engine with composition, complement, coalesce, zipped divide, recast, and TV layout construction. Tracing IR with SSA values, constant folding, and thread index decomposition. Metal emitter supporting scalar, float4 vector pointer, and TV layout vectorized codegen. Swift runtime with device management, buffer allocation, synchronous dispatch, and GPU timestamp measurement. Dialect TableGen definitions for 16 thread indexing and synchronization ops. 30 passing tests covering layout algebra, GPU execution across multiple sizes, Metal source export, and IR tracing correctness.
