# Enigma DSL — Implementation Plan

> A Python DSL for Apple Metal GPU kernels with CuTe-style layout algebra, targeting high-performance inference and training on Apple Silicon.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Module 1: Layout Algebra Engine](#3-module-1-layout-algebra-engine)
4. [Module 2: Type System](#4-module-2-type-system)
5. [Module 3: Tensor Abstraction](#5-module-3-tensor-abstraction)
6. [Module 4: Metal Atoms — SIMD, MMA, Copy](#6-module-4-metal-atoms--simd-mma-copy)
7. [Module 5: Algorithm Library](#7-module-5-algorithm-library)
8. [Module 6: AST Rewriting & Kernel Definition](#8-module-6-ast-rewriting--kernel-definition)
9. [Module 7: Compilation Pipeline](#9-module-7-compilation-pipeline)
10. [Module 8: Runtime Dispatch](#10-module-8-runtime-dispatch)
11. [Module 9: Fake Tensors & Dynamic Shapes](#11-module-9-fake-tensors--dynamic-shapes)
12. [Module 10: PyTorch Integration](#12-module-10-pytorch-integration)
13. [Metal Feature Coverage Matrix](#13-metal-feature-coverage-matrix)
14. [Dialect Interface Contract](#14-dialect-interface-contract)
15. [Testing Strategy](#15-testing-strategy)
16. [Dependency Graph & Build Order](#16-dependency-graph--build-order)

---

## 1. Architecture Overview

Enigma DSL follows the same layered design as CuTe DSL but targets Apple Metal instead of CUDA. The pipeline has 7 stages:

```
 STAGE 0    Python DSL source (@enigma.kernel decorated functions)
    │
    │  AST rewriter: for/if → context managers
    │  Execution with MetalTensor proxies → each op builds IR
    ▼
 STAGE 1    High-level MLIR (func + memref + scf + metal.* ops)
    │       Layout lives as #metal.layout<"shape:stride"> type attribute
    │
    │  MLIR PassManager (canonicalize, lower-affine, scf→cf, cse, licm)
    ▼
 STAGE 2    Lowered MLIR (arith + memref + cf + metal.* ops)
    │
    │  metal-translate --mlir-to-metal (TypeSwitch emitter → text)
    ▼
 STAGE 3    .metal shader source (Metal C++ text)
    │
    │  xcrun -sdk macosx metal -c kernel.metal -o kernel.air
    ▼
 STAGE 4    AIR bitcode (Apple's LLVM IR, closed)
    │
    │  xcrun -sdk macosx metallib kernel.air -o kernel.metallib
    ▼
 STAGE 5    .metallib container
    │
    │  Python ctypes → Swift dylib (libenigma_runtime.dylib)
    │  MTLDevice.makeLibrary(data:) → makeComputePipelineState
    ▼
 STAGE 6    Metal runtime dispatch on Apple Silicon GPU
```

**Key principle**: Stages 0-3 are what Enigma DSL implements. Stages 4-5 use Apple's closed toolchain. Stage 6 is a thin Swift runtime (~150 lines) with a C API.

### What Enigma DSL owns vs. what it gets for free

| Component | Owner | Effort |
|-----------|-------|--------|
| Layout algebra (pure Python) | Enigma DSL | ~1500 LOC |
| Tensor/proxy/type system | Enigma DSL | ~800 LOC |
| Metal atoms (SIMD, MMA, copy) | Enigma DSL | ~600 LOC |
| Algorithm library (gemm, copy, reduce) | Enigma DSL | ~500 LOC |
| AST rewriter + kernel decorator | Enigma DSL | ~400 LOC |
| Compiler driver (PassManager + subprocess) | Enigma DSL | ~300 LOC |
| Runtime (ctypes → Swift dylib) | Enigma DSL | ~200 LOC Python + ~150 LOC Swift |
| Fake tensor / dynamic shapes | Enigma DSL | ~300 LOC |
| Dialect (MetalOps.td, metal-translate) | Enigma-Dialect (separate repo) | ~900 LOC C++ |
| MLIR passes (lower-affine, scf→cf, cse...) | Upstream MLIR | Free |
| Metal compiler (xcrun metal) | Apple | Free |
| Metal runtime (MTLDevice, MTLCommandBuffer) | Apple | Free |

---

## 2. Repository Structure

```
enigma/                              # Python package root
├── __init__.py                      # Public API: enigma.kernel, enigma.compile, ...
├── core.py                          # Layout algebra engine (~1500 LOC)
├── typing.py                        # Type system (Float32, BFloat16, UInt8, ...)
├── tensor.py                        # Tensor abstraction + TensorSSA
├── tuple.py                         # Hierarchical tuple utilities
├── math.py                          # Metal math wrappers (fma, exp, rsqrt, ...)
├── atom.py                          # Metal atoms: SimdAtom, MmaAtom, CopyAtom
├── algorithm.py                     # gemm, copy, autovec_copy, reduce, softmax
├── testing.py                       # Test utilities, assertions
├── runtime.py                       # Fake tensor, dynamic tensor, DLPack
│
├── arch/                            # Metal architecture abstractions
│   ├── __init__.py
│   ├── simdgroup.py                 # simdgroup_matrix, simd_sum, simd_max, ...
│   ├── threadgroup.py               # threadgroup_barrier, shared memory
│   ├── atomic.py                    # atomic_fetch_add, atomic_store, ...
│   └── numeric_conversion.py        # bfloat↔float, quantized type conversions
│
├── metal/                           # Metal-specific hardware ops
│   ├── __init__.py
│   ├── simdgroup/
│   │   ├── copy.py                  # simdgroup_async_copy
│   │   ├── mma.py                   # simdgroup_multiply_accumulate (8x8)
│   │   └── reduce.py               # simd_sum, simd_max, simd_min
│   ├── threadgroup/
│   │   ├── memory.py                # threadgroup memory allocation
│   │   └── barrier.py              # threadgroup_barrier
│   └── common.py                    # Universal Metal ops
│
├── export/                          # Code generation
│   ├── __init__.py
│   ├── metal_emitter.py             # MLIR → .metal source text
│   └── aot_config.py               # AOT compilation configuration
│
├── compiler/                        # Compilation pipeline
│   ├── __init__.py
│   ├── compiler.py                  # PassManager + xcrun subprocess
│   ├── ast_rewriter.py              # ast.NodeTransformer for/if rewriting
│   ├── kernel.py                    # @enigma.kernel, @enigma.jit decorators
│   └── tvm_ffi_provider.py          # TVM FFI integration (optional)
│
├── runtime_dispatch/                # Metal runtime
│   ├── __init__.py
│   ├── runtime.py                   # Python ctypes wrapper
│   ├── swift/
│   │   └── libenigma_runtime.swift  # Swift dylib (~150 lines)
│   └── buffer.py                    # MetalBuffer abstraction
│
└── torch_integration/               # PyTorch interop
    ├── __init__.py
    ├── custom_op.py                 # torch.library custom ops
    └── dlpack.py                    # DLPack bridge
```

**Enigma-Dialect** (separate repo, interfaces with this DSL):
```
Enigma-Dialect/
├── CMakeLists.txt
├── include/enigma/
│   ├── EnigmaOps.td                 # MLIR operation definitions
│   ├── EnigmaTypes.td               # Metal layout attribute, memspace types
│   └── EnigmaDialect.td             # Dialect registration
├── lib/
│   ├── EnigmaDialect.cpp
│   ├── EnigmaOps.cpp
│   ├── Transforms/
│   │   └── LoweringPasses.cpp       # metal.* → standard MLIR
│   └── Translation/
│       └── MetalTranslate.cpp       # MLIR → .metal source (TypeSwitch)
├── tools/
│   ├── enigma-opt/                  # MLIR opt with Enigma dialect
│   └── enigma-translate/            # MLIR → Metal text
└── python/
    └── EnigmaModule.cpp             # nanobind/pybind11 bindings
```

---

## 3. Module 1: Layout Algebra Engine

**File**: `enigma/core.py` (~1500 LOC)
**Dependencies**: None (pure Python math)

This is the mathematical heart of the DSL. It implements the CuTe layout algebra adapted for Metal's memory hierarchy.

### 3.1 Core Data Structures

```python
# --- Fundamental types ---
Int = Union[int, "SymInt"]
Shape = Union[Int, Tuple["Shape", ...]]
Stride = Union[Int, "ScaledBasis", Tuple["Stride", ...]]

class Layout:
    """Functional mapping: coordinates → memory offsets.
    Layout = (Shape, Stride)
    L(c) = Stride ∘ Shape(c)"""
    shape: Shape
    stride: Stride

class ComposedLayout:
    """R(c) = inner(offset + outer(c))
    Used for thread-value partitioning and tiled access."""
    inner: Layout
    offset: Int
    outer: Layout

class ScaledBasis:
    """Represents a symbolic stride element: scale @ mode.
    Used for tracking thread/value dimensions in partitioning."""
    scale: int
    mode: int
```

### 3.2 Layout Construction Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `make_layout` | `(shape, stride=None) → Layout` | Create layout. If stride omitted, compute compact column-major strides. |
| `make_identity_layout` | `(shape) → Layout` | Identity mapping where `L(c) = c`. Stride = `(1@0, 1@1, ...)` |
| `make_ordered_layout` | `(shape, order) → Layout` | Compact layout with custom dimension ordering |
| `make_col_major` | `(shape) → Layout` | Column-major compact layout |
| `make_row_major` | `(shape) → Layout` | Row-major compact layout |

### 3.3 Layout Query Functions

| Function | Description |
|----------|-------------|
| `rank(layout, mode=[])` | Number of modes (dimensions). Hierarchical: `rank((4,(3,2))) = 2`, `rank((4,(3,2)), mode=[1]) = 2` |
| `depth(layout)` | Maximum nesting depth. `depth(4) = 0`, `depth((4,(3,2))) = 2` |
| `size(layout, mode=[])` | Total element count or per-mode count |
| `shape(layout)` | Extract shape tuple |
| `stride(layout)` | Extract stride tuple |
| `cosize(layout)` | Maximum offset + 1. The memory footprint in elements. |

### 3.4 Coordinate Conversion

| Function | Description |
|----------|-------------|
| `crd2idx(coord, shape, stride)` | Hierarchical coordinate → linear offset. `crd2idx((2,1), (4,8), (1,4)) = 6` |
| `idx2crd(index, shape, stride)` | Linear offset → hierarchical coordinate (colexicographic) |
| `compact_col_major(shape)` | Compute column-major strides for shape |
| `compact_row_major(shape)` | Compute row-major strides for shape |

### 3.5 Layout Algebra Operations

These are the core algebraic operations from the CuTe paper.

#### Coalesce
```python
def coalesce(layout: Layout) -> Layout:
    """Flatten hierarchical layout while preserving semantics.
    Merges adjacent modes with compatible strides.

    coalesce((2,(1,6)) : (1,(6,2))) → 12:1
    """
```

#### Composition
```python
def composition(lhs: Layout, rhs: Layout) -> Layout:
    """Functional composition: R(c) = lhs(rhs(c))

    The fundamental operation. rhs determines domain, lhs determines codomain.

    Divisibility conditions (from CuTe paper):
    - Stride divisibility: Sr | d  or  d | Sr
    - Shape divisibility: ceil(Sr/d) | s

    Example:
        A = (4,8) : (1,4)   # 32-element layout
        B = (2,4) : (1,8)   # 8-element layout
        A ∘ B maps 8 elements through A's offset function
    """
```

#### Complement
```python
def complement(layout: Layout, cotarget: int = None) -> Layout:
    """Find the complementary layout — elements NOT in image of layout.

    Properties:
    - Weak congruence: cotarget ≲ complement
    - Disjoint images: image(layout) ∩ image(complement) = {0}
    - Ordered: complement(a-1) < complement(a)

    Example:
        complement((4,8):(1,4)) → 1:32  (the gap between columns)
    """
```

#### Inverse Operations
```python
def right_inverse(layout: Layout) -> Layout:
    """Right inverse: maps offsets back to coordinates.
    L(L‡(L(c))) = L(c) for all c.
    Used for finding contiguous elements and vectorization."""

def left_inverse(layout: Layout) -> Layout:
    """Left inverse: recovers input coordinates from output.
    L(L†(L(c))) = L(c) for all c.
    Used for instruction admissibility checking."""
```

#### Product Operations (Tiling)
```python
def logical_product(block: Layout, tiler: Layout) -> Layout:
    """Tile block across grid defined by tiler.
    Result = (block, complement(block) ∘ tiler)
    First mode: one tile. Second mode: grid of tiles."""

def blocked_product(a: Layout, b: Layout) -> Layout:
    """Blocked tiling: each element of b gets a full copy of a."""

def raked_product(a: Layout, b: Layout) -> Layout:
    """Raked tiling: elements of a are interleaved with elements of b."""

def zipped_product(a: Layout, b: Layout) -> Layout:
    """Interleaved product preserving mode structure."""

def tiled_product(a: Layout, b: Layout) -> Layout:
    """Block-level tiling with explicit tile boundaries."""

def flat_product(a: Layout, b: Layout) -> Layout:
    """Flat concatenation of layouts."""
```

#### Divide Operations (Partitioning)
```python
def logical_divide(layout: Layout, tiler) -> Layout:
    """Split layout into elements pointed to by tiler + remainder.
    R = layout ∘ (tiler, complement(tiler, size(layout)))
    First mode: one tile's worth. Second mode: how many tiles."""

def zipped_divide(layout: Layout, tiler) -> Layout:
    """By-mode divide with mode zipping. Core operation for thread partitioning."""

def tiled_divide(layout: Layout, tiler) -> Layout:
    """Block-based decomposition."""

def flat_divide(layout: Layout, tiler) -> Layout:
    """Flat decomposition."""
```

### 3.6 Layout Predicates

```python
def is_congruent(a, b) -> bool:
    """Strict shape matching: same rank and same shape at every level."""

def is_weakly_congruent(a, b) -> bool:
    """Loose shape matching: sizes match but structure may differ."""

def is_compatible(a, b) -> bool:
    """Same total size, potentially different structure."""
```

### 3.7 Metal-Specific Layout Utilities

```python
def threadgroup_layout(shape, element_type) -> Layout:
    """Create layout for threadgroup (shared) memory.
    Accounts for Metal's 16-byte alignment requirements
    and bank conflict avoidance for simdgroup access."""

def simdgroup_matrix_layout(rows, cols, dtype) -> Layout:
    """Layout for simdgroup_matrix (8x8 tiles on Apple Silicon).
    Maps thread indices within a 32-thread simdgroup to matrix elements."""

def device_layout(shape, stride=None) -> Layout:
    """Layout for device (global) memory with alignment hints
    for Metal's coalesced access patterns."""
```

---

## 4. Module 2: Type System

**File**: `enigma/typing.py` (~400 LOC)
**Dependencies**: None

### 4.1 Numeric Types

```python
# Floating-point types
class Float32(Numeric):    width = 32; metal_name = "float"
class Float16(Numeric):    width = 16; metal_name = "half"
class BFloat16(Numeric):   width = 16; metal_name = "bfloat"

# Integer types
class Int8(Numeric):       width = 8;  metal_name = "char"
class Int16(Numeric):      width = 16; metal_name = "short"
class Int32(Numeric):      width = 32; metal_name = "int"
class Int64(Numeric):      width = 64; metal_name = "long"
class UInt8(Numeric):      width = 8;  metal_name = "uchar"
class UInt16(Numeric):     width = 16; metal_name = "ushort"
class UInt32(Numeric):     width = 32; metal_name = "uint"
class UInt64(Numeric):     width = 64; metal_name = "ulong"

# Boolean
class Boolean(Numeric):    width = 1;  metal_name = "bool"

# Quantized types (matching gpt-oss-metal-kernels)
class Float8E5M2(Numeric):   width = 8
class Float8E4M3(Numeric):   width = 8
class Float8E8M0(Numeric):   width = 8
class Float4E2M1(Numeric):   width = 4

# Aliases for convenience
f32 = Float32; f16 = Float16; bf16 = BFloat16
i32 = Int32; u32 = UInt32; u64 = UInt64
```

### 4.2 Vector Types

```python
class VectorType:
    """Metal SIMD vector types: float2, float4, bfloat4, uint4, etc."""
    element_type: Type[Numeric]
    count: int  # 2, 3, or 4

    @property
    def metal_name(self) -> str:
        return f"{self.element_type.metal_name}{self.count}"

# Pre-built vector types
float2 = VectorType(Float32, 2)
float4 = VectorType(Float32, 4)
bfloat4 = VectorType(BFloat16, 4)
uint4 = VectorType(UInt32, 4)
half4 = VectorType(Float16, 4)
```

### 4.3 Pointer Types

```python
class AddressSpace(Enum):
    device = 0       # Global GPU memory (Metal device)
    threadgroup = 3  # Shared memory within threadgroup (Metal addrspace 3)
    constant = 2     # Constant memory (Metal constant)
    thread = 0       # Per-thread private (registers)
    generic = 0      # Unspecified

class Pointer:
    value_type: Type[Numeric]
    address_space: AddressSpace
    assumed_align: int = 1

    def __add__(self, offset: int) -> "Pointer"
    def align(self, min_align: int) -> "Pointer"
```

### 4.4 Struct Types (for kernel arguments)

```python
class StructType:
    """Maps to Metal kernel argument structs.
    Example: gptoss_control, gptoss_expert_prediction."""
    name: str
    fields: List[Tuple[str, Type[Numeric]]]
    packed: bool = False  # GPTOSS_DENSELY_PACKED_STRUCTURE
```

---

## 5. Module 3: Tensor Abstraction

**File**: `enigma/tensor.py` (~600 LOC)
**Dependencies**: `core.py`, `typing.py`

### 5.1 Tensor Class

```python
class Tensor:
    """Core tensor: an accessor (pointer) composed with a layout.

    Tensor = Pointer ∘ Layout
    T[coord] = memory[pointer + layout(coord)]
    """

    @property
    def iterator(self) -> Pointer       # Base pointer
    @property
    def layout(self) -> Layout           # Data layout
    @property
    def element_type(self) -> Type[Numeric]  # Element dtype
    @property
    def memspace(self) -> AddressSpace   # device / threadgroup / thread
    @property
    def shape(self) -> Shape
    @property
    def stride(self) -> Stride

    def __getitem__(self, coord):        # Indexing → value or subtensor
        """Slicing: partial evaluation creates sub-tensor with reduced layout."""

    def load(self) -> "TensorSSA":       # Load to registers
    def store(self, data):               # Store from registers
    def fill(self, value):               # Fill with constant
```

### 5.2 Tensor Construction Functions

```python
def make_tensor(pointer, layout) -> Tensor:
    """Create tensor from pointer + layout."""

def make_identity_tensor(shape) -> Tensor:
    """Tensor with identity layout — used for coordinate generation."""

def make_fragment(shape, dtype, init=None) -> Tensor:
    """Register-file fragment (thread-private memory).
    Maps to: float rC[N]; in emitted Metal."""

def make_fragment_like(src: Tensor) -> Tensor:
    """Create fragment matching source tensor's shape and type."""

def make_smem_tensor(layout, dtype) -> Tensor:
    """Threadgroup (shared) memory tensor.
    Maps to: threadgroup float sA[N]; in emitted Metal."""

def make_rmem_tensor(layout, dtype, init=None) -> Tensor:
    """Register memory tensor (alias for make_fragment).
    Maps to: float rC[N] = {init}; in emitted Metal."""
```

### 5.3 Tensor Operations

```python
def domain_offset(tensor: Tensor, offset) -> Tensor:
    """Apply coordinate offset to tensor (pointer arithmetic)."""

def recast_tensor(tensor: Tensor, new_dtype) -> Tensor:
    """Reinterpret tensor elements as different type.
    Used for: bfloat→float conversion, quantized access."""

def local_tile(tensor: Tensor, tile_shape, coord) -> Tensor:
    """Extract a tile from tensor at given coordinate.
    Maps to: affine.apply + memref.subview in MLIR."""

def local_partition(tensor: Tensor, thread_layout, thread_idx) -> Tensor:
    """Partition tensor among threads according to thread_layout.
    Each thread gets its slice of the data."""

def group_modes(tensor: Tensor, mode_begin, mode_end) -> Tensor:
    """Fold multiple modes into a single mode (tensor reshaping without data movement)."""
```

### 5.4 TensorSSA (SSA Value Tensors)

```python
class TensorSSA:
    """Represents a tensor value in SSA form during IR construction.
    Used during kernel tracing — each operation produces a new TensorSSA.

    Unlike Tensor (which tracks pointers+layouts), TensorSSA represents
    the actual SSA values flowing through the MLIR graph."""

    ir_value: Any          # MLIR SSA value
    element_type: Type[Numeric]
    shape: Shape

    def __add__(self, other): ...   # arith.addf / arith.addi
    def __mul__(self, other): ...   # arith.mulf / arith.muli
    def __sub__(self, other): ...   # arith.subf / arith.subi
```

---

## 6. Module 4: Metal Atoms — SIMD, MMA, Copy

**Files**: `enigma/atom.py`, `enigma/metal/simdgroup/`, `enigma/arch/`
**Dependencies**: `core.py`, `typing.py`, `tensor.py`

Atoms represent the fundamental hardware operations on Apple Silicon. Unlike NVIDIA which has many MMA variants (Volta, Ampere, Hopper, Blackwell), Metal has a single simdgroup matrix system.

### 6.1 Atom Base Classes

```python
class Atom:
    """Base class for all hardware operation atoms."""
    @property
    def thr_id(self) -> Layout:
        """Thread ID layout — how threads are assigned to this operation."""

class MmaAtom(Atom):
    """Matrix multiply-accumulate atom.
    On Metal, this maps to simdgroup_multiply_accumulate with 8x8 tiles."""

    @property
    def shape_mnk(self) -> Tuple[int, int, int]:
        """(M, N, K) dimensions of the atom. On Metal: (8, 8, 8)."""

    @property
    def tv_layout_A(self) -> Layout:
        """Thread-Value layout for operand A."""

    @property
    def tv_layout_B(self) -> Layout:
        """Thread-Value layout for operand B."""

    @property
    def tv_layout_C(self) -> Layout:
        """Thread-Value layout for accumulator C."""

class CopyAtom(Atom):
    """Data copy atom — represents one copy instruction's behavior."""

    @property
    def src_memspace(self) -> AddressSpace: ...
    @property
    def dst_memspace(self) -> AddressSpace: ...
    @property
    def vector_width(self) -> int: ...
```

### 6.2 Metal Simdgroup MMA Atom

```python
class MetalSimdgroupMma8x8(MmaAtom):
    """Apple Silicon simdgroup matrix multiply-accumulate.

    Hardware: 32 threads in a simdgroup cooperate on 8x8 tiles.
    Operations:
      simdgroup_float8x8 tA, tB, tC;
      simdgroup_load(tA, src, stride);
      simdgroup_multiply_accumulate(tC, tA, tB, tC);
      simdgroup_store(tC, dst, stride);

    Supported element types:
      float (f32), half (f16), bfloat (bf16)

    Thread-value mapping:
      32 threads each own specific elements of the 8x8 = 64 result.
      Layout: ((4,8), 2) : ((16,1), 8) — matches Apple's hardware mapping.
    """

    shape_mnk = (8, 8, 8)

    supported_dtypes = [Float32, Float16, BFloat16]

    def make_fragment_A(self, shape_mk) -> Tensor:
        """Allocate register fragment for A operand."""

    def make_fragment_B(self, shape_nk) -> Tensor:
        """Allocate register fragment for B operand."""

    def make_fragment_C(self, shape_mn) -> Tensor:
        """Allocate register fragment for accumulator C."""
```

### 6.3 Tiled MMA

```python
class TiledMma:
    """Multi-simdgroup MMA built by tiling a base MmaAtom.

    Example: Tile 8x8 atom to cover 64x64 output:
      atom = MetalSimdgroupMma8x8(Float32)
      tiled = make_tiled_mma(atom, tiler_mnk=(8, 8, 1))
      # 8x8 = 64 simdgroups needed, each computing 8x8 of the output

    Matching gpt-oss-metal-kernels block sizes:
      QKV:         Bm=64, Bn=64, Bk=32, Sg_Bm=32, Sg_Bn=32
      Attn output: Bm=32, Bn=64, Bk=32, Sg_Bm=32, Sg_Bn=16
      MLP gate:    Bm=64, Bn=16, Bk=32, Sg_Bm=16, Sg_Bn=16
    """

    atom: MmaAtom
    tiler_mnk: Tuple[int, int, int]

    def get_slice(self, thr_idx) -> "ThrMma":
        """Get this thread's view of the tiled MMA."""

    def partition_A(self, tensor_mk) -> Tensor:
        """Partition A tensor for this tiled MMA."""

    def partition_B(self, tensor_nk) -> Tensor:
        """Partition B tensor for this tiled MMA."""

    def partition_C(self, tensor_mn) -> Tensor:
        """Partition C tensor for this tiled MMA."""

def make_tiled_mma(atom: MmaAtom, tiler_mnk, permutation=None) -> TiledMma:
    """Construct tiled MMA from base atom and tiling parameters."""
```

### 6.4 Copy Atoms for Metal

```python
class CooperativeCopy(CopyAtom):
    """Cooperative device→threadgroup copy using all threads in a threadgroup.

    NOTE: `simdgroup_async_copy` is NOT a public Metal API. It is an
    undocumented internal instruction used by Apple's Metal Performance Shaders.
    Metal has no public async copy API.

    Instead, gpt-oss-metal-kernels uses the standard cooperative copy pattern:
    all threads in the threadgroup cooperatively load data from device memory
    into threadgroup memory using vectorized loads (float4), then synchronize
    with threadgroup_barrier.

    Metal pattern:
      // Each thread loads its portion of the tile
      for (uint t = 0; t < ITERS; ++t) {
          uint i = t * thread_count + tid;
          if (i < total_elements) {
              *(threadgroup float4*)(dst + offset) = *(device const float4*)(src + offset);
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

    This is equivalent to CUDA's cooperative copy where threads in a CTA
    cooperatively load a tile from global → shared memory.
    """
    src_memspace = AddressSpace.device
    dst_memspace = AddressSpace.threadgroup

class SimtCopy(CopyAtom):
    """Simple per-thread element-wise copy.
    Each thread copies one or more elements independently.
    Used for non-vectorizable access patterns."""

class VectorizedCopy(CopyAtom):
    """Vectorized copy using float2/float4 loads.
    Used by autovec_copy when memory alignment allows.

    Metal:
      float4 val = *(device const float4*)(src + offset);
      *(threadgroup float4*)(dst + offset) = val;
    """
    vector_width: int  # 2 or 4
```

### 6.5 SIMD Reduction Atoms

```python
class SimdReduceAtom:
    """SIMD-level reduction within a 32-thread simdgroup.

    Operations:
      metal::simd_sum(value)    — sum across simdgroup
      metal::simd_max(value)    — max across simdgroup
      metal::simd_min(value)    — min across simdgroup
    """

class SimdPrefixSumAtom:
    """Inclusive prefix sum within simdgroup.
    metal::simd_prefix_inclusive_sum(value)"""

class SimdBroadcastAtom:
    """Broadcast from first thread of simdgroup.
    Used with metal::simd_is_first() for conditional execution."""
```

### 6.6 Atomic Operation Atoms

```python
class AtomicFetchAdd:
    """metal::atomic_fetch_add_explicit(&var, val, memory_order_relaxed)"""

class AtomicStore:
    """metal::atomic_store_explicit(&var, val, memory_order_relaxed)"""

class AtomicLoad:
    """metal::atomic_load_explicit(&var, memory_order_relaxed)"""

class AtomicMin:
    """metal::atomic_min_explicit(&var, val, memory_order_relaxed)
    Used for distributed argmax (as in unembedding kernel)."""
```

---

## 7. Module 5: Algorithm Library

**File**: `enigma/algorithm.py` (~500 LOC)
**Dependencies**: `core.py`, `tensor.py`, `atom.py`

### 7.1 Copy Algorithms

```python
def basic_copy(src: Tensor, dst: Tensor):
    """Element-wise copy. Fully unrolled for static shapes.
    Prerequisites: size(src) == size(dst)

    Generated Metal:
      for (int i = 0; i < N; i++) dst[i] = src[i];
    """

def basic_copy_if(pred: Tensor, src: Tensor, dst: Tensor):
    """Predicated copy — only copies where pred is true."""

def copy(atom: CopyAtom, src: Tensor, dst: Tensor, pred=None):
    """Copy using a specific copy atom.

    For CooperativeCopy (device → threadgroup):
      Each thread loads its slice using float4 vectorized reads:
        *(threadgroup float4*)(dst + offset) = *(device const float4*)(src + offset);
      Followed by threadgroup_barrier(mem_flags::mem_threadgroup);

    For VectorizedCopy:
      float4 v = *(device const float4*)(src + i);
      *(threadgroup float4*)(dst + i) = v;
    """

def autovec_copy(src: Tensor, dst: Tensor):
    """Auto-vectorized copy — the CuTe-style optimization for Metal.

    Algorithm:
    1. Find maximum common layout vector width between src and dst
    2. Check pointer alignment (4-byte for float, 2-byte for half)
    3. Compute vector width: min(common_width, 256 bits / element_width)
    4. If vectorizable:
       - logical_divide tensors by vector width
       - Use float4/float2 loads/stores
    5. Else: fallback to basic_copy

    Metal vectorization:
      float4 — 4 floats = 128 bits (max practical on Metal)
      half4  — 4 halfs  = 64 bits
      bfloat4 — 4 bfloats = 64 bits

    Unlike CUDA (256-bit max), Metal practical max is 128 bits.
    """
```

### 7.2 GEMM Algorithm

```python
def gemm(tiled_mma: TiledMma,
         tA: Tensor,  # Partitioned A fragment
         tB: Tensor,  # Partitioned B fragment
         tC: Tensor,  # Partitioned C accumulator
         k_tile_count: int = None):
    """Generic GEMM using tiled MMA atoms.

    Implements the CuTe generic GEMM:
      for k in range(K):
        for n in range(N):
          for m in range(M):
            C[m,n] += A[m,k] * B[n,k]

    On Metal, this generates:
      simdgroup_float8x8 tA, tB, tC;
      for k in range(K_tiles):
          simdgroup_load(tA, sA, stride, ulong2(k*8, m*8));
          simdgroup_load(tB, sB, stride, ulong2(k*8, n*8));
          simdgroup_multiply_accumulate(tC, tA, tB, tC);
    """

def gemm_with_staging(
    tiled_mma: TiledMma,
    A_device: Tensor, B_device: Tensor,
    A_smem: Tensor, B_smem: Tensor,
    C_rmem: Tensor,
    copy_atom: CopyAtom,
    k_tile_count: int):
    """GEMM with explicit staging through threadgroup memory.

    Pattern (matching gpt-oss-metal-kernels dense_matmul):
    1. All threads cooperatively load A tile: device → threadgroup (float4 vectorized)
    2. All threads cooperatively load B tile: device → threadgroup (float4 vectorized)
    3. threadgroup_barrier(mem_flags::mem_threadgroup)
    4. simdgroup_load from threadgroup → simdgroup_multiply_accumulate → accumulators
    5. threadgroup_barrier(mem_flags::mem_threadgroup)
    6. Advance to next K tile
    7. simdgroup_store accumulators → threadgroup → device memory
    """
```

### 7.3 Reduction Algorithms

```python
def simd_reduce_sum(value, simdgroup_tid):
    """Reduce sum within a 32-thread simdgroup.
    Metal: metal::simd_sum(value)"""

def simd_reduce_max(value, simdgroup_tid):
    """Reduce max within simdgroup.
    Metal: metal::simd_max(value)"""

def simd_reduce_min(value, simdgroup_tid):
    """Reduce min within simdgroup."""

def threadgroup_reduce_sum(value, tid, num_threads, smem_buffer):
    """Two-level reduction: simd_sum per simdgroup, then reduce across simdgroups.

    Pattern (matching gpt-oss-metal-kernels rmsnorm):
    1. simd_sum within each simdgroup
    2. if simd_is_first: write to threadgroup buffer
    3. threadgroup_barrier
    4. Reduce threadgroup buffer entries
    5. threadgroup_barrier
    """

def threadgroup_reduce_max(value, tid, num_threads, smem_buffer):
    """Two-level max reduction (used in softmax, SDPA)."""
```

### 7.4 Elementwise Algorithms

```python
def elementwise_unary(fn, src: Tensor, dst: Tensor):
    """Apply unary function elementwise. Auto-vectorized if possible.
    fn: one of metal::precise::exp, metal::precise::rsqrt, etc."""

def elementwise_binary(fn, a: Tensor, b: Tensor, dst: Tensor):
    """Apply binary function elementwise.
    fn: metal::fma, add, mul, etc."""

def fill(tensor: Tensor, value):
    """Fill tensor with constant value.
    Metal: for(int i=0; i<N; i++) tensor[i] = value;"""

def transform(src: Tensor, dst: Tensor, fn):
    """Transform copy: dst[i] = fn(src[i])"""
```

### 7.5 Specialized Metal Algorithms

```python
def rmsnorm(input: Tensor, weight: Tensor, output: Tensor, eps: float):
    """RMS normalization — full kernel algorithm.
    Matching gptoss_f32_rmsnorm:
    1. Compute sum of squares (threadgroup reduction)
    2. rsqrt(mean_sq + eps)
    3. Scale by weight
    """

def softmax(input: Tensor, output: Tensor):
    """Numerically stable softmax.
    Matching gptoss_f32_softmax:
    1. Find max (simd_max + threadgroup reduce)
    2. Compute exp(x - max) (precise::exp)
    3. Sum (simd_sum + threadgroup reduce)
    4. Divide
    """

def rope(input: Tensor, output: Tensor, positions, freqs, config):
    """Rotary position embedding.
    Matching gptoss_f32_rope:
    - Complex number rotation in 2D pairs
    - YaRN interpolation support via metal::mix
    - metal::precise::sincos for rotation angles
    """

def sdpa(Q: Tensor, K: Tensor, V: Tensor, output: Tensor, scale: float):
    """Scaled dot-product attention.
    Matching gptoss_f32_sdpa:
    1. Q·K scores via metal::dot
    2. Online softmax with max-subtract trick
    3. Score × V aggregation
    4. Multi-simdgroup hierarchical reduction
    """

def topk(input: Tensor, k: int, output_ids, output_scores):
    """Top-K selection for expert routing (MOE).
    Matching gptoss_f32_topk_softmax."""

def scatter(src: Tensor, indices: Tensor, dst: Tensor):
    """MOE scatter — copy tokens to expert buffers."""

def accumulate(expert_outputs: Tensor, scores: Tensor, dst: Tensor):
    """MOE weighted accumulation of expert outputs."""
```

---

## 8. Module 6: AST Rewriting & Kernel Definition

**Files**: `enigma/compiler/ast_rewriter.py`, `enigma/compiler/kernel.py`
**Dependencies**: Python `ast` module, `tensor.py`

### 8.1 AST Rewriter

The AST rewriter transforms Python control flow into context managers that the IR builder can intercept.

```python
class EnigmaASTRewriter(ast.NodeTransformer):
    """Transforms Python source before execution.

    Rewrites:
      for i in enigma.range(0, N):     →   with enigma._for_ctx(0, N) as i:
          body                                  body

      if condition:                    →   with enigma._if_ctx(condition):
          body                                  body

      while condition:                 →   with enigma._while_ctx(condition):
          body                                  body

    This enables the DSL to generate MLIR scf.for / scf.if / scf.while
    instead of executing Python loops at trace time.
    """

    def visit_For(self, node: ast.For) -> ast.AST:
        """Detect enigma.range() calls and rewrite to scf.for."""

    def visit_If(self, node: ast.If) -> ast.AST:
        """Rewrite if to scf.if (only when condition involves traced values)."""

    def visit_While(self, node: ast.While) -> ast.AST:
        """Rewrite while to scf.while."""
```

### 8.2 Kernel Decorators

```python
@enigma.kernel
def matmul(A: Tensor, B: Tensor, C: Tensor):
    """Declares a Metal compute kernel.

    The decorated function is:
    1. Source-extracted and AST-rewritten
    2. Executed with proxy tensors to trace the computation graph
    3. The trace builds an MLIR module with func.func @matmul(...) attributes {metal.kernel}

    Within the kernel body, these thread indices are available:
      enigma.threadgroup_position_in_grid    → uint2/uint3
      enigma.thread_position_in_threadgroup  → uint2/uint3
      enigma.thread_index_in_simdgroup       → uint
      enigma.simdgroup_index_in_threadgroup  → uint
      enigma.threads_per_threadgroup         → uint2/uint3
      enigma.simdgroups_per_threadgroup      → uint
    """

@enigma.jit
def helper_function(x: Tensor) -> Tensor:
    """JIT-compiled device function (not a kernel entry point).
    Can be called from within @enigma.kernel functions.
    Maps to a Metal device function rather than a kernel function."""
```

### 8.3 Thread Hierarchy

```python
# Thread index accessors (within @enigma.kernel)
class ThreadIdx:
    """Maps to Metal kernel attributes."""

    # Threadgroup position in grid
    threadgroup_position_in_grid_x: Int32    # [[threadgroup_position_in_grid]].x
    threadgroup_position_in_grid_y: Int32    # [[threadgroup_position_in_grid]].y
    threadgroup_position_in_grid_z: Int32    # [[threadgroup_position_in_grid]].z

    # Thread position within threadgroup
    thread_position_in_threadgroup_x: Int32  # [[thread_position_in_threadgroup]].x
    thread_position_in_threadgroup_y: Int32  # [[thread_position_in_threadgroup]].y

    # SIMD-specific
    thread_index_in_simdgroup: Int32         # [[thread_index_in_simdgroup]]
    simdgroup_index_in_threadgroup: Int32    # [[simdgroup_index_in_threadgroup]]
    simdgroups_per_threadgroup: Int32        # [[simdgroups_per_threadgroup]]
    threads_per_threadgroup: Int32           # [[threads_per_threadgroup]]

# Synchronization
def threadgroup_barrier(mem_flags="mem_threadgroup"):
    """metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup)"""

def simd_is_first() -> Boolean:
    """metal::simd_is_first() — true only for thread 0 of simdgroup."""
```

### 8.4 Control Flow

```python
def range(start, stop, step=1):
    """Enigma range — generates scf.for in MLIR.
    Static bounds: fully unrolled at compile time.
    Dynamic bounds: generates actual loop in Metal."""

def static_range(start, stop, step=1):
    """Always-unrolled range. Guaranteed compile-time expansion."""

def if_then(condition):
    """Context manager for conditional execution → scf.if"""

def while_loop(condition_fn):
    """Context manager for while loops → scf.while"""
```

---

## 9. Module 7: Compilation Pipeline

**Files**: `enigma/compiler/compiler.py`
**Dependencies**: `ast_rewriter.py`, `kernel.py`, Enigma-Dialect (C++/pybind11)

### 9.1 Compilation API

```python
def compile(
    kernel_fn,
    # Compilation options
    opt_level: int = 2,              # 0=no opt, 1=basic, 2=standard, 3=aggressive
    metal_language_version: str = "3.1",  # Metal Shading Language version
    target_os: str = "macosx",       # Target OS for xcrun

    # Debug options
    keep_metal_source: bool = False,  # Save .metal file
    keep_air: bool = False,           # Save .air file
    generate_line_info: bool = False,  # Debug info in Metal
    dump_mlir: bool = False,          # Print MLIR at each stage

    # TVM FFI (optional fast path)
    enable_tvm_ffi: bool = False,
) -> "CompiledKernel":
    """Compile an @enigma.kernel function to a Metal library.

    Pipeline:
    1. AST rewrite the kernel function
    2. Execute with proxy tensors → build MLIR module
    3. Run MLIR optimization passes
    4. Emit .metal source text (metal-translate)
    5. Invoke xcrun metal to compile → .air
    6. Invoke xcrun metallib to package → .metallib
    7. Return CompiledKernel handle
    """
```

### 9.2 MLIR Pass Pipeline

```python
def build_pass_pipeline(opt_level: int) -> List[str]:
    """Construct the MLIR pass pipeline.

    Standard pipeline (opt_level=2):
    1. canonicalize                    — fold constants, simplify ops
    2. affine-loop-normalize           — normalize affine loops
    3. lower-affine                    — affine.apply → arith ops
    4. convert-scf-to-cf               — scf.for → cf.br/cf.cond_br
    5. loop-invariant-code-motion      — hoist invariants out of loops
    6. cse                             — common subexpression elimination

    NOTE: metal.* ops are deliberately untouched by standard passes.
    They pass through to the metal-translate stage.

    Aggressive pipeline (opt_level=3) adds:
    7. memref-expand                   — expand complex memref ops
    8. normalize-memrefs               — normalize memref layouts
    """
```

### 9.3 Metal Source Emission

```python
class MetalSourceEmitter:
    """Walks the lowered MLIR and emits Metal C++ text.

    This is the TypeSwitch-based emitter that converts MLIR ops
    to Metal source code. It handles:

    Indexing:
      metal.threadgroup_position_in_grid_x  →  tgid.x
      metal.thread_position_in_threadgroup_x →  tid.x
      arith.muli %a, %b                     →  a * b
      arith.addi %a, %b                     →  a + b

    Memory:
      memref.alloc addrspace(3)  →  threadgroup float name[N];
      memref.alloc (private)     →  float name[N];
      memref.subview             →  pointer arithmetic

    Hardware ops:
      metal.threadgroup_barrier
        → threadgroup_barrier(mem_flags::mem_threadgroup);
      metal.simdgroup_load
        → simdgroup_load(mat, ptr, elements_per_row, ulong2(x,y), transpose);
      metal.simdgroup_store
        → simdgroup_store(mat, ptr, elements_per_row, ulong2(x,y));
      metal.simdgroup_multiply
        → simdgroup_multiply(d, a, b);
      metal.simdgroup_multiply_accumulate
        → simdgroup_multiply_accumulate(d, a, b, c);
      metal.simd_sum       → simd_sum(value);
      metal.simd_max       → simd_max(value);
      metal.simd_shuffle    → simd_shuffle(value, lane);
      metal.simd_broadcast  → simd_broadcast(value, lane);
      metal.fma             → fma(a, b, c);
      metal.dot             → dot(a, b);
      metal.precise_exp     → precise::exp(value);
      metal.precise_rsqrt   → precise::rsqrt(value);
      metal.atomic_fetch_add → atomic_fetch_add_explicit(...);

    Control flow:
      cf.br / cf.cond_br blocks  →  reconstructed for/while/if
    """
```

### 9.4 Compiled Kernel Object

```python
class CompiledKernel:
    """The result of enigma.compile(). Ready for dispatch."""

    metallib_bytes: bytes          # .metallib binary data
    kernel_name: str               # Entry point name

    # Metadata from compilation
    threadgroup_memory_size: int   # Bytes of threadgroup memory needed
    max_threads_per_threadgroup: int  # Hardware limit

    # Optional debug artifacts
    metal_source: Optional[str]    # .metal text if keep_metal_source=True
    mlir_text: Optional[str]       # MLIR text if dump_mlir=True

    def __call__(self, *args, grid, threads, **kwargs):
        """Dispatch the kernel with given arguments and launch config.

        Example:
          kernel = enigma.compile(matmul)
          kernel(A, B, C, grid=(2,2,1), threads=(8,8,1))
        """
```

### 9.5 TVM FFI Integration (Optional Fast Path)

```python
class TVMFFIProvider:
    """Optional TVM FFI integration for fast kernel execution.

    Instead of going through .metal → xcrun → .metallib,
    TVM FFI can compile and execute kernels directly.

    This is the 'cute.compile with TVM FFI' equivalent:
    1. MLIR module → TVM relay/TIR
    2. TVM builds Metal kernel directly
    3. Execute via TVM runtime (no xcrun subprocess)

    Tradeoff: faster compile time, but less control over Metal optimizations.
    """

    def compile_via_tvm(self, mlir_module) -> "TVMCompiledKernel":
        """Compile MLIR to executable kernel via TVM."""

    def execute(self, kernel, *args, grid, threads):
        """Execute TVM-compiled kernel."""
```

---

## 10. Module 8: Runtime Dispatch

**Files**: `enigma/runtime_dispatch/runtime.py`, `enigma/runtime_dispatch/swift/libenigma_runtime.swift`
**Dependencies**: `ctypes` (Python), Metal framework (Swift)

### 10.1 Swift Runtime Dylib

```swift
// libenigma_runtime.swift (~150 lines)
// Compiled with: swiftc -emit-library -o libenigma_runtime.dylib

import Metal
import Foundation

// --- C API exported functions ---

@_cdecl("enigma_create_device")
func enigma_create_device() -> OpaquePointer {
    let device = MTLCreateSystemDefaultDevice()!
    return Unmanaged.passRetained(device).toOpaque()
}

@_cdecl("enigma_load_library")
func enigma_load_library(
    _ devicePtr: OpaquePointer,
    _ data: UnsafeRawPointer,
    _ len: Int,
    _ name: UnsafePointer<CChar>
) -> OpaquePointer {
    // MTLDevice.makeLibrary(data:) → makeFunction → makeComputePipelineState
}

@_cdecl("enigma_create_buffer")
func enigma_create_buffer(
    _ devicePtr: OpaquePointer,
    _ data: UnsafeRawPointer?,
    _ len: Int,
    _ options: UInt
) -> OpaquePointer {
    // MTLDevice.makeBuffer(bytes:length:options:)
}

@_cdecl("enigma_dispatch")
func enigma_dispatch(
    _ psoPtr: OpaquePointer,
    _ queuePtr: OpaquePointer,
    _ bufPtrs: UnsafePointer<OpaquePointer>,
    _ bufCount: Int,
    _ gridX: Int, _ gridY: Int, _ gridZ: Int,
    _ threadsX: Int, _ threadsY: Int, _ threadsZ: Int,
    _ threadgroupMemSize: Int
) {
    // MTLCommandBuffer → MTLComputeCommandEncoder → dispatch
}

@_cdecl("enigma_synchronize")
func enigma_synchronize(_ queuePtr: OpaquePointer) {
    // cmdBuf.waitUntilCompleted()
}
```

### 10.2 Python Runtime Wrapper

```python
class MetalRuntime:
    """Python-side runtime using ctypes to call Swift dylib.

    Usage:
        runtime = MetalRuntime()
        kernel = runtime.load_kernel(metallib_bytes, "matmul")
        buf_a = runtime.create_buffer(numpy_array_a)
        buf_b = runtime.create_buffer(numpy_array_b)
        buf_c = runtime.create_buffer(size=output_size)
        runtime.dispatch(kernel, [buf_a, buf_b, buf_c],
                        grid=(2,2,1), threads=(8,8,1))
        runtime.synchronize()
        result = runtime.read_buffer(buf_c, dtype=np.float32)
    """

    def __init__(self):
        self._lib = ctypes.CDLL("libenigma_runtime.dylib")
        self._device = self._lib.enigma_create_device()
        self._queue = self._lib.enigma_create_queue(self._device)

    def load_kernel(self, metallib_bytes: bytes, name: str) -> "KernelHandle":
        """Load .metallib and create compute pipeline state."""

    def create_buffer(self, data=None, size=None, dtype=None) -> "BufferHandle":
        """Create MTLBuffer from numpy array or raw size."""

    def dispatch(self, kernel, buffers, grid, threads,
                 threadgroup_mem_size=0):
        """Dispatch kernel to GPU."""

    def synchronize(self):
        """Wait for all GPU work to complete."""

    def read_buffer(self, buffer, dtype, shape=None) -> np.ndarray:
        """Read MTLBuffer contents back to numpy array."""
```

### 10.3 Metal Buffer Abstraction

```python
class MetalBuffer:
    """Wraps MTLBuffer with numpy-compatible interface.

    Supports:
    - Zero-copy sharing between CPU and GPU (unified memory on Apple Silicon)
    - DLPack protocol for framework interop
    - Automatic dtype tracking
    """

    handle: OpaquePointer          # MTLBuffer opaque pointer
    size_bytes: int
    dtype: Type[Numeric]

    def to_numpy(self) -> np.ndarray:
        """Zero-copy view as numpy array (Apple unified memory)."""

    def __dlpack__(self):
        """DLPack protocol for cross-framework sharing."""

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "MetalBuffer":
        """Create buffer from numpy array."""

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> "MetalBuffer":
        """Create buffer from PyTorch MPS tensor."""
```

---

## 11. Module 9: Fake Tensors & Dynamic Shapes

**File**: `enigma/runtime.py` (~300 LOC)
**Dependencies**: `tensor.py`, `typing.py`

### 11.1 Fake Tensors (for compilation)

```python
class FakeTensor(Tensor):
    """Tensor with known shape/layout but no data — for tracing.

    Used in enigma.compile():
        # Create fake tensors matching expected input shapes
        A_fake = enigma.fake_tensor(shape=(128, 128), dtype=enigma.f32)
        B_fake = enigma.fake_tensor(shape=(128, 128), dtype=enigma.f32)
        C_fake = enigma.fake_tensor(shape=(128, 128), dtype=enigma.f32)

        # Compile with fake tensors (no actual data needed)
        kernel = enigma.compile(matmul, A_fake, B_fake, C_fake)

    All layout operations work on fake tensors (they're pure math).
    Only actual memory access (load/store) generates IR ops.
    """

    shape: Shape
    stride: Stride
    dtype: Type[Numeric]
    memspace: AddressSpace = AddressSpace.device
    _is_fake: bool = True

    def mark_dynamic(self, dim: int, divisibility: int = 1):
        """Mark a dimension as dynamic (unknown at compile time).
        The generated kernel will use a runtime parameter for this dimension."""
```

### 11.2 Dynamic Shapes

```python
class SymInt:
    """Symbolic integer for dynamic shapes.

    Tracks:
    - Bit width (32 or 64)
    - Known divisibility (e.g., divisible by 16 for alignment)
    - Optional symbol name for debugging

    Example:
        M = SymInt(32, divisibility=64, symbol="M")  # M is a multiple of 64
        N = SymInt(32, divisibility=64, symbol="N")
        A_fake = enigma.fake_tensor(shape=(M, N), dtype=enigma.f32)
    """

    width: int
    divisibility: int
    symbol: Optional[str]
```

### 11.3 DLPack Bridge

```python
def from_dlpack(tensor) -> Tensor:
    """Create Enigma tensor from any DLPack-compatible source.

    Supports: numpy arrays, PyTorch tensors, JAX arrays, etc.

    Extracts:
    - Data pointer
    - Shape and strides → Layout
    - Element type → Numeric type
    - Device info → AddressSpace
    """

def to_dlpack(tensor: Tensor):
    """Export Enigma tensor via DLPack protocol."""
```

---

## 12. Module 10: PyTorch Integration

**File**: `enigma/torch_integration/` (~200 LOC)
**Dependencies**: `runtime.py`, PyTorch

### 12.1 PyTorch Custom Op

```python
@torch.library.custom_op("enigma::matmul", mutates_args=("C",))
def enigma_matmul(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
    """Register Enigma kernels as PyTorch custom ops.

    Works with torch.compile and MPS backend:
    1. Get MPS buffer pointers from PyTorch tensors
    2. Dispatch Enigma-compiled kernel on same Metal queue
    3. Results visible immediately (same unified memory)
    """
```

### 12.2 MPS Tensor Bridge

```python
class MPSTensorBridge:
    """Bridge between PyTorch MPS tensors and Enigma buffers.

    On Apple Silicon, PyTorch MPS and Enigma share unified memory.
    This bridge extracts the MTLBuffer pointer from a PyTorch MPS tensor
    so Enigma kernels can operate on it directly without copying.
    """

    @staticmethod
    def get_metal_buffer(tensor: torch.Tensor) -> MetalBuffer:
        """Extract MTLBuffer from PyTorch MPS tensor."""

    @staticmethod
    def wrap_metal_buffer(buffer: MetalBuffer, shape, dtype) -> torch.Tensor:
        """Wrap MetalBuffer as PyTorch MPS tensor."""
```

---

## 13. Metal Feature Coverage Matrix

Every Metal feature used in gpt-oss-metal-kernels mapped to its Enigma DSL representation:

### 13.1 Math Intrinsics

| Metal API | Enigma DSL | Emitted Metal | Used In |
|-----------|-----------|---------------|---------|
| `metal::fma(a,b,c)` | `enigma.fma(a, b, c)` | `metal::fma(a, b, c)` | matmul, SDPA, accumulate |
| `metal::dot(a,b)` | `enigma.dot(a, b)` | `metal::dot(q, kval)` | SDPA (Q·K scores) |
| `metal::precise::exp(x)` | `enigma.precise.exp(x)` | `metal::precise::exp(x)` | softmax, SDPA, RoPE |
| `metal::fast::exp(x)` | `enigma.fast.exp(x)` | `metal::fast::exp(x)` | SDPA fast path |
| `metal::precise::rsqrt(x)` | `enigma.precise.rsqrt(x)` | `metal::precise::rsqrt(x)` | RMSNorm |
| `metal::precise::sincos(x,s,c)` | `enigma.precise.sincos(x)` | `metal::precise::sincos(x, s, c)` | RoPE |
| `metal::mix(a,b,t)` | `enigma.mix(a, b, t)` | `metal::mix(a, b, t)` | RoPE (YaRN) |
| `metal::saturate(x)` | `enigma.saturate(x)` | `metal::saturate(x)` | RoPE (YaRN alpha) |
| `metal::clamp(x,lo,hi)` | `enigma.clamp(x, lo, hi)` | `metal::clamp(x, lo, hi)` | MOE SwiGLU |
| `metal::rotate(x,n)` | `enigma.rotate(x, n)` | `metal::rotate(x, n)` | RNG |
| `metal::abs(x)` | `enigma.abs(x)` | `metal::abs(x)` | General |
| `metal::min(a,b)` | `enigma.min(a, b)` | `metal::min(a, b)` | General |
| `metal::max(a,b)` | `enigma.max(a, b)` | `metal::max(a, b)` | General |

### 13.2 SIMD Group Operations

| Metal API | Enigma DSL | Description | Since |
|-----------|-----------|-------------|-------|
| `simd_sum(v)` | `enigma.simd_sum(v)` | Sum across 32-thread simdgroup | MSL 2.1 |
| `simd_max(v)` | `enigma.simd_max(v)` | Max across simdgroup | MSL 2.1 |
| `simd_min(v)` | `enigma.simd_min(v)` | Min across simdgroup | MSL 2.1 |
| `simd_is_first()` | `enigma.simd_is_first()` | True for thread 0 of simdgroup | MSL 2.1 |
| `simd_prefix_inclusive_sum(v)` | `enigma.simd_prefix_inclusive_sum(v)` | Parallel prefix sum (scan) | MSL 2.1 |
| `simd_shuffle(v, lane)` | `enigma.simd_shuffle(v, lane)` | Read from specific lane | MSL 2.0 |
| `simd_shuffle_down(v, delta)` | `enigma.simd_shuffle_down(v, delta)` | Read from lane + delta | MSL 2.0 |
| `simd_shuffle_up(v, delta)` | `enigma.simd_shuffle_up(v, delta)` | Read from lane - delta | MSL 2.0 |
| `simd_shuffle_xor(v, mask)` | `enigma.simd_shuffle_xor(v, mask)` | Read from lane ^ mask | MSL 2.0 |
| `simd_broadcast(v, lane)` | `enigma.simd_broadcast(v, lane)` | Broadcast from one lane to all | MSL 2.0 |

> All SIMD-group functions are in the `metal` namespace but idiomatic Metal uses
> `using namespace metal;` so they are called without the prefix.

### 13.3 Simdgroup Matrix Operations

| Metal API | Enigma DSL | Description | Since |
|-----------|-----------|-------------|-------|
| `make_filled_simdgroup_matrix<T,R,C>(val)` | `enigma.make_simdgroup_matrix(dtype, rows, cols, fill)` | Create 8x8 matrix | MSL 2.3 |
| `simdgroup_load(mat, ptr, stride)` | `enigma.simdgroup_load(mat, tensor)` | Load 8x8 tile from threadgroup/device | MSL 2.3 |
| `simdgroup_load(mat, ptr, stride, ulong2, transpose)` | `enigma.simdgroup_load(mat, tensor, offset, transpose)` | Load with offset/transpose | MSL 2.3 |
| `simdgroup_store(mat, ptr, stride)` | `enigma.simdgroup_store(mat, tensor)` | Store 8x8 tile to threadgroup/device | MSL 2.3 |
| `simdgroup_multiply(D,A,B)` | `enigma.simdgroup_multiply(D, A, B)` | D = A @ B (8x8) | MSL 2.3 |
| `simdgroup_multiply_accumulate(D,A,B,C)` | `enigma.simdgroup_multiply_accumulate(D, A, B, C)` | D = A @ B + C (8x8) | MSL 2.3 |

> **NOTE**: `simdgroup_async_copy` is **NOT** a public Metal API. It is an undocumented
> internal instruction. Data movement from device→threadgroup must use cooperative
> per-thread vectorized copies (float4) followed by `threadgroup_barrier`, which is
> exactly what gpt-oss-metal-kernels does.

### 13.4 Threadgroup Operations

| Metal API | Enigma DSL | Description |
|-----------|-----------|-------------|
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `enigma.threadgroup_barrier()` | Full threadgroup sync |
| `threadgroup float buf[N]` | `enigma.make_smem_tensor(layout, dtype)` | Shared memory allocation |

### 13.5 Atomic Operations

| Metal API | Enigma DSL | Description |
|-----------|-----------|-------------|
| `atomic_fetch_add_explicit(&v,x,relaxed)` | `enigma.atomic_fetch_add(ptr, val)` | Atomic add |
| `atomic_store_explicit(&v,x,relaxed)` | `enigma.atomic_store(ptr, val)` | Atomic store |
| `atomic_load_explicit(&v,relaxed)` | `enigma.atomic_load(ptr)` | Atomic load |
| `atomic_min_explicit(&v,x,relaxed)` | `enigma.atomic_min(ptr, val)` | Atomic min (argmax) |

### 13.6 Type System Coverage

| Metal Type | Enigma Type | Vector Variants |
|-----------|-------------|-----------------|
| `float` | `enigma.f32` | `float2`, `float4` |
| `half` | `enigma.f16` | `half4` |
| `bfloat` | `enigma.bf16` | `bfloat4` |
| `uint` | `enigma.u32` | `uint2`, `uint4` |
| `int` | `enigma.i32` | — |
| `ulong` / `uint64_t` | `enigma.u64` | — |
| `uchar` | `enigma.u8` | — |
| `bool` | `enigma.boolean` | — |
| `atomic_uint` | `enigma.AtomicUInt32` | — |
| `atomic_ulong` | `enigma.AtomicUInt64` | — |

### 13.7 Kernel Attributes

| Metal Attribute | Enigma DSL | Description |
|----------------|-----------|-------------|
| `[[buffer(n)]]` | Automatic from argument order | Buffer binding index |
| `[[threadgroup_position_in_grid]]` | `enigma.threadgroup_position_in_grid` | Threadgroup ID |
| `[[thread_position_in_threadgroup]]` | `enigma.thread_position_in_threadgroup` | Thread ID in group |
| `[[thread_index_in_simdgroup]]` | `enigma.thread_index_in_simdgroup` | Thread ID in SIMD |
| `[[simdgroup_index_in_threadgroup]]` | `enigma.simdgroup_index_in_threadgroup` | Which SIMD in group |
| `[[simdgroups_per_threadgroup]]` | `enigma.simdgroups_per_threadgroup` | SIMD count |
| `[[threads_per_threadgroup]]` | `enigma.threads_per_threadgroup` | Thread count |
| `[[max_total_threads_per_threadgroup(N)]]` | `@enigma.kernel(max_threads=N)` | Max threads hint |

### 13.8 Pragmas & Compiler Hints

| Metal Pragma | Enigma DSL | Description |
|-------------|-----------|-------------|
| `#pragma METAL fp math_mode(safe)` | `enigma.compile(..., math_mode="safe")` | Disable fast math |
| `#pragma METAL fp contract(off)` | `enigma.compile(..., fp_contract=False)` | Disable FMA contraction |

---

## 14. Dialect Interface Contract

The Python DSL communicates with the Enigma-Dialect (separate C++ repo) through a pybind11/nanobind boundary. This section defines the contract.

### 14.1 Operations the Dialect Must Define (EnigmaOps.td)

```
// Thread indexing ops
metal.threadgroup_position_in_grid_x : () -> index
metal.threadgroup_position_in_grid_y : () -> index
metal.threadgroup_position_in_grid_z : () -> index
metal.thread_position_in_threadgroup_x : () -> index
metal.thread_position_in_threadgroup_y : () -> index
metal.thread_position_in_threadgroup_z : () -> index
metal.thread_index_in_simdgroup : () -> index
metal.simdgroup_index_in_threadgroup : () -> index
metal.simdgroups_per_threadgroup : () -> index
metal.threads_per_threadgroup : () -> index

// Synchronization
metal.threadgroup_barrier : () -> ()

// Simdgroup matrix operations (public Metal API, MSL 2.3+)
metal.simdgroup_load : (memref, index, index, bool) -> simdgroup_matrix
metal.simdgroup_store : (simdgroup_matrix, memref, index, index) -> ()
metal.simdgroup_multiply : (simdgroup_matrix, simdgroup_matrix) -> simdgroup_matrix
metal.simdgroup_multiply_accumulate : (simdgroup_matrix, simdgroup_matrix, simdgroup_matrix) -> simdgroup_matrix

// SIMD shuffle operations (public Metal API, MSL 2.0+)
metal.simd_shuffle : (f32, i16) -> f32
metal.simd_shuffle_down : (f32, i16) -> f32
metal.simd_shuffle_up : (f32, i16) -> f32
metal.simd_shuffle_xor : (f32, i16) -> f32
metal.simd_broadcast : (f32, i16) -> f32

// SIMD reductions
metal.simd_sum : (f32) -> f32
metal.simd_max : (f32) -> f32
metal.simd_min : (f32) -> f32
metal.simd_is_first : () -> i1
metal.simd_prefix_inclusive_sum : (f32) -> f32

// Math operations
metal.fma : (f32, f32, f32) -> f32
metal.dot : (vector<Nxf32>, vector<Nxf32>) -> f32
metal.precise_exp : (f32) -> f32
metal.fast_exp : (f32) -> f32
metal.precise_rsqrt : (f32) -> f32
metal.precise_sincos : (f32) -> (f32, f32)
metal.mix : (f32, f32, f32) -> f32
metal.saturate : (f32) -> f32
metal.clamp : (f32, f32, f32) -> f32

// Atomic operations
metal.atomic_fetch_add : (memref, f32) -> f32
metal.atomic_store : (memref, f32) -> ()
metal.atomic_load : (memref) -> f32
metal.atomic_min : (memref, u32) -> u32
```

### 14.2 Types the Dialect Must Define (EnigmaTypes.td)

```
// Layout attribute — lives in memref type
#metal.layout<"(shape):(stride)">

// Address spaces
// 0 = device, 2 = constant, 3 = threadgroup

// Simdgroup matrix type
!metal.simdgroup_matrix<element_type, rows, cols>
```

### 14.3 Python Bindings the Dialect Must Expose

```python
# The dialect's nanobind/pybind11 module must expose:
class EnigmaIRBuilder:
    """Builds MLIR ops from Python calls."""

    def create_module(self, name: str) -> mlir.Module
    def create_kernel_func(self, name, arg_types, attrs) -> mlir.FuncOp

    # Thread indexing
    def threadgroup_position_in_grid(self, dim: int) -> mlir.Value
    def thread_position_in_threadgroup(self, dim: int) -> mlir.Value
    def thread_index_in_simdgroup(self) -> mlir.Value
    def simdgroup_index_in_threadgroup(self) -> mlir.Value

    # Memory
    def alloc_threadgroup(self, shape, dtype) -> mlir.Value  # addrspace 3
    def alloc_private(self, shape, dtype) -> mlir.Value
    def subview(self, memref, offsets, sizes, strides) -> mlir.Value

    # Simdgroup matrix ops (MSL 2.3+)
    def simdgroup_load(self, memref, stride, offset, transpose=False) -> mlir.Value
    def simdgroup_store(self, matrix, memref, stride, offset)
    def simdgroup_multiply(self, a, b) -> mlir.Value
    def simdgroup_multiply_accumulate(self, a, b, c) -> mlir.Value

    # SIMD shuffle ops (MSL 2.0+)
    def simd_shuffle(self, value, lane) -> mlir.Value
    def simd_shuffle_down(self, value, delta) -> mlir.Value
    def simd_shuffle_up(self, value, delta) -> mlir.Value
    def simd_shuffle_xor(self, value, mask) -> mlir.Value
    def simd_broadcast(self, value, lane) -> mlir.Value

    # SIMD reductions
    def simd_sum(self, value) -> mlir.Value
    def simd_max(self, value) -> mlir.Value
    def simd_min(self, value) -> mlir.Value
    def simd_is_first(self) -> mlir.Value

    # Math
    def fma(self, a, b, c) -> mlir.Value
    def precise_exp(self, x) -> mlir.Value
    def precise_rsqrt(self, x) -> mlir.Value
    # ... etc for all math ops

    # Synchronization
    def threadgroup_barrier(self)

    # Control flow
    def scf_for(self, lb, ub, step) -> mlir.ForOp
    def scf_if(self, condition) -> mlir.IfOp

    # Pass pipeline
    def run_passes(self, module, passes: List[str])

    # Translation
    def translate_to_metal(self, module) -> str  # MLIR → .metal source text
```

---

## 15. Testing Strategy

### 15.1 Unit Tests (No GPU Required)

```
tests/
├── test_layout.py          # Layout algebra: make_layout, composition, complement, etc.
├── test_tuple.py           # Hierarchical tuple operations
├── test_types.py           # Type system, conversions
├── test_tensor.py          # Tensor construction, slicing, partitioning
├── test_ast_rewriter.py    # AST transformation correctness
├── test_fake_tensor.py     # Fake tensor tracing
```

**Layout algebra tests** (port from CuTe's test suite):
- `composition(A, identity) == A`
- `composition(identity, B) == B`
- `complement(complement(A)) ≠ A` (complement is not involutory)
- `size(logical_product(A, B)) == size(A) * size(B)`
- `size(logical_divide(A, B)) == size(A)`
- `right_inverse(A)` produces valid coordinates
- `coalesce` preserves functional semantics
- Divisibility condition violations raise errors at trace time

### 15.2 Integration Tests (Require Apple Silicon GPU)

```
tests/
├── test_compile_basic.py    # Compile simple kernels, verify .metallib output
├── test_compile_gemm.py     # Compile GEMM with various tile sizes
├── test_runtime.py          # Metal runtime: buffer create, dispatch, sync
├── test_metal_ops.py        # Each Metal op: simd_sum, fma, barrier, etc.
├── test_algorithms.py       # gemm, softmax, rmsnorm, rope, sdpa
├── test_autovec_copy.py     # Vectorized copy correctness
├── test_pytorch.py          # PyTorch MPS interop
├── test_quantized.py        # Quantized type support
├── test_moe.py              # MOE pipeline: topk, scatter, accumulate
```

**Numerical correctness**: Every algorithm test compares GPU output against numpy reference:
```python
def test_gemm_correctness():
    A = np.random.randn(128, 128).astype(np.float32)
    B = np.random.randn(128, 128).astype(np.float32)
    C_ref = A @ B

    kernel = enigma.compile(gemm_kernel)
    C_gpu = run_kernel(kernel, A, B)
    np.testing.assert_allclose(C_gpu, C_ref, rtol=1e-5)
```

### 15.3 Validation Tests (Match gpt-oss-metal-kernels Output)

Port each kernel from gpt-oss-metal-kernels as a reference:
- `test_match_matmul.py` — Compare against `gptoss_f32_bf16w_matmul`
- `test_match_rmsnorm.py` — Compare against `gptoss_f32_rmsnorm`
- `test_match_sdpa.py` — Compare against `gptoss_f32_sdpa_q8_d64`
- `test_match_rope.py` — Compare against `gptoss_f32_rope`
- `test_match_dense_matmul.py` — Compare against `gptoss_f32_bf16w_dense_matmul_*`

---

## 16. Dependency Graph & Build Order

### 16.1 Module Dependencies

```
                      ┌──────────┐
                      │ tuple.py │  (no dependencies)
                      └────┬─────┘
                           │
                      ┌────▼─────┐
                      │ core.py  │  (layout algebra, depends on tuple.py)
                      └────┬─────┘
                           │
                 ┌─────────┼──────────┐
                 │         │          │
            ┌────▼───┐ ┌──▼────┐ ┌───▼────┐
            │typing.py│ │math.py│ │tuple.py│
            └────┬───┘ └──┬────┘ └────────┘
                 │         │
            ┌────▼─────────▼──┐
            │    tensor.py    │
            └────┬────────────┘
                 │
         ┌───────┼──────────┐
         │       │          │
    ┌────▼──┐ ┌──▼───┐ ┌───▼──────┐
    │atom.py│ │arch/* │ │runtime.py│
    └────┬──┘ └──┬───┘ └───┬──────┘
         │       │          │
    ┌────▼───────▼──┐  ┌───▼──────────────┐
    │ algorithm.py  │  │ fake tensors /    │
    └───────┬───────┘  │ dynamic shapes    │
            │          └───┬──────────────┘
    ┌───────▼───────┐      │
    │compiler/      │◄─────┘
    │ ast_rewriter  │
    │ kernel.py     │
    │ compiler.py   │
    └───────┬───────┘
            │                    ┌──────────────────┐
            │                    │  Enigma-Dialect   │
            ├────────────────────► (C++ / pybind11)  │
            │                    │  EnigmaOps.td     │
            │                    │  MetalTranslate   │
            │                    └──────────────────┘
    ┌───────▼───────┐
    │runtime_dispatch│
    │ runtime.py    │
    │ swift dylib   │
    │ buffer.py     │
    └───────┬───────┘
            │
    ┌───────▼──────────┐
    │torch_integration/│
    │ custom_op.py     │
    │ dlpack.py        │
    └──────────────────┘
```

### 16.2 Implementation Order

**Phase 1 — Foundation (pure Python, no C++ dependencies)**
1. `tuple.py` — Hierarchical tuple utilities
2. `core.py` — Layout algebra engine (largest module)
3. `typing.py` — Type system
4. `tensor.py` — Tensor abstraction
5. Unit tests for all of the above

**Phase 2 — Hardware Abstraction**
6. `arch/simdgroup.py` — Simdgroup matrix abstraction
7. `arch/threadgroup.py` — Threadgroup memory and barriers
8. `atom.py` — MMA and Copy atoms for Metal
9. `math.py` — Metal math function wrappers

**Phase 3 — Algorithms**
10. `algorithm.py` — copy, autovec_copy, gemm, reduce, softmax, etc.

**Phase 4 — Compilation**
11. `compiler/ast_rewriter.py` — AST transformation
12. `compiler/kernel.py` — @enigma.kernel decorator
13. `compiler/compiler.py` — Full compile pipeline (needs Enigma-Dialect)

**Phase 5 — Runtime**
14. `runtime_dispatch/swift/libenigma_runtime.swift` — Swift dylib
15. `runtime_dispatch/runtime.py` — Python ctypes wrapper
16. `runtime_dispatch/buffer.py` — MetalBuffer abstraction

**Phase 6 — Integration**
17. `runtime.py` — Fake tensors, DLPack, dynamic shapes
18. `torch_integration/` — PyTorch custom ops and MPS bridge
19. End-to-end integration tests

### 16.3 Example: Complete Matmul Kernel

This shows what a user writes vs. what gets generated:

```python
import enigma

@enigma.kernel(max_threads=256)
def matmul(A: enigma.Tensor, B: enigma.Tensor, C: enigma.Tensor):
    # Thread indices
    tgid = enigma.threadgroup_position_in_grid
    tid = enigma.thread_position_in_threadgroup
    simd_tid = enigma.thread_index_in_simdgroup
    simd_idx = enigma.simdgroup_index_in_threadgroup

    # Layout algebra: partition A into 64x64 tiles
    gA = A.zipped_divide((64, 64))
    blk_A = gA.local_tile((64, 64), coord=(tgid.y, 0))  # select tile

    gB = B.zipped_divide((64, 64))
    blk_B = gB.local_tile((64, 64), coord=(0, tgid.x))

    # Allocate threadgroup memory
    sA = enigma.make_smem_tensor(
        enigma.make_layout((64, 64), (64, 1)), enigma.f32)
    sB = enigma.make_smem_tensor(
        enigma.make_layout((64, 64), (64, 1)), enigma.f32)

    # Register tile for accumulator
    rC = enigma.make_rmem_tensor(
        enigma.make_layout((8, 8), (8, 1)), enigma.f32, init=0.0)

    # Main loop over K tiles
    K_tiles = A.shape[1] // 64
    for k in enigma.range(0, K_tiles):
        # Cooperative copy: all threads load tiles into threadgroup memory
        # (vectorized float4 loads, matching gpt-oss-metal-kernels pattern)
        enigma.cooperative_copy(blk_A[:, k], sA)  # device → threadgroup
        enigma.cooperative_copy(blk_B[k, :], sB)  # device → threadgroup
        enigma.threadgroup_barrier()

        # Simdgroup matrix multiply-accumulate (8x8 tiles)
        enigma.simdgroup_multiply_accumulate(sA, sB, rC)
        enigma.threadgroup_barrier()

    # Store result
    gC = C.zipped_divide((64, 64))
    blk_C = gC.local_tile((64, 64), coord=(tgid.y, tgid.x))
    enigma.simdgroup_store(rC, blk_C)

# Compile with fake tensors
A = enigma.fake_tensor(shape=(128, 128), dtype=enigma.f32)
B = enigma.fake_tensor(shape=(128, 128), dtype=enigma.f32)
C = enigma.fake_tensor(shape=(128, 128), dtype=enigma.f32)

kernel = enigma.compile(matmul, A, B, C)

# Execute with real data
import numpy as np
runtime = enigma.MetalRuntime()
A_buf = runtime.create_buffer(np.random.randn(128, 128).astype(np.float32))
B_buf = runtime.create_buffer(np.random.randn(128, 128).astype(np.float32))
C_buf = runtime.create_buffer(size=128*128*4)

kernel(A_buf, B_buf, C_buf, grid=(2, 2, 1), threads=(8, 8, 1))
runtime.synchronize()

result = runtime.read_buffer(C_buf, dtype=np.float32, shape=(128, 128))
```

---

## Appendix A: Key Differences from CuTe DSL

| Aspect | CuTe DSL (CUDA) | Enigma DSL (Metal) |
|--------|-----------------|-------------------|
| **MMA unit** | Many variants (Volta, Ampere, Hopper, Blackwell) | Single: simdgroup 8x8 matrix |
| **Warp size** | 32 threads (warp) | 32 threads (simdgroup) |
| **Shared memory** | CUDA shared memory (addrspace 3) | Metal threadgroup memory (addrspace 3) |
| **Async copy** | cp.async / TMA | None public — use cooperative threadgroup copy (float4 vectorized) |
| **Tensor cores** | Dedicated tensor core units | SIMD ALU with matrix extensions |
| **Backend compiler** | nvcc / ptxas | xcrun metal / metallib |
| **IR target** | PTX → SASS | Metal C++ → AIR → GPU ISA |
| **Cache control** | Explicit (L1/L2 hints) | Implicit (Metal manages caches) |
| **Cluster** | Thread block cluster (Hopper+) | Not available |
| **Tensor memory** | TMEM (Blackwell) | Not available |
| **Vector width** | Up to 256 bits | Up to 128 bits practical (float4) |
| **Quantization** | INT4/INT8 tensor cores | Software dequantization (gpt-oss pattern) |
| **Runtime API** | CUDA driver API | Metal API via Swift dylib |

## Appendix B: Metal Shading Language Version Support

Minimum target: **MSL 2.3** (macOS 11+, Apple Silicon M1+)
Recommended target: **MSL 3.1** (macOS 14+) for bfloat support

**MSL 2.0+ (macOS 10.13+)**:
- `simd_shuffle`, `simd_shuffle_down`, `simd_shuffle_up`, `simd_shuffle_xor`
- `simd_broadcast`

**MSL 2.1+ (macOS 10.14+)**:
- `simd_sum`, `simd_max`, `simd_min`, `simd_is_first`
- `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum`

**MSL 2.3+ (macOS 11+, Apple Silicon M1+)**:
- `simdgroup_float8x8` / `simdgroup_half8x8` — simdgroup matrix types
- `simdgroup_load`, `simdgroup_store` — 8x8 tile load/store
- `simdgroup_multiply`, `simdgroup_multiply_accumulate` — matrix MMA
- `atomic_fetch_add_explicit` with relaxed ordering

**MSL 3.1+ (macOS 14+, Apple Silicon M1+)**:
- `simdgroup_bfloat8x8` — bfloat16 simdgroup matrix type
- `bfloat` native type
- `precise::exp`, `precise::rsqrt`, `precise::sincos`
- `fma`, `dot`, `mix`, `saturate`

**MSL 4.0+ (macOS 26+, Apple Silicon M4+)** — future consideration:
- `cooperative_tensor` — higher-level tensor abstraction (may supersede simdgroup_matrix)
- Investigate when available; Enigma DSL should be designed to support both backends

> **IMPORTANT**: `simdgroup_async_copy` is NOT a public Metal API. It is an undocumented
> internal instruction. Data movement uses cooperative per-thread vectorized copies.

## Appendix C: Complete File Listing with Estimated LOC

| File | LOC | Description |
|------|-----|-------------|
| `enigma/__init__.py` | ~100 | Public API exports |
| `enigma/core.py` | ~1500 | Layout algebra engine |
| `enigma/typing.py` | ~400 | Type system |
| `enigma/tensor.py` | ~600 | Tensor abstraction |
| `enigma/tuple.py` | ~300 | Hierarchical tuple utils |
| `enigma/math.py` | ~200 | Metal math wrappers |
| `enigma/atom.py` | ~400 | MMA and Copy atom definitions |
| `enigma/algorithm.py` | ~500 | Algorithms (gemm, copy, reduce, ...) |
| `enigma/testing.py` | ~100 | Test utilities |
| `enigma/runtime.py` | ~300 | Fake tensors, DLPack, SymInt |
| `enigma/arch/simdgroup.py` | ~200 | Simdgroup abstractions |
| `enigma/arch/threadgroup.py` | ~100 | Threadgroup abstractions |
| `enigma/arch/atomic.py` | ~100 | Atomic operations |
| `enigma/arch/numeric_conversion.py` | ~100 | Type conversion helpers |
| `enigma/metal/simdgroup/copy.py` | ~100 | Async copy atoms |
| `enigma/metal/simdgroup/mma.py` | ~200 | MMA atoms |
| `enigma/metal/simdgroup/reduce.py` | ~100 | Reduction atoms |
| `enigma/metal/threadgroup/memory.py` | ~100 | Threadgroup memory |
| `enigma/metal/threadgroup/barrier.py` | ~50 | Barriers |
| `enigma/metal/common.py` | ~150 | Universal Metal ops |
| `enigma/compiler/ast_rewriter.py` | ~300 | AST transformation |
| `enigma/compiler/kernel.py` | ~200 | @kernel decorator |
| `enigma/compiler/compiler.py` | ~300 | Compile pipeline + pass mgr |
| `enigma/compiler/tvm_ffi_provider.py` | ~150 | TVM FFI (optional) |
| `enigma/export/metal_emitter.py` | ~400 | MLIR → .metal text |
| `enigma/runtime_dispatch/runtime.py` | ~200 | Python ctypes wrapper |
| `enigma/runtime_dispatch/swift/*.swift` | ~150 | Swift dylib |
| `enigma/runtime_dispatch/buffer.py` | ~150 | MetalBuffer abstraction |
| `enigma/torch_integration/custom_op.py` | ~100 | PyTorch custom op |
| `enigma/torch_integration/dlpack.py` | ~100 | DLPack bridge |
| **Total Python** | **~7150** | |
| **Total Swift** | **~150** | |
| **Total (excl. dialect C++)** | **~7300** | |
