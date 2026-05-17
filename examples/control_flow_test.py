#!/usr/bin/env python3
"""Control flow examples: for_range, if_, while_.

Tests IR tracing correctness for all three control flow constructs.
Once the dialect wheel registers the SCF Python bindings, these will
compile to valid MSL and run on the GPU.

Kernels tested:
  1. array_sum        — for_range: sum N elements
  2. clamp_kernel     — if_/else: clamp values to [lo, hi]
  3. prefix_sum       — for_range + if_: inclusive prefix sum
  4. nested_matmul    — nested for_range: naive matmul inner loop
  5. while_search     — while_: linear search for threshold
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import enigma
from enigma._tracing import KernelBuilder, TracingTensor

# ============================================================================
# Helper: walk all ops (including nested regions) and collect op types
# ============================================================================

def collect_op_types(ops):
    """Recursively collect all op_type strings from an op tree."""
    result = []
    for op in ops:
        result.append(op.op_type)
        for region in op.regions:
            result.extend(collect_op_types(region))
    return result


def count_ops(ops, op_type):
    """Count occurrences of op_type in the full op tree."""
    return collect_op_types(ops).count(op_type)


# ============================================================================
# Test 1: for_range — sum N elements of an array
# ============================================================================
# Each thread sums A[tid*stride .. tid*stride+count) into Out[tid].
#
# Expected MSL (once dialect supports scf):
#   float acc = 0.0;
#   for (int i = 0; i < count; i++) {
#       acc = acc + A[tid * count + i];
#   }
#   Out[tid] = acc;

print("=" * 60)
print("Test 1: for_range — array sum")
print("=" * 60)

@enigma.kernel
def array_sum(A: enigma.f32, Out: enigma.f32):
    tid = enigma.thread_position_in_grid
    acc = enigma.metal_cast(0, "float")
    count = enigma.metal_cast(4, "int")
    with enigma.for_range(0, count) as i:
        idx = tid * count + i
        val = A[idx]
        acc = acc + val
    Out[tid] = acc

# Trace it
builder = KernelBuilder("array_sum")
builder.args.append(("A", 0, "float"))
builder.args.append(("Out", 1, "float"))
with builder:
    A_proxy = TracingTensor("A", 0, "float")
    Out_proxy = TracingTensor("Out", 1, "float")
    tid = enigma.thread_position_in_grid
    acc = enigma.metal_cast(0, "float")
    count = enigma.metal_cast(4, "int")
    with enigma.for_range(0, count) as i:
        idx = tid * count + i
        val = A_proxy[idx]
        acc = acc + val
    Out_proxy[tid] = acc

# Verify IR structure
for_ops = [op for op in builder.ops if op.op_type == "scf_for"]
assert len(for_ops) == 1, f"Expected 1 scf_for, got {len(for_ops)}"
body = for_ops[0].regions[0]
assert count_ops(body, "load") == 1, "For body should have 1 load"
assert count_ops(body, "add") >= 1, "For body should have adds"

# Verify that store is OUTSIDE the for (at top level)
top_types = [op.op_type for op in builder.ops]
assert "store" in top_types, "Store should be at top level (after for)"
assert "scf_for" in top_types, "scf_for should be at top level"

print(f"  IR: {len(builder.ops)} top-level ops")
print(f"  for body: {len(body)} ops")
print(f"  All op types in tree: {collect_op_types(builder.ops)}")

# Simulate expected result (Python reference)
N = 4
CHUNK = 4
a_data = np.arange(N * CHUNK, dtype=np.float32)
expected = np.array([a_data[i*CHUNK:(i+1)*CHUNK].sum() for i in range(N)])
print(f"  Reference (N={N}, chunk={CHUNK}): {expected}")
print("  PASSED\n")


# ============================================================================
# Test 2: if_/else — clamp values
# ============================================================================
# Out[tid] = lo if A[tid] < lo, hi if A[tid] > hi, else A[tid]

print("=" * 60)
print("Test 2: if_/else — clamp values")
print("=" * 60)

builder2 = KernelBuilder("clamp_kernel")
builder2.args.append(("A", 0, "float"))
builder2.args.append(("Out", 1, "float"))
with builder2:
    A_proxy = TracingTensor("A", 0, "float")
    Out_proxy = TracingTensor("Out", 1, "float")
    tid = enigma.thread_position_in_grid
    val = A_proxy[tid]
    lo = enigma.metal_cast(0, "float")
    hi = enigma.metal_cast(1, "float")

    too_low = enigma.cmp_lt(val, lo)
    with enigma.if_(too_low) as (then_lo, else_lo):
        with then_lo:
            Out_proxy[tid] = lo
        with else_lo:
            too_high = enigma.cmp_gt(val, hi)
            with enigma.if_(too_high) as (then_hi, else_hi):
                with then_hi:
                    Out_proxy[tid] = hi
                with else_hi:
                    Out_proxy[tid] = val

# Verify IR structure
if_ops = [op for op in builder2.ops if op.op_type == "scf_if"]
assert len(if_ops) == 1, f"Expected 1 top-level scf_if, got {len(if_ops)}"
assert if_ops[0].attrs["has_else"] == True, "Outer if should have else"
# The else branch should contain the nested if
else_ops = if_ops[0].regions[1]
nested_ifs = [op for op in else_ops if op.op_type == "scf_if"]
assert len(nested_ifs) == 1, f"Expected nested if in else, got {len(nested_ifs)}"
assert nested_ifs[0].attrs["has_else"] == True, "Inner if should have else"

print(f"  IR: {len(builder2.ops)} top-level ops")
print(f"  Outer if: then={len(if_ops[0].regions[0])} ops, else={len(else_ops)} ops")
print(f"  Nested if in else: then={len(nested_ifs[0].regions[0])}, else={len(nested_ifs[0].regions[1])}")

# Reference
test_vals = np.array([-0.5, 0.3, 0.7, 1.5], dtype=np.float32)
expected2 = np.clip(test_vals, 0.0, 1.0)
print(f"  Reference: clamp({test_vals}) = {expected2}")
print("  PASSED\n")


# ============================================================================
# Test 3: for_range + if_ — conditional accumulation
# ============================================================================
# Sum only positive elements: for each element, add to acc only if > 0.

print("=" * 60)
print("Test 3: for_range + if_ — sum positive elements")
print("=" * 60)

builder3 = KernelBuilder("sum_positive")
builder3.args.append(("A", 0, "float"))
builder3.args.append(("Out", 1, "float"))
with builder3:
    A_proxy = TracingTensor("A", 0, "float")
    Out_proxy = TracingTensor("Out", 1, "float")
    tid = enigma.thread_position_in_grid
    acc = enigma.metal_cast(0, "float")
    zero = enigma.metal_cast(0, "float")
    n = enigma.metal_cast(8, "int")

    with enigma.for_range(0, n) as i:
        val = A_proxy[i]
        is_positive = enigma.cmp_gt(val, zero)
        with enigma.if_(is_positive):
            acc = acc + val

    Out_proxy[tid] = acc

# Verify nesting: for -> if
for_ops3 = [op for op in builder3.ops if op.op_type == "scf_for"]
assert len(for_ops3) == 1
body3 = for_ops3[0].regions[0]
ifs_in_for = [op for op in body3 if op.op_type == "scf_if"]
assert len(ifs_in_for) == 1, "Should have if inside for"

all_types = collect_op_types(builder3.ops)
print(f"  Total ops in tree: {len(all_types)}")
print(f"  for body: {len(body3)} ops (including nested if)")
print(f"  if inside for: then has {len(ifs_in_for[0].regions[0])} ops")

# Reference
test_a = np.array([-2.0, 3.0, -1.0, 5.0, 0.0, -4.0, 2.0, 1.0], dtype=np.float32)
expected3 = test_a[test_a > 0].sum()
print(f"  Reference: sum_positive({test_a}) = {expected3}")
print("  PASSED\n")


# ============================================================================
# Test 4: nested for_range — naive matmul inner loop
# ============================================================================
# C[row, col] = sum_k(A[row, k] * B[k, col])
# Two nested for loops over the tile dimensions.

print("=" * 60)
print("Test 4: nested for_range — matmul accumulation")
print("=" * 60)

K_SIZE = 4

builder4 = KernelBuilder("matmul_loop")
builder4.args.append(("A", 0, "float"))
builder4.args.append(("B", 1, "float"))
builder4.args.append(("C", 2, "float"))
with builder4:
    A_proxy = TracingTensor("A", 0, "float")
    B_proxy = TracingTensor("B", 1, "float")
    C_proxy = TracingTensor("C", 2, "float")
    tid = enigma.thread_position_in_grid
    N_val = enigma.metal_cast(8, "int")
    K_val = enigma.metal_cast(K_SIZE, "int")

    # row = tid / N, col = tid % N
    row = tid // N_val
    col = tid % N_val

    acc = enigma.metal_cast(0, "float")
    with enigma.for_range(0, K_val) as k:
        a_idx = row * K_val + k
        b_idx = k * N_val + col
        a_val = A_proxy[a_idx]
        b_val = B_proxy[b_idx]
        acc = acc + a_val * b_val

    C_proxy[tid] = acc

# Verify structure
for_ops4 = [op for op in builder4.ops if op.op_type == "scf_for"]
assert len(for_ops4) == 1
body4 = for_ops4[0].regions[0]
assert count_ops(body4, "load") == 2, f"Expected 2 loads in for body, got {count_ops(body4, 'load')}"
assert count_ops(body4, "mul") >= 1, "Expected multiply in for body"
assert count_ops(body4, "add") >= 1, "Expected add in for body"

print(f"  IR: {len(builder4.ops)} top-level ops")
print(f"  for body: {len(body4)} ops (2 loads, mul, add)")

# Reference
M, N, K = 2, 8, K_SIZE
A_mat = np.random.randn(M, K).astype(np.float32)
B_mat = np.random.randn(K, N).astype(np.float32)
C_expected = A_mat @ B_mat
print(f"  Reference: ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")
print(f"  Sample C[0,0] = {C_expected[0,0]:.4f}")
print("  PASSED\n")


# ============================================================================
# Test 5: while_ — linear search
# ============================================================================
# Find the first index where A[idx] > threshold.

print("=" * 60)
print("Test 5: while_ — linear search for threshold")
print("=" * 60)

builder5 = KernelBuilder("while_search")
builder5.args.append(("A", 0, "float"))
builder5.args.append(("Out", 1, "float"))
with builder5:
    A_proxy = TracingTensor("A", 0, "float")
    Out_proxy = TracingTensor("Out", 1, "float")
    tid = enigma.thread_position_in_grid
    i = enigma.metal_cast(0, "int")
    n = enigma.metal_cast(16, "int")
    threshold = enigma.metal_cast(0, "float")  # placeholder

    with enigma.while_(lambda: enigma.cmp_lt(i, n)):
        val = A_proxy[i]

# Verify structure
while_ops = [op for op in builder5.ops if op.op_type == "scf_while"]
assert len(while_ops) == 1
before = while_ops[0].regions[0]  # condition
after = while_ops[0].regions[1]   # body
assert len(before) > 0, "Condition region should have ops"
assert len(after) > 0, "Body region should have ops"

# Check condition has a comparison
cond_types = collect_op_types(before)
assert "cmp_lt" in cond_types, f"Condition should have cmp_lt, got {cond_types}"

print(f"  IR: {len(builder5.ops)} top-level ops")
print(f"  while condition: {len(before)} ops ({cond_types})")
print(f"  while body: {len(after)} ops")

# Reference
test_a5 = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.5], dtype=np.float32)
threshold_val = 1.0
idx = np.argmax(test_a5 > threshold_val)
print(f"  Reference: first index > {threshold_val} in {test_a5} = {idx}")
print("  PASSED\n")


# ============================================================================
# Test 6: dump_ir to show the traced op tree
# ============================================================================

print("=" * 60)
print("Test 6: IR dump of matmul kernel")
print("=" * 60)

def dump_ir(ops, indent=0):
    """Pretty-print the traced IR tree."""
    prefix = "  " * indent
    for op in ops:
        res = op.result.name if op.result else "(void)"
        operands = ", ".join(getattr(o, "name", str(o)) for o in op.operands)
        attrs_str = ""
        if op.attrs:
            # Filter out large/internal attrs
            show = {k: v for k, v in op.attrs.items()
                    if k not in ("iv",) and not callable(v)}
            if show:
                attrs_str = f"  {show}"
        print(f"{prefix}{res:>8} = {op.op_type}({operands}){attrs_str}")
        for i, region in enumerate(op.regions):
            label = ["body", "then", "else", "before", "after"]
            rname = label[i] if i < len(label) else f"region{i}"
            print(f"{prefix}  [{rname}]:")
            dump_ir(region, indent + 2)

dump_ir(builder4.ops)
print()


# ============================================================================
# Summary
# ============================================================================

print("=" * 60)
print("All control flow tracing tests passed!")
print("=" * 60)
print()
print("Once the dialect wheel registers SCF Python bindings, these")
print("kernels will compile to MSL and run on the GPU.  The only")
print("dialect change needed:")
print()
print('  // EnigmaModule.cpp')
print('  #include "mlir-c/Dialect/SCF.h"')
print('  mlirDialectHandleRegisterDialect(')
print('      mlirGetDialectHandle__scf__(), context);')
