"""Emit MLIR from traced IR using the Enigma dialect python bindings,
then translate to MSL via the dialect's TranslateToMSL binding.

The DSL package is `enigma`; the dialect submodule is `mlir.dialects.enigma`.
Separate namespaces, no collision.
"""

from __future__ import annotations

from .._tracing import KernelBuilder


_DIM_XYZ = {"x": 0, "y": 1, "z": 2}

_MEM_FLAGS = {
    "mem_none": 0, "mem_device": 1, "mem_threadgroup": 2,
    "mem_device_and_threadgroup": 3, "mem_texture": 4,
}

_MEM_ORDER = {"relaxed": 0, "acquire": 1, "release": 2, "acq_rel": 3}

_ATOMIC_RMW = {
    "atomic_exchange":  "AtomicExchangeOp",
    "atomic_fetch_add": "AtomicFetchAddOp",
    "atomic_fetch_sub": "AtomicFetchSubOp",
    "atomic_fetch_min": "AtomicFetchMinOp",
    "atomic_fetch_max": "AtomicFetchMaxOp",
    "atomic_fetch_and": "AtomicFetchAndOp",
    "atomic_fetch_or":  "AtomicFetchOrOp",
    "atomic_fetch_xor": "AtomicFetchXorOp",
}

_DTYPE_NAMES = {
    "float": "float", "f32": "float",
    "half": "half", "f16": "half",
    "int": "int", "int32": "int", "i32": "int",
    "uint": "uint", "uint32": "uint", "u32": "uint",
    "i1": "i1",
}


def _build_module(builder: KernelBuilder):
    from mlir import ir
    from mlir.dialects import arith, memref
    from mlir.dialects import enigma as en

    ctx = ir.Context()
    en.register_dialect(ctx)
    ctx.load_all_available_dialects()
    loc = ir.Location.unknown(ctx)

    with ctx, loc:
        module = ir.Module.create()

        def _elem_type(metal_dtype: str):
            if metal_dtype == "float":
                return ir.F32Type.get()
            if metal_dtype == "half":
                return ir.F16Type.get()
            if metal_dtype in ("int", "int32", "uint", "uint32"):
                return ir.IntegerType.get_signless(32)
            return ir.F32Type.get()

        i32 = ir.IntegerType.get_signless(32)
        i1 = ir.IntegerType.get_signless(1)
        index_t = ir.IndexType.get()

        _UNARY_MATH = {
            "abs": en.AbsOp, "ceil": en.CeilOp, "floor": en.FloorOp,
            "round": en.RoundOp, "trunc": en.TruncOp, "sign": en.SignOp,
            "saturate": en.SaturateOp, "fract": en.FractOp,
            "sqrt": en.SqrtOp, "rsqrt": en.RsqrtOp,
            "exp": en.ExpOp, "exp2": en.Exp2Op,
            "log": en.LogOp, "log2": en.Log2Op, "log10": en.Log10Op,
            "sin": en.SinOp, "cos": en.CosOp, "tan": en.TanOp,
            "asin": en.AsinOp, "acos": en.AcosOp, "atan": en.AtanOp,
            "sinh": en.SinhOp, "cosh": en.CoshOp, "tanh": en.TanhOp,
        }
        _BINARY_MATH = {
            "fmin": en.FminOp, "fmax": en.FmaxOp, "pow": en.PowOp,
            "fmod": en.FmodOp, "atan2": en.Atan2Op, "step": en.StepOp,
            "copysign": en.CopysignOp,
        }
        _TERNARY_MATH = {
            "clamp": en.ClampOp, "fma": en.FmaOp,
            "mix": en.MixOp, "smoothstep": en.SmoothstepOp,
        }
        _FLOAT_PREDICATES = {
            "isnan": en.IsNanOp, "isinf": en.IsInfOp,
            "isfinite": en.IsFiniteOp, "signbit": en.SignbitOp,
            "isnormal": en.IsNormalOp,
        }
        _UNARY_INT = {
            "popcount": en.PopcountOp, "clz": en.ClzOp, "ctz": en.CtzOp,
            "reverse_bits": en.ReverseBitsOp,
            "abs_diff_unary": en.AbsDiffUnaryOp,
        }
        _BINARY_INT = {
            "abs_diff": en.AbsDiffBinOp, "add_sat": en.AddSatOp,
            "sub_sat": en.SubSatOp, "mul_hi": en.MulHiOp,
            "rotate": en.RotateOp,
            "imin": en.IMinOp, "imax": en.IMaxOp,
        }
        _SIMD_UNARY = {
            "simd_sum": en.SimdSumOp, "simd_product": en.SimdProductOp,
            "simd_min": en.SimdMinOp, "simd_max": en.SimdMaxOp,
            "simd_and": en.SimdAndOp, "simd_or": en.SimdOrOp,
            "simd_xor": en.SimdXorOp,
            "simd_prefix_exclusive_sum": en.SimdPrefixExclusiveSumOp,
            "simd_prefix_inclusive_sum": en.SimdPrefixInclusiveSumOp,
            "simd_prefix_exclusive_product": en.SimdPrefixExclusiveProductOp,
            "simd_prefix_inclusive_product": en.SimdPrefixInclusiveProductOp,
        }
        _SIMD_SHUFFLE = {
            "simd_shuffle": en.SimdShuffleOp,
            "simd_shuffle_up": en.SimdShuffleUpOp,
            "simd_shuffle_down": en.SimdShuffleDownOp,
            "simd_shuffle_xor": en.SimdShuffleXorOp,
            "simd_broadcast": en.SimdBroadcastOp,
        }
        _QUAD_UNARY = {
            "quad_sum": en.QuadSumOp, "quad_product": en.QuadProductOp,
            "quad_min": en.QuadMinOp, "quad_max": en.QuadMaxOp,
            "quad_and": en.QuadAndOp, "quad_or": en.QuadOrOp,
            "quad_xor": en.QuadXorOp,
            "quad_prefix_exclusive_sum": en.QuadPrefixExclusiveSumOp,
            "quad_prefix_inclusive_sum": en.QuadPrefixInclusiveSumOp,
        }
        _QUAD_SHUFFLE = {
            "quad_shuffle": en.QuadShuffleOp,
            "quad_shuffle_up": en.QuadShuffleUpOp,
            "quad_shuffle_down": en.QuadShuffleDownOp,
            "quad_shuffle_xor": en.QuadShuffleXorOp,
            "quad_broadcast": en.QuadBroadcastOp,
        }

        def _mlir_type_from_dtype(dt: str):
            if dt in ("float", "f32"): return ir.F32Type.get()
            if dt in ("half", "f16"): return ir.F16Type.get()
            if dt == "i1": return i1
            return ir.IntegerType.get_signless(32)

        arg_types = [
            ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], _elem_type(dt))
            for _n, _bi, dt in builder.args
        ]

        with ir.InsertionPoint(module.body):
            fn_type = ir.FunctionType.get(arg_types, [])
            kernel_op = en.KernelOp(
                sym_name=ir.StringAttr.get(builder.kernel_name),
                function_type=ir.TypeAttr.get(fn_type),
            )
            block = kernel_op.regions[0].blocks.append(*arg_types)

            with ir.InsertionPoint(block):
                ssa: dict[str, ir.Value] = {}
                buf_of: dict[str, ir.Value] = {}
                for (name, _bi, _dt), blk_arg in zip(builder.args, block.arguments):
                    buf_of[name] = blk_arg

                def _to_index(v):
                    if str(v.type) == "index":
                        return v
                    return arith.IndexCastOp(index_t, v).result

                for op in builder.ops:
                    t = op.op_type

                    if t == "thread_position_in_grid":
                        v = en.ThreadPositionInGridOp(
                            dimension=ir.IntegerAttr.get(i32, _DIM_XYZ["x"])
                        ).result
                        ssa[op.result.name] = v

                    elif t == "thread_position_in_threadgroup":
                        v = en.ThreadPositionInThreadgroupOp(
                            dimension=ir.IntegerAttr.get(i32, _DIM_XYZ["x"])
                        ).result
                        ssa[op.result.name] = v

                    elif t == "threadgroup_position_in_grid":
                        v = en.ThreadgroupPositionInGridOp(
                            dimension=ir.IntegerAttr.get(i32, _DIM_XYZ["x"])
                        ).result
                        ssa[op.result.name] = v

                    elif t == "threads_per_threadgroup":
                        v = en.ThreadsPerThreadgroupOp(
                            dimension=ir.IntegerAttr.get(i32, _DIM_XYZ["x"])
                        ).result
                        ssa[op.result.name] = v

                    elif t == "const":
                        val = int(op.attrs["value"])
                        c = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, val)).result
                        ssa[op.result.name] = c

                    elif t == "load":
                        buf = buf_of[op.attrs["buffer"]]
                        idx = _to_index(ssa[op.operands[0].name])
                        ssa[op.result.name] = memref.LoadOp(buf, [idx]).result

                    elif t == "store":
                        buf = buf_of[op.attrs["buffer"]]
                        idx = _to_index(ssa[op.operands[0].name])
                        val = ssa[op.operands[1].name]
                        memref.StoreOp(val, buf, [idx])

                    elif t == "neg":
                        a = ssa[op.operands[0].name]
                        if str(a.type) in ("f32", "f16", "bf16"):
                            ssa[op.result.name] = arith.NegFOp(a).result
                        else:
                            zero = arith.ConstantOp(a.type, ir.IntegerAttr.get(a.type, 0)).result
                            ssa[op.result.name] = arith.SubIOp(zero, a).result

                    elif t in ("add", "sub", "mul", "div", "mod"):
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        ty_str = str(a.type)
                        is_float = ty_str in ("f32", "f16", "bf16")
                        if is_float:
                            cls = {"add": arith.AddFOp, "sub": arith.SubFOp,
                                   "mul": arith.MulFOp, "div": arith.DivFOp,
                                   "mod": arith.RemFOp}[t]
                        else:
                            cls = {"add": arith.AddIOp, "sub": arith.SubIOp,
                                   "mul": arith.MulIOp, "div": arith.DivSIOp,
                                   "mod": arith.RemSIOp}[t]
                        ssa[op.result.name] = cls(a, b).result

                    elif t in _UNARY_MATH:
                        a = ssa[op.operands[0].name]
                        ssa[op.result.name] = _UNARY_MATH[t](a).result

                    elif t in _BINARY_MATH:
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        ssa[op.result.name] = _BINARY_MATH[t](a, b).result

                    elif t in _TERNARY_MATH:
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        c = ssa[op.operands[2].name]
                        ssa[op.result.name] = _TERNARY_MATH[t](a, b, c).result

                    elif t in _FLOAT_PREDICATES:
                        a = ssa[op.operands[0].name]
                        ssa[op.result.name] = _FLOAT_PREDICATES[t](a).result

                    elif t in _UNARY_INT:
                        a = ssa[op.operands[0].name]
                        ssa[op.result.name] = _UNARY_INT[t](a).result

                    elif t in _BINARY_INT:
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        ssa[op.result.name] = _BINARY_INT[t](a, b).result

                    elif t == "iclamp":
                        a = ssa[op.operands[0].name]
                        lo = ssa[op.operands[1].name]
                        hi = ssa[op.operands[2].name]
                        ssa[op.result.name] = en.IClampOp(a, lo, hi).result

                    elif t == "mad_sat":
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        c = ssa[op.operands[2].name]
                        ssa[op.result.name] = en.MadSatOp(a, b, c).result

                    elif t == "select":
                        fv = ssa[op.operands[0].name]
                        tv = ssa[op.operands[1].name]
                        cond = ssa[op.operands[2].name]
                        ssa[op.result.name] = en.SelectOp(fv, tv, cond).result

                    elif t == "extract_bits":
                        a = ssa[op.operands[0].name]
                        offset = ir.IntegerAttr.get(i32, int(op.attrs["offset"]))
                        bits = ir.IntegerAttr.get(i32, int(op.attrs["bits"]))
                        ssa[op.result.name] = en.ExtractBitsOp(a, offset, bits).result

                    elif t == "insert_bits":
                        base = ssa[op.operands[0].name]
                        ins = ssa[op.operands[1].name]
                        offset = ir.IntegerAttr.get(i32, int(op.attrs["offset"]))
                        bits = ir.IntegerAttr.get(i32, int(op.attrs["bits"]))
                        ssa[op.result.name] = en.InsertBitsOp(base, ins, offset, bits).result

                    elif t in _SIMD_UNARY:
                        a = ssa[op.operands[0].name]
                        ssa[op.result.name] = _SIMD_UNARY[t](a).result

                    elif t in _SIMD_SHUFFLE:
                        a = ssa[op.operands[0].name]
                        idx = _to_index(ssa[op.operands[1].name])
                        ssa[op.result.name] = _SIMD_SHUFFLE[t](a, idx).result

                    elif t in _QUAD_UNARY:
                        a = ssa[op.operands[0].name]
                        ssa[op.result.name] = _QUAD_UNARY[t](a).result

                    elif t in _QUAD_SHUFFLE:
                        a = ssa[op.operands[0].name]
                        idx = _to_index(ssa[op.operands[1].name])
                        ssa[op.result.name] = _QUAD_SHUFFLE[t](a, idx).result

                    elif t == "metal_cast":
                        a = ssa[op.operands[0].name]
                        target = _mlir_type_from_dtype(op.attrs["target_dtype"])
                        ssa[op.result.name] = en.MetalCastOp(target, a).result

                    elif t == "as_type":
                        a = ssa[op.operands[0].name]
                        target = _mlir_type_from_dtype(op.attrs["target_dtype"])
                        ssa[op.result.name] = en.AsTypeOp(target, a).result

                    elif t == "threadgroup_alloc":
                        size = int(op.attrs["size"])
                        dtype = op.attrs["dtype"]
                        elem = _elem_type(dtype)
                        tg_space = ir.IntegerAttr.get(i32, 2)
                        mtype = ir.MemRefType.get([size], elem, memory_space=tg_space)
                        res = en.ThreadgroupAllocOp(mtype).result
                        buf_of[op.attrs["buffer"]] = res

                    elif t == "atomic_load":
                        buf = buf_of[op.attrs["buffer"]]
                        idx = _to_index(ssa[op.operands[0].name])
                        order = ir.IntegerAttr.get(i32, _MEM_ORDER[op.attrs["memory_order"]])
                        elem = _elem_type(op.attrs["dtype"])
                        ssa[op.result.name] = en.AtomicLoadOp(elem, buf, [idx], order).result

                    elif t == "atomic_store":
                        buf = buf_of[op.attrs["buffer"]]
                        idx = _to_index(ssa[op.operands[0].name])
                        val = ssa[op.operands[1].name]
                        order = ir.IntegerAttr.get(i32, _MEM_ORDER[op.attrs["memory_order"]])
                        en.AtomicStoreOp(val, buf, [idx], order)

                    elif t in _ATOMIC_RMW:
                        buf = buf_of[op.attrs["buffer"]]
                        idx = _to_index(ssa[op.operands[0].name])
                        val = ssa[op.operands[1].name]
                        order = ir.IntegerAttr.get(i32, _MEM_ORDER[op.attrs["memory_order"]])
                        cls = getattr(en, _ATOMIC_RMW[t])
                        ssa[op.result.name] = cls(buf, [idx], val, order).result

                    elif t == "atomic_compare_exchange_weak":
                        buf = buf_of[op.attrs["buffer"]]
                        idx = _to_index(ssa[op.operands[0].name])
                        expected = ssa[op.operands[1].name]
                        desired = ssa[op.operands[2].name]
                        so = ir.IntegerAttr.get(i32, _MEM_ORDER[op.attrs["success_order"]])
                        fo = ir.IntegerAttr.get(i32, _MEM_ORDER[op.attrs["failure_order"]])
                        ssa[op.result.name] = en.AtomicCompareExchangeWeakOp(
                            buf, [idx], expected, desired, so, fo).result

                    elif t == "threadgroup_barrier":
                        flags = ir.IntegerAttr.get(
                            i32, _MEM_FLAGS[op.attrs.get("mem_flags", "mem_threadgroup")])
                        en.ThreadgroupBarrierOp(mem_flags=flags)

                    elif t == "simdgroup_barrier":
                        flags = ir.IntegerAttr.get(
                            i32, _MEM_FLAGS[op.attrs.get("mem_flags", "mem_threadgroup")])
                        en.SimdgroupBarrierOp(mem_flags=flags)

                    else:
                        # tv_load / tv_store / tv_add don't have a clean enigma-dialect
                        # representation yet. Raise so the caller can fall back.
                        raise _UnsupportedMLIROp(t)

                en.ReturnOp()

        return module


class _UnsupportedMLIROp(Exception):
    pass


def emit_mlir(builder: KernelBuilder) -> str:
    """Return textual MLIR for the traced kernel."""
    return str(_build_module(builder))


def emit_msl(builder: KernelBuilder) -> str:
    """Trace -> MLIR -> MSL, using the dialect's TranslateToMSL binding."""
    from mlir.dialects import enigma as en

    module = _build_module(builder)
    return en.translate_to_msl(module.operation)
