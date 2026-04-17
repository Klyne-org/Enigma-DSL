"""Emit MLIR from traced IR using the Enigma dialect python bindings,
then translate to MSL via the dialect's TranslateToMSL binding.

The DSL package is `enigma`; the dialect submodule is `mlir.dialects.enigma`.
Separate namespaces, no collision.
"""

from __future__ import annotations

from .._tracing import KernelBuilder


_DIM_XYZ = {"x": 0, "y": 1, "z": 2}

_CMP_PRED_INT = {
    "cmp_eq": 0, "cmp_ne": 1,
    "cmp_lt": 2, "cmp_le": 3, "cmp_gt": 4, "cmp_ge": 5,  # signed
    "cmp_ult": 6, "cmp_ule": 7, "cmp_ugt": 8, "cmp_uge": 9,  # unsigned
}
_CMP_PRED_FLOAT = {
    # OGT, OGE, OLT, OLE, OEQ, ONE — "ordered" family for Metal (fast math).
    "cmp_eq": 1, "cmp_gt": 2, "cmp_ge": 3,
    "cmp_lt": 4, "cmp_le": 5, "cmp_ne": 6,
    # Unsigned float forms don't exist; map u_* to the same ordered preds.
    "cmp_ult": 4, "cmp_ule": 5, "cmp_ugt": 2, "cmp_uge": 3,
}

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


def _build_module(builder: KernelBuilder, vec_width: int = 0):
    from mlir import ir
    from mlir.dialects import arith, memref
    from mlir.dialects import enigma as en

    # Try to import scf; it may not be available if the dialect wheel
    # doesn't expose the SCF Python bindings yet.
    try:
        from mlir.dialects import scf as scf_dialect
        _has_scf = True
    except ImportError:
        scf_dialect = None
        _has_scf = False

    ctx = ir.Context()
    en.register_dialect(ctx)
    ctx.load_all_available_dialects()
    loc = ir.Location.unknown(ctx)

    with ctx, loc:
        module = ir.Module.create()

        def _scalar_type(metal_dtype: str):
            if metal_dtype in ("float", "f32"):
                return ir.F32Type.get()
            if metal_dtype in ("half", "f16"):
                return ir.F16Type.get()
            if metal_dtype in ("bfloat", "bf16"):
                return ir.BF16Type.get()
            if metal_dtype in ("char", "int8", "i8", "uchar", "uint8", "u8"):
                return ir.IntegerType.get_signless(8)
            if metal_dtype in ("short", "int16", "i16", "ushort", "uint16", "u16"):
                return ir.IntegerType.get_signless(16)
            if metal_dtype in ("int", "int32", "i32", "uint", "uint32", "u32"):
                return ir.IntegerType.get_signless(32)
            if metal_dtype in ("long", "int64", "i64", "ulong", "uint64", "u64"):
                return ir.IntegerType.get_signless(64)
            if metal_dtype == "i1" or metal_dtype == "bool":
                return ir.IntegerType.get_signless(1)
            return ir.F32Type.get()

        def _parse_vec(dtype: str):
            if isinstance(dtype, str) and dtype.startswith("vec<") and dtype.endswith(">"):
                body = dtype[4:-1]
                n_s, elem = body.split(",", 1)
                return int(n_s), elem.strip()
            return None

        def _elem_type(metal_dtype: str):
            """Returns the MLIR type for a dtype string (scalar or vec<>)."""
            parsed = _parse_vec(metal_dtype)
            if parsed is not None:
                n, elem = parsed
                return ir.VectorType.get([n], _scalar_type(elem))
            return _scalar_type(metal_dtype)

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
        _PACK = {
            "pack_float_to_snorm4x8":      en.PackSnorm4x8Op,
            "pack_float_to_unorm4x8":      en.PackUnorm4x8Op,
            "pack_float_to_snorm2x16":     en.PackSnorm2x16Op,
            "pack_float_to_unorm2x16":     en.PackUnorm2x16Op,
            "pack_float_to_srgb_unorm4x8": en.PackSrgbUnorm4x8Op,
            "pack_float_to_unorm10a2":     en.PackUnorm10a2Op,
        }
        _UNPACK = {
            "unpack_snorm4x8_to_float":     en.UnpackSnorm4x8ToFloatOp,
            "unpack_unorm4x8_to_float":     en.UnpackUnorm4x8ToFloatOp,
            "unpack_snorm2x16_to_float":    en.UnpackSnorm2x16ToFloatOp,
            "unpack_unorm2x16_to_float":    en.UnpackUnorm2x16ToFloatOp,
            "unpack_srgb_unorm4x8_to_float":en.UnpackSrgbUnorm4x8ToFloatOp,
            "unpack_unorm10a2_to_float":    en.UnpackUnorm10a2ToFloatOp,
        }

        def _mlir_type_from_dtype(dt: str):
            if dt in ("float", "f32"): return ir.F32Type.get()
            if dt in ("half", "f16"): return ir.F16Type.get()
            if dt in ("bfloat", "bf16"): return ir.BF16Type.get()
            if dt == "i1" or dt == "bool": return i1
            if dt in ("char", "int8", "i8", "uchar", "uint8", "u8"):
                return ir.IntegerType.get_signless(8)
            if dt in ("short", "int16", "i16", "ushort", "uint16", "u16"):
                return ir.IntegerType.get_signless(16)
            if dt in ("long", "int64", "i64", "ulong", "uint64", "u64"):
                return ir.IntegerType.get_signless(64)
            return ir.IntegerType.get_signless(32)

        def _buffer_elem_type(dt: str):
            base = _elem_type(dt)
            # For vec_width > 0 on float-like buffers, promote element type to
            # vector<Nx_base> so MSL emits `device floatN*` in the signature and
            # scalar load/add/store ops operate on vectors transparently.
            if vec_width > 0 and dt in ("float", "f32", "half", "f16", "bfloat", "bf16"):
                return ir.VectorType.get([vec_width], base)
            return base

        arg_types = [
            ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], _buffer_elem_type(dt))
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

                # For TV-vectorized values (tv_load / tv_add / tv_store), we
                # lower to per-element scalar MLIR ops. `tv_elems` maps a
                # traced-IR name to the list of per-element SSA values in
                # flattened-layout order.
                tv_elems: dict[str, list] = {}

                def _to_index(v):
                    if str(v.type) == "index":
                        return v
                    return arith.IndexCastOp(index_t, v).result

                def _const_index(n: int):
                    return arith.ConstantOp(index_t, ir.IntegerAttr.get(index_t, int(n))).result

                def _resolve_base(base):
                    """Return an index SSA value for a base_offset (int or IRValue)."""
                    if isinstance(base, int):
                        return _const_index(base)
                    # Traced IRValue passed in as base_offset
                    from .._tracing import IRValue as _IRV
                    if isinstance(base, _IRV):
                        return _to_index(ssa[base.name])
                    # Should not happen, but fall back to 0.
                    return _const_index(0)

                _GRID_QUERY_OPS = {
                    "thread_position_in_grid": en.ThreadPositionInGridOp,
                    "thread_position_in_threadgroup": en.ThreadPositionInThreadgroupOp,
                    "threadgroup_position_in_grid": en.ThreadgroupPositionInGridOp,
                    "threads_per_threadgroup": en.ThreadsPerThreadgroupOp,
                    "threads_per_grid": en.ThreadsPerGridOp,
                    "threadgroups_per_grid": en.ThreadgroupsPerGridOp,
                    "grid_size": en.GridSizeOp,
                    "thread_index_in_threadgroup": en.ThreadIndexInThreadgroupOp,
                    "thread_index_in_simdgroup": en.ThreadIndexInSimdgroupOp,
                    "simdgroup_index_in_threadgroup": en.SimdgroupIndexInThreadgroupOp,
                    "threads_per_simdgroup": en.ThreadsPerSimdgroupOp,
                    "simdgroups_per_threadgroup": en.SimdgroupsPerThreadgroupOp,
                }

                def _emit_ops(ops_list):
                  """Emit a list of traced IR ops into the current MLIR insertion point.

                  This function is recursive: control-flow ops call it for their
                  nested regions.
                  """
                  for op in ops_list:
                    t = op.op_type

                    if t in _GRID_QUERY_OPS:
                        dim = op.attrs.get("dim", "x")
                        cls = _GRID_QUERY_OPS[t]
                        v = cls(
                            dimension=ir.IntegerAttr.get(i32, _DIM_XYZ[dim])
                        ).result
                        ssa[op.result.name] = v

                    elif t == "const":
                        dt = op.attrs.get("dtype", "int")
                        raw = op.attrs["value"]
                        if dt in ("float", "f32"):
                            f = ir.F32Type.get()
                            c = arith.ConstantOp(f, ir.FloatAttr.get(f, float(raw))).result
                        elif dt in ("half", "f16"):
                            f = ir.F16Type.get()
                            c = arith.ConstantOp(f, ir.FloatAttr.get(f, float(raw))).result
                        elif dt == "i1":
                            c = arith.ConstantOp(i1, ir.IntegerAttr.get(i1, int(raw))).result
                        else:
                            c = arith.ConstantOp(i32, ir.IntegerAttr.get(i32, int(raw))).result
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
                        is_float = ty_str in ("f32", "f16", "bf16") or (
                            ty_str.startswith("vector<")
                            and any(f in ty_str for f in ("xf32", "xf16", "xbf16"))
                        )
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

                    elif t == "vec_make":
                        elem = op.attrs["elem"]
                        n = int(op.attrs["n"])
                        vt = ir.VectorType.get([n], _scalar_type(elem))
                        elems = [ssa[o.name] for o in op.operands]
                        ssa[op.result.name] = en.VecMakeOp(vt, elems).result

                    elif t == "vec_extract":
                        v = ssa[op.operands[0].name]
                        lane = int(op.attrs["lane"])
                        elem_t = v.type.element_type
                        ssa[op.result.name] = en.VecExtractOp(
                            elem_t, v, ir.IntegerAttr.get(i32, lane)).result

                    # --- Pack / Unpack ---
                    elif t in _PACK:
                        a = ssa[op.operands[0].name]
                        res_t = _scalar_type("uint")
                        ssa[op.result.name] = _PACK[t](res_t, a).result

                    elif t in _UNPACK:
                        a = ssa[op.operands[0].name]
                        elem = op.attrs["elem"]
                        n = int(op.attrs["n"])
                        res_t = ir.VectorType.get([n], _scalar_type(elem))
                        ssa[op.result.name] = _UNPACK[t](res_t, a).result

                    # --- Geometry ---
                    elif t == "dot":
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        res_t = a.type.element_type
                        ssa[op.result.name] = en.DotOp(res_t, a, b).result

                    elif t == "length":
                        v = ssa[op.operands[0].name]
                        res_t = v.type.element_type
                        ssa[op.result.name] = en.LengthOp(res_t, v).result

                    elif t == "distance":
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        res_t = a.type.element_type
                        ssa[op.result.name] = en.DistanceOp(res_t, a, b).result

                    elif t == "cross":
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        ssa[op.result.name] = en.CrossOp(a, b).result

                    elif t == "normalize":
                        v = ssa[op.operands[0].name]
                        ssa[op.result.name] = en.NormalizeOp(v).result

                    elif t == "reflect":
                        i = ssa[op.operands[0].name]
                        n = ssa[op.operands[1].name]
                        ssa[op.result.name] = en.ReflectOp(i, n).result

                    elif t == "refract":
                        i = ssa[op.operands[0].name]
                        n = ssa[op.operands[1].name]
                        eta = ssa[op.operands[2].name]
                        ssa[op.result.name] = en.RefractOp(i, n, eta).result

                    elif t == "faceforward":
                        n = ssa[op.operands[0].name]
                        i = ssa[op.operands[1].name]
                        nref = ssa[op.operands[2].name]
                        ssa[op.result.name] = en.FaceforwardOp(n, i, nref).result

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

                    # --- Comparisons (arith.cmpi / arith.cmpf) ---
                    elif t in _CMP_PRED_INT:
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        ty_str = str(a.type)
                        is_float = ty_str in ("f32", "f16", "bf16") or ty_str.startswith("vector<") and any(
                            f in ty_str for f in ("xf32", "xf16", "xbf16"))
                        if is_float:
                            pred = ir.IntegerAttr.get(
                                ir.IntegerType.get_signless(64), _CMP_PRED_FLOAT[t])
                            ssa[op.result.name] = arith.CmpFOp(pred, a, b).result
                        else:
                            pred = ir.IntegerAttr.get(
                                ir.IntegerType.get_signless(64), _CMP_PRED_INT[t])
                            ssa[op.result.name] = arith.CmpIOp(pred, a, b).result

                    # --- Function constants ---
                    elif t == "function_constant":
                        dt = op.attrs["dtype"]
                        mlir_ty = _mlir_type_from_dtype(dt)
                        idx_attr = ir.IntegerAttr.get(i32, int(op.attrs["index"]))
                        ssa[op.result.name] = en.FunctionConstantOp(
                            mlir_ty, idx_attr).result

                    # --- TV-layout ops: per-element scalar lowering ---
                    # Each TV group is (start, count): `count` contiguous
                    # elements starting at buf[base_offset + start]. We expand
                    # to per-element memref.load/store and store the list of
                    # per-element SSA values in tv_elems[name] so downstream
                    # tv_add / tv_store can consume them.
                    elif t == "tv_load":
                        buf = buf_of[op.attrs["buffer"]]
                        base = op.attrs["base_offset"]
                        groups = op.attrs["groups"]
                        elem_t = _elem_type(op.attrs["dtype"])
                        base_idx = _resolve_base(base)
                        loaded = []
                        for start, count in groups:
                            for k in range(count):
                                off = start + k
                                if off == 0:
                                    idx_v = base_idx
                                else:
                                    off_v = _const_index(off)
                                    idx_v = arith.AddIOp(base_idx, off_v).result
                                loaded.append(memref.LoadOp(buf, [idx_v]).result)
                        tv_elems[op.result.name] = loaded
                        # Keep a sentinel in ssa so lookups don't KeyError
                        # on the result name (e.g. if used in non-TV ops, we
                        # point at element 0 — though tv ops are the normal path).
                        if loaded:
                            ssa[op.result.name] = loaded[0]

                    elif t == "tv_add":
                        a_elems = tv_elems[op.operands[0].name]
                        b_elems = tv_elems[op.operands[1].name]
                        assert len(a_elems) == len(b_elems)
                        ty_str = str(a_elems[0].type)
                        is_float = ty_str in ("f32", "f16", "bf16")
                        cls = arith.AddFOp if is_float else arith.AddIOp
                        out = [cls(x, y).result for x, y in zip(a_elems, b_elems)]
                        tv_elems[op.result.name] = out
                        if out:
                            ssa[op.result.name] = out[0]

                    elif t == "tv_store":
                        buf = buf_of[op.attrs["buffer"]]
                        base = op.attrs["base_offset"]
                        groups = op.attrs["groups"]
                        val_elems = tv_elems[op.operands[0].name]
                        base_idx = _resolve_base(base)
                        i = 0
                        for start, count in groups:
                            for k in range(count):
                                off = start + k
                                if off == 0:
                                    idx_v = base_idx
                                else:
                                    off_v = _const_index(off)
                                    idx_v = arith.AddIOp(base_idx, off_v).result
                                memref.StoreOp(val_elems[i], buf, [idx_v])
                                i += 1

                    # --- Simdgroup matrix ops ---
                    elif t == "simdgroup_matrix_load":
                        buf = buf_of[op.attrs["buffer"]]
                        rows = int(op.attrs["rows"])
                        cols = int(op.attrs["cols"])
                        elem = op.attrs["elem"]
                        mat_t = ir.VectorType.get([rows, cols], _scalar_type(elem))
                        epr = _const_index(int(op.attrs["elements_per_row"]))
                        ssa[op.result.name] = en.SimdgroupMatLoadOp(
                            mat_t, buf, epr).result

                    elif t == "simdgroup_matrix_store":
                        buf = buf_of[op.attrs["buffer"]]
                        mat = ssa[op.operands[0].name]
                        epr = _const_index(int(op.attrs["elements_per_row"]))
                        en.SimdgroupMatStoreOp(mat, buf, epr)

                    elif t == "simdgroup_multiply_accumulate":
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        c = ssa[op.operands[2].name]
                        ssa[op.result.name] = en.SimdgroupMatMulAccOp(
                            c.type, a, b, c).result

                    elif t == "make_filled_simdgroup_matrix":
                        val = ssa[op.operands[0].name]
                        rows = int(op.attrs["rows"])
                        cols = int(op.attrs["cols"])
                        elem = op.attrs["elem"]
                        mat_t = ir.VectorType.get([rows, cols], _scalar_type(elem))
                        ssa[op.result.name] = en.MakeFilledSimdgroupMatOp(
                            mat_t, val).result

                    # --- Regular matrix ops on vector<CxRxT> ---
                    elif t == "matmul":
                        a = ssa[op.operands[0].name]
                        b = ssa[op.operands[1].name]
                        ssa[op.result.name] = en.MatMulOp(a.type, a, b).result

                    elif t == "transpose":
                        m = ssa[op.operands[0].name]
                        ssa[op.result.name] = en.TransposeOp(m.type, m).result

                    elif t == "determinant":
                        m = ssa[op.operands[0].name]
                        # Result dtype recorded on the traced op
                        res_ty = m.type.element_type if hasattr(m.type, "element_type") else m.type
                        ssa[op.result.name] = en.DeterminantOp(res_ty, m).result

                    # --- Control flow: scf.for ---
                    elif t == "scf_for":
                        if not _has_scf:
                            raise NotImplementedError(
                                "Control flow (for_range) requires the SCF dialect. "
                                "Install a dialect wheel with SCF Python bindings."
                            )
                        lo = _to_index(ssa[op.operands[0].name])
                        hi = _to_index(ssa[op.operands[1].name])
                        step = _to_index(ssa[op.operands[2].name])

                        iter_args_ir = op.attrs.get("iter_args", [])
                        yield_vals_ir = op.attrs.get("yield_vals", [])
                        results_ir = op.attrs.get("results", [])
                        init_operands = [
                            ssa[o.name] for o in op.operands[3:]
                        ]

                        for_op = scf_dialect.ForOp(
                            lo, hi, step,
                            iter_args=init_operands if init_operands else None,
                        )
                        iv_ir = op.attrs["iv"]
                        ssa[iv_ir.name] = for_op.induction_variable

                        # Bind iter_arg names (body-local SSA) before emitting body.
                        for ia_ir, ia_val in zip(iter_args_ir, for_op.inner_iter_args):
                            ssa[ia_ir.name] = ia_val

                        with ir.InsertionPoint(for_op.body):
                            _emit_ops(op.regions[0])
                            yield_operands = [ssa[v.name] for v in yield_vals_ir]
                            scf_dialect.YieldOp(yield_operands)

                        # Bind loop results (for use after the loop).
                        for res_ir, res_val in zip(results_ir, for_op.results):
                            ssa[res_ir.name] = res_val

                    # --- Control flow: scf.if ---
                    elif t == "scf_if":
                        if not _has_scf:
                            raise NotImplementedError(
                                "Control flow (if_) requires the SCF dialect. "
                                "Install a dialect wheel with SCF Python bindings."
                            )
                        cond = ssa[op.operands[0].name]
                        has_else = op.attrs.get("has_else", False)

                        if_op = scf_dialect.IfOp(cond, has_else=has_else)

                        with ir.InsertionPoint(if_op.then_block):
                            _emit_ops(op.regions[0])
                            scf_dialect.YieldOp([])

                        if has_else and len(op.regions) > 1:
                            with ir.InsertionPoint(if_op.else_block):
                                _emit_ops(op.regions[1])
                                scf_dialect.YieldOp([])

                    # --- Control flow: scf.while ---
                    elif t == "scf_while":
                        if not _has_scf:
                            raise NotImplementedError(
                                "Control flow (while_) requires the SCF dialect. "
                                "Install a dialect wheel with SCF Python bindings."
                            )
                        cond_result = op.attrs["cond_result"]

                        while_op = scf_dialect.WhileOp([], [])

                        # "before" region: evaluates the condition
                        before_block = while_op.before.blocks.append()
                        with ir.InsertionPoint(before_block):
                            _emit_ops(op.regions[0])
                            cond_val = ssa[cond_result.name]
                            scf_dialect.ConditionOp(cond_val, [])

                        # "after" region: loop body
                        after_block = while_op.after.blocks.append()
                        with ir.InsertionPoint(after_block):
                            _emit_ops(op.regions[1])
                            scf_dialect.YieldOp([])

                    else:
                        raise NotImplementedError(
                            f"No MLIR lowering for traced op {t!r}"
                        )

                _emit_ops(builder.ops)
                en.ReturnOp()

        return module


def emit_mlir(builder: KernelBuilder, vec_width: int = 0) -> str:
    """Return textual MLIR for the traced kernel."""
    return str(_build_module(builder, vec_width=vec_width))


def emit_msl(builder: KernelBuilder, vec_width: int = 0) -> str:
    """Trace -> MLIR -> MSL, using the dialect's TranslateToMSL binding."""
    from mlir.dialects import enigma as en

    module = _build_module(builder, vec_width=vec_width)
    return en.translate_to_msl(module.operation)
