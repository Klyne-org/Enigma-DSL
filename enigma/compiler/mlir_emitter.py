"""Emit MLIR from traced IR using the Enigma dialect python bindings,
then translate to MSL via the dialect's TranslateToMSL binding.

The DSL package is `enigma`; the dialect submodule is `mlir.dialects.enigma`.
Separate namespaces, no collision.
"""

from __future__ import annotations

from .._tracing import KernelBuilder


_DIM_XYZ = {"x": 0, "y": 1, "z": 2}


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
        index_t = ir.IndexType.get()

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
