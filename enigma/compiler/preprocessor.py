"""AST preprocessor for @enigma.kernel functions.

Rewrites `for i in enigma.range(...)` into closures with automatic
loop-carried variable tracking, similar to CuTe DSL.

`for i in enigma.range_constexpr(N)` is rewritten to `for i in range(N)`
(fully unrolled at Python trace time).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable, Set


def preprocess_kernel(fn: Callable) -> Callable:
    """Preprocess a kernel function: rewrite enigma.range() for loops.

    If no enigma.range() calls are found, returns the original function.
    Falls back to original if source code is unavailable (e.g. -c mode).
    """
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return fn

    source = textwrap.dedent(source)
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn.__name__:
            func_def = node
            break

    if func_def is None:
        return fn

    rewriter = _ForLoopRewriter(func_def)
    new_tree = rewriter.visit(tree)

    if not rewriter.did_rewrite:
        return fn

    func_def.decorator_list = []

    ast.fix_missing_locations(new_tree)
    code = compile(new_tree, inspect.getfile(fn), "exec")

    globs = dict(fn.__globals__)
    from .._tracing import _enigma_for_range
    globs["_enigma_for_range"] = _enigma_for_range

    local_ns = {}
    exec(code, globs, local_ns)
    new_fn = local_ns[fn.__name__]
    new_fn.__module__ = fn.__module__
    return new_fn


class _ForLoopRewriter(ast.NodeTransformer):
    """Rewrite `for i in enigma.range(...)` with automatic variable tracking."""

    def __init__(self, func_def: ast.FunctionDef):
        self.func_def = func_def
        self.param_names = {a.arg for a in func_def.args.args}
        self._counter = 0
        self.did_rewrite = False

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)

        if _is_enigma_range_constexpr(node.iter):
            self.did_rewrite = True
            node.iter = ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=node.iter.args,
                keywords=[],
            )
            return node

        if not _is_enigma_range(node.iter):
            return node

        self.did_rewrite = True

        start, stop, step = _extract_range_args(node.iter)
        target_name = node.target.id if isinstance(node.target, ast.Name) else None
        if target_name is None:
            raise SyntaxError("enigma.range() loop variable must be a simple name")

        assigned_in_body = _analyze_assigned_vars(node.body)
        live_before = self._compute_live_before(node)

        if target_name in assigned_in_body:
            assigned_in_body.discard(target_name)

        carried = sorted(assigned_in_body & live_before)

        body_name = f"_loop_body_{self._counter}"
        self._counter += 1

        body_params = [ast.arg(arg=target_name)] + [ast.arg(arg=n) for n in carried]
        body_stmts = list(node.body)

        if carried:
            ret = ast.Return(value=ast.Tuple(
                elts=[ast.Name(id=n, ctx=ast.Load()) for n in carried],
                ctx=ast.Load(),
            ))
            body_stmts.append(ret)

        body_func = ast.FunctionDef(
            name=body_name,
            args=ast.arguments(
                posonlyargs=[], args=body_params, vararg=None,
                kwonlyargs=[], kw_defaults=[], kwargs=None, defaults=[],
            ),
            body=body_stmts,
            decorator_list=[],
            returns=None,
        )

        call_args = [start, stop, step,
                     ast.Name(id=body_name, ctx=ast.Load())]
        call_args += [ast.Name(id=n, ctx=ast.Load()) for n in carried]

        call = ast.Call(
            func=ast.Name(id="_enigma_for_range", ctx=ast.Load()),
            args=call_args,
            keywords=[],
        )

        if carried:
            assign = ast.Assign(
                targets=[ast.Tuple(
                    elts=[ast.Name(id=n, ctx=ast.Store()) for n in carried],
                    ctx=ast.Store(),
                )],
                value=call,
            )
        else:
            assign = ast.Expr(value=call)

        return [body_func, assign]

    def _compute_live_before(self, target_node: ast.For) -> Set[str]:
        live = set(self.param_names)
        for stmt in self.func_def.body:
            if stmt is target_node:
                break
            live |= _analyze_assigned_vars([stmt])
        return live


def _analyze_assigned_vars(stmts: list) -> Set[str]:
    names = set()
    for node in ast.walk(ast.Module(body=stmts, type_ignores=[])):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                names |= _extract_target_names(t)
        elif isinstance(node, ast.AugAssign):
            names |= _extract_target_names(node.target)
    return names


def _extract_target_names(target) -> Set[str]:
    names = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
        for elt in target.elts:
            names |= _extract_target_names(elt)
    elif isinstance(target, ast.Subscript):
        if isinstance(target.value, ast.Name):
            names.add(target.value.id)
    elif isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Name):
            names.add(target.value.id)
    return names


def _is_enigma_range(node) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "range":
            if isinstance(func.value, ast.Name) and func.value.id == "enigma":
                return True
    return False


def _is_enigma_range_constexpr(node) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "range_constexpr":
            if isinstance(func.value, ast.Name) and func.value.id == "enigma":
                return True
    return False


def _extract_range_args(call_node: ast.Call):
    args = call_node.args
    if len(args) == 1:
        return ast.Constant(value=0), args[0], ast.Constant(value=1)
    elif len(args) == 2:
        return args[0], args[1], ast.Constant(value=1)
    elif len(args) == 3:
        return args[0], args[1], args[2]
    raise SyntaxError(f"enigma.range() expects 1-3 arguments, got {len(args)}")
