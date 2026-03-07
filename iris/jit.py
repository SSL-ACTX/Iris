# iris/jit.py
"""High-level Python helpers for the JIT/offload feature.

The Rust side exposes a low-level helper `register_offload` that simply
records (and eventually compiles) a Python function.  The :func:`offload`
wrapper makes it convenient to use from pure Python.
"""
from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import warnings
import copy
import array as _array
from typing import Callable, Optional, Any

try:
    from .iris import (
        register_offload,
        offload_call,
        call_jit,
        configure_jit_logging,
        is_jit_logging_enabled,
        configure_quantum_speculation,
        is_quantum_speculation_enabled,
    )  # pyo3 extension
except ImportError:  # allow tests to import without extension built
    register_offload = None  # type: ignore
    offload_call = None  # type: ignore
    call_jit = None  # type: ignore
    call_jit_step_loop_f64 = None  # type: ignore
    configure_jit_logging = None  # type: ignore
    is_jit_logging_enabled = None  # type: ignore
    configure_quantum_speculation = None  # type: ignore
    is_quantum_speculation_enabled = None  # type: ignore

try:
    from .iris import call_jit_step_loop_f64  # type: ignore
except Exception:
    call_jit_step_loop_f64 = None  # type: ignore


def set_jit_logging(enabled: Optional[bool] = None, env_var: Optional[str] = None) -> bool:
    """Configure low-level Rust JIT logging.

    Parameters
    ----------
    enabled:
        - ``True``: force logs on
        - ``False``: force logs off
        - ``None``: use environment variable mode
    env_var:
        Environment variable name to read when ``enabled`` is ``None``.
        Default is ``IRIS_JIT_LOG``.
    """
    if configure_jit_logging is None:
        return False
    return bool(configure_jit_logging(enabled, env_var))


def get_jit_logging() -> bool:
    """Return whether Rust JIT logging is currently enabled."""
    if is_jit_logging_enabled is None:
        return False
    return bool(is_jit_logging_enabled())


def set_quantum_speculation(enabled: Optional[bool] = None, env_var: Optional[str] = None) -> bool:
    """Configure quantum-style multi-version JIT speculation.

    Parameters
    ----------
    enabled:
        - ``True``: force quantum speculation on
        - ``False``: force quantum speculation off
        - ``None``: use environment variable mode
    env_var:
        Environment variable name to read when ``enabled`` is ``None``.
        Default is ``IRIS_JIT_QUANTUM``.
    """
    if configure_quantum_speculation is None:
        return False
    return bool(configure_quantum_speculation(enabled, env_var))


def get_quantum_speculation() -> bool:
    """Return whether quantum-style multi-version JIT speculation is enabled."""
    if is_quantum_speculation_enabled is None:
        return False
    return bool(is_quantum_speculation_enabled())


def _strip_docstring(stmts: list[ast.stmt]) -> list[ast.stmt]:
    if (
        stmts
        and isinstance(stmts[0], ast.Expr)
        and isinstance(getattr(stmts[0], "value", None), ast.Constant)
        and isinstance(stmts[0].value.value, str)
    ):
        return stmts[1:]
    return stmts


class _NameSubstituter(ast.NodeTransformer):
    def __init__(self, var_name: str, replacement: ast.AST) -> None:
        self.var_name = var_name
        self.replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self.var_name:
            return copy.deepcopy(self.replacement)
        return node


def _extract_inline_template_from_callable(
    fn_obj: Any,
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[tuple[list[str], ast.AST]]:
    if not inspect.isfunction(fn_obj):
        return None

    try:
        helper_src = textwrap.dedent(inspect.getsource(fn_obj))
        helper_tree = ast.parse(helper_src)
    except Exception:
        return None

    helper_node: Optional[ast.FunctionDef] = None
    for node in helper_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_obj.__name__:
            helper_node = node
            break

    if helper_node is None:
        return None

    args = helper_node.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
        return None
    if args.defaults or args.kw_defaults:
        return None

    inlined = _extract_inlined_expr_plan(helper_node, fn_globals, inline_cache)
    if inlined is None:
        return None

    try:
        expr_ast = ast.parse(inlined, mode="eval").body
    except Exception:
        return None

    params = [arg.arg for arg in helper_node.args.args]
    return params, expr_ast


class _JitExprNormalizer(ast.NodeTransformer):
    def __init__(
        self,
        fn_globals: Optional[dict[str, Any]] = None,
        inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
        active_inline: Optional[set[str]] = None,
    ) -> None:
        self.fn_globals = fn_globals or {}
        self.inline_cache = inline_cache if inline_cache is not None else {}
        self.active_inline = active_inline if active_inline is not None else set()

    def _maybe_inline_call(self, node: ast.Call) -> Optional[ast.AST]:
        if node.keywords or not isinstance(node.func, ast.Name):
            return None

        name = node.func.id
        if name in self.active_inline:
            return None

        target = self.fn_globals.get(name)
        if not inspect.isfunction(target):
            return None

        if name in self.inline_cache:
            template = self.inline_cache[name]
        else:
            template = _extract_inline_template_from_callable(target, self.fn_globals, self.inline_cache)
            self.inline_cache[name] = template

        if template is None:
            return None

        params, body_expr = template
        if len(params) != len(node.args):
            return None

        env = {param: copy.deepcopy(arg) for param, arg in zip(params, node.args)}
        inlined = _subst_expr(
            body_expr,
            env,
            self.fn_globals,
            self.inline_cache,
            self.active_inline | {name},
        )
        return ast.copy_location(inlined, node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        inlined = self._maybe_inline_call(node)
        if inlined is not None:
            return inlined

        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "pow" and len(node.args) == 2 and not node.keywords:
                return ast.BinOp(
                    left=copy.deepcopy(node.args[0]),
                    op=ast.Pow(),
                    right=copy.deepcopy(node.args[1]),
                )
        return node


def _subst_expr(
    expr: ast.AST,
    env: dict[str, ast.AST],
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
    active_inline: Optional[set[str]] = None,
) -> ast.AST:
    out = copy.deepcopy(expr)
    for name, value in env.items():
        out = _NameSubstituter(name, value).visit(out)
    out = _JitExprNormalizer(fn_globals, inline_cache, active_inline).visit(out)
    return ast.fix_missing_locations(out)


def _merge_branch_env(cond_expr: ast.AST, left: dict[str, ast.AST], right: dict[str, ast.AST]) -> Optional[dict[str, ast.AST]]:
    merged: dict[str, ast.AST] = {}
    keys = set(left.keys()) | set(right.keys())
    for key in keys:
        if key not in left or key not in right:
            return None
        lv = left[key]
        rv = right[key]
        if ast.dump(lv) == ast.dump(rv):
            merged[key] = lv
        else:
            merged[key] = ast.IfExp(
                test=copy.deepcopy(cond_expr),
                body=copy.deepcopy(lv),
                orelse=copy.deepcopy(rv),
            )
    return merged


def _lower_block(
    stmts: list[ast.stmt],
    env: dict[str, ast.AST],
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
    active_inline: Optional[set[str]] = None,
) -> tuple[dict[str, ast.AST], Optional[ast.AST]]:
    current = dict(env)
    for stmt in stmts:
        if isinstance(stmt, ast.Pass):
            continue

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return current, None
            return current, _subst_expr(stmt.value, current, fn_globals, inline_cache, active_inline)

        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            current[stmt.targets[0].id] = _subst_expr(stmt.value, current, fn_globals, inline_cache, active_inline)
            continue

        if (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
        ):
            current[stmt.target.id] = _subst_expr(stmt.value, current, fn_globals, inline_cache, active_inline)
            continue

        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            target_name = stmt.target.id
            left_expr = copy.deepcopy(current.get(target_name, ast.Name(id=target_name, ctx=ast.Load())))
            right_expr = _subst_expr(stmt.value, current, fn_globals, inline_cache, active_inline)
            current[target_name] = ast.BinOp(left=left_expr, op=copy.deepcopy(stmt.op), right=right_expr)
            current[target_name] = ast.fix_missing_locations(current[target_name])
            continue

        if isinstance(stmt, ast.If):
            cond_expr = _subst_expr(stmt.test, current, fn_globals, inline_cache, active_inline)
            then_env, then_ret = _lower_block(list(stmt.body), dict(current), fn_globals, inline_cache, active_inline)
            if stmt.orelse:
                else_env, else_ret = _lower_block(list(stmt.orelse), dict(current), fn_globals, inline_cache, active_inline)
            else:
                else_env, else_ret = dict(current), None

            if then_ret is not None and else_ret is not None:
                return current, ast.IfExp(test=cond_expr, body=then_ret, orelse=else_ret)
            if (then_ret is None) != (else_ret is None):
                return current, None

            merged = _merge_branch_env(cond_expr, then_env, else_env)
            if merged is None:
                return current, None
            current = merged
            continue

        return current, None

    return current, None


def _extract_return_expr_plan(
    fn_node: ast.FunctionDef,
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[tuple[str, list[str]]]:
    stmts = _strip_docstring(list(fn_node.body))
    if len(stmts) == 1 and isinstance(stmts[0], ast.Return) and stmts[0].value is not None:
        return ast.unparse(_subst_expr(stmts[0].value, {}, fn_globals, inline_cache)), []
    return None


def _extract_inlined_expr_plan(
    fn_node: ast.FunctionDef,
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[str]:
    stmts = _strip_docstring(list(fn_node.body))
    _env, ret_expr = _lower_block(stmts, {}, fn_globals, inline_cache)
    if ret_expr is None:
        return None
    return ast.unparse(ret_expr)


def _extract_last_return_expr(
    fn_node: ast.FunctionDef,
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[str]:
    stmts = _strip_docstring(list(fn_node.body))
    for stmt in reversed(stmts):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return ast.unparse(_subst_expr(stmt.value, {}, fn_globals, inline_cache))
    return None


def _extract_stateful_loop_plan(
    fn_node: ast.FunctionDef,
    arg_names: list[str],
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[dict[str, Any]]:
    if len(arg_names) < 2:
        return None

    stmts = _strip_docstring(list(fn_node.body))
    if len(stmts) != 4:
        return None

    list_assign, state_assign, loop_stmt, ret_stmt = stmts
    if not (
        isinstance(list_assign, ast.Assign)
        and len(list_assign.targets) == 1
        and isinstance(list_assign.targets[0], ast.Name)
        and isinstance(list_assign.value, ast.List)
        and len(list_assign.value.elts) == 0
    ):
        return None
    list_var = list_assign.targets[0].id

    if not (
        isinstance(state_assign, ast.Assign)
        and len(state_assign.targets) == 1
        and isinstance(state_assign.targets[0], ast.Name)
        and isinstance(state_assign.value, ast.Name)
    ):
        return None
    state_var = state_assign.targets[0].id
    seed_src = ast.unparse(_subst_expr(state_assign.value, {}, fn_globals, inline_cache))

    if not isinstance(loop_stmt, ast.For) or not isinstance(loop_stmt.target, ast.Name):
        return None
    iter_var = loop_stmt.target.id
    if not (
        isinstance(loop_stmt.iter, ast.Call)
        and isinstance(loop_stmt.iter.func, ast.Name)
        and loop_stmt.iter.func.id == "range"
        and len(loop_stmt.iter.args) == 1
    ):
        return None
    range_arg = loop_stmt.iter.args[0]
    count_arg: Optional[str] = None
    if isinstance(range_arg, ast.Name):
        if range_arg.id not in arg_names:
            return None
        count_arg = range_arg.id
    elif (
        isinstance(range_arg, ast.Call)
        and isinstance(range_arg.func, ast.Name)
        and range_arg.func.id == "int"
        and len(range_arg.args) == 1
        and isinstance(range_arg.args[0], ast.Name)
    ):
        if range_arg.args[0].id not in arg_names:
            return None
        count_arg = range_arg.args[0].id
    else:
        return None

    if count_arg is None:
        return None

    if not (
        isinstance(ret_stmt, ast.Return)
        and isinstance(ret_stmt.value, ast.Name)
        and ret_stmt.value.id == list_var
    ):
        return None

    loop_body = list(loop_stmt.body)
    if not loop_body:
        return None
    append_stmt = loop_body[-1]
    if not (
        isinstance(append_stmt, ast.Expr)
        and isinstance(append_stmt.value, ast.Call)
        and isinstance(append_stmt.value.func, ast.Attribute)
        and isinstance(append_stmt.value.func.value, ast.Name)
        and append_stmt.value.func.value.id == list_var
        and append_stmt.value.func.attr == "append"
        and len(append_stmt.value.args) == 1
        and isinstance(append_stmt.value.args[0], ast.Name)
        and append_stmt.value.args[0].id == state_var
    ):
        return None

    state_expr: ast.AST = ast.Name(id=state_var, ctx=ast.Load())
    for stmt in loop_body[:-1]:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == state_var
        ):
            rhs = ast.fix_missing_locations(copy.deepcopy(stmt.value))
            state_expr = _subst_expr(
                rhs,
                {state_var: state_expr},
                fn_globals,
                inline_cache,
            )
            continue

        if (
            isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == state_var
        ):
            rhs = ast.BinOp(
                left=ast.Name(id=state_var, ctx=ast.Load()),
                op=stmt.op,
                right=copy.deepcopy(stmt.value),
            )
            state_expr = _subst_expr(
                rhs,
                {state_var: state_expr},
                fn_globals,
                inline_cache,
            )
            continue

        return None

    step_src = ast.unparse(state_expr)
    return {
        "count_arg": count_arg,
        "seed_src": seed_src,
        "iter_var": iter_var,
        "step_src": step_src,
        "step_args": [state_var, iter_var],
    }


def _extract_scalar_while_plan(
    fn_node: ast.FunctionDef,
    arg_names: list[str],
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[dict[str, Any]]:
    if len(arg_names) < 1:
        return None

    stmts = _strip_docstring(list(fn_node.body))
    if len(stmts) < 4:
        return None

    state_init = stmts[0]
    iter_init = stmts[1]
    while_stmt = stmts[2]
    ret_stmt = stmts[-1]

    if not (
        isinstance(state_init, ast.Assign)
        and len(state_init.targets) == 1
        and isinstance(state_init.targets[0], ast.Name)
    ):
        return None
    state_var = state_init.targets[0].id
    seed_expr = state_init.value

    if not (
        isinstance(iter_init, ast.Assign)
        and len(iter_init.targets) == 1
        and isinstance(iter_init.targets[0], ast.Name)
        and isinstance(iter_init.value, ast.Constant)
        and iter_init.value.value == 0
    ):
        return None
    iter_var = iter_init.targets[0].id

    if not isinstance(while_stmt, ast.While):
        return None

    test = while_stmt.test
    count_arg: Optional[str] = None
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == iter_var
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Lt)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Name)
    ):
        return None
    if test.comparators[0].id not in arg_names:
        return None
    count_arg = test.comparators[0].id

    if not (
        isinstance(ret_stmt, ast.Return)
        and isinstance(ret_stmt.value, ast.Name)
        and ret_stmt.value.id == state_var
    ):
        return None

    body = list(while_stmt.body)
    if not body:
        return None

    inc_stmt = body[-1]
    valid_inc = (
        isinstance(inc_stmt, ast.AugAssign)
        and isinstance(inc_stmt.target, ast.Name)
        and inc_stmt.target.id == iter_var
        and isinstance(inc_stmt.op, ast.Add)
        and isinstance(inc_stmt.value, ast.Constant)
        and inc_stmt.value.value == 1
    )
    if not valid_inc:
        return None

    state_expr: ast.AST = ast.Name(id=state_var, ctx=ast.Load())
    for stmt in body[:-1]:
        if isinstance(stmt, ast.Pass):
            continue
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == state_var
        ):
            rhs = ast.fix_missing_locations(copy.deepcopy(stmt.value))
            state_expr = _subst_expr(
                rhs,
                {state_var: state_expr},
                fn_globals,
                inline_cache,
            )
            continue
        if (
            isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == state_var
        ):
            rhs = ast.BinOp(
                left=ast.Name(id=state_var, ctx=ast.Load()),
                op=stmt.op,
                right=copy.deepcopy(stmt.value),
            )
            state_expr = _subst_expr(
                rhs,
                {state_var: state_expr},
                fn_globals,
                inline_cache,
            )
            continue
        return None

    return {
        "count_arg": count_arg,
        "iter_var": iter_var,
        "state_var": state_var,
        "seed_src": ast.unparse(_subst_expr(seed_expr, {}, fn_globals, inline_cache)),
        "step_src": ast.unparse(state_expr),
        "step_args": [state_var, iter_var],
    }


def _extract_scalar_for_plan(
    fn_node: ast.FunctionDef,
    arg_names: list[str],
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[dict[str, Any]]:
    if len(arg_names) < 1:
        return None

    stmts = _strip_docstring(list(fn_node.body))
    if len(stmts) < 3:
        return None

    state_init = stmts[0]
    for_idx: Optional[int] = None
    for idx, stmt in enumerate(stmts[1:-1], start=1):
        if isinstance(stmt, ast.For):
            for_idx = idx
            break
    if for_idx is None:
        return None

    for_stmt = stmts[for_idx]
    ret_stmt = stmts[-1]

    if not (
        isinstance(state_init, ast.Assign)
        and len(state_init.targets) == 1
        and isinstance(state_init.targets[0], ast.Name)
    ):
        return None
    state_var = state_init.targets[0].id
    seed_expr = state_init.value

    if not (isinstance(for_stmt, ast.For) and isinstance(for_stmt.target, ast.Name)):
        return None

    alias_count_map: dict[str, str] = {}
    for pre_stmt in stmts[1:for_idx]:
        if not (
            isinstance(pre_stmt, ast.Assign)
            and len(pre_stmt.targets) == 1
            and isinstance(pre_stmt.targets[0], ast.Name)
        ):
            return None
        alias_name = pre_stmt.targets[0].id
        if alias_name == state_var:
            return None

        value = pre_stmt.value
        if isinstance(value, ast.Name) and value.id in arg_names:
            alias_count_map[alias_name] = value.id
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "int"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in arg_names
        ):
            alias_count_map[alias_name] = value.args[0].id
            continue

        return None

    iter_var = for_stmt.target.id
    if not (
        isinstance(for_stmt.iter, ast.Call)
        and isinstance(for_stmt.iter.func, ast.Name)
        and for_stmt.iter.func.id == "range"
        and len(for_stmt.iter.args) == 1
    ):
        return None

    range_arg = for_stmt.iter.args[0]
    count_arg: Optional[str] = None
    if isinstance(range_arg, ast.Name):
        if range_arg.id in arg_names:
            count_arg = range_arg.id
        elif range_arg.id in alias_count_map:
            count_arg = alias_count_map[range_arg.id]
        else:
            return None
    elif (
        isinstance(range_arg, ast.Call)
        and isinstance(range_arg.func, ast.Name)
        and range_arg.func.id == "int"
        and len(range_arg.args) == 1
        and isinstance(range_arg.args[0], ast.Name)
    ):
        if range_arg.args[0].id not in arg_names:
            return None
        count_arg = range_arg.args[0].id
    else:
        return None

    if count_arg is None:
        return None

    if not (
        isinstance(ret_stmt, ast.Return)
        and isinstance(ret_stmt.value, ast.Name)
        and ret_stmt.value.id == state_var
    ):
        return None

    body = list(for_stmt.body)
    if not body:
        return None

    state_expr: ast.AST = ast.Name(id=state_var, ctx=ast.Load())
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == state_var
        ):
            rhs = ast.fix_missing_locations(copy.deepcopy(stmt.value))
            state_expr = _subst_expr(
                rhs,
                {state_var: state_expr},
                fn_globals,
                inline_cache,
            )
            continue
        if (
            isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == state_var
        ):
            rhs = ast.BinOp(
                left=ast.Name(id=state_var, ctx=ast.Load()),
                op=stmt.op,
                right=copy.deepcopy(stmt.value),
            )
            state_expr = _subst_expr(
                rhs,
                {state_var: state_expr},
                fn_globals,
                inline_cache,
            )
            continue
        return None

    return {
        "count_arg": count_arg,
        "iter_var": iter_var,
        "state_var": state_var,
        "seed_src": ast.unparse(_subst_expr(seed_expr, {}, fn_globals, inline_cache)),
        "step_src": ast.unparse(state_expr),
        "step_args": [state_var, iter_var],
    }


def _is_vector_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _vectorized_python_fallback(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    vector_positions: list[int] = []
    vector_len: Optional[int] = None
    for idx, arg in enumerate(args):
        if not _is_vector_like(arg):
            continue
        current_len = len(arg)
        if vector_len is None:
            vector_len = current_len
        elif current_len != vector_len:
            raise ValueError("vectorized fallback inputs must have matching lengths")
        vector_positions.append(idx)

    if vector_len is None:
        return func(*args, **kwargs)

    out: list[Any] = []
    for i in range(vector_len):
        iter_args = [arg[i] if idx in vector_positions else arg for idx, arg in enumerate(args)]
        out.append(func(*iter_args, **kwargs))

    try:
        return _array.array("d", (float(v) for v in out))
    except Exception:
        return out


def offload(strategy: str = "actor", return_type: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that marks a function for execution on the Iris JIT/actor pool.

    The decorated function is returned unchanged; the runtime keeps track of
    metadata and may later compile or route the call to native code.

    Example
    -------
    >>> @iris.offload(strategy="actor", return_type="float")
    ... def add(a: float, b: float) -> float:
    ...     return a + b

    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        inline_cache: dict[str, Optional[tuple[list[str], ast.AST]]] = {}
        jit_eval_globals: dict[str, Any] = func.__globals__
        src: Optional[str] = None
        arg_names: Optional[list[str]] = None
        loop_plan: Optional[dict[str, Any]] = None
        scalar_while_plan: Optional[dict[str, Any]] = None
        scalar_for_plan: Optional[dict[str, Any]] = None
        aggressive_src: Optional[str] = None
        sig: Optional[inspect.Signature] = None

        if strategy == "jit":
            try:
                src_txt = inspect.getsource(func)
                src_txt = textwrap.dedent(src_txt)
                tree = ast.parse(src_txt)
                sig = inspect.signature(func)
                arg_names = list(sig.parameters.keys())
                try:
                    closure_vars = inspect.getclosurevars(func)
                    inline_ns = dict(func.__globals__)
                    inline_ns.update(closure_vars.nonlocals)
                    jit_eval_globals = inline_ns
                except Exception:
                    jit_eval_globals = func.__globals__

                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.body:
                        expr_plan = _extract_return_expr_plan(node, jit_eval_globals, inline_cache)
                        if expr_plan is not None:
                            src, _ = expr_plan
                        else:
                            inlined = _extract_inlined_expr_plan(node, jit_eval_globals, inline_cache)
                            if inlined is not None:
                                src = inlined
                            else:
                                src = None
                                if arg_names is not None:
                                    loop_plan = _extract_stateful_loop_plan(node, arg_names, jit_eval_globals, inline_cache)
                                    if loop_plan is None:
                                        scalar_while_plan = _extract_scalar_while_plan(node, arg_names, jit_eval_globals, inline_cache)
                                        if scalar_while_plan is None:
                                            scalar_for_plan = _extract_scalar_for_plan(node, arg_names, jit_eval_globals, inline_cache)
                                aggressive_src = _extract_last_return_expr(node, jit_eval_globals, inline_cache)
                        break
            except Exception:
                pass

        if register_offload is not None:
            try:
                register_offload(func, strategy, return_type, src, arg_names)
            except Exception as e:  # pragma: no cover - defensive
                warnings.warn(f"offload registration failed: {e}")

        # Wrap with runtime call depending on strategy
        if strategy == "actor" and offload_call is not None:
            @functools.wraps(func)
            def actor_wrapper(*args: Any, **kwargs: Any) -> Any:
                return offload_call(func, args, kwargs)
            return actor_wrapper
            
        elif strategy == "jit" and call_jit is not None:
            if src is None and loop_plan is None and scalar_while_plan is None and scalar_for_plan is None:
                if aggressive_src is not None and arg_names is not None and register_offload is not None:
                    try:
                        register_offload(func, strategy, return_type, aggressive_src, arg_names)
                    except Exception:
                        pass

                    @functools.wraps(func)
                    def aggressive_vector_wrapper(*args: Any, **kwargs: Any) -> Any:
                        has_vector = any(_is_vector_like(a) for a in args)
                        if has_vector:
                            try:
                                return call_jit(func, args, kwargs)
                            except RuntimeError as e:
                                msg = str(e)
                                if (
                                    "no JIT entry" in msg
                                    or "failed to compile" in msg
                                    or "jit panic" in msg
                                    or "wrong argument count" in msg
                                ):
                                    return _vectorized_python_fallback(func, args, kwargs)
                                raise
                        return func(*args, **kwargs)

                    return aggressive_vector_wrapper

                @functools.wraps(func)
                def py_fallback_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return _vectorized_python_fallback(func, args, kwargs)

                return py_fallback_wrapper

            if src is not None:
                @functools.wraps(func)
                def jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        res = call_jit(func, args, kwargs)
                    except RuntimeError as e:
                        msg = str(e)
                        if (
                            "no JIT entry" in msg
                            or "failed to compile" in msg
                            or "jit panic" in msg
                        ):
                            if any(_is_vector_like(a) for a in args):
                                return _vectorized_python_fallback(func, args, kwargs)
                            return func(*args, **kwargs)
                        raise
                    reduction_mode: Optional[str] = None
                    src_s = src.strip()
                    if src_s.startswith("sum("):
                        reduction_mode = "sum"
                    elif src_s.startswith("any("):
                        reduction_mode = "any"
                    elif src_s.startswith("all("):
                        reduction_mode = "all"

                    try:
                        if reduction_mode is not None and hasattr(res, "__iter__") and not isinstance(res, (float, int)):
                            if reduction_mode == "any":
                                return 1.0 if any(float(v) != 0.0 for v in res) else 0.0
                            if reduction_mode == "all":
                                return 1.0 if all(float(v) != 0.0 for v in res) else 0.0
                            total = 0.0
                            for v in res:
                                total += float(v)
                            return total
                    except Exception:
                        pass
                    return res

                return jit_wrapper

            if loop_plan is not None and register_offload is not None and sig is not None:
                step_src = loop_plan["step_src"]
                step_args = loop_plan["step_args"]
                seed_src = loop_plan["seed_src"]
                count_arg = loop_plan["count_arg"]
                iter_var = loop_plan["iter_var"]

                def _iris_step(x: float, i: float) -> float:
                    namespace = {step_args[0]: x, step_args[1]: i}
                    return float(eval(step_src, jit_eval_globals, namespace))

                try:
                    register_offload(_iris_step, "jit", "float", step_src, step_args)
                except Exception:
                    return func

                @functools.wraps(func)
                def loop_jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                    try:
                        bound = sig.bind_partial(*args, **kwargs)
                    except Exception:
                        return func(*args, **kwargs)
                    bound.apply_defaults()
                    if count_arg not in bound.arguments:
                        return func(*args, **kwargs)

                    local_seed_ns = dict(bound.arguments)
                    try:
                        state = float(eval(seed_src, jit_eval_globals, local_seed_ns))
                        count = int(bound.arguments[count_arg])
                    except Exception:
                        return func(*args, **kwargs)

                    if count <= 0:
                        return []

                    out = []
                    for i in range(count):
                        iter_val = float(i)
                        try:
                            state = float(call_jit(_iris_step, (state, iter_val), None))
                        except RuntimeError as e:
                            msg = str(e)
                            if (
                                "no JIT entry" in msg
                                or "failed to compile" in msg
                                or "jit panic" in msg
                            ):
                                local_ns = {step_args[0]: state, iter_var: i, step_args[1]: iter_val}
                                state = float(eval(step_src, jit_eval_globals, local_ns))
                            else:
                                raise
                        out.append(state)
                    return out

                return loop_jit_wrapper

            if scalar_while_plan is not None and register_offload is not None and sig is not None:
                step_src = scalar_while_plan["step_src"]
                step_args = scalar_while_plan["step_args"]
                count_arg = scalar_while_plan["count_arg"]
                iter_var = scalar_while_plan["iter_var"]
                seed_src = scalar_while_plan["seed_src"]

                def _iris_step(x: float, i: float) -> float:
                    namespace = {step_args[0]: x, step_args[1]: i}
                    return float(eval(step_src, jit_eval_globals, namespace))

                try:
                    register_offload(_iris_step, "jit", "float", step_src, step_args)
                except Exception:
                    return func

                @functools.wraps(func)
                def while_jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if any(_is_vector_like(a) for a in args):
                        return _vectorized_python_fallback(func, args, kwargs)

                    try:
                        bound = sig.bind_partial(*args, **kwargs)
                    except Exception:
                        return func(*args, **kwargs)
                    bound.apply_defaults()
                    if count_arg not in bound.arguments:
                        return func(*args, **kwargs)

                    local_seed_ns = dict(bound.arguments)
                    try:
                        state = float(eval(seed_src, jit_eval_globals, local_seed_ns))
                        count = int(bound.arguments[count_arg])
                    except Exception:
                        return func(*args, **kwargs)

                    if count <= 0:
                        return state

                    if call_jit_step_loop_f64 is not None:
                        try:
                            return float(call_jit_step_loop_f64(_iris_step, state, count))
                        except RuntimeError as e:
                            msg = str(e)
                            if not (
                                "no JIT entry" in msg
                                or "failed to compile" in msg
                                or "jit panic" in msg
                                or "step loop requires" in msg
                            ):
                                raise

                    for i in range(count):
                        iter_val = float(i)
                        try:
                            state = float(call_jit(_iris_step, (state, iter_val), None))
                        except RuntimeError as e:
                            msg = str(e)
                            if (
                                "no JIT entry" in msg
                                or "failed to compile" in msg
                                or "jit panic" in msg
                            ):
                                local_ns = {step_args[0]: state, iter_var: i, step_args[1]: iter_val}
                                state = float(eval(step_src, jit_eval_globals, local_ns))
                            else:
                                raise
                    return state

                return while_jit_wrapper

            if scalar_for_plan is not None and register_offload is not None and sig is not None:
                step_src = scalar_for_plan["step_src"]
                step_args = scalar_for_plan["step_args"]
                count_arg = scalar_for_plan["count_arg"]
                iter_var = scalar_for_plan["iter_var"]
                seed_src = scalar_for_plan["seed_src"]

                def _iris_step(x: float, i: float) -> float:
                    namespace = {step_args[0]: x, step_args[1]: i}
                    return float(eval(step_src, jit_eval_globals, namespace))

                try:
                    register_offload(_iris_step, "jit", "float", step_src, step_args)
                except Exception:
                    return func

                @functools.wraps(func)
                def for_jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                    if any(_is_vector_like(a) for a in args):
                        return _vectorized_python_fallback(func, args, kwargs)

                    try:
                        bound = sig.bind_partial(*args, **kwargs)
                    except Exception:
                        return func(*args, **kwargs)
                    bound.apply_defaults()
                    if count_arg not in bound.arguments:
                        return func(*args, **kwargs)

                    local_seed_ns = dict(bound.arguments)
                    try:
                        state = float(eval(seed_src, jit_eval_globals, local_seed_ns))
                        count = int(bound.arguments[count_arg])
                    except Exception:
                        return func(*args, **kwargs)

                    if count <= 0:
                        return state

                    if call_jit_step_loop_f64 is not None:
                        try:
                            return float(call_jit_step_loop_f64(_iris_step, state, count))
                        except RuntimeError as e:
                            msg = str(e)
                            if not (
                                "no JIT entry" in msg
                                or "failed to compile" in msg
                                or "jit panic" in msg
                                or "step loop requires" in msg
                            ):
                                raise

                    for i in range(count):
                        iter_val = float(i)
                        try:
                            state = float(call_jit(_iris_step, (state, iter_val), None))
                        except RuntimeError as e:
                            msg = str(e)
                            if (
                                "no JIT entry" in msg
                                or "failed to compile" in msg
                                or "jit panic" in msg
                            ):
                                local_ns = {step_args[0]: state, iter_var: i, step_args[1]: iter_val}
                                state = float(eval(step_src, jit_eval_globals, local_ns))
                            else:
                                raise
                    return state

                return for_jit_wrapper

            return func

        return func

    return decorator