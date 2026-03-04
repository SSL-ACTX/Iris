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
from typing import Callable, Optional, Any

try:
    from .iris import (
        register_offload,
        offload_call,
        call_jit,
        configure_jit_logging,
        is_jit_logging_enabled,
    )  # pyo3 extension
except ImportError:  # allow tests to import without extension built
    register_offload = None  # type: ignore
    offload_call = None  # type: ignore
    call_jit = None  # type: ignore
    configure_jit_logging = None  # type: ignore
    is_jit_logging_enabled = None  # type: ignore


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
        src: Optional[str] = None
        arg_names: Optional[list[str]] = None

        if strategy == "jit":
            try:
                src_txt = inspect.getsource(func)
                src_txt = textwrap.dedent(src_txt)
                
                # Parse function and find the return expression, skipping docstrings
                tree = ast.parse(src_txt)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.body:
                        for stmt in node.body:
                            if isinstance(stmt, ast.Return) and stmt.value is not None:
                                # ast.unparse available in py3.9+
                                src = ast.unparse(stmt.value)
                                break
                        if src is not None:
                            break
                            
                sig = inspect.signature(func)
                arg_names = list(sig.parameters.keys())
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
            @functools.wraps(func)
            def jit_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    res = call_jit(func, args, kwargs)
                except RuntimeError as e:
                    # common failure when JIT entry is missing; fall back to
                    # executing the original Python function.  This keeps the
                    # decorator non‑fatal when compilation is not possible.
                    msg = str(e)
                    if "no JIT entry" in msg or "failed to compile" in msg:
                        return func(*args, **kwargs)
                    raise
                reduction_mode: Optional[str] = None
                if isinstance(src, str):
                    src_s = src.strip()
                    if src_s.startswith("sum("):
                        reduction_mode = "sum"
                    elif src_s.startswith("any("):
                        reduction_mode = "any"
                    elif src_s.startswith("all("):
                        reduction_mode = "all"

                # If JIT returned a sequence (vectorized run), reduce according
                # to generator semantics.
                try:
                    if hasattr(res, "__iter__") and not isinstance(res, (float, int)):
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

        return func

    return decorator