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
    from .iris import register_offload, offload_call, call_jit  # pyo3 extension
except ImportError:  # allow tests to import without extension built
    register_offload = None  # type: ignore
    offload_call = None  # type: ignore
    call_jit = None  # type: ignore


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
                return call_jit(func, args, kwargs)
            return jit_wrapper

        return func

    return decorator