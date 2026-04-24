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
import logging
import textwrap
import warnings
import copy
import array as _array
import hashlib
import os
import platform
import sys
import time
import zlib
import threading
import atexit
from typing import Callable, Optional, Any

try:
    import msgpack as _msgpack
except Exception:
    _msgpack = None

try:
    from .iris import (
        register_offload,
        offload_call,
        call_jit,
        configure_jit_logging,
        is_jit_logging_enabled,
        configure_quantum_speculation,
        is_quantum_speculation_enabled,
        get_quantum_profile as _get_quantum_profile,
        seed_quantum_profile as _seed_quantum_profile,
        configure_quantum_speculation_threshold as _configure_quantum_speculation_threshold,
        get_quantum_speculation_threshold as _get_quantum_speculation_threshold,
        configure_quantum_log_threshold as _configure_quantum_log_threshold,
        get_quantum_log_threshold as _get_quantum_log_threshold,
        configure_quantum_compile_budget as _configure_quantum_compile_budget,
        get_quantum_compile_budget as _get_quantum_compile_budget,
        configure_quantum_cooldown as _configure_quantum_cooldown,
        get_quantum_cooldown as _get_quantum_cooldown,
    )
except ImportError:
    register_offload = None
    offload_call = None
    call_jit = None
    call_jit_step_loop_f64 = None
    configure_jit_logging = None
    is_jit_logging_enabled = None
    configure_quantum_speculation = None
    is_quantum_speculation_enabled = None
    _get_quantum_profile = None
    _seed_quantum_profile = None
    _configure_quantum_speculation_threshold = None
    _get_quantum_speculation_threshold = None
    _configure_quantum_log_threshold = None
    _get_quantum_log_threshold = None
    _configure_quantum_compile_budget = None
    _get_quantum_compile_budget = None
    _configure_quantum_cooldown = None
    _get_quantum_cooldown = None

try:
    from .iris import call_jit_step_loop_f64
except Exception:
    call_jit_step_loop_f64 = None


_IRIS_META_SCHEMA = 1
_IRIS_META_FILENAME = ".iris.meta.bin"
_IRIS_META_MAGIC = b"IRSMETA1"
_IRIS_META_FLAG_COMPRESSED = 0x1
_IRIS_META_FLUSH_EVERY = 16
_IRIS_META_TTL_NS = int(
    os.environ.get("IRIS_JIT_META_TTL_NS", str(7 * 24 * 60 * 60 * 1_000_000_000))
)
_IRIS_META_MAX_ENTRIES = int(os.environ.get("IRIS_JIT_META_MAX_ENTRIES", "256"))
_IRIS_META_FLUSH_MIN = int(os.environ.get("IRIS_JIT_META_FLUSH_MIN", "8"))
_IRIS_META_FLUSH_MAX = int(os.environ.get("IRIS_JIT_META_FLUSH_MAX", "128"))
_IRIS_META_COMPRESS_MIN_BYTES = int(
    os.environ.get("IRIS_JIT_META_COMPRESS_MIN_BYTES", "4096")
)
_IRIS_META_REFRESH_NS = int(
    os.environ.get("IRIS_JIT_META_REFRESH_NS", str(6 * 60 * 60 * 1_000_000_000))
)
_IRIS_META_COUNTERS: dict[str, int] = {}
_IRIS_META_FLUSH_INTERVALS: dict[str, int] = {}
_IRIS_META_LAST_SIGNATURES: dict[str, str] = {}
_IRIS_META_LAST_DEFER_COUNTS: dict[str, int] = {}
_IRIS_META_PENDING: dict[
    str, tuple[Callable[..., Any], str, list[str], Optional[str]]
] = {}
_IRIS_META_STATE_LOCK = threading.RLock()


_jit_meta_logger = logging.getLogger("iris.jit.meta")


def _jit_meta_log(message: str) -> None:
    try:
        enabled = bool(get_jit_logging())
    except Exception:
        enabled = False
    if not enabled:
        return
    try:
        if not bool(get_quantum_speculation()):
            return
    except Exception:
        pass

    try:
        if not logging.root.handlers and not _jit_meta_logger.handlers:
            logging.basicConfig(level=logging.INFO)
        if not _jit_meta_logger.isEnabledFor(logging.INFO):
            _jit_meta_logger.setLevel(logging.INFO)
        _jit_meta_logger.info(message)
    except Exception:
        try:
            sys.stderr.write(f"[Iris][jit][meta] {message}\n")
        except Exception:
            pass


def _empty_metadata_doc() -> dict[str, Any]:
    return {"schema": _IRIS_META_SCHEMA, "entries": {}}


def _normalize_metadata_policy() -> tuple[int, int]:
    ttl_ns = (
        _IRIS_META_TTL_NS if _IRIS_META_TTL_NS > 0 else 7 * 24 * 60 * 60 * 1_000_000_000
    )
    max_entries = _IRIS_META_MAX_ENTRIES if _IRIS_META_MAX_ENTRIES > 0 else 256
    return ttl_ns, max_entries


def _normalize_flush_window() -> tuple[int, int]:
    min_flush = (
        _IRIS_META_FLUSH_MIN if _IRIS_META_FLUSH_MIN > 0 else _IRIS_META_FLUSH_EVERY
    )
    max_flush = (
        _IRIS_META_FLUSH_MAX
        if _IRIS_META_FLUSH_MAX > 0
        else max(min_flush, _IRIS_META_FLUSH_EVERY)
    )
    if max_flush < min_flush:
        max_flush = min_flush
    return min_flush, max_flush


def _prune_metadata_entries(entries: dict[str, Any], now_ns: int) -> dict[str, Any]:
    ttl_ns, max_entries = _normalize_metadata_policy()
    min_updated = now_ns - ttl_ns
    kept: list[tuple[str, dict[str, Any], int]] = []

    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        updated_ns_raw = value.get("updated_ns", 0)
        try:
            updated_ns = int(updated_ns_raw)
        except Exception:
            updated_ns = 0
        if updated_ns < min_updated:
            continue
        kept.append((key, value, updated_ns))

    kept.sort(key=lambda item: item[2], reverse=True)
    if len(kept) > max_entries:
        kept = kept[:max_entries]

    return {key: value for key, value, _ in kept}


def _profile_variant_shape(rows: Any) -> tuple[int, ...]:
    if not isinstance(rows, list):
        return tuple()
    shape: list[int] = []
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        try:
            shape.append(int(row[0]))
        except Exception:
            continue
    return tuple(shape)


def _normalize_profile_rows(rows: Any) -> list[list[Any]]:
    if not isinstance(rows, list):
        return []
    out: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            continue
        try:
            out.append([int(row[0]), float(row[1]), int(row[2]), int(row[3])])
        except Exception:
            continue
    out.sort(key=lambda item: item[0])
    return out


def _merge_profile_rows(
    existing_rows: Any, incoming_rows: list[list[Any]]
) -> list[list[Any]]:
    merged: dict[int, list[Any]] = {}
    for row in _normalize_profile_rows(existing_rows):
        merged[int(row[0])] = row
    for row in incoming_rows:
        merged[int(row[0])] = row
    return [merged[idx] for idx in sorted(merged.keys())]


def _metadata_cache_path(func: Callable[..., Any]) -> Optional[str]:
    try:
        source_file = inspect.getsourcefile(func) or inspect.getfile(func)
    except Exception:
        return None
    if not source_file:
        return None
    module_dir = os.path.dirname(os.path.abspath(source_file))
    pycache_dir = os.path.join(module_dir, "__pycache__")
    return os.path.join(pycache_dir, _IRIS_META_FILENAME)


def _metadata_key(
    func: Callable[..., Any],
    src: str,
    arg_names: list[str],
    return_type: Optional[str],
) -> str:
    fn_module = getattr(func, "__module__", "")
    fn_qualname = getattr(func, "__qualname__", getattr(func, "__name__", ""))
    stable_payload = (
        fn_module,
        fn_qualname,
        src,
        tuple(arg_names),
        return_type or "float",
        f"{sys.version_info.major}.{sys.version_info.minor}",
        platform.machine(),
    )
    if _msgpack is not None:
        payload_bytes = _msgpack.packb(stable_payload, use_bin_type=True)
    else:
        payload_bytes = repr(stable_payload).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()


def _pack_metadata_doc(doc: dict[str, Any]) -> bytes:
    if _msgpack is None:
        return b""
    payload = _msgpack.packb(doc, use_bin_type=True)
    flags = 0
    if (
        _IRIS_META_COMPRESS_MIN_BYTES > 0
        and len(payload) >= _IRIS_META_COMPRESS_MIN_BYTES
    ):
        payload = zlib.compress(payload)
        flags |= _IRIS_META_FLAG_COMPRESSED
    return _IRIS_META_MAGIC + bytes([flags]) + payload


def _unpack_metadata_doc(raw: bytes) -> dict[str, Any]:
    if _msgpack is None or len(raw) < len(_IRIS_META_MAGIC) + 1:
        return _empty_metadata_doc()
    if raw[: len(_IRIS_META_MAGIC)] != _IRIS_META_MAGIC:
        return _empty_metadata_doc()

    flags = raw[len(_IRIS_META_MAGIC)]
    payload = raw[len(_IRIS_META_MAGIC) + 1 :]
    if flags & _IRIS_META_FLAG_COMPRESSED:
        payload = zlib.decompress(payload)

    doc = _msgpack.unpackb(payload, raw=False)
    if not isinstance(doc, dict):
        return _empty_metadata_doc()
    return doc


def _read_metadata(path: str) -> dict[str, Any]:
    if _msgpack is None:
        _jit_meta_log("read skipped: msgpack unavailable")
        return _empty_metadata_doc()
    try:
        with open(path, "rb") as f:
            raw = f.read()
        doc = _unpack_metadata_doc(raw)
        if not isinstance(doc, dict):
            return _empty_metadata_doc()
        if int(doc.get("schema", 0)) != _IRIS_META_SCHEMA:
            return _empty_metadata_doc()
        entries = doc.get("entries")
        if not isinstance(entries, dict):
            return _empty_metadata_doc()
        doc["entries"] = _prune_metadata_entries(entries, time.time_ns())
        _jit_meta_log(
            f"read ok: path={path} entries={len(doc.get('entries', {}))} bytes={len(raw)}"
        )
        return doc
    except Exception as exc:
        _jit_meta_log(f"read failed: path={path} err={exc}")
        return _empty_metadata_doc()


def _write_metadata(path: str, doc: dict[str, Any]) -> None:
    if _msgpack is None:
        _jit_meta_log("write skipped: msgpack unavailable")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"
    payload = _pack_metadata_doc(doc)
    if not payload:
        _jit_meta_log("write skipped: empty payload")
        return

    # Avoid rewriting the file when the content is identical to preserve mtime.
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                if f.read() == payload:
                    _jit_meta_log(f"write skipped: unchanged path={path}")
                    return
        except Exception:
            pass

    try:
        with open(temp_path, "wb") as f:
            f.write(payload)
        os.replace(temp_path, path)
        _jit_meta_log(
            f"write ok: path={path} entries={len(doc.get('entries', {}))} bytes={len(payload)}"
        )
    except Exception as exc:
        _jit_meta_log(f"write failed: path={path} err={exc}")


def _seed_quantum_from_metadata(
    func: Callable[..., Any],
    src: Optional[str],
    arg_names: Optional[list[str]],
    return_type: Optional[str],
) -> bool:
    if _msgpack is None:
        _jit_meta_log("seed skipped: msgpack unavailable")
        return False
    if _seed_quantum_profile is None:
        _jit_meta_log("seed skipped: seed API unavailable")
        return False
    if src is None:
        _jit_meta_log("seed skipped: source expression unavailable")
        return False
    if arg_names is None:
        _jit_meta_log("seed skipped: arg_names unavailable")
        return False
    path = _metadata_cache_path(func)
    if path is None:
        _jit_meta_log("seed skipped: no source path for function")
        return False
    if not os.path.exists(path):
        _jit_meta_log(f"seed miss: cache file not found path={path}")
        return False
    key = _metadata_key(func, src, arg_names, return_type)
    doc = _read_metadata(path)
    entry = doc.get("entries", {}).get(key)
    if not isinstance(entry, dict):
        _jit_meta_log(f"seed miss: key not present key={key[:12]} path={path}")
        return False
    rows = entry.get("profile")
    if not isinstance(rows, list) or not rows:
        _jit_meta_log(f"seed miss: empty profile key={key[:12]} path={path}")
        return False
    normalized: list[tuple[int, float, int, int]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            continue
        try:
            normalized.append((int(row[0]), float(row[1]), int(row[2]), int(row[3])))
        except Exception:
            continue
    if not normalized:
        _jit_meta_log(
            f"seed miss: profile normalization empty key={key[:12]} path={path}"
        )
        return False
    try:
        ok = bool(_seed_quantum_profile(func, normalized))
        _jit_meta_log(
            f"seed {'ok' if ok else 'failed'}: rows={len(normalized)} key={key[:12]} path={path}"
        )
        return ok
    except Exception as exc:
        _jit_meta_log(f"seed failed: key={key[:12]} path={path} err={exc}")
        return False


def _maybe_persist_quantum_metadata(
    func: Callable[..., Any],
    src: Optional[str],
    arg_names: Optional[list[str]],
    return_type: Optional[str],
    force: bool = False,
) -> None:
    if _msgpack is None:
        _jit_meta_log("persist skipped: msgpack unavailable")
        return
    if _get_quantum_profile is None:
        _jit_meta_log("persist skipped: profile API unavailable")
        return
    if src is None:
        _jit_meta_log("persist skipped: source expression unavailable")
        return
    if arg_names is None:
        _jit_meta_log("persist skipped: arg_names unavailable")
        return
    path = _metadata_cache_path(func)
    if path is None:
        _jit_meta_log("persist skipped: no source path for function")
        return
    key = _metadata_key(func, src, arg_names, return_type)
    with _IRIS_META_STATE_LOCK:
        _IRIS_META_PENDING[key] = (func, src, list(arg_names), return_type)
        min_flush, max_flush = _normalize_flush_window()
        flush_interval = _IRIS_META_FLUSH_INTERVALS.get(key, min_flush)
        if flush_interval < min_flush:
            flush_interval = min_flush
        if flush_interval > max_flush:
            flush_interval = max_flush

        if not force:
            count = _IRIS_META_COUNTERS.get(key, 0) + 1
            _IRIS_META_COUNTERS[key] = count
            remainder = count % flush_interval
            if remainder != 0:
                remaining = flush_interval - remainder
                last_logged = _IRIS_META_LAST_DEFER_COUNTS.get(key, -1)
                should_log = (
                    count == 1
                    or remaining == 1
                    or (count - last_logged) >= flush_interval
                )
                if should_log:
                    _IRIS_META_LAST_DEFER_COUNTS[key] = count
                    _jit_meta_log(
                        f"persist deferred: count={count} interval={flush_interval} in={remaining} key={key[:12]}"
                    )
                return
        else:
            _jit_meta_log(f"persist forced: key={key[:12]}")
    try:
        rows = _get_quantum_profile(func)
    except Exception as exc:
        _jit_meta_log(f"persist skipped: profile fetch failed key={key[:12]} err={exc}")
        return
    if not rows:
        _jit_meta_log(f"persist skipped: empty profile rows key={key[:12]}")
        return
    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            continue
        try:
            normalized_rows.append(
                [int(row[0]), float(row[1]), int(row[2]), int(row[3])]
            )
        except Exception:
            continue
    if not normalized_rows:
        _jit_meta_log(f"persist skipped: normalized profile empty key={key[:12]}")
        return

    try:
        doc = _read_metadata(path)
        now_ns = time.time_ns()
        entries = doc.setdefault("entries", {})
        if not isinstance(entries, dict):
            entries = {}
            doc["entries"] = entries

        existing = entries.get(key)
        if isinstance(existing, dict):
            existing_rows = existing.get("profile")
            normalized_rows = _merge_profile_rows(existing_rows, normalized_rows)
            existing_return_type = existing.get("return_type")
            existing_arg_count = existing.get("arg_count")
            existing_updated_raw = existing.get("updated_ns", 0)
            try:
                existing_updated_ns = int(existing_updated_raw)
            except Exception:
                existing_updated_ns = 0

            refresh_ns = _IRIS_META_REFRESH_NS if _IRIS_META_REFRESH_NS > 0 else 0
            existing_rows_normalized = _normalize_profile_rows(existing_rows)
            existing_shape = _profile_variant_shape(existing_rows_normalized)
            current_shape = _profile_variant_shape(normalized_rows)
            unchanged = (
                existing_shape == current_shape
                and existing_return_type == (return_type or "float")
                and int(existing_arg_count) == len(arg_names)
            )
            still_fresh = refresh_ns == 0 or (now_ns - existing_updated_ns) < refresh_ns
            if unchanged and still_fresh:
                _jit_meta_log(
                    f"persist skipped: unchanged-shape key={key[:12]} age_ns={max(0, now_ns - existing_updated_ns)}"
                )
                return

        row_sig = hashlib.sha256(
            _msgpack.packb(normalized_rows, use_bin_type=True)
        ).hexdigest()
        with _IRIS_META_STATE_LOCK:
            min_flush, max_flush = _normalize_flush_window()
            flush_interval = _IRIS_META_FLUSH_INTERVALS.get(key, min_flush)
            if flush_interval < min_flush:
                flush_interval = min_flush
            if flush_interval > max_flush:
                flush_interval = max_flush

            prev_sig = _IRIS_META_LAST_SIGNATURES.get(key)
            if prev_sig == row_sig:
                _IRIS_META_FLUSH_INTERVALS[key] = min(max_flush, flush_interval * 2)
            else:
                _IRIS_META_FLUSH_INTERVALS[key] = max(
                    min_flush, flush_interval // 2 if flush_interval > 1 else 1
                )
            _IRIS_META_LAST_SIGNATURES[key] = row_sig

        entries[key] = {
            "updated_ns": now_ns,
            "return_type": return_type or "float",
            "arg_count": len(arg_names),
            "profile": normalized_rows,
        }
        doc["entries"] = _prune_metadata_entries(entries, now_ns)
        _write_metadata(path, doc)
        _jit_meta_log(
            f"persist ok: rows={len(normalized_rows)} entries={len(doc.get('entries', {}))} key={key[:12]}"
        )
    except Exception as exc:
        _jit_meta_log(f"persist failed: key={key[:12]} path={path} err={exc}")


def _flush_pending_quantum_metadata() -> None:
    with _IRIS_META_STATE_LOCK:
        pending_items = list(_IRIS_META_PENDING.items())
        _IRIS_META_PENDING.clear()
    if not pending_items:
        return

    _jit_meta_log(f"flush pending start: entries={len(pending_items)}")
    forced = 0
    failed = 0
    written_paths: set[str] = set()

    for key, (fn, src, arg_names, return_type) in pending_items:
        forced += 1
        try:
            _maybe_persist_quantum_metadata(fn, src, arg_names, return_type, force=True)
        except Exception as exc:
            failed += 1
            _jit_meta_log(f"flush pending failed: err={exc}")
            continue

        try:
            path = _metadata_cache_path(fn)
            if path is not None and os.path.exists(path):
                written_paths.add(path)
        except Exception:
            pass

        _jit_meta_log(f"flush pending key done: key={key[:12]}")

    _jit_meta_log(
        f"flush pending done: forced={forced} failed={failed} files={len(written_paths)}"
    )


atexit.register(_flush_pending_quantum_metadata)


def set_jit_logging(
    enabled: Optional[bool] = None, env_var: Optional[str] = None
) -> bool:
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


def set_quantum_speculation(
    enabled: Optional[bool] = None, env_var: Optional[str] = None
) -> bool:
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


def set_quantum_speculation_threshold(
    threshold_ns: Optional[int] = None, env_var: Optional[str] = None
) -> int:
    """Configure how slow a function must be before quantum speculation kicks in.

    Parameters
    ----------
    threshold_ns:
        If provided, sets the minimum duration (in nanoseconds) required before
        the runtime starts doing multi-variant speculation.
        Set to 0 to always speculate.
    env_var:
        Optional env var name to use (defaults to ``IRIS_JIT_QUANTUM_SPECULATION_NS``).
    """
    if _configure_quantum_speculation_threshold is None:
        return 0
    return int(_configure_quantum_speculation_threshold(threshold_ns, env_var))


def get_quantum_speculation_threshold() -> int:
    """Return the current quantum speculation threshold in nanoseconds."""
    if _get_quantum_speculation_threshold is None:
        return 0
    return int(_get_quantum_speculation_threshold())


def set_quantum_log_threshold(
    threshold_ns: Optional[int] = None, env_var: Optional[str] = None
) -> int:
    """Configure how long a JIT quantum execution must run before emitting a log.

    Parameters
    ----------
    threshold_ns:
        If provided, sets the minimum duration (in nanoseconds) needed before
        quantum decision logging emits.  Set to 0 to log all quantum invocations.
    env_var:
        Optional env var name to use (defaults to ``IRIS_JIT_QUANTUM_LOG_NS``).
    """
    if _configure_quantum_log_threshold is None:
        return 0
    return int(_configure_quantum_log_threshold(threshold_ns, env_var))


def get_quantum_log_threshold() -> int:
    """Return the current quantum log threshold in nanoseconds."""
    if _get_quantum_log_threshold is None:
        return 0
    return int(_get_quantum_log_threshold())


def set_quantum_compile_budget(
    budget_ns: Optional[int] = None,
    window_ns: Optional[int] = None,
    budget_env_var: Optional[str] = None,
    window_env_var: Optional[str] = None,
) -> tuple[int, int]:
    """Configure quantum compile-time budget and accounting window.

    Returns
    -------
    tuple[int, int]
        ``(budget_ns, window_ns)`` currently in effect.
    """
    if _configure_quantum_compile_budget is None:
        return (0, 0)
    budget, window = _configure_quantum_compile_budget(
        budget_ns,
        window_ns,
        budget_env_var,
        window_env_var,
    )
    return int(budget), int(window)


def get_quantum_compile_budget() -> tuple[int, int]:
    """Return the current quantum compile budget tuple ``(budget_ns, window_ns)``."""
    if _get_quantum_compile_budget is None:
        return (0, 0)
    budget, window = _get_quantum_compile_budget()
    return int(budget), int(window)


def set_quantum_cooldown(
    base_ns: Optional[int] = None,
    max_ns: Optional[int] = None,
    base_env_var: Optional[str] = None,
    max_env_var: Optional[str] = None,
) -> tuple[int, int]:
    """Configure quantum compile cooldown backoff bounds.

    Returns
    -------
    tuple[int, int]
        ``(base_ns, max_ns)`` currently in effect.
    """
    if _configure_quantum_cooldown is None:
        return (0, 0)
    base, maxv = _configure_quantum_cooldown(
        base_ns,
        max_ns,
        base_env_var,
        max_env_var,
    )
    return int(base), int(maxv)


def get_quantum_cooldown() -> tuple[int, int]:
    """Return the current quantum cooldown tuple ``(base_ns, max_ns)``."""
    if _get_quantum_cooldown is None:
        return (0, 0)
    base, maxv = _get_quantum_cooldown()
    return int(base), int(maxv)


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
            template = _extract_inline_template_from_callable(
                target, self.fn_globals, self.inline_cache
            )
            self.inline_cache[name] = template

        if template is None:
            return None

        params, body_expr = template
        if len(params) != len(node.args):
            return None

        inlined = copy.deepcopy(body_expr)
        inlined = _JitExprNormalizer(
            self.fn_globals, self.inline_cache, self.active_inline | {name}
        ).visit(inlined)

        for param, arg in reversed(list(zip(params, node.args))):
            inlined = ast.Call(
                func=ast.Name(id="let_bind", ctx=ast.Load()),
                args=[ast.Name(id=param, ctx=ast.Load()), copy.deepcopy(arg), inlined],
                keywords=[],
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


def _lower_to_expr(
    stmts: list[ast.stmt],
    rest_expr: ast.AST,
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
    active_inline: Optional[set[str]] = None,
) -> Optional[ast.AST]:
    if not stmts:
        return rest_expr

    stmt = stmts[0]
    next_stmts = stmts[1:]
    normalizer = _JitExprNormalizer(fn_globals, inline_cache, active_inline)

    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        target = stmt.targets[0].id
        val = copy.deepcopy(stmt.value)
        val = normalizer.visit(val)
        inner = _lower_to_expr(
            next_stmts, rest_expr, fn_globals, inline_cache, active_inline
        )
        if inner is None:
            return None
        return ast.Call(
            func=ast.Name(id="let_bind", ctx=ast.Load()),
            args=[ast.Name(id=target, ctx=ast.Load()), val, inner],
            keywords=[],
        )

    if (
        isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.value is not None
    ):
        target = stmt.target.id
        val = copy.deepcopy(stmt.value)
        val = normalizer.visit(val)
        inner = _lower_to_expr(
            next_stmts, rest_expr, fn_globals, inline_cache, active_inline
        )
        if inner is None:
            return None
        return ast.Call(
            func=ast.Name(id="let_bind", ctx=ast.Load()),
            args=[ast.Name(id=target, ctx=ast.Load()), val, inner],
            keywords=[],
        )

    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        target = stmt.target.id
        right_val = copy.deepcopy(stmt.value)
        right_val = normalizer.visit(right_val)
        new_val = ast.BinOp(
            left=ast.Name(id=target, ctx=ast.Load()),
            op=copy.deepcopy(stmt.op),
            right=right_val,
        )
        inner = _lower_to_expr(
            next_stmts, rest_expr, fn_globals, inline_cache, active_inline
        )
        if inner is None:
            return None
        return ast.Call(
            func=ast.Name(id="let_bind", ctx=ast.Load()),
            args=[ast.Name(id=target, ctx=ast.Load()), new_val, inner],
            keywords=[],
        )

    if isinstance(stmt, ast.If):
        cond = copy.deepcopy(stmt.test)
        cond = normalizer.visit(cond)
        then_expr = _lower_to_expr(
            list(stmt.body) + next_stmts,
            copy.deepcopy(rest_expr),
            fn_globals,
            inline_cache,
            active_inline,
        )
        orelse_stmts = list(stmt.orelse) + next_stmts if stmt.orelse else next_stmts
        else_expr = _lower_to_expr(
            orelse_stmts,
            copy.deepcopy(rest_expr),
            fn_globals,
            inline_cache,
            active_inline,
        )
        if then_expr is None or else_expr is None:
            return None
        return ast.IfExp(test=cond, body=then_expr, orelse=else_expr)

    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return rest_expr
        val = copy.deepcopy(stmt.value)
        return normalizer.visit(val)

    if isinstance(stmt, ast.Pass) or isinstance(stmt, ast.Expr):
        return _lower_to_expr(
            next_stmts, rest_expr, fn_globals, inline_cache, active_inline
        )

    return None


def _extract_return_expr_plan(
    fn_node: ast.FunctionDef,
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[tuple[str, list[str]]]:
    stmts = _strip_docstring(list(fn_node.body))
    if (
        len(stmts) == 1
        and isinstance(stmts[0], ast.Return)
        and stmts[0].value is not None
    ):
        return ast.unparse(
            _subst_expr(stmts[0].value, {}, fn_globals, inline_cache)
        ), []
    return None


def _contains_unsupported_ast(node: ast.AST) -> bool:
    """Check if node contains constructs unsupported by JIT parser."""
    for child in ast.walk(node):
        if isinstance(
            child, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
        ):
            return True
    return False


class _RuntimeLetBindTransformer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "let_bind"
            and len(node.args) == 3
            and isinstance(node.args[0], ast.Name)
        ):
            bind_name = node.args[0].id
            val_expr = node.args[1]
            body_expr = node.args[2]
            assign_expr = ast.NamedExpr(
                target=ast.Name(id=bind_name, ctx=ast.Store()),
                value=val_expr,
            )
            seq_expr = ast.Tuple(elts=[assign_expr, body_expr], ctx=ast.Load())
            return ast.Subscript(
                value=seq_expr, slice=ast.Constant(value=1), ctx=ast.Load()
            )
        return node


def _compile_runtime_expr(expr_src: str) -> Optional[Any]:
    try:
        tree = ast.parse(expr_src, mode="eval")
        tree = _RuntimeLetBindTransformer().visit(tree)
        tree = ast.fix_missing_locations(tree)
        return compile(tree, "<iris-jit-step>", "eval")
    except Exception:
        return None


def _eval_runtime_expr(
    expr_src: str,
    globals_ns: dict[str, Any],
    locals_ns: dict[str, Any],
    compiled: Optional[Any],
) -> Any:
    if compiled is not None:
        return eval(compiled, globals_ns, locals_ns)
    return eval(expr_src, globals_ns, locals_ns)


def _extract_inlined_expr_plan(
    fn_node: ast.FunctionDef,
    fn_globals: Optional[dict[str, Any]] = None,
    inline_cache: Optional[dict[str, Optional[tuple[list[str], ast.AST]]]] = None,
) -> Optional[str]:
    stmts = _strip_docstring(list(fn_node.body))
    ret_expr = _lower_to_expr(stmts, ast.Constant(value=0.0), fn_globals, inline_cache)
    if ret_expr is None:
        return None
    # Reject if lowered expression contains constructs JIT parser doesn't support
    if _contains_unsupported_ast(ret_expr):
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
    seed_src = ast.unparse(
        _subst_expr(state_assign.value, {}, fn_globals, inline_cache)
    )

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

    pre_append_body = loop_body[:-1]
    allowed_stmt_types = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass)
    if any(not isinstance(stmt, allowed_stmt_types) for stmt in pre_append_body):
        return None
    if any(isinstance(stmt, ast.If) for stmt in pre_append_body):
        return None

    state_expr: ast.AST = ast.Name(id=state_var, ctx=ast.Load())
    step_expr = _lower_to_expr(pre_append_body, state_expr, fn_globals, inline_cache)
    if step_expr is None:
        return None

    step_src = ast.unparse(step_expr)
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
    step_expr = _lower_to_expr(body[:-1], state_expr, fn_globals, inline_cache)
    if step_expr is None:
        return None

    return {
        "count_arg": count_arg,
        "iter_var": iter_var,
        "state_var": state_var,
        "seed_src": ast.unparse(_subst_expr(seed_expr, {}, fn_globals, inline_cache)),
        "step_src": ast.unparse(step_expr),
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
    step_expr = _lower_to_expr(body, state_expr, fn_globals, inline_cache)
    if step_expr is None:
        return None

    return {
        "count_arg": count_arg,
        "iter_var": iter_var,
        "state_var": state_var,
        "seed_src": ast.unparse(_subst_expr(seed_expr, {}, fn_globals, inline_cache)),
        "step_src": ast.unparse(step_expr),
        "step_args": [state_var, iter_var],
    }


def _is_vector_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__getitem__")


def _vectorized_python_fallback(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
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
        iter_args = [
            arg[i] if idx in vector_positions else arg for idx, arg in enumerate(args)
        ]
        out.append(func(*iter_args, **kwargs))

    try:
        return _array.array("d", (float(v) for v in out))
    except Exception:
        return out


def offload(
    strategy: str = "actor", return_type: Optional[str] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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
                        expr_plan = _extract_return_expr_plan(
                            node, jit_eval_globals, inline_cache
                        )
                        if expr_plan is not None:
                            src, _ = expr_plan
                        else:
                            inlined = _extract_inlined_expr_plan(
                                node, jit_eval_globals, inline_cache
                            )
                            if inlined is not None:
                                src = inlined
                            else:
                                src = None
                                if arg_names is not None:
                                    loop_plan = _extract_stateful_loop_plan(
                                        node, arg_names, jit_eval_globals, inline_cache
                                    )
                                    if loop_plan is None:
                                        scalar_while_plan = _extract_scalar_while_plan(
                                            node,
                                            arg_names,
                                            jit_eval_globals,
                                            inline_cache,
                                        )
                                        if scalar_while_plan is None:
                                            scalar_for_plan = _extract_scalar_for_plan(
                                                node,
                                                arg_names,
                                                jit_eval_globals,
                                                inline_cache,
                                            )
                                aggressive_src = _extract_last_return_expr(
                                    node, jit_eval_globals, inline_cache
                                )
                        break
            except Exception:
                pass

        effective_src: Optional[str] = src if src is not None else aggressive_src

        preseeded_quantum = False
        if strategy == "jit":
            preseeded_quantum = _seed_quantum_from_metadata(
                func, effective_src, arg_names, return_type
            )

        if register_offload is not None:
            try:
                register_offload(func, strategy, return_type, effective_src, arg_names)
                if strategy == "jit" and not preseeded_quantum:
                    _seed_quantum_from_metadata(
                        func, effective_src, arg_names, return_type
                    )
            except Exception as e:  # pragma: no cover - defensive
                warnings.warn(f"offload registration failed: {e}")

        # Wrap with runtime call depending on strategy
        if strategy == "actor" and offload_call is not None:

            @functools.wraps(func)
            def actor_wrapper(*args: Any, **kwargs: Any) -> Any:
                return offload_call(func, args, kwargs)

            return actor_wrapper

        elif strategy == "jit" and call_jit is not None:
            if (
                src is None
                and loop_plan is None
                and scalar_while_plan is None
                and scalar_for_plan is None
            ):
                if aggressive_src is not None and arg_names is not None:

                    @functools.wraps(func)
                    def aggressive_vector_wrapper(*args: Any, **kwargs: Any) -> Any:
                        has_vector = any(_is_vector_like(a) for a in args)
                        if has_vector:
                            try:
                                res = call_jit(func, args, kwargs)
                                _maybe_persist_quantum_metadata(
                                    func, aggressive_src, arg_names, return_type
                                )
                                return res
                            except RuntimeError as e:
                                msg = str(e)
                                if (
                                    "no JIT entry" in msg
                                    or "failed to compile" in msg
                                    or "jit panic" in msg
                                    or "wrong argument count" in msg
                                ):
                                    return _vectorized_python_fallback(
                                        func, args, kwargs
                                    )
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
                        if (
                            reduction_mode is not None
                            and hasattr(res, "__iter__")
                            and not isinstance(res, (float, int))
                        ):
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
                    _maybe_persist_quantum_metadata(func, src, arg_names, return_type)
                    return res

                return jit_wrapper

            if (
                loop_plan is not None
                and register_offload is not None
                and sig is not None
            ):
                step_src = loop_plan["step_src"]
                step_args = loop_plan["step_args"]
                seed_src = loop_plan["seed_src"]
                count_arg = loop_plan["count_arg"]
                iter_var = loop_plan["iter_var"]
                step_code = _compile_runtime_expr(step_src)
                seed_code = _compile_runtime_expr(seed_src)

                def _iris_step(x: float, i: float) -> float:
                    namespace = {step_args[0]: x, step_args[1]: i}
                    return float(
                        _eval_runtime_expr(
                            step_src, jit_eval_globals, namespace, step_code
                        )
                    )

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
                        state = float(
                            _eval_runtime_expr(
                                seed_src, jit_eval_globals, local_seed_ns, seed_code
                            )
                        )
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
                                local_ns = {
                                    step_args[0]: state,
                                    iter_var: i,
                                    step_args[1]: iter_val,
                                }
                                state = float(
                                    _eval_runtime_expr(
                                        step_src, jit_eval_globals, local_ns, step_code
                                    )
                                )
                            else:
                                raise
                        out.append(state)
                    return out

                return loop_jit_wrapper

            if (
                scalar_while_plan is not None
                and register_offload is not None
                and sig is not None
            ):
                step_src = scalar_while_plan["step_src"]
                step_args = scalar_while_plan["step_args"]
                count_arg = scalar_while_plan["count_arg"]
                iter_var = scalar_while_plan["iter_var"]
                seed_src = scalar_while_plan["seed_src"]
                step_code = _compile_runtime_expr(step_src)
                seed_code = _compile_runtime_expr(seed_src)

                def _iris_step(x: float, i: float) -> float:
                    namespace = {step_args[0]: x, step_args[1]: i}
                    return float(
                        _eval_runtime_expr(
                            step_src, jit_eval_globals, namespace, step_code
                        )
                    )

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
                        state = float(
                            _eval_runtime_expr(
                                seed_src, jit_eval_globals, local_seed_ns, seed_code
                            )
                        )
                        count = int(bound.arguments[count_arg])
                    except Exception:
                        return func(*args, **kwargs)

                    if count <= 0:
                        return state

                    if call_jit_step_loop_f64 is not None:
                        try:
                            return float(
                                call_jit_step_loop_f64(_iris_step, state, count)
                            )
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
                                local_ns = {
                                    step_args[0]: state,
                                    iter_var: i,
                                    step_args[1]: iter_val,
                                }
                                state = float(
                                    _eval_runtime_expr(
                                        step_src, jit_eval_globals, local_ns, step_code
                                    )
                                )
                            else:
                                raise
                    return state

                return while_jit_wrapper

            if (
                scalar_for_plan is not None
                and register_offload is not None
                and sig is not None
            ):
                step_src = scalar_for_plan["step_src"]
                step_args = scalar_for_plan["step_args"]
                count_arg = scalar_for_plan["count_arg"]
                iter_var = scalar_for_plan["iter_var"]
                seed_src = scalar_for_plan["seed_src"]
                step_code = _compile_runtime_expr(step_src)
                seed_code = _compile_runtime_expr(seed_src)

                def _iris_step(x: float, i: float) -> float:
                    namespace = {step_args[0]: x, step_args[1]: i}
                    return float(
                        _eval_runtime_expr(
                            step_src, jit_eval_globals, namespace, step_code
                        )
                    )

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
                        state = float(
                            _eval_runtime_expr(
                                seed_src, jit_eval_globals, local_seed_ns, seed_code
                            )
                        )
                        count = int(bound.arguments[count_arg])
                    except Exception:
                        return func(*args, **kwargs)

                    if count <= 0:
                        return state

                    if call_jit_step_loop_f64 is not None:
                        try:
                            return float(
                                call_jit_step_loop_f64(_iris_step, state, count)
                            )
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
                                local_ns = {
                                    step_args[0]: state,
                                    iter_var: i,
                                    step_args[1]: iter_val,
                                }
                                state = float(
                                    _eval_runtime_expr(
                                        step_src, jit_eval_globals, local_ns, step_code
                                    )
                                )
                            else:
                                raise
                    return state

                return for_jit_wrapper

            return func

        return func

    return decorator
