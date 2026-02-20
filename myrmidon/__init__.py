# myrmidon/__init__.py
import asyncio
from typing import Optional, Callable, Union, Awaitable

try:
    from .myrmidon import PyRuntime, PySystemMessage, version, allocate_buffer, PyMailbox
except ImportError:
    from myrmidon import PyRuntime, PySystemMessage, version, allocate_buffer, PyMailbox

class Runtime:
    def __init__(self):
        self._inner = PyRuntime()

    def spawn(self, handler, budget: int = 100, release_gil: bool = False) -> int:
        """
        Spawn a new push-based actor (Green Thread).

        The handler must be a callable that accepts a single argument (the message).
        The actor will be called repeatedly for each incoming message.

        Args:
            handler: Callable(message)
            budget: Reduction budget for cooperative scheduling.
            release_gil: If True, the runtime will execute the Python callback
                         in a blocking thread that acquires the GIL, avoiding
                         holding the GIL on the async worker. Defaults to False.
        """
        return self._inner.spawn_py_handler(handler, budget, release_gil)

    def spawn_with_mailbox(self, handler, budget: int = 100) -> int:
        """
        Spawn a new pull-based actor in a dedicated OS thread.
        
        The handler must be a callable that accepts a single argument: the `mailbox` object.
        The handler is responsible for running its own loop and calling `mailbox.recv()` (blocking).
        
        Note: This consumes a generic thread pool worker. Bounded by system resources.
        
        Args:
            handler: A callable taking (mailbox: PyMailbox).
            budget: Reduction budget for cooperative scheduling.
            
        Returns:
            int: The PID of the spawned actor.
        """
        return self._inner.spawn_with_mailbox(handler, budget)

    def send(self, pid: int, data: bytes) -> bool:
        """Send data to a specific local PID."""
        return self._inner.send(pid, data)

    def send_named(self, name: str, data: bytes) -> bool:
        """Send data to an actor by its registered name."""
        pid = self.resolve(name)
        if pid:
            return self._inner.send(pid, data)
        return False

    def register(self, name: str, pid: int):
        """Assign a human-readable name to a PID."""
        self._inner.register(name, pid)

    def unregister(self, name: str):
        """Unregister a named PID."""
        self._inner.unregister(name)

    def resolve(self, name: str) -> Optional[int]:
        """Look up the PID associated with a name locally."""
        return self._inner.resolve(name)

    def whereis(self, name: str) -> Optional[int]:
        """Alias for resolve (Erlang style)."""
        return self._inner.whereis(name)

    # --- Path-based registry helpers ---
    def register_path(self, path: str, pid: int):
        """Register an actor under a hierarchical path (e.g. /system/service/one)."""
        self._inner.register_path(path, pid)

    def unregister_path(self, path: str):
        """Remove a hierarchical path registration."""
        self._inner.unregister_path(path)

    def whereis_path(self, path: str) -> Optional[int]:
        """Resolve an exact hierarchical path to a PID."""
        return self._inner.whereis_path(path)

    def list_children(self, prefix: str):
        """List registered entries under a path prefix. Returns list of (path, pid)."""
        return self._inner.list_children(prefix)

    def list_children_direct(self, prefix: str):
        """List only direct children one level below `prefix`. Returns list of (path, pid)."""
        return self._inner.list_children_direct(prefix)

    def watch_path(self, prefix: str):
        """Register (shallow) watch on all direct children under `prefix`."""
        self._inner.watch_path(prefix)

    def children_count(self) -> int:
        """Return number of children registered with the supervisor."""
        return self._inner.children_count()

    def child_pids(self):
        """Return a list of child PIDs currently registered with the supervisor."""
        return self._inner.child_pids()

    # --- Path-scoped supervisors ---
    def create_path_supervisor(self, path: str):
        """Create a path-scoped supervisor object for `path`."""
        self._inner.create_path_supervisor(path)

    def remove_path_supervisor(self, path: str):
        """Remove the path-scoped supervisor for `path` if present."""
        self._inner.remove_path_supervisor(path)

    def path_supervisor_watch(self, path: str, pid: int):
        """Register `pid` with the path-scoped supervisor if it exists, otherwise falls back to global supervise."""
        self._inner.path_supervisor_watch(path, pid)

    def path_supervisor_children(self, path: str):
        """Return child PIDs supervised by the path-scoped supervisor."""
        return self._inner.path_supervisor_children(path)

    def spawn_with_path_observed(self, budget: int, path: str) -> int:
        """Spawn an observed handler and register it under `path`.

        This spawns an observed actor (for monitoring/debugging) and registers
        it at the given hierarchical `path`.
        """
        return self._inner.spawn_with_path_observed(budget, path)

    def path_supervise_with_factory(self, path: str, pid: int, factory, strategy: str):
        """Attach a Python factory to the path-scoped supervisor for restart semantics.

        Args:
            path: The path prefix for the supervisor.
            pid: The current PID to supervise.
            factory: A callable returning a new PID when called.
            strategy: Restart strategy string: 'restartone' or 'restartall'.
        """
        self._inner.path_supervise_with_factory(path, pid, factory, strategy)

    def resolve_remote(self, addr: str, name: str) -> Optional[int]:
        """Query a remote node for a PID by name (Blocking)."""
        return self._inner.resolve_remote(addr, name)

    def resolve_remote_py(self, addr: str, name: str) -> Awaitable[Optional[int]]:
        """Query a remote node for a PID by name (Async/Awaitable)."""
        return self._inner.resolve_remote_py(addr, name)

    def is_node_up(self, addr: str) -> bool:
        """Quick network probe to check if a remote node is reachable."""
        return self._inner.is_node_up(addr)

    def send_buffer(self, pid: int, buffer_id: int) -> bool:
        """Zero-Copy send via Buffer ID."""
        return self._inner.send_buffer(pid, buffer_id)

    def hot_swap(self, pid: int, new_handler):
        """Update actor logic at runtime."""
        self._inner.hot_swap(pid, new_handler)

    def selective_recv(self, pid: int, matcher: Callable, timeout: Optional[float] = None) -> Awaitable[Optional[Union[bytes, PySystemMessage]]]:
        """
        Return an awaitable that resolves when `matcher(msg)` is True.
        
        NOTE: This is for 'observed' actors (debug/monitoring) spawned via 
        `spawn_observed_handler`, NOT for standard mailbox actors.
        
        Standard mailbox actors should use `mailbox.selective_recv()` directly.

        Args:
            pid: The PID of the observed actor.
            matcher: A callable accepting (bytes | PySystemMessage) -> bool.
            timeout: Optional timeout in seconds.

        Returns:
            The matching message, or None if timed out.
        """
        return self._inner.selective_recv_observed_py(pid, matcher, timeout)

    def selective_recv_blocking(self, pid: int, matcher: Callable, timeout: Optional[float] = None) -> Optional[Union[bytes, PySystemMessage]]:
        """
        Blocking convenience wrapper around `selective_recv` for sync code.
        Runs a new asyncio event loop to await the result.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            fut = self.selective_recv(pid, matcher, timeout)
            return loop.run_until_complete(fut)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def listen(self, addr: str):
        """Start TCP server for remote messages and name resolution."""
        self._inner.listen(addr)

    def set_release_gil_limits(self, max_threads: int, pool_size: int):
        """Set programmatic limits for `release_gil` behavior.

        Args:
            max_threads: Max number of dedicated per-actor GIL threads.
            pool_size: Size of the shared GIL worker pool used as fallback.
        """
        self._inner.set_release_gil_limits(max_threads, pool_size)

    def set_release_gil_strict(self, strict: bool):
        """When True, spawning with `release_gil=True` returns an error if limits are exceeded."""
        self._inner.set_release_gil_strict(strict)

    def send_remote(self, addr: str, pid: int, data: bytes):
        """Send data to a PID on a remote node."""
        self._inner.send_remote(addr, pid, data)

    def monitor_remote(self, addr: str, pid: int):
        """Watch a remote PID; triggers local supervisor on failure."""
        self._inner.monitor_remote(addr, pid)

    def stop(self, pid: int):
        """Stop an actor and close its mailbox."""
        self._inner.stop(pid)

    def join(self, pid: int):
        """Block until the specified actor exits."""
        self._inner.join(pid)

__all__ = ["Runtime", "PySystemMessage", "version", "allocate_buffer", "PyMailbox"]