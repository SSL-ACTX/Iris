# myrmidon/__init__.py
try:
    from .myrmidon import PyRuntime, version, allocate_buffer
except ImportError:
    from myrmidon import PyRuntime, version, allocate_buffer

class Runtime:
    def __init__(self):
        self._inner = PyRuntime()

    def spawn(self, handler, budget=100):
        return self._inner.spawn_py_handler(handler, budget)

    def send(self, pid, data):
        """Send data to a specific PID."""
        return self._inner.send(pid, data)

    def send_named(self, name: str, data: bytes):
        """Send data to an actor by its registered name."""
        pid = self.resolve(name)
        if pid:
            return self._inner.send(pid, data)
        return False

    def register(self, name: str, pid: int):
        """Assign a human-readable name to a PID."""
        self._inner.register(name, pid)

    def resolve(self, name: str):
        """Look up the PID associated with a name."""
        return self._inner.resolve(name)

    def resolve_remote(self, addr: str, name: str):
        """Phase 7: Query a remote node for a PID by name."""
        return self._inner.resolve_remote(addr, name)

    def send_buffer(self, pid, buffer_id):
        """Zero-Copy send via Buffer ID."""
        return self._inner.send_buffer(pid, buffer_id)

    def hot_swap(self, pid, new_handler):
        """Update actor logic at runtime."""
        self._inner.hot_swap(pid, new_handler)

    def listen(self, addr: str):
        """Start TCP server for remote messages and name resolution."""
        self._inner.listen(addr)

    def send_remote(self, addr: str, pid: int, data: bytes):
        """Send data to a PID on a remote node."""
        self._inner.send_remote(addr, pid, data)

    def monitor_remote(self, addr: str, pid: int):
        """Watch a remote PID; triggers local supervisor on failure."""
        self._inner.monitor_remote(addr, pid)

    def stop(self, pid):
        self._inner.stop(pid)

    def join(self, pid):
        self._inner.join(pid)

__all__ = ["Runtime", "version", "allocate_buffer"]
