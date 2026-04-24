# Usage Guide — Complete API Reference

This document lists the public runtime APIs exposed by the Python bindings and the common Node.js equivalents where available. It is intended to be a comprehensive reference for everyday usage: spawning actors, messaging, supervision, timers, JIT/offload helpers, registry, and network helpers.

Notes:
- All Python examples assume `from iris import Runtime` or `import iris` as appropriate.
- Node.js examples use the shipped `NodeRuntime` wrapper where shown.

---

## Runtime: creating and basic flow

Python:

```python
from iris import Runtime
rt = Runtime()
```

Node.js (conceptual):

```javascript
const { NodeRuntime } = require('./index.js');
const rt = new NodeRuntime();
```

---

## Spawning actors

- `spawn(handler, budget: int = 100, release_gil: bool = False) -> int`
  - Spawn a push-based actor (green-thread). `handler(message)` is called for each incoming message.

```python
def handler(msg):
    print('got', msg)

pid = rt.spawn(handler, budget=50)
```

- `spawn_py_handler_bounded(handler, budget: int, capacity: int, release_gil: bool = False) -> int`
  - Spawn a Python handler with a bounded mailbox capacity.

- `spawn_with_mailbox(handler, budget: int = 100) -> int`
  - Spawn a pull-based actor running in a dedicated OS thread. `handler(mailbox)` should call `mailbox.recv()`.

- `spawn_child(parent: int, handler, budget: int = 100, release_gil: bool = False) -> int`
  - Spawn a child actor that is automatically stopped when `parent` exits.

- `spawn_child_pool(parent, handler, workers: int, budget: int = 100, release_gil: bool = False) -> list[int]`
  - Spawn a persistent pool of child actors tied to `parent` (returns worker PIDs).

- `spawn_child_with_mailbox(parent, handler, budget: int = 100) -> int`
  - Child variant of `spawn_with_mailbox`.

- `spawn_virtual(handler, budget: int = 100, idle_timeout_ms: Optional[int] = None) -> int`
  - Reserve a PID and lazily activate the actor on first message.

- `spawn_with_path_observed(budget: int, path: str) -> int`
  - Spawn an observed handler and register it under `path` (useful for monitoring/testing).

Example — push vs pull:

```python
# push actor
rt.spawn(lambda m: print('push', m))

# pull actor
def mailbox_worker(mailbox):
    while True:
        msg = mailbox.recv()
        if not msg:
            break
        print('pull', msg)

rt.spawn_with_mailbox(mailbox_worker)
```

---

## Sending messages

- `send(pid: int, data: bytes-like) -> bool` — send user payload (`bytes`, `bytearray`, `memoryview`) to a PID.
- `call(pid: int, data: bytes-like, timeout: Optional[float] = None) -> Awaitable[bytes]` — send a request and await a response (async).
- `send_many(pid: int, payloads: Iterable[bytes-like]) -> int` — batch-send payloads and return accepted count.
- `send_named(name: str, data: bytes-like) -> bool` — resolve and send by registered name.
- `send_buffer(pid: int, buffer_id: int) -> bool` — zero-copy send via buffer ID (use `allocate_buffer`).
- `send_after(pid: int, delay_ms: int, data: bytes-like) -> int` — schedule one-shot message; returns timer id.
- `send_interval(pid: int, interval_ms: int, data: bytes-like) -> int` — schedule repeating message; returns timer id.
- `cancel_timer(timer_id: int) -> bool` — cancel timer/interval.

Example:

```python
# Async request-reply
response = await rt.call(pid, b"ping", timeout=2.0)
print("response", response)

timer = rt.send_after(pid, 200, b'tick')
```

---

## Observability & Introspection

Iris provides deep visibility into the live runtime state via the following APIs:

- `list_actors() -> list[int]` — list all active local actor PIDs.
- `actor_info(pid: int) -> Optional[dict[str, str]]` — get metadata, health, and status for a specific actor.
- `set_actor_health(pid: int, status: str)` — manually update actor health (e.g. `'busy'`, `'degraded'`, `'ready'`).
- `get_metrics() -> dict[str, int]` — retrieve system-wide metrics (actor count, message throughput).

Example:

```python
for pid in rt.list_actors():
    info = rt.actor_info(pid)
    print(f"Actor {pid}: {info['health']}")

metrics = rt.get_metrics()
print(f"Throughput: {metrics['messages_received']} msgs")
```

---

## Resilience & Load Shedding

To prevent cascade failures under extreme load, Iris can shed traffic by rejecting new actors and messages.

- `set_system_capacity(cap: int)` — set the maximum number of concurrent actors allowed.
- `set_load_shedding(enabled: bool)` — enable or disable the load shedding subsystem.
- `is_load_shedding_active() -> bool` — check if the system is currently rejecting work due to capacity.

Example:

```python
rt.set_system_capacity(10_000)
rt.set_load_shedding(True)

if rt.is_load_shedding_active():
    print("Warning: System at capacity, load shedding active.")
```

---

## Mailbox helpers and introspection

- `mailbox_size(pid: int) -> Optional[int]` — number of queued user messages, or `None` if PID unknown.
- `mailbox_backpressure(pid: int) -> Optional[str]` — inferred backpressure level for the mailbox: `"NORMAL"`, `"HIGH"`, or `"CRITICAL"`.
- `send_with_backpressure(pid: int, data: bytes) -> Tuple[bool, str]` — send with immediate pressure result (tight loop use case; returns send success and current `BackpressureLevel`).
- `send_user_with_backpressure(pid: int, buffer_id: int) -> Tuple[bool, str]` — same as above for zero-copy buffer path.
- `selective_recv(pid, matcher: Callable, timeout: Optional[float] = None) -> Awaitable[Optional[bytes | PySystemMessage]]` — await a message that satisfies `matcher` (for observed actors).
- `selective_recv_blocking(pid, matcher, timeout=None)` — blocking convenience wrapper.

Example:

```python
size = rt.mailbox_size(pid)
print('size=', size)
```

---

## Registration & name registry

- `register(name: str, pid: int)` — register a local name.
- `unregister(name: str)` — remove registration.
- `resolve(name: str) -> Optional[int]` — local resolve.
- `whereis(name: str) -> Optional[int]` — alias for `resolve`.

Path-based registry:
- `register_path(path: str, pid: int)` — register under hierarchical path (e.g., `/svc/payments/one`).
- `unregister_path(path: str)` — remove a path registration.
- `whereis_path(path: str) -> Optional[int]` — exact path resolution.
- `list_children(prefix: str)` — list all descendant registrations under `prefix` (returns list of `(path, pid)`).
- `list_children_direct(prefix: str)` — list immediate children.

Example:

```python
rt.register('auth', pid)
assert rt.whereis('auth') == pid

rt.register_path('/svc/test/one', pid)
print(rt.list_children('/svc/test'))
```

---

## Path-scoped supervisors & supervision helpers

- `create_path_supervisor(path: str)` — create path-scoped supervisor.
- `remove_path_supervisor(path: str)` — remove it.
- `path_supervisor_watch(path: str, pid: int)` — register pid with path supervisor.
- `path_supervisor_children(path: str)` — list children supervised by path supervisor.
- `path_supervise_with_factory(path: str, pid: int, factory, strategy: str)` — attach a Python factory to restart actors (`restartone` / `restartall`).
- `children_count()` / `child_pids()` — global supervised children count and list.

Example:

```python
rt.create_path_supervisor('/svc/test')
rt.path_supervisor_watch('/svc/test', pid)
print(rt.path_supervisor_children('/svc/test'))
```

---

## Hot-swap, behavior versions, and exits

- `hot_swap(pid: int, new_handler)` — atomically swap actor logic.
- `behavior_version(pid: int) -> int` — current behavior version.
- `rollback_behavior(pid: int, steps: int = 1) -> int` — roll back behavior by `steps` versions; returns new version.
- `get_messages(observer_pid)` — (guest API) fetch observed messages (see language bindings).

Exit reasons and system messages are delivered as structured `PySystemMessage` objects with fields such as `type_name`, `from_pid`, `target_pid`, `reason`, and optional `metadata`.

Example:

```python
def a(msg):
    print('got', msg)

pid = rt.spawn(a)
rt.hot_swap(pid, lambda m: print('upgraded', m))
```

---

## Remote / network helpers

- `listen(addr: str)` — start TCP server to accept remote messages and resolves.
- `resolve_remote(addr: str, name: str) -> Optional[int]` — blocking remote resolve (returns local proxy PID).
- `resolve_remote_py(addr: str, name: str) -> Awaitable[Optional[int]]` — async remote resolve.
- `is_node_up(addr: str) -> bool` — quick probe for remote node reachability.
- `send_remote(addr: str, pid: int, data: bytes-like)` — send bytes-like payload to a remote PID (creates/reuses proxy internally).
- `monitor_remote(addr: str, pid: int)` — watch remote PID; triggers local supervisor on failure.

Example:

```python
proxy = await rt.resolve_remote_py('192.168.1.5:9000', 'auth_worker')
if proxy:
    rt.send(proxy, b'login')
```

---

## Process & thread / GIL controls

- `set_release_gil_limits(max_threads: int, pool_size: int)` — configure dedicated GIL threads and shared pool.
- `set_release_gil_strict(strict: bool)` — when `True`, spawning with `release_gil=True` returns an error if limits exceeded.

Example:

```python
rt.set_release_gil_limits(8, 16)
rt.set_release_gil_strict(True)
```

---

## Vortex controls (Python Runtime, experimental)

When Iris is built with the `vortex` feature, `PyRuntime` exposes automatic ghost-arbitration controls and telemetry.

- `vortex_set_auto_ghost_policy(policy: str) -> bool`
  - Accepted values: `FirstSafePointWins`, `PreferPrimary`.
- `vortex_get_auto_ghost_policy() -> Optional[str]`
- `vortex_get_auto_resolution_counts() -> Tuple[int, int]`
  - Returns `(primary_wins, ghost_wins)`.
- `vortex_get_auto_replay_count() -> int`
- `vortex_reset_auto_telemetry() -> None`

Example:

```python
rt = Runtime()
rt.vortex_reset_auto_telemetry()
rt.vortex_set_auto_ghost_policy("PreferPrimary")

primary_wins, ghost_wins = rt.vortex_get_auto_resolution_counts()
replayed = rt.vortex_get_auto_replay_count()
print(primary_wins, ghost_wins, replayed)
```

---

## Lifecycle helpers

- `stop(pid: int)` — stop an actor and close mailbox.
- `join(pid: int)` — block until actor exits (useful in tests).
- `is_alive(pid: int) -> bool` — check actor liveness.

Example:

```python
rt.stop(pid)
rt.join(pid)
assert not rt.is_alive(pid)
```

---

## JIT / Offload helpers (Python)

Imported into package namespace:
- `offload` decorator: `@iris.offload(strategy='jit' | 'actor', return_type=...)` — intercepts a Python function to either route to Rust actor pool or compile to native where supported.
- `set_jit_logging`, `get_jit_logging`, `set_quantum_speculation`, `get_quantum_speculation` — runtime controls for the JIT subsystem.

Example:

```python
@iris.offload(strategy='jit', return_type='float')
def vec_mag(x: float, y: float, z: float) -> float:
    return (x*x + y*y + z*z) ** 0.5

print(vec_mag(1.0, 2.0, 3.0))
```

---

## Bounded mailboxes & overflow policies

Use `spawn_py_handler_bounded` to set capacity, then configure overflow policy:

- `set_overflow_policy(pid, policy, target=None)` — `policy` is one of: `dropnew`, `dropold`, `redirect`, `spill`, `block`.
  - `redirect` / `spill` require `target` PID.

Example (Python):

```python
primary = rt.spawn_py_handler_bounded(lambda m: print(m), 100, capacity=1)
fallback = rt.spawn(lambda m: print('fallback', m), 100)
rt.set_overflow_policy(primary, 'redirect', fallback)
```

---

## Examples: small end-to-end

Python — spawn, register, send, timers:

```python
rt = Runtime()

def hello(msg):
    print('hello', msg)

pid = rt.spawn(hello)
rt.register('greeter', pid)
rt.send_named('greeter', b'world')

tid = rt.send_interval(pid, 1000, b'ping')
# later
rt.cancel_timer(tid)
```

Node.js — conceptual:

```javascript
const rt = new NodeRuntime();
const pid = rt.spawn((msg) => console.log('got', msg));
rt.send(pid, Buffer.from('hello'));
```

---

## Where to look next

- Architecture details: [docs/architecture.md](architecture.md)
- JIT internals and controls: [docs/jit.md](jit.md)
- Distributed mesh & discovery: [docs/distributed.md](distributed.md)