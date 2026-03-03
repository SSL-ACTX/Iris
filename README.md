<div align="center">

![Iris Banner](https://svg-banners.vercel.app/api?type=luminance&text1=Iris%20🌸&width=800&height=200&color=FFB6C1)

![Version](https://img.shields.io/badge/version-0.4.0-blue.svg?style=for-the-badge)
![Language](https://img.shields.io/badge/language-Rust%20%7C%20Python%20%7C%20Node.js-orange.svg?style=for-the-badge&logo=rust)
![License](https://img.shields.io/badge/license-AGPL_3.0-green.svg?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS%20%7C%20Android-lightgrey.svg?style=for-the-badge&logo=linux)

**Hybrid distributed runtime fabric for actors, native compute offload, and cross-language services.**

[Features](#-core-capabilities) • [Architecture](#-technical-deep-dive) • [Installation](#-quick-start) • [Usage](#-usage-examples) • [Distributed Mesh](#-the-distributed-mesh)

</div>

---

## Overview

**Iris** is a hybrid distributed runtime built in Rust with first-class **Python** and **Node.js** bindings.
It combines three execution styles in one system:
- an actor mesh for stateful, message-driven workflows,
- native compute offload/JIT for CPU-heavy hot paths,
- cross-language runtime APIs for service-oriented applications.

So Iris is not only an actor runtime—it is a runtime fabric that lets you mix coordination, messaging, and high-performance compute under a single operational model.

At its core, Iris uses a **cooperative reduction-based scheduler** for fairness and high concurrency, while providing built-in supervision, hot swapping, discovery, and location-transparent messaging across nodes.

> [!NOTE]
> Node.js bindings are still in very early phases and are not yet feature-parity with Python.

## Core Capabilities

Iris is designed as a hybrid platform, not a single-paradigm engine.

### ⚡ Hybrid Execution Model (Push & Pull)
Iris provides two complementary execution patterns:
* **Push Actors (Green Threads):** ultra-lightweight handlers triggered only when messages arrive.
* **Pull Actors (OS Threads):** blocking mailbox workers for synchronous control flow.
    * **Python pull actors:** run on dedicated OS threads and block on `recv()` while releasing the GIL.

### ⚡ Cooperative Reduction Scheduler
Inspired by the BEAM (Erlang VM), Iris uses a cooperative reduction scheduler for fairness.
* **Reduction budgets:** each actor gets a budget, then yields to Tokio via `yield_now()` when it is exhausted.
* **Starvation resistance:** no single high-throughput actor can monopolize a core.

### 🔄 Atomic Hot-Code Swapping
Update live application logic without stopping the runtime.
* **Zero downtime:** replace Python or Node.js handlers in memory without losing mailbox state.
* **Safe transition:** in-flight work completes on old logic; new messages use new logic.

### 🌐 Global Service Discovery (Phase 7 Enhanced)
Actors are first-class network services.
* **Name registry:** register human-readable names (for example, `"auth-provider"`) with `register`/`unregister` and resolve with `whereis`.
* **Async discovery:** resolve remote service PIDs with Python `await` or Node.js Promises without blocking runtime progress.
* **Location transparency:** message actors the same way whether local or remote.

### 🛡️ Distributed Supervision & Self-Healing
Built-in fault tolerance follows the “Let it Crash” model.
* **Heartbeat monitoring:** automatic `PING`/`PONG` (0x02/0x03) detects silent failures such as GIL stalls and half-open links.
* **Structured system messages:** exits, hot swaps, and heartbeats are surfaced as system events for supervisors.
* **Self-healing factories:** restart logic can re-resolve and reconnect automatically when remote nodes recover.

### 🧠 Experimental JIT & Compute Offload
A Python decorator (`@iris.offload`) marks CPU-heavy functions for native execution.
Under the hood Iris either:
- compiles expression bodies to machine code via Cranelift (`strategy="jit"`), or
- routes calls to a dedicated Rust actor pool (`strategy="actor"`).

Current JIT support (concise):
- **Arithmetic:** `+ - * / % **`, unary `+/-`, constants `pi/e`.
- **Logic:** `and`, `or`, `not`, boolean literals `True/False`.
- **Comparisons:** `< > <= >= == !=`, including chained comparisons (`a < b < c`).
- **Conditionals:** Python ternary (`a if cond else b`).
- **Loops:** `sum(expr for i in range(...))`, with optional `step` and `if` predicate,
    plus container form `sum(expr for x_i in x)` via wrapper vectorization.
- **Math calls:** `sin cos tan sinh cosh tanh exp log sqrt pow abs`, and `math.*` variants.

Optimizer highlights:
- constant folding + algebraic simplification,
- closed-form rewrites for common linear/quadratic range sums,
- safe constant evaluation for many bounded loop cases,
- automatic fallback to Python when compilation is not possible.

Logging is environment-aware and runtime-configurable:
- env mode: `IRIS_JIT_LOG=1` enables Rust-side JIT debug logs,
- Python API: `iris.jit.set_jit_logging(...)` and `iris.jit.get_jit_logging()`.

> [!NOTE] 
> Cranelift’s JIT backend historically relied on x86_64‑only PLT support. When running on
> aarch64 hardware, the runtime automatically disables PIC mode to avoid PLT relocation
> generation; offloaded functions still execute correctly but are not position‑independent.


---

## Technical Deep Dive

### The Actor Lifecycle
Iris actor internals vary by execution pattern:

1. **Push actors:** state-machine driven futures in Tokio, typically ~2KB each.
2. **Pull actors (Python):** dedicated blocking threads with higher footprint but simpler synchronous flow.

### Distributed Mesh Protocol
Iris uses a length-prefixed binary TCP protocol for inter-node communication.

| Packet Type | Function | Payload Structure |
| :--- | :--- | :--- |
| `0x00` | **User Message** | `[PID: u64][LEN: u32][DATA: Bytes]` |
| `0x01` | **Resolve Request** | `[LEN: u32][NAME: String]` → Returns `[PID: u64]` |
| `0x02` | **Heartbeat (Ping)** | `[Empty]` — Probe remote node health |
| `0x03` | **Heartbeat (Pong)** | `[Empty]` — Acknowledge health |

### Memory Safety & FFI
Iris bridges Rust safety with dynamic-language ergonomics through **PyO3** (Python) and **N-API** (Node.js):
* **Membrane hardening:** uses `block_in_place` and `ThreadSafeFunction` queues for safe sync/async boundaries.
* **GIL management:** Python `recv()` releases the GIL so other Python threads can continue running.
* **Atomic RwLocks:** behavior pointers are swapped safely for thread-safe hot swapping.

---

## Quick Start

### Requirements
* **Rust** 1.70+
* **Python** 3.8+ OR **Node.js** 14+
* **Maturin** (for Python) / **NAPI-RS** (for Node)
* **Cranelift JIT backend** is included by default; needed only if you use the experimental offload/JIT APIs.

### Installation

#### 🐍 Python
```bash
# Clone the repository
git clone https://github.com/SSL-ACTX/iris.git
cd iris

# Build and install the Python extension
maturin develop --release

```

#### 📦 Node.js

```bash
# Clone the repository
git clone https://github.com/SSL-ACTX/iris.git
cd iris

# Build the N-API binding
npm install
npm run build

```

---

## Usage Examples

### 🧠 Experimental JIT & Compute Offload (Python only)
```python
import iris
rt = iris.Runtime()

@iris.offload(strategy="jit", return_type="float")
def vector_magnitude(x: float, y: float, z: float) -> float:
    # simple expression gets compiled or dispatched via actor pool
    return (x*x + y*y + z*z) ** 0.5

# call just like a normal Python function; the heavy work runs in native code
result = vector_magnitude(1.0, 2.0, 3.0)
print(result)  # 3.7416573867739416
```

> [!NOTE] 
> The offload API also supports `strategy="actor"` for routing to a dedicated compute pool; see `JIT.md` for lower-level internals.

#### JIT Capability Snapshot
- Arithmetic + power: `+ - * / % **`, unary `+/-`
- Booleans: `and/or/not`, `True/False`
- Comparisons: `< > <= >= == !=`, including chains (`x < y < z`)
- Conditionals: `a if cond else b`
- Loops: `sum(...)` over `range(...)` (with step/predicate) and container generators
- Math: `sin cos tan sinh cosh tanh exp log sqrt pow abs`, including `math.*`

#### Optimizer Behavior
- Constant folding + simplification (including boolean/relation folding)
- Loop rewrites for common linear/quadratic forms
- Constant-bound loop evaluation when safe
- Exponent shortcuts (`x**0.5 -> sqrt(x)`, `x**-1 -> 1.0/x`)

#### Fallback + Logging
- JIT compile failure auto-falls back to normal Python execution.
- Enable logs by env: `IRIS_JIT_LOG=1`.
- Control at runtime from Python:
    - `iris.jit.set_jit_logging(True|False|None, env_var=None)`
    - `iris.jit.get_jit_logging()`

Iris provides a unified API across both supported languages.

### 1. High-Performance Push Actors (Recommended)

Use `spawn` for maximum throughput (100k+ actors). Rust owns the scheduling and only invokes the guest language when a message arrives.

#### Python

```python
import iris
rt = iris.Runtime()

def fast_worker(msg):
    print(f"Processed: {msg}")

# Spawn 1000 workers instantly (Green Threads)
for _ in range(1000):
    rt.spawn(fast_worker, budget=50)

```

#### Node.js

```javascript
const { NodeRuntime } = require('./index.js');
const rt = new NodeRuntime();

const fastWorker = (msg) => {
    // msg is a Buffer
    console.log(`Processed: ${msg.toString()}`);
};

for (let i = 0; i < 1000; i++) {
    rt.spawn(fastWorker, 50);
}

```

### 2. Synchronous Pull Actors (Erlang Style)

Use `spawn_with_mailbox` for complex logic where you want to block and wait for specific messages. **No async/await required in Python.**

#### Python

```python
# Runs in a dedicated OS thread. Blocking is safe.
def saga_coordinator(mailbox):
    # Blocks thread, releases GIL
    msg = mailbox.recv() 
    print("Starting Saga...")
    
    # Wait up to 5 seconds for next message
    confirm = mailbox.recv(timeout=5.0)
    if confirm: 
        print("Confirmed")
    else:
        print("Timed out")

rt.spawn_with_mailbox(saga_coordinator, budget=100)

```

#### Node.js (Promise Based)

```javascript
const sagaCoordinator = async (mailbox) => {
    const msg = await mailbox.recv();
    console.log("Starting Saga...");
    
    // Wait up to 5 seconds
    const confirm = await mailbox.recv(5.0);
    if (confirm) {
        console.log("Confirmed");
    } else {
        console.log("Timed out");
    }
};

rt.spawnWithMailbox(sagaCoordinator, 100);

```

### 3. Structured Concurrency

Actors can now be spawned with a **parent** PID. When the parent exits (normal or crash) all of its direct children are automatically stopped as well. This mirrors the behaviour of many functional runtimes and makes it easy to manage lifetimes for short‑lived helper tasks.

#### Rust

```rust
let rt = Runtime::new();
let parent = rt.spawn_actor(|mut rx| async move { /* ... */ });
let child = rt.spawn_child(parent, |mut rx| async move { /* will die with parent */ });

rt.send(parent, Message::User(Bytes::from("quit"))).unwrap();
// after the parent exits the child mailbox is closed as well
assert!(!rt.is_alive(child));
```

#### Python

```python
rt = iris.Runtime()
parent = rt.spawn(lambda msg: print("parent got", msg))
child = rt.spawn_child(parent, lambda msg: print("child got", msg))

rt.send(parent, b"quit")
import time; time.sleep(0.1)
assert not rt.is_alive(child)
```

There are three variants of the API:

- `spawn_child(parent, handler)` – mailbox‑based actor.
- `spawn_child_with_budget(parent, handler, budget)` – same but with a reduction budget.
- `spawn_child_handler_with_budget(parent, handler, budget)` – message‑style handler (used by Python/Node wrappers).

Python and Node bindings expose matching helpers (`spawn_child`, `spawn_child_with_mailbox`, etc.) which accept the same arguments as their non‑child counterparts plus the parent PID.

---

### 4. Service Discovery & Registry

> [!NOTE]
> **Network hardening:** the underlying TCP protocol now imposes a 1 MiB
> payload ceiling, per-operation timeouts, and diligent logging. Malformed or
> oversized messages are dropped rather than crashing the node, and remote
> resolution/send operations will fail fast instead of hanging indefinitely.


#### Python

```python
# 1. Register a local actor
pid = rt.spawn(my_handler)
rt.register("auth_worker", pid)

# 2. Look it up later (Local)
target = rt.whereis("auth_worker")

# 3. Look it up remotely (Network)
async def find_remote():
    addr = "192.168.1.5:9000"
    # Non-blocking resolution
    remote_pid = await rt.resolve_remote_py(addr, "auth_worker")
    if remote_pid:
        rt.send_remote(addr, remote_pid, b"login")

```

#### Node.js

```javascript
// 1. Register
const pid = rt.spawn(myHandler);
rt.register("auth_worker", pid);

// 2. Resolve Remote
async function findAndQuery() {
    const addr = "192.168.1.5:9000";
    const targetPid = await rt.resolveRemote(addr, "auth_worker");
    if (targetPid) {
        rt.sendRemote(addr, targetPid, Buffer.from("login"));
    }
}

```

---

### Path-Scoped Supervisors

Iris supports hierarchical path registrations (e.g. `/svc/payment/processor`) and allows you to create a supervisor that is scoped to a path prefix. This is useful for grouping related actors and applying supervision policies per-service or per-tenant.

Key Python APIs:
- `rt.create_path_supervisor(path)` — create a per-path supervisor instance.
- `rt.path_supervisor_watch(path, pid)` — register an actor PID with the path supervisor.
- `rt.path_supervisor_children(path)` — list PIDs currently supervised for the path.
- `rt.remove_path_supervisor(path)` — remove the path supervisor.
- `rt.spawn_with_path_observed(budget, path)` — spawn and register an observed actor under `path` (useful for testing/monitoring).

Python example:

```python
rt = iris.Runtime()

# spawn an observed actor and register it under a hierarchical path
pid = rt.spawn_with_path_observed(10, "/svc/test/one")

# create a supervisor for the '/svc/test' prefix and register the pid
rt.create_path_supervisor("/svc/test")
rt.path_supervisor_watch("/svc/test", pid)

# inspect supervised children
children = rt.path_supervisor_children("/svc/test")
print(children)  # [pid]

# remove supervisor when done
rt.remove_path_supervisor("/svc/test")

```

This mechanism makes it easy to apply restart strategies or monitoring rules to logical groups of actors without affecting the global supervisor.

### 4. Structured System Messages

#### Python

```python
messages = rt.get_messages(observer_pid)
for msg in messages:
    if isinstance(msg, iris.PySystemMessage):
        if msg.type_name == "EXIT":
            print(f"Actor {msg.target_pid} has crashed!")

```

#### Node.js

```javascript
// Node wrappers return wrapped objects { data: Buffer, system: Object }
const messages = rt.getMessages(observerPid);
messages.forEach(msg => {
    if (msg.system) {
        if (msg.system.typeName === "EXIT") {
            console.log(`Actor ${msg.system.targetPid} has crashed!`);
        }
    } else {
        console.log(`User Data: ${msg.data.toString()}`);
    }
});

```

---

### 5. Hot-Swapping Logic

#### Python

```python
def behavior_a(msg): print("Logic A")
def behavior_b(msg): print("Logic B (Upgraded!)")

pid = rt.spawn(behavior_a, budget=10)
rt.send(pid, b"test") # Prints "Logic A"

rt.hot_swap(pid, behavior_b)
rt.send(pid, b"test") # Prints "Logic B (Upgraded!)"

```

### 6. Mailbox Introspection & Timers

Iris exposes lightweight mailbox introspection and actor-local timers so guest languages can inspect queue sizes and schedule timed messages.

#### Mailbox Introspection

- **Rust:** `Runtime::mailbox_size(pid: u64) -> Option<usize>` returns the number of user messages queued for `pid` (excludes system messages).
- **Python:** `rt.mailbox_size(pid)` mirrors the Rust API and returns `None` if the PID is unknown.

Python example:

```python
size = rt.mailbox_size(pid)
print(f"Mailbox size for {pid}: {size}")
```

#### Actor-local Timers

You can schedule one-shot or repeating messages to an actor's mailbox. Timers are cancellable via an id returned at creation.

- **Rust APIs:** `send_after(pid, delay_ms, payload)`, `send_interval(pid, interval_ms, payload)`, `cancel_timer(timer_id)`
- **Python:** `rt.send_after(pid, ms, b'data')`, `rt.send_interval(pid, ms, b'data')`, `rt.cancel_timer(timer_id)`

Python example (one-shot):

```python
timer_id = rt.send_after(pid, 200, b'tick')  # send 'tick' after 200ms

# cancel if needed
rt.cancel_timer(timer_id)
```

Python example (repeating):

```python
timer_id = rt.send_interval(pid, 1000, b'heartbeat')  # every 1s

# stop later
rt.cancel_timer(timer_id)
```

---

### 7. Exit Reasons (Structured)

When an actor exits the runtime sends a structured `EXIT` system message that includes the reason and optional metadata. This allows supervisors and link/watch logic to make informed decisions.

Common `ExitReason` variants:
- `Normal` — actor finished cleanly.
- `Killed` — requested shutdown.
- `Panic` — runtime detected a panic; `ExitInfo` may include panic metadata.
- `Crash` — user code returned an unrecoverable error.

Python example receiving exit info:

```python
for msg in rt.get_messages(supervisor_pid):
    if isinstance(msg, iris.PySystemMessage) and msg.type_name == 'EXIT':
        print('from:', msg.from_pid)
        print('target:', msg.target_pid)
        print('reason:', msg.reason)        # e.g. 'Normal' | 'Panic' | 'Killed'
        print('metadata:', msg.metadata)    # optional dict/bytes with extra info

```

Node.js receives `system` objects with the same fields: `fromPid`, `targetPid`, `reason`, and optional `metadata`.

---

### 8. Runtime Configuration APIs (programmatic)

In addition to the environment variables documented above, the Python runtime exposes programmatic setters:

- `rt.set_release_gil_limits(max_threads: int, gil_pool_size: int)` — set the per-process cap and fallback pool size at runtime.
- `rt.set_release_gil_strict(strict: bool)` — when `true` a `spawn(..., release_gil=True)` will return an error if the dedicated-thread cap is reached instead of falling back to the shared pool.

---

### Bounded Mailboxes

By default, all Iris mailboxes are unbounded. For some high-throughput
workloads you may want a fixed capacity with a *drop-new* policy — useful for
rate-limiting fast producers. Use `Runtime.spawn_py_handler_bounded` in Python
or `runtime.spawn_bounded` in Node and specify the mailbox capacity (in
messages).

Python example:

```python
from iris import Runtime
rt = Runtime()

# handler simply prints messages
pid = rt.spawn_py_handler_bounded(lambda m: print('got', m), budget=100, capacity=2)

# first two sends succeed
assert rt.send(pid, b'one')
assert rt.send(pid, b'two')

# third message is dropped and send returns False
assert not rt.send(pid, b'three')
```

### Python example — toggling GIL release

You can control whether push-based Python actors run their callbacks on a blocking thread
that acquires the GIL (avoiding holding the GIL on the async worker) using the
`release_gil` flag on `Runtime.spawn`:

```python
from iris import Runtime
import time

rt = Runtime()

def handler_no(msg):
    print('no release', __import__('threading').get_ident())

def handler_yes(msg):
    print('release', __import__('threading').get_ident())

pid_no = rt.spawn(handler_no, budget=10, release_gil=False)
pid_yes = rt.spawn(handler_yes, budget=10, release_gil=True)

rt.send(pid_no, b'ping')
rt.send(pid_yes, b'ping')

time.sleep(0.2)
```

When `release_gil=True` the handler runs inside a `spawn_blocking` worker that
acquires the GIL; `release_gil=False` keeps the previous behavior (callback runs
directly while holding the GIL on the async worker).

---

### Path-based Registry & Supervision

Iris supports hierarchical path registrations (for example `/system/service/one`) so you can group, query and supervise actors by logical paths.

Python APIs (examples): `rt.register_path(path, pid)`, `rt.unregister_path(path)`, `rt.whereis_path(path)`, `rt.list_children(prefix)`, `rt.list_children_direct(prefix)`, `rt.watch_path(prefix)`, `rt.spawn_with_path_observed(budget, path)`, `rt.child_pids()`, `rt.children_count()`.

Python example:

```python
from iris import Runtime
rt = Runtime()

# Spawn and register
pid = rt.spawn(lambda m: None, 10)
rt.register_path("/system/service/one", pid)

print(rt.whereis_path("/system/service/one"))
print(rt.list_children("/system/service"))
print(rt.list_children_direct("/system"))

# Shallow watch: registers current direct children with the supervisor
rt.watch_path("/system/service")
print(rt.child_pids(), rt.children_count())
```

Node.js example (conceptual):

```javascript
const { NodeRuntime } = require('./index.js');
const rt = new NodeRuntime();

const pid = rt.spawn(myHandler, 10);
rt.registerPath('/system/service/one', pid);
const children = rt.listChildren('/system/service');
```

Notes:
- `list_children` returns all descendant registrations under a prefix.
- `list_children_direct` returns only immediate children one level below the prefix.
- `watch_path` performs a shallow registration of direct children with the supervisor — path-scoped supervisors are planned as a next step.

## Platform Notes

<details>
<summary><strong>Linux / macOS / Android (Termux)</strong></summary>
Fully supported. High performance via multi-threaded Tokio runtime.
<em>Note: Android builds require NDK or local clang configuration.</em>
</details>

<details>
<summary><strong>Windows</strong></summary>
Supported. Ensure you have the latest Microsoft C++ Build Tools installed for PyO3/N-API compilation.
</details>

---

## Disclaimer

> [!IMPORTANT]
> **Production Status:** Iris is currently in **Alpha**.
> * **Experimental Features:** The JIT/offload APIs are extremely new and may change or break between releases. Use with caution.
> * **Performance Metrics (v0.3.0):**
> * **Push Actors:** Validated to scale to **100k+ concurrent actors** with message throughput exceeding **~1.2M+ msgs/sec** on modern cloud vCPUs and **~409k msgs/sec** on single-core legacy hardware.
> * **Pull Actors:** High-performance threaded actors supporting **100k+ concurrent instances** with throughput reaching **~1.5M+ msgs/sec**, demonstrating massive scaling beyond traditional thread-pool limitations.
> * **Hot-Swapping:** Logic upgrades validated at **~136k swaps/sec** while maintaining active message processing.
> 
> 
> * **Notice:** The binary protocol is subject to change.
> * **Stability:** Always use the `Supervisor` for critical actor lifecycles to ensure automatic recovery and location transparency.
> 
> 

---

<div align="center">

**Author:** Seuriin ([SSL-ACTX](https://github.com/SSL-ACTX))

*v0.4.0*

</div>
