<div align="center">

![Myrmidon Banner](https://capsule-render.vercel.app/api?type=waving&color=0:000000,100:2E3440&height=220&section=header&text=Myrmidon&fontSize=90&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&rotate=-2&stroke=4C566A&strokeWidth=2&desc=Distributed%20Polyglot%20Actor%20Runtime&descSize=20&descAlignY=60)

![Version](https://img.shields.io/badge/version-0.1.3-blue.svg?style=for-the-badge)
![Language](https://img.shields.io/badge/language-Rust%20%7C%20Python%20%7C%20Node.js-orange.svg?style=for-the-badge&logo=rust)
![License](https://img.shields.io/badge/license-AGPL_3.0-green.svg?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS%20%7C%20Android-lightgrey.svg?style=for-the-badge&logo=linux)

**High-performance, hot-swappable distributed actor mesh for the modern era.**

[Features](#-core-capabilities) • [Architecture](#-technical-deep-dive) • [Installation](#-quick-start) • [Usage](#-usage-examples) • [Distributed Mesh](#-the-distributed-mesh)

</div>

---

## Overview

**Myrmidon** is a distributed actor-model runtime built in Rust with deep **Python** and **Node.js** integration. It is designed for systems that require the extreme concurrency of Erlang, the raw speed of Rust, and the flexibility of high-level scripting languages.

Unlike standard message queues or microservice frameworks, Myrmidon implements a **cooperative reduction-based scheduler**. This allows the runtime to manage millions of "actors" (lightweight processes) with microsecond latency, providing built-in fault tolerance, hot code swapping, and location-transparent messaging across a global cluster.

## Core Capabilities

### ⚡ Hybrid Actor Model (Push & Pull)
Myrmidon supports two distinct actor patterns to balance throughput and logic complexity:
* **Push Actors (Green Threads):** Extremely lightweight. Rust "pushes" messages to a callback only when data arrives.
    * **Ideal for:** High-throughput workers (100k+ concurrent), stateless processing, I/O handling.
* **Pull Actors (OS Threads):** Specialized blocking actors that "pull" messages from a `Mailbox`.
    * **Python:** Runs in a **dedicated OS thread**. Blocks on `recv()` (releasing the GIL), acting like a true Erlang/Go process. Zero `asyncio` overhead.
    * **Node.js:** Integrates with the V8 Event Loop via Promises.

### ⚡ Reduction-Based Scheduler
Inspired by the BEAM (Erlang VM), Myrmidon uses a **Cooperative Reduction Scheduler**.
* **Fairness:** Every actor is assigned a "reduction budget." Once exhausted, the runtime automatically yields to ensure no single actor can starve the system.
* **Pre-emptive Feel:** Provides the responsiveness of pre-emptive multitasking with the efficiency of async/await.

### 🔄 Atomic Hot-Code Swapping
Update your application logic while it is running.
* **Zero Downtime:** Swap out a Python handler or a Node.js function in memory without stopping the actor or losing its mailbox state.
* **Safe Transition:** Future messages are processed by the new logic instantly; current messages finish processing under the old logic.

### 🌐 Global Service Discovery (Phase 7 Enhanced)
Actors are first-class citizens of the network.
* **Name Registry:** Register actors with human-readable strings (e.g., `"auth-provider"`) using `register`/`unregister` and look them up via `whereis`.
* **Async Discovery:** Resolve remote service PIDs using native `await` (Python) or Promises (Node.js) without blocking the runtime.
* **Location Transparency:** Send messages to actors whether they reside on the local CPU or a server across the globe.

### 🛡️ Distributed Supervision & Self-Healing
Built-in fault tolerance modeled after the "Let it Crash" philosophy.
* **Heartbeat Monitoring:** The mesh automatically sends `PING`/`PONG` signals (0x02/0x03) to detect silent failures (e.g., GIL freezes or half-open TCP connections).
* **Structured System Messages:** Actor exits, hot-swaps, and heartbeats are delivered as System Messages, providing rich context for supervisor logic.
* **Self-Healing Factories:** Define closures that automatically re-resolve and restart connections when a remote node comes back online.

---

## Technical Deep Dive

### The Actor Lifecycle
Myrmidon actors are extremely lightweight, but the implementation differs by type:

1. **Push Actors:** Purely state-machine driven. They consume ~2KB of RAM and exist only as futures in the Tokio runtime.
2. **Pull Actors (Python):** Allocated a dedicated stack and OS thread (via Tokio's blocking pool). They are heavier but allow for straightforward, blocking, synchronous logic without "colored functions" (async/await).

### Distributed Mesh Protocol
Myrmidon uses a proprietary length-prefixed binary protocol over TCP for inter-node communication.

| Packet Type | Function | Payload Structure |
| :--- | :--- | :--- |
| `0x00` | **User Message** | `[PID: u64][LEN: u32][DATA: Bytes]` |
| `0x01` | **Resolve Request** | `[LEN: u32][NAME: String]` → Returns `[PID: u64]` |
| `0x02` | **Heartbeat (Ping)** | `[Empty]` — Probe remote node health |
| `0x03` | **Heartbeat (Pong)** | `[Empty]` — Acknowledge health |

### Memory Safety & FFI
Myrmidon bridges the gap between Rust’s memory safety and dynamic languages using **PyO3** (Python) and **N-API** (Node.js):
* **Membrane Hardening:** The runtime uses `block_in_place` and `ThreadSafeFunction` queues to safely handle synchronous calls from within asynchronous Rust contexts.
* **GIL Management:** * Python `recv()` calls release the GIL, allowing other threads to run in parallel while the actor waits for messages.
* **Atomic RwLocks:** Actor behaviors are protected by thread-safe pointer swaps, ensuring hot-swapping is thread-safe.

---

## Quick Start

### Requirements
* **Rust** 1.70+
* **Python** 3.8+ OR **Node.js** 14+
* **Maturin** (for Python) / **NAPI-RS** (for Node)

### Installation

#### 🐍 Python
```bash
# Clone the repository
git clone https://github.com/SSL-ACTX/myrmidon.git
cd myrmidon

# Build and install the Python extension
maturin develop --release

```

#### 📦 Node.js

```bash
# Clone the repository
git clone https://github.com/SSL-ACTX/myrmidon.git
cd myrmidon

# Build the N-API binding
npm install
npm run build

```

---

## Usage Examples

Myrmidon provides a unified API across both supported languages.

### 1. High-Performance Push Actors (Recommended)

Use `spawn` for maximum throughput (100k+ actors). Rust owns the scheduling and only invokes the guest language when a message arrives.

#### Python

```python
import myrmidon
rt = myrmidon.Runtime()

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

### 3. Service Discovery & Registry

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

### 4. Structured System Messages

#### Python

```python
messages = rt.get_messages(observer_pid)
for msg in messages:
    if isinstance(msg, myrmidon.PySystemMessage):
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

### 5. Mailbox Introspection & Timers

Myrmidon exposes lightweight mailbox introspection and actor-local timers so guest languages can inspect queue sizes and schedule timed messages.

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

### 6. Exit Reasons (Structured)

When an actor exits the runtime sends a structured `EXIT` system message that includes the reason and optional metadata. This allows supervisors and link/watch logic to make informed decisions.

Common `ExitReason` variants:
- `Normal` — actor finished cleanly.
- `Killed` — requested shutdown.
- `Panic` — runtime detected a panic; `ExitInfo` may include panic metadata.
- `Crash` — user code returned an unrecoverable error.

Python example receiving exit info:

```python
for msg in rt.get_messages(supervisor_pid):
    if isinstance(msg, myrmidon.PySystemMessage) and msg.type_name == 'EXIT':
        print('from:', msg.from_pid)
        print('target:', msg.target_pid)
        print('reason:', msg.reason)        # e.g. 'Normal' | 'Panic' | 'Killed'
        print('metadata:', msg.metadata)    # optional dict/bytes with extra info

```

Node.js receives `system` objects with the same fields: `fromPid`, `targetPid`, `reason`, and optional `metadata`.

---

### 7. Runtime Configuration APIs (programmatic)

In addition to the environment variables documented above, the Python runtime exposes programmatic setters:

- `rt.set_release_gil_limits(max_threads: int, gil_pool_size: int)` — set the per-process cap and fallback pool size at runtime.
- `rt.set_release_gil_strict(strict: bool)` — when `true` a `spawn(..., release_gil=True)` will return an error if the dedicated-thread cap is reached instead of falling back to the shared pool.

These mirror the behavior of `MYRMIDON_MAX_RELEASE_GIL_THREADS` and `MYRMIDON_GIL_POOL_SIZE` but allow dynamic tuning from the host language.

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

---

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
> **Production Status:** Myrmidon is currently in **Alpha**.
> * **Performance:** >   * **Push Actors:** Scale linearly with CPU cores (100k+ actors).
> * **Pull Actors:** Bound by OS thread limits (~500 concurrent threads default). Use for heavy logic only.
> 
> 
> * The binary protocol is subject to change.
> * Always use the `Supervisor` for critical actor lifecycles to ensure automatic recovery.
> 
> 

---

### Python example — toggling GIL release

You can control whether push-based Python actors run their callbacks on a blocking thread
that acquires the GIL (avoiding holding the GIL on the async worker) using the
`release_gil` flag on `Runtime.spawn`:

```python
from myrmidon import Runtime
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

## Configuration (environment)

You can tune `release_gil` behavior with environment variables:

- `MYRMIDON_MAX_RELEASE_GIL_THREADS` (integer, default 256): maximum number of
    dedicated per-actor threads created when `release_gil=True`. When exceeded,
    the runtime falls back to a shared GIL worker pool.
- `MYRMIDON_GIL_POOL_SIZE` (integer, default 8): number of threads in the
    shared GIL worker pool used as a fallback when the dedicated-thread limit is
    reached.

Example (bash):

```bash
export MYRMIDON_MAX_RELEASE_GIL_THREADS=128
export MYRMIDON_GIL_POOL_SIZE=16
python your_app.py
```

<div align="center">

**Author:** Seuriin ([SSL-ACTX](https://www.google.com/search?q=https://github.com/SSL-ACTX))

*v0.1.3*

</div>