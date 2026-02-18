<div align="center">

![Myrmidon Banner](https://capsule-render.vercel.app/api?type=waving&color=0:000000,100:2E3440&height=220&section=header&text=Myrmidon&fontSize=90&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&rotate=-2&stroke=4C566A&strokeWidth=2&desc=Distributed%20Polyglot%20Actor%20Runtime&descSize=20&descAlignY=60)

![Version](https://img.shields.io/badge/version-0.1.1-blue.svg?style=for-the-badge)
![Language](https://img.shields.io/badge/language-Rust%20%7C%20Python-orange.svg?style=for-the-badge&logo=rust)
![License](https://img.shields.io/badge/license-AGPL_3.0-green.svg?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg?style=for-the-badge&logo=linux)

**High-performance, hot-swappable distributed actor mesh for the modern era.**

[Features](#-core-capabilities) • [Architecture](#-technical-deep-dive) • [Installation](#-quick-start) • [Usage](#-usage-examples) • [Distributed Mesh](#-the-distributed-mesh)

</div>

---

## Overview

**Myrmidon** is a distributed actor-model runtime built in Rust with deep Python integration. It is designed for systems that require the extreme concurrency of Erlang, the raw speed of Rust, and the flexibility of Python.

Unlike standard message queues or microservice frameworks, Myrmidon implements a **cooperative reduction-based scheduler**. This allows the runtime to manage millions of "actors" (lightweight processes) with microsecond latency, providing built-in fault tolerance, hot code swapping, and location-transparent messaging across a global cluster.

## Core Capabilities

### ⚡ Hybrid Actor Model (Push & Pull)
Myrmidon supports two distinct actor patterns to balance raw speed with developer convenience:
* **Push Actors (Default):** Extremely lightweight. Rust "pushes" messages to a Python callback only when data arrives. Zero idle overhead; ideal for high-throughput workers.
* **Pull Actors (Mailbox):** Specialized `async/await` actors that "pull" messages from a `PyMailbox`. These run inside their own `asyncio` loop, allowing for complex orchestration flows (e.g., "send request, sleep 5s, await response").

### ⚡ Reduction-Based Scheduler
Inspired by the BEAM (Erlang VM), Myrmidon uses a **Cooperative Reduction Scheduler**.
* **Fairness:** Every actor is assigned a "reduction budget." Once exhausted, the runtime automatically yields the thread to ensure no single actor can starve the system.
* **Pre-emptive Feel:** Provides the responsiveness of pre-emptive multitasking with the efficiency of async/await.

### 🔄 Atomic Hot-Code Swapping
Update your application logic while it is running.
* **Zero Downtime:** Swap out a Python handler function in memory without stopping the actor or losing its mailbox state.
* **Safe Transition:** Future messages are processed by the new logic instantly; current messages finish processing under the old logic.

### 🌐 Global Service Discovery (Phase 7 Enhanced)
Actors are first-class citizens of the network.
* **Name Registry:** Register actors with human-readable strings (e.g., `"auth-provider"`) instead of tracking numeric PIDs.
* **Async Discovery:** Resolve remote service PIDs using native Python `await` without blocking the event loop.
* **Location Transparency:** Send messages to actors whether they reside on the local CPU or a server across the globe.

### 🛡️ Distributed Supervision & Self-Healing
Built-in fault tolerance modeled after the "Let it Crash" philosophy.
* **Heartbeat Monitoring:** The mesh automatically sends `PING`/`PONG` signals (0x02/0x03) to detect silent failures (e.g., GIL freezes or half-open TCP connections).
* **Structured System Messages:** Actor exits, hot-swaps, and heartbeats are delivered as `PySystemMessage` objects, providing rich context for supervisor logic.
* **Self-Healing Factories:** Use `supervise_with_factory` to define Python closures that automatically re-resolve and restart connections when a remote node comes back online.

---

## Technical Deep Dive

### The Actor Lifecycle
Myrmidon actors are extremely lightweight, consuming only a few kilobytes of RAM. 

1. **Spawn:** An actor is initialized with a mailbox and a PID from the Slab Allocator.
2. **Execute:** The scheduler executes the actor's handler until its reduction budget hits zero.
3. **Message Loop:** Actors remain suspended until a message arrives in their lock-free MPSC mailbox.
4. **Exit:** Upon completion or failure, the Supervisor cleans up the PID and notifies any linked observers.

### Distributed Mesh Protocol
Myrmidon uses a proprietary length-prefixed binary protocol over TCP for inter-node communication.

| Packet Type | Function | Payload Structure |
| :--- | :--- | :--- |
| `0x00` | **User Message** | `[PID: u64][LEN: u32][DATA: Bytes]` |
| `0x01` | **Resolve Request** | `[LEN: u32][NAME: String]` → Returns `[PID: u64]` |
| `0x02` | **Heartbeat (Ping)** | `[Empty]` — Probe remote node health |
| `0x03` | **Heartbeat (Pong)** | `[Empty]` — Acknowledge health |

### Memory Safety & FFI
By using **PyO3** and **Tokio**, Myrmidon bridges the gap between Rust’s memory safety and Python’s rapid development:
* **Membrane Hardening:** The runtime uses `block_in_place` to safely handle synchronous Python calls from within asynchronous Rust contexts.
* **GIL Awareness:** The runtime carefully manages the Python Global Interpreter Lock to ensure Rust networking threads never block Python execution.
* **Atomic RwLocks:** Actor behaviors are protected by thread-safe pointer swaps, ensuring hot-swapping is thread-safe.

---

## Quick Start

### Requirements
* **Rust** 1.70+
* **Python** 3.8+
* **Maturin** (for building Python bindings)

### Installation
```bash
# Clone the repository
git clone https://github.com/SSL-ACTX/myrmidon.git
cd myrmidon

# Build and install the Python extension
maturin develop --release

```

---

## Usage Examples

### 1. High-Performance Push Actors (Recommended)

Use `spawn` for maximum throughput. Rust owns the scheduling and only invokes Python when a message arrives.

```python
import myrmidon

rt = myrmidon.Runtime()

def fast_worker(msg):
    # Rust calls this function efficiently
    print(f"Processed: {msg}")

# Spawn 1000 workers instantly
for _ in range(1000):
    rt.spawn(fast_worker, budget=50)

```

### 2. Async Pull Actors (Orchestration)

Use `spawn_with_mailbox` for complex logic that requires `await`, timers, or specific message ordering.
*Note: These actors hold a dedicated thread and should be used sparingly for orchestration.*

```python
import asyncio

async def saga_coordinator(mailbox):
    # 1. Wait for initial trigger
    msg = await mailbox.recv()
    print("Starting Saga...")

    # 2. Selective Receive: Wait specifically for 'CONFIRM' message, skipping others
    def confirmation_matcher(m):
        return m == b"CONFIRM"
    
    try:
        # Wait up to 5 seconds for confirmation
        confirm = await mailbox.selective_recv(confirmation_matcher, timeout=5.0)
        if confirm:
            print("Saga Confirmed!")
    except Exception:
        print("Saga Timed Out")

# Spawn the async actor
rt.spawn_with_mailbox(saga_coordinator, budget=100)

```

### 3. Async Service Discovery

```python
import asyncio

rt = myrmidon.Runtime()

async def find_and_query():
    # Resolve a remote service PID without blocking the asyncio loop
    addr = "127.0.0.1:9000"
    target_pid = await rt.resolve_remote_py(addr, "auth-service")
    
    if target_pid:
        rt.send_remote(addr, target_pid, b"verify_token")

asyncio.run(find_and_query())

```

### 4. Structured System Messages

```python
# Messages from an observed actor can be data or system events
messages = rt._inner.get_messages(observer_pid)

for msg in messages:
    if isinstance(msg, myrmidon.PySystemMessage):
        if msg.type_name == "EXIT":
            print(f"Actor {msg.target_pid} has crashed!")
        elif msg.type_name == "PING":
            pass # Ignore heartbeats
    else:
        print(f"User Data: {msg.decode()}")

```

### 5. Hot-Swapping Logic

```python
def behavior_a(msg):
    print("Logic A")

def behavior_b(msg):
    print("Logic B (Upgraded!)")

pid = rt.spawn(behavior_a, budget=10)
rt.send(pid, b"test") # Prints "Logic A"

# Atomic swap to behavior_b
rt.hot_swap(pid, behavior_b)
rt.send(pid, b"test") # Prints "Logic B (Upgraded!)"

```

---

## Platform Notes

<details>
<summary><strong>Linux / macOS</strong></summary>
Fully supported. High performance via multi-threaded Tokio runtime.
</details>

<details>
<summary><strong>Windows</strong></summary>
Supported. Ensure you have the latest Microsoft C++ Build Tools installed for PyO3 compilation.
</details>

---

## Disclaimer

> [!IMPORTANT]
> **Production Status:** Myrmidon is currently in **Alpha**.
> * **Push vs Pull:** Prefer `spawn()` (Push) for performance. `spawn_with_mailbox()` (Pull) creates a per-actor Python event loop and pins a thread; use it only for specialized orchestration actors.
> * The binary protocol is subject to change.
> * Always use the `Supervisor` for critical actor lifecycles to ensure automatic recovery.
> 
> 

---

<div align="center">

**Author:** Seuriin ([SSL-ACTX](https://www.google.com/search?q=))

*v0.1.1*

</div>