<div align="center">

![Myrmidon Banner](https://capsule-render.vercel.app/api?type=waving&color=0:000000,100:2E3440&height=220&section=header&text=Myrmidon&fontSize=90&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&rotate=-2&stroke=4C566A&strokeWidth=2&desc=Distributed%20Polyglot%20Actor%20Runtime&descSize=20&descAlignY=60)

![Version](https://img.shields.io/badge/version-0.1.2-blue.svg?style=for-the-badge)
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
Myrmidon supports two distinct actor patterns across both Python and Node.js:
* **Push Actors (Default):** Extremely lightweight. Rust "pushes" messages to a callback (Python Function or JS `ThreadSafeFunction`) only when data arrives. Zero idle overhead; ideal for high-throughput workers.
* **Pull Actors (Mailbox):** Specialized `async/await` actors that "pull" messages from a `Mailbox`. 
    * **Python:** Runs inside a dedicated `asyncio` loop.
    * **Node.js:** Integrates with the V8 Event Loop via Promises.

### ⚡ Reduction-Based Scheduler
Inspired by the BEAM (Erlang VM), Myrmidon uses a **Cooperative Reduction Scheduler**.
* **Fairness:** Every actor is assigned a "reduction budget." Once exhausted, the runtime automatically yields the thread to ensure no single actor can starve the system.
* **Pre-emptive Feel:** Provides the responsiveness of pre-emptive multitasking with the efficiency of async/await.

### 🔄 Atomic Hot-Code Swapping
Update your application logic while it is running.
* **Zero Downtime:** Swap out a Python handler or a Node.js function in memory without stopping the actor or losing its mailbox state.
* **Safe Transition:** Future messages are processed by the new logic instantly; current messages finish processing under the old logic.

### 🌐 Global Service Discovery (Phase 7 Enhanced)
Actors are first-class citizens of the network.
* **Name Registry:** Register actors with human-readable strings (e.g., `"auth-provider"`) instead of tracking numeric PIDs.
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
Myrmidon bridges the gap between Rust’s memory safety and dynamic languages using **PyO3** (Python) and **N-API** (Node.js):
* **Membrane Hardening:** The runtime uses `block_in_place` and `ThreadSafeFunction` queues to safely handle synchronous calls from within asynchronous Rust contexts.
* **GIL & V8 Integration:** * **Python:** Manages the GIL to ensure Rust networking threads never block Python execution.
    * **Node.js:** Respects the single-threaded Event Loop by offloading actor logic to the Libuv queue while keeping the heavy lifting in Rust threads.
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

Use `spawn` for maximum throughput. Rust owns the scheduling and only invokes the guest language when a message arrives.

#### Python

```python
import myrmidon
rt = myrmidon.Runtime()

def fast_worker(msg):
    print(f"Processed: {msg}")

# Spawn 1000 workers instantly
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

### 2. Async Pull Actors (Orchestration)

Use `spawn_with_mailbox` for complex logic that requires `await`, timers, or specific message ordering.

#### Python

```python
async def saga_coordinator(mailbox):
    msg = await mailbox.recv()
    print("Starting Saga...")
    # Wait up to 5 seconds for next message
    try:
        confirm = await mailbox.recv(timeout=5.0)
        if confirm: print("Confirmed")
    except:
        print("Timed out")

rt.spawn_with_mailbox(saga_coordinator, budget=100)

```

#### Node.js

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

### 3. Async Service Discovery

#### Python

```python
async def find_and_query():
    addr = "127.0.0.1:9000"
    target_pid = await rt.resolve_remote_py(addr, "auth-service")
    if target_pid:
        rt.send_remote(addr, target_pid, b"verify_token")

```

#### Node.js

```javascript
async function findAndQuery() {
    const addr = "127.0.0.1:9000";
    const targetPid = await rt.resolveRemote(addr, "auth-service");
    if (targetPid) {
        rt.sendRemote(addr, targetPid, Buffer.from("verify_token"));
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

#### Node.js

```javascript
const behaviorA = (msg) => console.log("Logic A");
const behaviorB = (msg) => console.log("Logic B (Upgraded!)");

const pid = rt.spawn(behaviorA, 10);
rt.send(pid, Buffer.from("test"));

rt.hotSwap(pid, behaviorB);
rt.send(pid, Buffer.from("test"));

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
> * **Performance:** Node.js throughput is bound by the single-threaded Event Loop (~138k msgs/sec on mobile), whereas Python/Rust scales with CPU cores via the GIL release mechanics.
> * The binary protocol is subject to change.
> * Always use the `Supervisor` for critical actor lifecycles to ensure automatic recovery.
> 
> 

---

<div align="center">

**Author:** Seuriin ([SSL-ACTX](https://www.google.com/search?q=))

*v0.1.2*

</div>