<div align="center">

![Myrmidon Banner](https://capsule-render.vercel.app/api?type=waving&color=0:000000,100:2E3440&height=220&section=header&text=Myrmidon&fontSize=90&fontColor=FFFFFF&animation=fadeIn&fontAlignY=35&rotate=-2&stroke=4C566A&strokeWidth=2&desc=Distributed%20Polyglot%20Actor%20Runtime&descSize=20&descAlignY=60)

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?style=for-the-badge)
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

### ⚡ Reduction-Based Scheduler
Inspired by the BEAM (Erlang VM), Myrmidon uses a **Cooperative Reduction Scheduler**.
* **Fairness:** Every actor is assigned a "reduction budget." Once exhausted, the runtime automatically yields the thread to ensure no single actor can starve the system.
* **Pre-emptive Feel:** Provides the responsiveness of pre-emptive multitasking with the efficiency of async/await.

### 🔄 Atomic Hot-Code Swapping
Update your application logic while it is running.
* **Zero Downtime:** Swap out a Python handler function in memory without stopping the actor or losing its mailbox state.
* **Safe Transition:** Future messages are processed by the new logic instantly; current messages finish processing under the old logic.

### 🌐 Global Service Discovery
Actors are first-class citizens of the network.
* **Name Registry:** Register actors with human-readable strings (e.g., `"auth-provider"`) instead of tracking numeric PIDs.
* **Remote Resolution:** Query any node in the cluster to find a service PID.
* **Location Transparency:** Send messages to actors whether they reside on the local CPU or a server across the globe.

### 🛡️ Distributed Supervision
Built-in fault tolerance modeled after the "Let it Crash" philosophy.
* **Remote Monitoring:** A node can "watch" a remote PID. If the network drops or the remote process crashes, the local supervisor is notified instantly.
* **Restart Strategies:** Define `RestartOne` or `RestartAll` logic to automatically recover from failures.

### 🚀 Zero-Copy Memory Management
Optimized for high-throughput data processing.
* **Slab Allocation:** Uses a high-performance slab allocator for PIDs and internal buffers to minimize heap fragmentation.
* **Shared Buffers:** Move large datasets between actors using `BufferID` handles, avoiding expensive serialization/deserialization.

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
| `0x02` | **System Signal** | High-priority signals (Exit, Link, Monitor) |

### Memory Safety & FFI
By using **PyO3** and **Tokio**, Myrmidon bridges the gap between Rust’s memory safety and Python’s rapid development:
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

### 1. Simple Local Actor

```python
import myrmidon

rt = myrmidon.Runtime()

def my_handler(msg):
    print(f"Received: {msg.decode()}")

# Spawn with a budget of 100 reductions
pid = rt.spawn(my_handler, budget=100)
rt.send(pid, b"Hello Myrmidon!")

```

### 2. Hot-Swapping Logic

```python
def behavior_a(msg):
    print("Logic A")

def behavior_b(msg):
    print("Logic B (Upgraded!)")

pid = rt.spawn(behavior_a)
rt.send(pid, b"test") # Prints "Logic A"

# Atomic swap to behavior_b
rt.hot_swap(pid, behavior_b)
rt.send(pid, b"test") # Prints "Logic B (Upgraded!)"

```

### 3. Service Discovery

```python
# On Node A (127.0.0.1:9000)
rt.listen("127.0.0.1:9000")
pid = rt.spawn(my_handler)
rt.register("logger", pid)

# On Node B
target_pid = rt.resolve_remote("127.0.0.1:9000", "logger")
rt.send_remote("127.0.0.1:9000", target_pid, b"Log this message")

```

---

## The Distributed Mesh

Myrmidon nodes automatically form a mesh network.

* **Resilience:** If Node A monitors Node B and the connection fails, Node A can trigger an automatic failover.
* **Scalability:** Scale your Python application horizontally by simply spawning actors on new nodes and registering them in the global namespace.

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
> * The binary protocol is subject to change.
> * Performance tuning for million-actor benchmarks is ongoing.
> * Always use the `Supervisor` for critical actor lifecycles to ensure automatic recovery.
> 
> 

---

<div align="center">

**Author:** Seuriin ([SSL-ACTX]())

*v0.1.0*

</div>
