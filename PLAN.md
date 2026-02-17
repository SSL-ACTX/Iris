# 🐜 Project Myrmidon: The Embedded Actor Mesh

> **The Concept:** A high-performance, "Cooperative Preemption" Actor system implemented in Rust. It provides BEAM-like semantics (Lightweight processes, Supervision, Hot-Swapping) with a distributed networking layer, exposed via a high-performance FFI membrane to host languages.
> **The Goal:** A unified runtime where Python and Node.js can spawn 100,000+ Rust-managed actors that communicate transparently across a global cluster, bypassing the GIL and scaling horizontally with microsecond latency.

---

## 🏗️ The Architecture

### 1. The Core (Rust)

* **Slab-Allocated PIDs:** Actors are referenced by `u64` IDs via a high-performance Slab Allocator. This ensures memory safety across FFI boundaries and prevents dangling pointers.
* **Cooperative Preemption:** Implements a **Reduction Budget** scheduler. Every actor yields after a set number of operations, ensuring fair CPU distribution and preventing "heavy" actors from starving the system.
* **Dual-Channel Mailboxes:** Priority-based messaging where **System Messages** (Exit, HotSwap, Link) bypass the standard user message queue for instant lifecycle control.

### 2. The Distributed Mesh (Phase 5-7)

* **Location Transparency:** Actors can be addressed via `PID` or human-readable **Names**.
* **Service Discovery:** A global **Name Registry** allows nodes to resolve service PIDs across the network.
* **Transparent Routing:** The runtime automatically determines if a message is local or remote, handling TCP serialization and delivery via a background `NetworkManager`.

### 3. The Membrane (FFI)

* **GIL-Aware Execution:** Python handlers run on a Rust-managed thread pool. The runtime acquires the GIL only during callback execution, effectively breaking the Python bottleneck.
* **Zero-Copy Memory:** High-speed data transfer via `allocate_buffer` and `send_buffer`, moving raw memory addresses between actors instead of serializing large payloads.

---

## 🗓️ The Roadmap (Updated)

### ✅ Phase 1: The Hive (Rust Core) - *COMPLETED*

* Implemented **Slab Allocator** for `u64` PIDs.
* Built the **ReductionLimiter** for cooperative multitasking.
* Established basic **Link/Monitor** supervision primitives.

### ✅ Phase 2: The Python Membrane (PyO3) - *COMPLETED*

* Created the `PyRuntime` wrapper and `Runtime` Python class.
* Implemented the **GIL-Safe Callback** actor.
* Developed **Zero-Copy** buffer management using Python MemoryViews.

### ⏳ Phase 3: The JavaScript Membrane (NAPI-RS) - *PENDING*

* Expose the same `ActorHandle` and `Runtime` logic to Node.js.
* Implement `Promise`-based message resolution.

### ✅ Phase 4: Hot Code Swapping - *COMPLETED*

* Implemented atomic behavior swapping in memory.
* Actors can update their logic pointer mid-execution without losing state or mailbox data.

### ✅ Phase 5-7: The Distributed Mesh & Registry - *COMPLETED*

* **Networking:** TCP server/client for cross-node messaging.
* **Remote Monitoring:** Monitor a PID on a different machine; local supervisor reacts if the network drops.
* **Name Registry:** String-to-PID mapping for local and remote service discovery.

### 🚀 Phase 8: The Mesh Cloud (Next Steps)

* **Auto-Gossip:** Implement UDP discovery so nodes find each other without manual IP configuration.
* **CRDT Registry:** Synchronize the Name Registry across all nodes using Conflict-free Replicated Data Types.

---

## 🧪 Updated Tech Stack

* **Concurrency:** `Tokio` (Backbone) + Custom `ReductionLimiter` (Fairness).
* **Storage:** `DashMap` (Concurrent PID and Registry storage).
* **Networking:** `Bytes` + `Tokio TCP` (Binary Protocol).
* **Python Bindings:** `PyO3`.
* **State Safety:** `parking_lot::RwLock` for atomic behavior swapping.

---

## 📝 Current Usage Example

**Node A (The Provider):**

```python
import myrmidon

rt = myrmidon.Runtime()
rt.listen("0.0.0.0:9000")

def auth_logic(msg):
    print(f"Authenticating: {msg}")

pid = rt.spawn(auth_logic)
rt.register("auth-service", pid)

```

**Node B (The Consumer):**

```python
import myrmidon

rt = myrmidon.Runtime()
# Discover Node A's service by name
target_pid = rt.resolve_remote("192.168.1.10:9000", "auth-service")

# Send data across the network
rt.send_remote("192.168.1.10:9000", target_pid, b"CREDENTIALS_DATA")

```
