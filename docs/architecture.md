# Iris Architecture

Iris is a hybrid distributed runtime fabric designed for actors, native compute offload, and cross-language services. It combines the safety and performance of Rust with the ergonomics of Python and Node.js.

## The Actor Model in Iris

Iris implements a lightweight actor model with two distinct execution styles:

### ⚡ Push Actors (Green Threads)
- **Mechanism:** Ultra-lightweight handlers triggered only when messages arrive.
- **Scheduling:** Managed by a cooperative reduction-based scheduler on top of Tokio.
- **Scaling:** Can scale to 100k+ concurrent actors.
- **Budget:** Each actor has a "reduction budget" and must yield when exhausted to ensure fairness.

### ⚡ Pull Actors (OS Threads)
- **Mechanism:** Blocking mailbox workers for synchronous control flow.
- **Python usage:** Run on dedicated OS threads and block on `recv()` while releasing the GIL.
- **Benefit:** Simplifies complex synchronous logic without needing `async/await`.

## Cooperative Reduction Scheduler

Inspired by the BEAM (Erlang VM), Iris ensures that no single high-throughput actor can monopolize a CPU core.
- **Reductions:** Every operation (sending a message, processing a callback) costs "reductions".
- **Yielding:** When the budget is hit, the actor's future yields to the Tokio executor.

## Atomic Hot-Code Swapping

Iris allows updating live application logic without stopping the runtime.
- **Zero downtime:** Replace Python or Node.js handlers in memory.
- **State Preservation:** In-flight messages remain in the mailbox; only the "behavior" pointer is swapped.
- **Versioning:** Supports rolling back to previous behavior versions.

## Telemetry & Observability Layer

Iris includes a built-in `TelemetryManager` that provides real-time instrumentation of the actor mesh.
- **Event Stream:** Captures actor life-cycles (`Spawned`, `Stopped`, `Crashed`) and message throughput.
- **Health Tracking:** Tracks actor health status (`Starting`, `Ready`, `Busy`, `Degraded`) for cluster-wide health checks.
- **Mailbox Depth:** Provides visibility into queue depths for identifying bottlenecks.
- **Metrics API:** Aggregated system-wide statistics available via a high-performance concurrent registry.

## Distributed Mesh Protocol

Iris nodes communicate over TCP using a length-prefixed binary protocol.

| Packet Type | Function | Payload Structure |
| :--- | :--- | :--- |
| `0x00` | **User Message** | `[PID: u64][LEN: u32][DATA: Bytes]` |
| `0x01` | **Resolve Request** | `[LEN: u32][NAME: String]` → Returns `[PID: u64]` |
| `0x02` | **Heartbeat (Ping)** | `[Empty]` — Probe remote node health |
| `0x03` | **Heartbeat (Pong)** | `[Empty]` — Acknowledge health |

## Memory Safety & FFI

Iris bridges Rust with guest languages using high-performance membranes:
- **Python:** Built with [PyO3](https://github.com/PyO3/pyo3).
- **Node.js:** Built with [N-API](https://nodejs.org/api/n-api.html).
- **GIL Management:** Specifically designed to release the Python GIL during blocking operations or when offloading to Rust.
