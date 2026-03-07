# 🐜 Iris – Generalization Feature Checklist

> **Goal:** Evolve Iris from a high-performance actor runtime into a **general, language-agnostic, BEAM-class foundation**.

---

## 🟣 Core Actor Semantics (BEAM-Inspired)

* [x] **Selective Receive**
  Actors can pattern-match and defer messages until a desired one arrives.

* [x] **Exit Reasons & Crash Metadata**
  Structured failure reasons (`:panic`, `:timeout`, `:killed`, `:oom`, etc.).

* [x] **Mailbox Introspection**
  Query mailbox size / pressure for adaptive behavior.

* [x] **Actor-Local Timers**
  Timers belong to actors (`send_after`, `interval`) — no global sleeps.

---

## 🔵 Supervision & Structure (Akka-Inspired)

* [x] **Actor Hierarchies / Paths**
  Structured actor paths (`/system/http/router/worker-17`).

* [x] **Structured Concurrency**
  Child actors automatically terminate with their parent.

* [ ] **Behavior Versioning**
  Track hot-swapped logic versions with rollback support.

---

## 🟢 Mailbox & Flow Control (Go / CSP-Inspired)

* [x] **Bounded Mailboxes**
  Configurable capacity per actor with drop-new semantic (this phase 1 implementation).
  - Rust API: `spawn_actor_bounded` / `spawn_actor_with_budget_bounded`
  - Python: `Runtime.spawn_py_handler_bounded` (with tests)
  - Node.js: `Runtime.spawn_bounded`

* [x] **Overflow Policies**
  Drop-new, drop-old, block, redirect, or spill to actor.  (Implemented with tests across Rust and Python.)

* [ ] **Backpressure Signals**
  Runtime can refuse messages or slow producers under load.

---

## 🟠 Scale & Abstraction (Orleans-Inspired)

* [x] **Virtual / Lazy Actors**
  Actors instantiated on first message, destroyed when idle.

* [x] **Location Transparency**
  Local vs remote actors are indistinguishable to callers.  Automatic
  proxies forward messages and monitoring across nodes.
---

## 🔴 Distribution & Resilience

* [ ] **Cluster Health Probes**
  Actors expose liveness / readiness.

* [ ] **Network-Aware Supervision**
  Distinguish crash vs disconnect vs partition.

* [ ] **Graceful Degradation**
  Load shedding instead of OOM or cascade failure.

---

## 🟤 Observability & Introspection

* [ ] **Tracing Hooks**
  Spawn, send, receive, crash, restart.

* [ ] **Metrics Export**
  Actor count, mailbox depth, memory per actor, throughput.

* [ ] **Runtime Introspection API**
  List actors, supervisors, memory usage, queues.

---

## ⚫ Safety & Correctness (Functional Runtimes)

* [ ] **Pure Actor Mode (Optional)**
  Forbid shared mutable state across actors.

* [ ] **Deterministic Scheduling Mode**
  Debug/replay mode for race-free testing.

---

## 🧩 Language-Specific Integrations

### 🐍 Python

* [ ] `async/await` ↔ actor bridge
* [ ] `contextvars` propagation
* [ ] NumPy / buffer zero-copy pipelines

### 🟨 Node.js

* [ ] Promise-based `call()` semantics
* [ ] Async iterator message streams
* [ ] Zero-copy `ArrayBuffer` / `Uint8Array` messaging

---

## 🚫 Explicit Non-Goals (Keep the Runtime Clean)

* [ ] ❌ Shared-memory actors
* [ ] ❌ Actor inheritance
* [ ] ❌ Blocking APIs in core runtime
* [ ] ❌ Language-level thread management
* [ ] ❌ Reflection-heavy APIs

---

## 🏁 Guiding Principle (Pin This)

* [-] **Runtime First, Language Second**
  Languages are guests.
  Rust owns scheduling, memory, safety, and correctness.
