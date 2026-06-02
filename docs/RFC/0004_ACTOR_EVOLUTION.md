# RFC 0004: Evolution of the Iris Actor Model

Status: PROPOSAL
Date: 2026-05-12

## Abstract

Iris currently utilizes a traditional Actor Model (PID-based, isolated state, mailbox-driven). While robust, it follows established patterns (Erlang/Akka). This RFC proposes four experimental "Divergence Tracks" to evolve Iris into a unique runtime fabric that breaks the traditional actor mold.

---

## track 1: Spatial substrate (Proximity Actors)

### The Shift
Ditch PIDs for Coordinates. Communication is based on logical or physical "distance" rather than identity.

### Mechanism
- **Addressing:** Actors are spawned at `(x, y, z)` or logical tags.
- **Diffusion:** `send_to_radius(center, radius, msg)`.
- **Topography:** The runtime manages a spatial hash grid. Latency and reliability are functions of logical "distance."
- **Use Case:** Swarm intelligence, edge computing, digital twins.

---

## Track 2: Log-Stream "Projection" Actors

### The Shift
Actors are not long-running loops; they are materialized "views" over immutable event streams.

### Mechanism
- **State:** Not private/local, but a projection of a shared log.
- **Receiving:** An actor is a pointer on a log. Processing a message is moving the pointer.
- **Concurrency:** Multiple actors can project the same log into different state shapes (Read Models).
- **Use Case:** Distributed consistency, audit-logging, zero-cost state recovery.

---

## Track 3: Capability-First (Link-Oriented)

### The Shift
No global registry. Communication is impossible without an explicit "Link" (Object Capability).

### Mechanism
- **Links:** First-class objects passed between actors. Links define the *how* (e.g., `SecureLink`, `LossyLink`, `BatchedLink`).
- **Session Types:** Links enforce communication protocols at the runtime level (e.g., "Must send A, then receive B").
- **Security:** Actors are naturally isolated. No `lookup("service")`; only "use the link I was given."
- **Use Case:** High-security systems, complex multi-step protocols.

---

## Track 4: Relational (Declarative) Actors

### The Shift
Actor logic is a set of relational rules (Datalog/SQL-like) rather than imperative `recv` loops.

### Mechanism
- **Data as State:** Actor state is a set of relations (tables).
- **Triggers:** Messages are "row inserts" that trigger rule evaluations.
- **Optimization:** The Iris runtime acts as a query optimizer, parallelizing rule execution across available cores.
- **Use Case:** Large-scale data processing, complex state machines.

---

## Recommendation

Iris should pursue **Track 3 (Capability)** for its core security and **Track 1 (Spatial)** as an extension for distributed coordination. This creates a "Connected & Situated" model unique to Iris.
