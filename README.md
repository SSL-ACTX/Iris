<div align="center">

![Iris Banner](https://svg-banners.vercel.app/api?type=luminance&text1=Iris%20🌸&width=800&height=200&color=FFB6C1)

![Version](https://img.shields.io/badge/version-0.5.0-blue.svg?style=flat-square)
![Language](https://img.shields.io/badge/language-Rust%20%7C%20Python%20%7C%20Node.js-orange.svg?style=flat-square&logo=rust)
![License](https://img.shields.io/badge/license-AGPL_3.0-green.svg?style=flat-square)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SSL-ACTX/Iris)

**Hybrid distributed runtime fabric for actors, cross-language services, and experimental native compute offload.**

[Architecture](docs/architecture.md) • [Usage Guide](docs/usage.md) • [JIT & Offload](docs/jit.md) • [Distributed Mesh](docs/distributed.md) • [Vortex-Transmuter](docs/vortex.md)

</div>

---

## Overview

**Iris** is a hybrid distributed runtime built in Rust with first-class **Python** bindings. It combines three execution styles:
- **Actor Mesh:** Stateful, message-driven workflows with high concurrency.
- **Native Offload/JIT:** CPU-heavy hot paths accelerated via Cranelift. This path is experimental, currently paused, and may be dropped in future releases.
- **Cross-Language API:** Service-oriented apps mixing Rust and Python.

Iris uses a **cooperative reduction-based scheduler** for fairness, providing built-in supervision, hot swapping, discovery, and location-transparent messaging across nodes.

> [!NOTE]
> Node.js bindings are currently on hold and are not actively developed or supported.

---

## Core Capabilities

- **Hybrid Concurrency:** Mix "Push" green-thread actors with "Pull" OS-thread actors.
- **Atomic Hot-Swap:** Update live application logic (Python) with zero downtime.
- **Global Discovery:** Register and resolve named services locally or over the network.
- **Resilience:** Built-in supervision, load shedding, and network-aware exit reasons for self-healing systems.
- **Observability:** Deep runtime introspection, system metrics, and actor health tracking.
- **Async Request/Reply:** Built-in `call()` semantics for ergonomic, awaitable actor communication.
- **Vortex-Transmuter (Experimental):** Instruction-bound preemption, transactional ghosting primitives, and guarded bytecode transmutation with explicit fallback telemetry (see [Vortex-Transmuter Guide](docs/vortex.md)).
- **JIT Acceleration:** Transparently compile Python math functions to native machine code.
    - **Quantum Speculation:** Optional multi-variant JIT selection with runtime telemetry, bounded by compile budget and cooldown controls (see [JIT Internals & Configuration](docs/jit.md)).

> [!NOTE]
> JIT acceleration development is currently paused and may be dropped from the project, while the runtime focuses on actor and cross-language capabilities.

---

## Quick Start

### Installation

#### 🐍 Python
```bash
pip install maturin
maturin develop --release
```

#### 📦 Node.js
```bash
npm install
npm run build
```

### Basic Example (Python)

```python
import iris
rt = iris.Runtime()

# 1. Spawn a high-performance actor
def worker(msg):
    print(f"Got: {msg}")

pid = rt.spawn(worker, budget=50)

# 2. Transparently offload math to JIT
@iris.offload(strategy="jit", return_type="float")
def fast_math(x: float):
    return x * 1.5 + 42.0

# 3. Message the actor
rt.send(pid, b"hello world")
rt.send_many(pid, [b"a", bytearray(b"b"), memoryview(b"c")])
print(fast_math(10.0))
```

---

## Learn More

- [Full Architecture Reference](docs/architecture.md)
- [Usage Examples & API Guide](docs/usage.md)
- [JIT Internals & Configuration](docs/jit.md)
- [Distributed Mesh & Discovery](docs/distributed.md)
- [Vortex-Transmuter Guide & Roadmap](docs/vortex.md)

---

## Disclaimer

> [!IMPORTANT]
> **Production Status:** Iris is currently in **Beta**. 
>
> **Performance (v0.3.0):**
> - **Push Actors:** 100k+ concurrent actors, ~1.2M+ msgs/sec.
> - **Pull Actors:** 100k+ concurrent instances, ~1.5M+ msgs/sec.
> - **Hot-Swapping:** ~136k swaps/sec under load.
> - **See more at:** [v0.3.0 Benchmarks](docs/benchmarks/BENCHMARKS.md)

---

<div align="center">

**Author:** Seuriin ([SSL-ACTX](https://github.com/SSL-ACTX))

</div>
