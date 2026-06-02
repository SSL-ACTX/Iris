<div align="center">

![Iris Banner](https://svg-banners.vercel.app/api?type=luminance&text1=Iris%20🌸&width=800&height=200&color=FFB6C1)

![Version](https://img.shields.io/badge/version-0.5.0-blue.svg?style=flat-square)
![Language](https://img.shields.io/badge/language-Rust%20%7C%20Python-orange.svg?style=flat-square&logo=rust)
![License](https://img.shields.io/badge/license-AGPL_3.0-green.svg?style=flat-square)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SSL-ACTX/Iris)

**Hybrid distributed runtime fabric for actors, cross-language services, and experimental native compute offload.**


</div>

---

## Overview

**Iris** is a high-performance actor engine built in Rust with first-class **Python** bindings. It combines three execution styles:
- **Actor Mesh:** Stateful, message-driven workflows with high concurrency.
- **Cross-Language API:** Service-oriented apps mixing Rust and Python.

Iris uses a **cooperative reduction-based scheduler** for fairness, providing built-in supervision, hot swapping, discovery, and location-transparent messaging across nodes.

---

## Core Capabilities

- **Hybrid Concurrency:** Mix "Push" green-thread actors with "Pull" OS-thread actors.
- **Atomic Hot-Swap:** Update live application logic (Python) with zero downtime.
- **Global Discovery:** Register and resolve named services locally or over the network.
- **Resilience:** Built-in supervision, load shedding, and network-aware exit reasons for self-healing systems.
- **Observability:** Deep runtime introspection, system metrics, and actor health tracking.
- **Async Request/Reply:** Built-in `call()` semantics for ergonomic, awaitable actor communication.

---

## Quick Start

### Installation

#### 🐍 Python
```bash
pip install maturin
maturin develop --release
```

### Basic Example (Python)

```python
import iris
rt = iris.Runtime()

# 1. Spawn a high-performance actor
def worker(msg):
    print(f"Got: {msg}")

pid = rt.spawn(worker, budget=50)

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
- [Distributed Mesh & Discovery](docs/distributed.md)

---

## Disclaimer

> [!IMPORTANT]
> **Production Status:** Iris is currently in **Beta**. 
>
**Performance (v0.5.0):**
- **Messaging (Rust):** ~1.3M+ msgs/sec (Unbounded), ~750k+ msgs/sec (Bounded).
- **Messaging (Python):** ~827k+ msgs/sec (send), ~1.8M+ msgs/sec (send_many).
- **Density:** 100k+ concurrent actors on a 4GB mobile device (~10KB/actor).
- **PID Management:** ~7.8M allocations/sec (Lock-free).
- **See more at:** [Performance Report](docs/benchmarks/PERFORMANCE.md)


---

<div align="center">

**Author:** Seuriin ([SSL-ACTX](https://github.com/SSL-ACTX))

</div>
