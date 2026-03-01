# RFC: Iris Transparent Rust Offloading (`@iris.offload`)
> **Chosen: cranelift

## 1. Abstract

This proposal outlines the implementation of a new feature for the Iris runtime: a zero-friction compute offloading mechanism. By utilizing an `@iris.offload` decorator, developers can write mathematical or CPU-bound functions in pure Python, which the Iris runtime will intercept, compile or bind to Rust under the hood, and execute outside the Python Global Interpreter Lock (GIL).

## 2. Motivation

Iris currently excels at concurrency, bridging memory-safe, ultra-concurrent Rust operations with Python's flexibility. Features like `spawn_with_mailbox` and `release_gil=True` allow blocking OS threads to bypass the GIL for synchronous logic or I/O. However, heavy mathematical computations written in Python still suffer from interpreter overhead.

To maintain location transparency and developer velocity, developers shouldn't need to manually write PyO3 boilerplate or manage C-extensions for simple math. We need a way for Rust to "steal" the computation directly from the Python definition.

---

## 3. Proposed API

The developer experience remains entirely in Python. The decorator signals the runtime to analyze the function and route its execution to a Rust backend.

```python
import iris

# The decorator intercepts the function definition at module load.
@iris.offload(strategy="jit", return_type="float")
def compute_vector_magnitude(x: float, y: float, z: float):
    return (x**2 + y**2 + z**2) ** 0.5

# Usage remains identical to standard Python
result = compute_vector_magnitude(10.5, 20.1, 8.4)

```

---

## 4. Architectural Approaches

To make "Rust steal the math," we have two distinct architectural paths to implement under the `@iris.offload` hood.

### Approach A: The AST-to-Rust JIT (The "Holy Grail")

This approach parses the Python function, translates the abstract syntax tree (AST) to a Rust intermediate representation, and compiles it.

* **Inspection:** At import time, `@iris.offload` uses Python's `inspect.getsource()` to extract the function's code.
* **AST Parsing:** Python's `ast` module parses the code into a syntax tree.
* **Rust Compilation:** The AST is passed to the Iris Rust core. A lightweight JIT backend (e.g., Cranelift) translates the limited subset of Python math operations into native machine code.
* **Pointer Swap:** The Python decorator replaces the original callable with a PyO3-wrapped function pointer pointing directly to the compiled native code.

### Approach B: Actor-Pool FFI Routing (The Pragmatic Stepping Stone)

Instead of dynamic compilation, we leverage Iris's existing actor primitives to route workloads to pre-compiled Rust compute nodes.

* **Pre-compiled Primitives:** The Iris Rust backend ships with highly optimized math primitives (e.g., matrix multiplication, vector ops).
* **Zero-Copy Messaging:** The decorator serializes the inputs and performs a zero-copy send using a pre-allocated Buffer ID to a dedicated Rust worker pool.
* **GIL-Free Execution:** The execution happens entirely in Rust OS threads, completely freeing the Python async worker.

---

## 5. Comparison of Strategies

| Feature | Approach A (AST JIT) | Approach B (Actor Routing) |
| --- | --- | --- |
| **Performance** | Maximum (Native machine code) | High (Rust FFI + Zero-copy overhead) |
| **Supported Operations** | Strict subset (Math/Variables only) | Limited to pre-compiled Rust library |
| **Implementation Complexity** | Extremely High (Requires compiler engineering) | Low (Reuses existing Iris architecture) |
| **GIL Behavior** | Completely bypassed | Bypassed via Actor messaging |

---

## 6. Execution Plan

**Phase 1: Validation & Actor Routing**

1. Implement the `@iris.offload(strategy="actor")` decorator.
2. Build a dedicated pull-actor pool in Rust specifically for CPU-bound tasks to ensure cooperative reduction scheduler budgets aren't exhausted by heavy math.
3. Establish PyO3 bindings for zero-copy argument serialization.

**Phase 2: The JIT Prototype**

1. Define the restricted subset of Python AST that the JIT will support (e.g., `BinaryOp`, `UnaryOp`, `Assign`, `Return`). Type hinting will be mandatory for JIT parsing.
2. Integrate a Cranelift backend into the Iris Rust crate.
3. Map Python AST nodes to Cranelift IR.

**Phase 3: Fallback Mechanisms**

1. Implement automatic fallback. If the JIT parser encounters an unsupported Python construct (like an external library call or I/O), it silently falls back to standard Python execution and logs a warning.