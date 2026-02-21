# Contributing to Iris

Thank you for your interest in contributing to **Iris**! We are building a high-performance, language-agnostic, BEAM-class foundation for distributed systems. To maintain the project's performance and reliability, we follow strict guidelines for development.

## 🏁 Guiding Principles

Before contributing, please keep our core philosophy in mind:
* **Runtime First, Language Second**: Rust owns the scheduling, memory, safety, and correctness. Guest languages (Python, Node.js) are users of the runtime.
* **No Shared-Mutable State**: We forbid shared mutable state across actors to ensure safety.
* **Non-Blocking Core**: Blocking APIs are strictly prohibited in the core runtime to preserve microsecond-latency profiles.
* **Mechanical Sympathy**: We optimize for cache locality and minimal lock contention.

## 🛠️ Development Environment

### Prerequisites
* **Rust**: Latest stable version (installed via `rustup`).
* **Python**: 3.8+ (3.10+ recommended).
* **Maturin**: For building Python bindings.

### Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/seuriin/iris.git
    cd iris
    ```
2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Build the project in development mode**:
    ```bash
    maturin develop
    ```

## 🧪 Testing and Benchmarks

All contributions must pass existing stress tests and benchmarks. We rely on extreme boundary testing to validate the leanness of the architecture.

### Mandatory Feature Testing
For every new feature or significant update, you must ensure the following tests are included:
1.  **Rust Integration Tests**: You must write a `pyo3_test_*` file in the `tests/` directory (e.g., `tests/pyo3_test_my_feature.rs`). This test must validate the FFI boundary, ensuring the Rust core correctly exposes the feature to the Python runtime.
2.  **Python-Side Tests**: You must provide a Python script (typically in `tests/` or `xtest/`) that exercises the feature through the high-level `iris` Python API. This ensures end-to-end functionality from the perspective of a library user.

### Validation Procedures
* **Standard Stress Tests**: Run the 100k actor spawn tests to ensure no regressions in throughput.
* **Chaos Tests**: Use the `run.py` orchestrator to validate that remote monitoring and self-healing logic still handle node failures correctly.
* **Benchmarking**: If modifying the scheduler or network layer, run the benchmarks on your local hardware and compare against the baseline metrics (e.g., the ~563k msgs/sec threshold on single-core setups).

## 📜 Coding Standards

### Rust (The Core)
* **Format**: Use `cargo fmt`.
* **Lints**: Use `cargo clippy`.
* **Safety**: Avoid `unsafe` unless absolutely necessary for FFI or performance-critical operations that cannot be expressed safely.

### Python (The Bindings)
* **API Consistency**: Use the `Runtime` wrapper class defined in `iris/__init__.py` to provide a clean Pythonic API.
* **GIL Management**: Ensure any blocking Python operations use `release_gil=True` when spawning actors to prevent stalling the async worker.

## 📤 Pull Request Process

1.  **Fork the repo** and create your branch from `main`.
2.  **Ensure your branch builds** and passes all integration tests (including the mandatory new ones).
3.  **Update documentation**: If you change an API or add a feature, update the relevant `README.md` or `FEAT.md` entries.
4.  **Submit the PR**: Provide a clear description of the changes and any performance impacts observed.
