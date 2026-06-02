# Performance Benchmarks

This document records the performance metrics of Iris after the low-level optimization refactoring (lock-free structures and cache-line alignment).

## Benchmark Environment

*   **Device**: Infinix X669
*   **OS**: Android (Linux localhost 5.4.254-android12)
*   **Architecture**: aarch64
*   **Memory**: 4GB RAM (Total)
*   **Toolchain**: Rust 1.8x (Release Mode)
*   **Date**: June 2, 2026

## Results

The following metrics were obtained using `criterion` in release mode. Results are consistent across multiple runs with less than 3% variance.

| Metric | Average Latency | Throughput (est) |
| :--- | :--- | :--- |
| **PID Allocation** | **127.7 ns** | ~7.8 Million ops/sec |
| **Unbounded Messaging** | **769.9 ns** | ~1.3 Million msg/sec |
| **Bounded Messaging** | **1.33 µs** | ~750k msg/sec |

## Python Latency (Ping-Pong)

Measured using `benchmark_ping_pong.py` (Round-trip between two actors).

| Mode | Median Latency | Operations/sec (est) |
| :--- | :--- | :--- |
| **Pull (Mailbox)** | **32.2 µs** | ~31k RT/sec |
| **Push (Callback)** | **70.0 µs** | ~14k RT/sec |

*Note: Pull mode is significantly faster in Python as it allows the worker thread to loop tightly on `recv()`, minimizing GIL acquisition overhead compared to per-message callback dispatch.*

## Python Throughput

Measured using `benchmark_throughput.py` (Sending from Python to a Rust sink actor).

| Operation | Median Throughput | Latency (est) |
| :--- | :--- | :--- |
| **rt.send** | **827k msg/s** | ~1.2 µs |
| **rt.send_many** | **1.87M msg/s** | ~535 ns |

*Note: `send_many` achieves higher throughput by batching multiple messages into a single FFI call, significantly reducing GIL overhead.*

## Actor Density (Spawn Limit)

Measured using `benchmark_spawn_limit.py` on the Infinix X669 (4GB RAM).

| Metric | Value |
| :--- | :--- |
| **Stable Capacity** | **100,000 actors** |
| **Memory usage (100k)** | **~1.1 GB RSS** |
| **Peak Density (Observed)** | **150,000 actors** (Killed at 1.6GB RSS) |
| **Spawn Rate** | **~60,000 actors/sec** (Initial) |

*Note: The system maintains high spawn rates (60k/s) until memory pressure causes OS-level paging. Each actor occupies approximately 10-12 KB of RAM (including Python callback overhead and Rust mailbox state).*

## Optimization Impact

### Lock-Free PID Management
By replacing `Mutex<Vec<u32>>` with a lock-free `SegQueue` and an `AtomicUsize` for generational indices, we achieved sub-200ns allocation. This eliminates hot-path contention during high-frequency actor spawning and recycling.

### Cache-Line Alignment
Using `#[repr(align(64))]` and `Aligned<T>` wrappers on the `Runtime` struct prevented **False Sharing**. This ensures that concurrent access to different fields (e.g., `mailboxes` and `telemetry`) from multiple OS threads does not cause CPU cache invalidation loops.

### Zero-Copy Message Passing
The integration of `BufferRegistry` and `Bytes` allows for zero-copy binary messaging between actors. The messaging latency (~770ns) includes:
1.  `DashMap` lookup for the sender PID.
2.  Telemetry logging (Atomic increment).
3.  Atomic counter updates (Aligned).
4.  MPSC channel enqueue.

## How to Reproduce

Run the following command from the project root:

```bash
cargo bench -p iris-core --bench throughput
```

Note: Performance may vary based on CPU frequency scaling and thermal throttling on mobile devices.
