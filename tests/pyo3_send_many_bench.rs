use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn bench_send_vs_send_many() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", &rt).unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
import statistics
import time

def bench_send(rt, count=1000):
    def empty_handler(msg):
        pass
    
    pid = rt.spawn(empty_handler)
    msg = b"hello"
    
    start = time.perf_counter_ns()
    for _ in range(count):
        rt.send(pid, msg)
    end = time.perf_counter_ns()
    
    rt.stop(pid)
    return (end - start) / count

def bench_send_many(rt, count=1000):
    def empty_handler(msg):
        pass
    
    pid = rt.spawn(empty_handler)
    msg = b"hello"
    batch = [msg] * count
    
    start = time.perf_counter_ns()
    rt.send_many(pid, batch)
    end = time.perf_counter_ns()
    
    rt.stop(pid)
    return (end - start) / count

# Warmup
bench_send(rt, 100)
bench_send_many(rt, 100)

# Real bench
times_single = [bench_send(rt, 1000) for _ in range(5)]
times_many = [bench_send_many(rt, 1000) for _ in range(5)]

avg_single = statistics.mean(times_single)
avg_many = statistics.mean(times_many)
speedup = avg_single / avg_many

print(f"\n[Bench] Average send: {avg_single:.2f}ns")
print(f"[Bench] Average send_many (amortized): {avg_many:.2f}ns")
print(f"[Bench] Speedup: {speedup:.2f}x")

# We expect at least some speedup due to reduced GIL acquisitions and overhead
# in the high-frequency send_many path.
assert speedup > 1.0, f"send_many was not faster (speedup={speedup:.2f}x)"

bench_speedup = speedup
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        Ok::<(), PyErr>(())
    })
    .unwrap();
}
