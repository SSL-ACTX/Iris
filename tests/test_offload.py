import iris
import time
import math

# simple arithmetic function for JIT comparison
@iris.offload(strategy="jit", return_type="float")
def add(x: float, y: float) -> float:
    return x + y

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start

def heavy_calc(n):
    """A CPU-bound task: Check if a large number is prime."""
    if n < 2: return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def test_offload_performance_gain():
    # Large number to ensure the CPU actually has to work
    test_val = 10**12 + 39 

    # 1. Baseline: Standard Local Execution
    with Timer() as local_timer:
        local_res = heavy_calc(test_val)
    
    # 2. Offloaded: Using Iris Actor Strategy
    @iris.offload(strategy="actor", return_type="bool")
    def offloaded_calc(n):
        if n < 2: return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    with Timer() as offload_timer:
        offload_res = offloaded_calc(test_val)

    # Validation
    assert local_res == offload_res
    
    print(f"\nLocal Execution:  {local_timer.duration:.6f}s")
    print(f"Offload Execution: {offload_timer.duration:.6f}s")
    
    # Note: Depending on your 'iris' implementation (Rust/C++ backend),
    # the first call might have 'warm-up' overhead.
    if offload_timer.duration < local_timer.duration:
        print("🚀 Performance gain detected!")
    else:
        print("🐢 Offload was slower (likely due to serialization/startup overhead).")

    # additional JIT test: simple add should be faster than Python
    with Timer() as jit_timer:
        jit_res = add(1.234567, 2.345678)
    print(f"JIT call result {jit_res}, took {jit_timer.duration:.6f}s")

if __name__ == "__main__":
    test_offload_performance_gain()