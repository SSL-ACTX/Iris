# benchmark.py
import time
import array
from iris.jit import offload

def square_normal(x: float) -> float:
    return x * x

@offload(strategy="jit", return_type="float")
def square_jit(x: float) -> float:
    return x * x

def main() -> None:
    iterations = 1_000_000
    print(f"Preparing {iterations:,} elements...")
    
    # Standard Python list of floats
    py_list = [float(i) for i in range(iterations)]
    
    # Contiguous C-style array of doubles ('d' represents f64)
    # This natively exposes the C-buffer protocol to our Rust backend.
    c_array = array.array('d', py_list)

    # Warm-up JIT
    square_jit(1.0)

    print("\n--- Batch Processing (1,000,000 elements) ---")
    
    # 1. Standard Python Execution
    start = time.perf_counter()
    # Using a list comprehension to measure Python's fastest native loop
    _ = [square_normal(x) for x in py_list]
    norm_time = time.perf_counter() - start
    print(f"Normal Python (List Comprehension): {norm_time:.4f} seconds")

    # 2. Iris JIT Vectorized Offload
    start = time.perf_counter()
    # Pass the entire array at once. The FFI boundary is crossed exactly ONE time.
    _ = square_jit(c_array) 
    jit_time = time.perf_counter() - start
    print(f"Iris JIT (Vectorized Buffer)      : {jit_time:.4f} seconds")
    
    print(f"\nResult: {norm_time / jit_time:.2f}x faster with JIT")

if __name__ == "__main__":
    main()