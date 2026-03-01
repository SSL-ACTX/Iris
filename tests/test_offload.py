# test_offload.py
import time
import array
from iris.jit import offload

# --- Simple Math (1 argument) ---
def square_normal(x: float) -> float:
    return x * x

@offload(strategy="jit", return_type="float")
def square_jit(x: float) -> float:
    return x * x

# --- Heavy Math (3 arguments) ---
def heavy_normal(x: float, y: float, z: float) -> float:
    return (x * x + y * y + z * z) / (x + y + z + 1.0) * (x - y) + 42.0

@offload(strategy="jit", return_type="float")
def heavy_jit(x: float, y: float, z: float) -> float:
    return (x * x + y * y + z * z) / (x + y + z + 1.0) * (x - y) + 42.0

def main() -> None:
    iterations = 1_000_000
    print(f"Preparing {iterations:,} elements per array...")
    
    # Generate standard Python lists
    py_x = [float(i) for i in range(iterations)]
    py_y = [float(i + 1) for i in range(iterations)]
    py_z = [float(i + 2) for i in range(iterations)]
    
    # Convert to contiguous C-style arrays
    arr_x = array.array('d', py_x)
    arr_y = array.array('d', py_y)
    arr_z = array.array('d', py_z)

    # Warm-up JIT to trigger compilation before the timer starts
    square_jit(1.0)
    heavy_jit(1.0, 2.0, 3.0)

    print("\n--- Simple Math (1 array, 1,000,000 elements) ---")
    
    start = time.perf_counter()
    _ = [square_normal(x) for x in py_x]
    norm_simple_time = time.perf_counter() - start
    print(f"Normal Python : {norm_simple_time:.4f} seconds")

    start = time.perf_counter()
    _ = square_jit(arr_x) 
    jit_simple_time = time.perf_counter() - start
    print(f"Iris JIT      : {jit_simple_time:.4f} seconds")
    print(f"Result        : {norm_simple_time / jit_simple_time:.2f}x faster with JIT")


    print("\n--- Heavy Math (3 arrays, 1,000,000 elements each) ---")
    
    start = time.perf_counter()
    # Using zip to iterate through all three lists simultaneously in standard Python
    _ = [heavy_normal(x, y, z) for x, y, z in zip(py_x, py_y, py_z)]
    norm_heavy_time = time.perf_counter() - start
    print(f"Normal Python : {norm_heavy_time:.4f} seconds")

    start = time.perf_counter()
    # Passing all three C-arrays straight through the FFI boundary
    _ = heavy_jit(arr_x, arr_y, arr_z)
    jit_heavy_time = time.perf_counter() - start
    print(f"Iris JIT      : {jit_heavy_time:.4f} seconds")
    print(f"Result        : {norm_heavy_time / jit_heavy_time:.2f}x faster with JIT")

if __name__ == "__main__":
    main()