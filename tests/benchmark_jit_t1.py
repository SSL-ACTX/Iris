# benchmark.py
import time
import array
import math
from iris.jit import offload

# --- Complex Math with Functions & Negatives ---
# This tests the new AST parser's ability to handle function calls,
# module namespaces (math.sin), and negative numbers (-1.5).

def wave_normal(x: float, y: float, z: float) -> float:
    return math.sin(x) * math.cos(y) + math.sqrt(z) + -1.5

@offload(strategy="jit", return_type="float")
def wave_jit(x: float, y: float, z: float) -> float:
    return math.sin(x) * math.cos(y) + math.sqrt(z) + -1.5

def main() -> None:
    iterations = 1_000_000
    print(f"Preparing {iterations:,} elements per array...")
    
    # Generate standard Python lists
    # Note: Using absolute value for z to ensure math.sqrt doesn't fault on negative inputs
    py_x = [float(i * 0.1) for i in range(iterations)]
    py_y = [float(i * 0.2) for i in range(iterations)]
    py_z = [float(abs(i * 0.5)) for i in range(iterations)]
    
    # Convert to contiguous C-style arrays
    arr_x = array.array('d', py_x)
    arr_y = array.array('d', py_y)
    arr_z = array.array('d', py_z)

    # Warm-up JIT to trigger Cranelift compilation and C-math linkage
    wave_jit(1.0, 1.0, 1.0)

    print("\n--- Complex Math (Trig, Sqrt, Negatives) ---")
    print("Equation: math.sin(x) * math.cos(y) + math.sqrt(z) + -1.5\n")
    
    start = time.perf_counter()
    _ = [wave_normal(x, y, z) for x, y, z in zip(py_x, py_y, py_z)]
    norm_time = time.perf_counter() - start
    print(f"Normal Python : {norm_time:.4f} seconds")

    start = time.perf_counter()
    _ = wave_jit(arr_x, arr_y, arr_z)
    jit_time = time.perf_counter() - start
    print(f"Iris JIT      : {jit_time:.4f} seconds")
    
    print(f"\nResult: {norm_time / jit_time:.2f}x faster with JIT")

if __name__ == "__main__":
    main()