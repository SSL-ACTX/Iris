import myrmidon
import time
import random

# Configuration
SWAP_COUNT = 5000       # Total times we swap the behavior
MSG_COUNT = 20000       # Total messages to send during swaps
rt = myrmidon.PyRuntime()

print(f"--- Myrmidon Hot Swap Stress Test: {myrmidon.version()} ---")

# Define two distinct behaviors to swap between
def logic_a(msg):
    # Intentional no-op or simple math to keep it fast
    _ = msg

def logic_b(msg):
    # Slightly different path
    _ = len(msg)

# 1. Spawn the initial actor (Push-based for maximum FFI speed)
pid = rt.spawn_py_handler(logic_a, budget=100)

print(f"🚀 Spawning actor {pid}...")
print(f"🔥 Starting simultaneous Send/Swap blitz ({SWAP_COUNT} swaps, {MSG_COUNT} msgs)...")

start_time = time.time()

# We use two threads to create a race condition between sending and swapping
def sender_loop():
    for _ in range(MSG_COUNT):
        rt.send(pid, b"test_payload")
        # Tiny sleep to prevent saturating the GIL immediately
        if _ % 1000 == 0:
            time.sleep(0.001)

def swapper_loop():
    for i in range(SWAP_COUNT):
        # Rapidly toggle between behaviors
        new_logic = logic_b if i % 2 == 0 else logic_a
        rt.hot_swap(pid, new_logic)
        
        if (i + 1) % 500 == 0:
            print(f"   [Swap Progress] {i + 1}/{SWAP_COUNT} swaps completed...", end="\r")

# 2. Run the stress test
import threading
t1 = threading.Thread(target=sender_loop)
t2 = threading.Thread(target=swapper_loop)

t1.start()
t2.start()

t1.join()
t2.join()

total_time = time.time() - start_time

print(f"\n\n✅ Benchmark Complete!")
print(f"   Total Time: {total_time:.2f}s")
print(f"   Swap Rate: {SWAP_COUNT / total_time:.0f} swaps/sec")
print(f"   Message Rate: {MSG_COUNT / total_time:.0f} msgs/sec")

# 3. Final Verification
print("🛑 Stopping actor...")
rt.stop(pid)
print("Done.")
