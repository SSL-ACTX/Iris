# tests/test3.py
import myrmidon
import time

print("--- Phase 4: Hot Code Swapping Test ---")

rt = myrmidon.Runtime()

def behavior_a(msg):
    print(f"[A] Received: {msg.decode()}")

def behavior_b(msg):
    print(f"[B] Received: {msg.decode()} (UPGRADED)")

# 1. Spawn with A
print("Spawning actor with Behavior A...")
pid = rt.spawn(behavior_a)
rt.send(pid, b"msg1")
time.sleep(0.1)

# 2. Hot Swap to B
print("Attempting Hot Swap...")
rt.hot_swap(pid, behavior_b)
print("✅ Hot Swap signal sent.")

# 3. Verify B
print("Sending second message...")
rt.send(pid, b"msg2")
time.sleep(0.1)

rt.stop(pid)
rt.join(pid)
print("Test complete.")
