# chaos_test.py
import myrmidon
import time
import threading
import gc

rt = myrmidon.Runtime()

def initial_behavior(msg):
    pass

# 1. Spawn a target actor
pid = rt.spawn(initial_behavior, budget=100, release_gil=True)
print(f"🚀 Started Actor {pid} for Chaos Test...")

stop_event = threading.Event()

def behavior_factory(i):
    """Generates a transient function to swap into the actor."""
    def transient_logic(msg):
        # Do a tiny bit of work
        _ = len(msg) + i
    return transient_logic

def flooding_loop():
    """Floods the actor with messages to keep the mailbox busy."""
    while not stop_event.is_set():
        rt.send(pid, b"payload")

def aggressive_swap_loop():
    """Rapidly swaps behavior and manually triggers GC to find pointer bugs."""
    swaps = 0
    try:
        for i in range(10000):
            # Create a function, swap it, and immediately lose the reference
            new_func = behavior_factory(i)
            rt.hot_swap(pid, new_func)
            
            del new_func # Drop the local Python reference
            
            if i % 100 == 0:
                gc.collect() # Force GC to try and reap the function pointer
                print(f"🔄 Swaps: {i} (GC Triggered)", end="\r")
            swaps = i
    finally:
        print(f"\n✅ Finished {swaps} aggressive swaps.")

# 2. Run the chaos
t1 = threading.Thread(target=flooding_loop)
t2 = threading.Thread(target=aggressive_swap_loop)

t1.start()
t2.start()

t2.join()
stop_event.set()
t1.join()

print("🛑 Test complete. If no segfault occurred, the pointer management is robust.")
