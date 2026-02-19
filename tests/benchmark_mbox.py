import myrmidon
import time
import threading

# Configuration
TOTAL_GOAL = 10_000   # We will run this many TOTAL
CONCURRENCY_LIMIT = 500  # But only this many AT ONCE

print(f"--- Myrmidon Bounded Mailbox Test ---")
rt = myrmidon.PyRuntime()

# A semaphore to track available 'slots' in our thread pool
pool_sem = threading.Semaphore(CONCURRENCY_LIMIT)

def worker(mailbox):
    # Simulate work (blocking the thread)
    # The 'recv' here releases the GIL, so other threads can run.
    msg = mailbox.recv(timeout=0.1) 
    
    # When this function returns, the thread is freed.
    # We release the semaphore to let the main loop spawn another.
    pool_sem.release()

start_time = time.time()
active_pids = []

print(f"🚀 Running {TOTAL_GOAL} actors (max {CONCURRENCY_LIMIT} concurrent)...")

try:
    for i in range(TOTAL_GOAL):
        # 1. Acquire a slot. If 500 are running, this BLOCKS until one finishes.
        pool_sem.acquire()
        
        # 2. Spawn the actor
        pid = rt.spawn_with_mailbox(worker, budget=10)
        active_pids.append(pid)
        
        if (i + 1) % 100 == 0:
            print(f"   Spawned {i + 1}/{TOTAL_GOAL}...", end="\r", flush=True)

    # Wait for the last batch to finish
    for _ in range(CONCURRENCY_LIMIT):
        pool_sem.acquire()

    total_time = time.time() - start_time
    print(f"\n✅ Success! Ran {TOTAL_GOAL} threaded actors in {total_time:.2f}s")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted.")
