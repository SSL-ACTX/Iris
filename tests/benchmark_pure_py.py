import asyncio
import time
import sys

# Config to match your Myrmidon test
COUNT = 100000

async def actor(queue):
    """
    Standard asyncio worker.
    Waits for a message on its private queue.
    """
    while True:
        msg = await queue.get()
        if msg is None:  # Poison pill to stop
            break
        # Process message (no-op)
        pass

async def main():
    print(f"--- Pure Asyncio Stress Test ---")
    print(f"Goal: Spawn {COUNT} tasks (Queues + Workers).")
    print(f"Python Version: {sys.version.split()[0]}")
    
    queues = []
    tasks = []
    
    t0 = time.time()
    
    # --- 1. SPAWN PHASE ---
    for i in range(COUNT):
        # In pure Python, an "actor" is a Queue + a Coroutine Task
        q = asyncio.Queue()
        t = asyncio.create_task(actor(q))
        queues.append(q)
        tasks.append(t)
        
        if (i + 1) % 10000 == 0:
             elapsed = time.time() - t0
             rate = (i + 1) / elapsed
             print(f"   [{i + 1}/{COUNT}] Spawned. Rate: {rate:.0f} actors/sec")
             # Yield briefly so we don't freeze the printing
             await asyncio.sleep(0)

    t_spawn = time.time() - t0
    print(f"✅ Spawned {COUNT} actors in {t_spawn:.2f}s ({COUNT/t_spawn:.0f} actors/sec)")

    # --- 2. MESSAGE PHASE ---
    print(f"📨 Sending {COUNT} messages...")
    t1 = time.time()
    
    # Simulate "rt.send(pid, msg)" by putting into the specific queue
    for q in queues:
        q.put_nowait(b"test")
    
    # We force a context switch to ensure tasks actually get a chance to wake up 
    # and process, mimicking the scheduler work.
    await asyncio.sleep(0)

    t_send = time.time() - t1
    print(f"✅ Sent {COUNT} messages in {t_send:.2f}s ({COUNT/t_send:.0f} msgs/sec)")
    
    # --- 3. CLEANUP PHASE ---
    print("🛑 Cleaning up (stopping all actors)...")
    for q in queues:
        q.put_nowait(None)
    
    await asyncio.gather(*tasks)
    print("Done.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass