import myrmidon
import time
import sys

# Configuration
COUNT = 100_000  # The Goal
BATCH_SIZE = 10000

print(f"--- Myrmidon Stress Test: {myrmidon.version()} ---")
print(f"Goal: Spawn {COUNT} actors.")

rt = myrmidon.Runtime()

# A minimal handler that does nothing (to test pure overhead)
def no_op_handler(msg):
    pass

pids = []
start_time = time.time()

try:
    print("🚀 Spawning...", flush=True)
    for i in range(COUNT):
        # Spawn actor with a small budget since they are tiny
        pid = rt.spawn(no_op_handler, budget=10)
        pids.append(pid)

        if (i + 1) % BATCH_SIZE == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"   [{i + 1}/{COUNT}] Spawned. Rate: {rate:.0f} actors/sec", flush=True)

    spawn_time = time.time() - start_time
    print(f"✅ Spawned {COUNT} actors in {spawn_time:.2f}s ({COUNT / spawn_time:.0f} actors/sec)")

    # Phase 2: Message Blitz
    # We send a message to EVERY actor.
    # This floods the Tokio runtime and forces 100k GIL acquisitions.
    print(f"📨 Sending {COUNT} messages...", flush=True)
    msg_start = time.time()

    for pid in pids:
        rt.send(pid, b"ping")

    msg_time = time.time() - msg_start
    print(f"✅ Sent {COUNT} messages in {msg_time:.2f}s ({COUNT / msg_time:.0f} msgs/sec)")

    print("🛑 Cleaning up (stopping all actors)...")
    for pid in pids:
        rt.stop(pid)

    print("Done.")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user.")
except Exception as e:
    print(f"\n❌ Failed: {e}")
