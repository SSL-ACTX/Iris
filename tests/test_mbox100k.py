import myrmidon
import time
import sys

# Configuration
# NOTE: Mailbox actors use dedicated OS threads (via Tokio's blocking pool).
# The default Tokio limit is usually 512 blocking threads.
# Setting this to 100,000 WILL deadlock standard runtimes.
COUNT = 55_000  
BATCH_SIZE = 5000

print(f"--- Myrmidon Mailbox Stress Test: {myrmidon.version()} ---")
print(f"Goal: Spawn {COUNT} threaded mailbox actors.")

# Instantiate the runtime
rt = myrmidon.PyRuntime()

# A blocking handler that loops until the channel is closed.
# This mimics a standard "Erlang-style" actor loop.
def mailbox_handler(mailbox):
    while True:
        # Blocks the underlying thread until a message arrives.
        # Returns None if the actor is stopped (channel closed).
        msg = mailbox.recv()
        
        if msg is None:
            break
            
        # Do nothing with the message (simulate work)
        pass

pids = []
start_time = time.time()

try:
    print("🚀 Spawning (Threaded)...", flush=True)
    for i in range(COUNT):
        # Spawn a threaded actor.
        # This allocates a new buffer/channel and schedules a blocking task.
        pid = rt.spawn_with_mailbox(mailbox_handler, budget=10)
        pids.append(pid)

        if (i + 1) % BATCH_SIZE == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"   [{i + 1}/{COUNT}] Spawned. Rate: {rate:.0f} actors/sec", flush=True)

    spawn_time = time.time() - start_time
    print(f"✅ Spawned {COUNT} actors in {spawn_time:.2f}s ({COUNT / spawn_time:.0f} actors/sec)")

    # Phase 2: Message Blitz
    # Even though they are in threads, sending is still just pushing to a channel.
    print(f"📨 Sending {COUNT} messages...", flush=True)
    msg_start = time.time()

    for pid in pids:
        rt.send(pid, b"ping")

    msg_time = time.time() - msg_start
    print(f"✅ Sent {COUNT} messages in {msg_time:.2f}s ({COUNT / msg_time:.0f} msgs/sec)")

    print("🛑 Cleaning up (stopping all actors)...")
    # This closes the channels. The 'mailbox.recv()' calls in the threads
    # will return None, causing the loops to break and threads to exit.
    for pid in pids:
        rt.stop(pid)

    print("Done.")

except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user.")
except Exception as e:
    print(f"\n❌ Failed: {e}")
