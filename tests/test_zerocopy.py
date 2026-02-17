import myrmidon
import time

rt = myrmidon.Runtime()

# 1. Allocate a 10MB buffer in Rust (No Python memory usage yet)
buffer_size = 10 * 1024 * 1024  # 10 MB
print(f"Allocating {buffer_size / 1024 / 1024:.0f} MB buffer...")

# returns (id, memoryview, capsule)
(buf_id, view, cap) = myrmidon.allocate_buffer(buffer_size)

# Write to it from Python (Direct memory access, no copies)
view[0:5] = b"HEAVY"

def heavy_handler(msg):
    # Proof we got the data (first 5 bytes)
    if msg[0:5] == b"HEAVY":
        pass

pid = rt.spawn(heavy_handler)

print("Sending 10MB payload via Zero-Copy...")
start = time.time()

# 2. Send the ID. Rust takes the pointer. NO COPY happens here.
rt.send_buffer(pid, buf_id)

# 3. CRITICAL: Tell the actor to stop so we can measure the finish time
rt.stop(pid)

# 4. Wait for actor to finish processing and exit
rt.join(pid)
end = time.time()

print(f"✅ Transfer complete in {end - start:.4f}s")
