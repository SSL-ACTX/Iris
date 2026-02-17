# test4b.py
import myrmidon
import time

rt = myrmidon.Runtime()
# Target the ACTUAL PID from Node A's output
target_pid = 4294967296

print(f"Sending message to Node A (PID {target_pid})...")
rt.send_remote("127.0.0.1:9000", target_pid, b"Hello from Node B!")

# Wait briefly for the background thread to finish the TCP send
time.sleep(0.5)
print("Message sent.")
