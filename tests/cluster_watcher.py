# cluster_watcher.py
import myrmidon
import time

rt = myrmidon.Runtime()

# Node A details (change these based on your actual test4.py output)
NODE_A_ADDR = "127.0.0.1:9000"
NODE_A_PID = 4294967296

print(f"--- Cluster Guardian Starting ---")
print(f"Monitoring {NODE_A_ADDR} (PID: {NODE_A_PID})...")

# Start monitoring the remote process
rt.monitor_remote(NODE_A_ADDR, NODE_A_PID)

try:
    while True:
        # In a real app, you could check rt.is_alive(local_proxy_pid)
        # or wait for a supervisor callback.
        time.sleep(1)
except KeyboardInterrupt:
    print("Guardian shutting down.")
