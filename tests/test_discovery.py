# tests/test_discovery.py
import myrmidon
import time
import threading

def run_server():
    """Simulates Node A (The Provider)"""
    rt_a = myrmidon.Runtime()
    rt_a.listen("127.0.0.1:9000")

    def auth_handler(msg):
        print(f"\n[Node A] Auth Service received: {msg.decode()}")

    pid = rt_a.spawn(auth_handler)
    rt_a.register("auth-service", pid)
    print(f"[Node A] Listening on 127.0.0.1:9000...")
    print(f"[Node A] 'auth-service' registered with PID: {pid}")

    # Keep the server thread alive
    while True:
        time.sleep(1)

# 1. Start Node A in a background thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Give the server a moment to bind the port
time.sleep(1)

# 2. Setup Node B (The Client)
print("--- Starting Remote Discovery Test ---")
rt_b = myrmidon.Runtime()
node_a_addr = "127.0.0.1:9000"

print(f"[Node B] Querying {node_a_addr} for 'auth-service'...")

# 3. Remote Resolution
pid = rt_b.resolve_remote(node_a_addr, "auth-service")

if pid:
    print(f"[Node B] ✅ Resolved 'auth-service' to PID: {pid}")

    # 4. Send a message to the discovered remote PID
    print("[Node B] Sending credentials...")
    rt_b.send_remote(node_a_addr, pid, b"USER:seuriin_PASS:rust_is_awesome")

    # Small delay to ensure the network packet is processed before script exit
    time.sleep(0.5)
    print("[Node B] Test complete.")
else:
    print("[Node B] ❌ Failed: Could not resolve service.")
