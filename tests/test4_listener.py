import myrmidon
import time

rt = myrmidon.Runtime()
rt.listen("127.0.0.1:9000") # Starts the TCP server

def remote_handler(msg):
    print(f"Node A received remote message: {msg.decode()}")

pid = rt.spawn(remote_handler)
print(f"Actor started on Node A with PID: {pid}")

while True:
    time.sleep(1)
