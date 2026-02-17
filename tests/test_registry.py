import myrmidon
import time

rt = myrmidon.Runtime()

def my_handler(msg):
    print(f"Service received: {msg.decode()}")

# Spawn and give it a name
pid = rt.spawn(my_handler)
rt.register("auth-service", pid)

# Later, in a different part of the code (or even a different node)
target_pid = rt.resolve("auth-service")
if target_pid:
    rt.send(target_pid, b"Login Request")

time.sleep(0.1)
rt.stop(pid)
