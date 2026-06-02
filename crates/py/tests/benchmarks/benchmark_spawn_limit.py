import iris
import time
import os
import sys


def get_rss():
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return line.split()[1] + " " + line.split()[2]
    except:
        return "N/A"
    return "N/A"


def main():
    target = 300_000
    batch_size = 10_000

    rt = iris.PyRuntime()
    rt.set_system_capacity(target + 100)

    print(f"--- Iris High-Density Actor Spawn Test ---")
    print(f"Target: {target:,} actors")
    print(f"Initial RSS: {get_rss()}")

    def no_op(msg):
        pass

    pids = []
    t0 = time.time()

    try:
        for i in range(0, target, batch_size):
            t_batch_0 = time.time()
            for _ in range(batch_size):
                # We use spawn_py_handler which is the fastest path for Python
                # release_gil=False avoids thread creation overhead per actor
                pid = rt.spawn(no_op, budget=10, release_gil=False)
                pids.append(pid)

            t_batch_1 = time.time()
            elapsed_total = t_batch_1 - t0
            current_count = i + batch_size
            rate = batch_size / (t_batch_1 - t_batch_0)

            print(
                f"   [{current_count:,}/{target:,}] RSS: {get_rss()} | Batch Rate: {rate:,.0f} actors/s | Total Time: {elapsed_total:.2f}s",
                flush=True,
            )

            # Brief sleep to allow system to breathe/collect telemetry
            time.sleep(0.1)

        print(f"\n✅ SUCCESS! Reached {target:,} actors.")
        print(f"Final RSS: {get_rss()}")
        print(f"Total time: {time.time() - t0:.2f}s")

    except Exception as e:
        print(f"\n❌ FAILED at {len(pids):,} actors.")
        print(f"Error: {e}")
        print(f"Last RSS: {get_rss()}")
    except KeyboardInterrupt:
        print(f"\n⚠️ INTERRUPTED at {len(pids):,} actors.")
        print(f"Last RSS: {get_rss()}")
    finally:
        print("🛑 Cleaning up (stopping all actors)...")
        # Stopping many actors can take time
        stop_t0 = time.time()
        for pid in pids:
            rt.stop(pid)
        print(f"Cleanup took {time.time() - stop_t0:.2f}s")


if __name__ == "__main__":
    main()
