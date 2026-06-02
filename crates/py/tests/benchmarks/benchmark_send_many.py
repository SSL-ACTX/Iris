import statistics
import time

import iris

TOTAL_MESSAGES = 100_000
BATCH_SIZE = 2048
ROUNDS = 5
PAYLOAD = b"ping"


def run_send(rt, pid):
    t0 = time.perf_counter()
    for _ in range(TOTAL_MESSAGES):
        if not rt.send(pid, PAYLOAD):
            raise RuntimeError("send failed")
    return time.perf_counter() - t0


def run_send_many(rt, pid):
    full_batches = TOTAL_MESSAGES // BATCH_SIZE
    remainder = TOTAL_MESSAGES % BATCH_SIZE
    batch = [PAYLOAD] * BATCH_SIZE

    t0 = time.perf_counter()
    for _ in range(full_batches):
        accepted = rt.send_many(pid, batch)
        if accepted != BATCH_SIZE:
            raise RuntimeError(
                f"send_many failed: accepted={accepted} expected={BATCH_SIZE}"
            )

    if remainder:
        accepted = rt.send_many(pid, [PAYLOAD] * remainder)
        if accepted != remainder:
            raise RuntimeError(
                f"send_many remainder failed: accepted={accepted} expected={remainder}"
            )

    return time.perf_counter() - t0


def main():
    print("--- Iris PyRuntime send vs send_many ---")
    print(f"messages={TOTAL_MESSAGES:,}, batch_size={BATCH_SIZE}, rounds={ROUNDS}")

    rt = iris.PyRuntime()

    # No-op Python handler: keeps actor alive without storing all payloads.
    def sink(_msg):
        return None

    pid = rt.spawn_py_handler(sink, 1024, False)

    # Warmup
    run_send(rt, pid)
    run_send_many(rt, pid)

    send_times = []
    send_many_times = []

    for i in range(ROUNDS):
        t_send = run_send(rt, pid)
        t_many = run_send_many(rt, pid)
        send_times.append(t_send)
        send_many_times.append(t_many)
        print(
            f"round {i + 1}: send={t_send:.6f}s ({TOTAL_MESSAGES / t_send:,.0f} msg/s) | "
            f"send_many={t_many:.6f}s ({TOTAL_MESSAGES / t_many:,.0f} msg/s)"
        )

    send_med = statistics.median(send_times)
    many_med = statistics.median(send_many_times)
    speedup = send_med / many_med if many_med > 0 else float("inf")

    print("\n--- Median ---")
    print(f"send      : {send_med:.6f}s ({TOTAL_MESSAGES / send_med:,.0f} msg/s)")
    print(f"send_many : {many_med:.6f}s ({TOTAL_MESSAGES / many_med:,.0f} msg/s)")
    print(f"speedup   : {speedup:.2f}x")

    rt.stop(pid)


if __name__ == "__main__":
    main()
