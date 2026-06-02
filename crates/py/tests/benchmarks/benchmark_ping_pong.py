import time
import statistics
import iris
import argparse


def bench_push_ping_pong(rt, rounds=10000):
    """
    Two push actors sending messages back and forth.
    """
    results = []

    # We use a mutable list to share state with the closure
    state = {"count": 0, "start_time": 0, "done": False, "target": None}

    def pong_handler(msg):
        rt.send(state["target"], msg)

    def ping_handler(msg):
        state["count"] += 1
        if state["count"] >= rounds:
            state["latency"] = time.perf_counter() - state["start_time"]
            state["done"] = True
        else:
            rt.send(state["pong_pid"], msg)

    pong_pid = rt.spawn(pong_handler, budget=10)
    ping_pid = rt.spawn(ping_handler, budget=10)

    state["target"] = ping_pid
    state["pong_pid"] = pong_pid

    state["start_time"] = time.perf_counter()
    rt.send(pong_pid, b"ping")

    while not state["done"]:
        time.sleep(0.001)

    avg_rt = (state["latency"] / rounds) * 1_000_000

    rt.stop(ping_pid)
    rt.stop(pong_pid)

    return avg_rt


def bench_pull_ping_pong(rt, rounds=10000):
    """
    Two pull actors (mailbox) sending messages back and forth.
    """

    def pong_worker(mailbox):
        while True:
            msg = mailbox.recv()
            if msg is None:
                break
            # The message doesn't contain the sender PID in this simple API
            # so we'd need to bake it in or use a global.
            # For ping-pong we just send back to a known PID.
            rt.send(state["ping_pid"], msg)

    def ping_worker(mailbox):
        state["start_time"] = time.perf_counter()
        rt.send(state["pong_pid"], b"ping")
        for _ in range(rounds):
            msg = mailbox.recv()
            if msg is None:
                break
            if _ < rounds - 1:
                rt.send(state["pong_pid"], msg)
        state["latency"] = time.perf_counter() - state["start_time"]
        state["done"] = True

    state = {
        "done": False,
        "start_time": 0,
        "latency": 0,
        "ping_pid": None,
        "pong_pid": None,
    }

    pong_pid = rt.spawn_with_mailbox(pong_worker, budget=10)
    state["pong_pid"] = pong_pid

    ping_pid = rt.spawn_with_mailbox(ping_worker, budget=10)
    state["ping_pid"] = ping_pid

    while not state["done"]:
        time.sleep(0.01)

    avg_rt = (state["latency"] / rounds) * 1_000_000

    rt.stop(ping_pid)
    rt.stop(pong_pid)

    return avg_rt


def main():
    parser = argparse.ArgumentParser(description="Iris Python Ping-Pong Benchmark")
    parser.add_argument(
        "--rounds", type=int, default=10000, help="Number of round trips"
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of samples to take"
    )
    args = parser.parse_args()

    rt = iris.PyRuntime()
    print(f"--- Iris Python Ping-Pong Benchmark (rounds={args.rounds}) ---")

    push_latencies = []
    for i in range(args.samples):
        lat = bench_push_ping_pong(rt, args.rounds)
        push_latencies.append(lat)
        print(f"Push Sample {i + 1}: {lat:.2f} µs/round-trip")

    pull_latencies = []
    for i in range(args.samples):
        lat = bench_pull_ping_pong(rt, args.rounds)
        pull_latencies.append(lat)
        print(f"Pull Sample {i + 1}: {lat:.2f} µs/round-trip")

    print("\n--- Summary ---")
    print(f"Push Median: {statistics.median(push_latencies):.2f} µs/round-trip")
    print(f"Pull Median: {statistics.median(pull_latencies):.2f} µs/round-trip")


if __name__ == "__main__":
    main()
