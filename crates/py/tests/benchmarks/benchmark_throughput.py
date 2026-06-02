import time
import statistics
import iris
import argparse


def main():
    parser = argparse.ArgumentParser(description="Iris Python Throughput Benchmark")
    parser.add_argument(
        "--count", type=int, default=100000, help="Number of messages to send"
    )
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    args = parser.parse_args()

    rt = iris.PyRuntime()

    # A simple sink actor that just drops messages
    def sink(msg):
        pass

    pid = rt.spawn(sink, budget=100)
    payload = b"test_payload_64_bytes_" + b"0" * 42  # ~64 bytes

    print(
        f"--- Iris Python Throughput (count={args.count:,}, rounds={args.rounds}) ---"
    )

    send_rates = []
    for i in range(args.rounds):
        t0 = time.perf_counter()
        for _ in range(args.count):
            rt.send(pid, payload)
        elapsed = time.perf_counter() - t0
        rate = args.count / elapsed
        send_rates.append(rate)
        print(f"Round {i + 1}: {rate:,.0f} msg/s")

    # Benchmarking send_many
    batch_size = 1000
    batch = [payload] * batch_size
    batch_rounds = args.count // batch_size

    many_rates = []
    for i in range(args.rounds):
        t0 = time.perf_counter()
        for _ in range(batch_rounds):
            rt.send_many(pid, batch)
        elapsed = time.perf_counter() - t0
        rate = (batch_rounds * batch_size) / elapsed
        many_rates.append(rate)
        print(f"Round {i + 1} (send_many batch={batch_size}): {rate:,.0f} msg/s")

    rt.stop(pid)

    print("\n--- Summary ---")
    print(f"Python send median      : {statistics.median(send_rates):,.0f} msg/s")
    print(f"Python send_many median : {statistics.median(many_rates):,.0f} msg/s")


if __name__ == "__main__":
    main()
