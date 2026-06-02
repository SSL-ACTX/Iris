import argparse
import time
from statistics import mean

import iris


def spawn_push_actors(rt, count: int, budget: int):
    pids = []
    t0 = time.perf_counter()

    def no_op_handler(_msg):
        return None

    for _ in range(count):
        pids.append(rt.spawn(no_op_handler, budget=budget, release_gil=False))

    elapsed = time.perf_counter() - t0
    return pids, elapsed


def send_one_per_actor(rt, pids, payload: bytes):
    t0 = time.perf_counter()
    for pid in pids:
        ok = rt.send(pid, payload)
        if not ok:
            raise RuntimeError(f"send failed for pid={pid}")
    elapsed = time.perf_counter() - t0
    return elapsed


def spawn_pull_actors(rt, count: int, budget: int):
    pids = []
    t0 = time.perf_counter()

    def mailbox_handler(mailbox):
        while True:
            msg = mailbox.recv()
            if msg is None:
                break

    for _ in range(count):
        pids.append(rt.spawn_with_mailbox(mailbox_handler, budget=budget))

    elapsed = time.perf_counter() - t0
    return pids, elapsed


def cleanup(rt, pids):
    for pid in pids:
        rt.stop(pid)


def fmt_rate(total: int, secs: float) -> str:
    if secs <= 0:
        return "inf"
    return f"{total / secs:,.0f}"


def main():
    parser = argparse.ArgumentParser(
        description="One-shot benchmark for push actor send regression checks (ARM-friendly)."
    )
    parser.add_argument(
        "--count", type=int, default=100_000, help="Number of actors/messages"
    )
    parser.add_argument("--budget", type=int, default=10, help="Actor budget")
    parser.add_argument(
        "--payload-size",
        type=int,
        default=4,
        help="Payload size in bytes",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="How many rounds to run for the push path",
    )
    parser.add_argument(
        "--include-pull",
        action="store_true",
        help="Also benchmark pull/mailbox actors in same run",
    )
    args = parser.parse_args()

    payload = b"p" * args.payload_size

    print("--- Iris Push Regression Benchmark ---")
    print(f"version={iris.version()}")
    print(
        f"count={args.count:,}, budget={args.budget}, payload_size={args.payload_size}, rounds={args.rounds}"
    )
    print("Tip: run this script on two commits and compare send msg/s.")

    push_spawn_times = []
    push_send_times = []

    for i in range(args.rounds):
        rt = iris.Runtime()
        pids = []
        try:
            pids, spawn_t = spawn_push_actors(rt, args.count, args.budget)
            send_t = send_one_per_actor(rt, pids, payload)
            push_spawn_times.append(spawn_t)
            push_send_times.append(send_t)

            print(
                f"round {i + 1} push: spawn={spawn_t:.3f}s ({fmt_rate(args.count, spawn_t)} actors/s) | "
                f"send={send_t:.3f}s ({fmt_rate(args.count, send_t)} msg/s)"
            )
        finally:
            cleanup(rt, pids)

    print("\n--- Push Summary (mean) ---")
    mean_spawn = mean(push_spawn_times)
    mean_send = mean(push_send_times)
    print(f"spawn: {mean_spawn:.3f}s ({fmt_rate(args.count, mean_spawn)} actors/s)")
    print(f"send : {mean_send:.3f}s ({fmt_rate(args.count, mean_send)} msg/s)")

    if args.include_pull:
        rt = iris.Runtime()
        pids = []
        try:
            pids, pull_spawn_t = spawn_pull_actors(rt, args.count, args.budget)
            pull_send_t = send_one_per_actor(rt, pids, payload)
            print("\n--- Pull (mailbox) One Shot ---")
            print(
                f"spawn={pull_spawn_t:.3f}s ({fmt_rate(args.count, pull_spawn_t)} actors/s) | "
                f"send={pull_send_t:.3f}s ({fmt_rate(args.count, pull_send_t)} msg/s)"
            )
        finally:
            cleanup(rt, pids)


if __name__ == "__main__":
    main()
