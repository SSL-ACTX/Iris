use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iris_core::{mailbox, pid, Runtime, RUNTIME};
use std::sync::Arc;

fn bench_pid_allocation(c: &mut Criterion) {
    let slab = pid::SlabAllocator::new();
    c.bench_function("pid_allocate_deallocate_lock_free", |b| {
        b.iter(|| {
            let pid = slab.allocate();
            slab.deallocate(pid);
        })
    });
}

fn bench_actor_throughput_unbounded(c: &mut Criterion) {
    let rt = Arc::new(Runtime::new());

    // Spawn a sink actor that just consumes messages
    let pid = rt.spawn_actor(|mut rx| async move {
        while let Some(_msg) = rx.recv().await {
            // sink
        }
    });

    let payload = Bytes::from_static(&[0u8; 64]);

    c.bench_function("actor_send_throughput_unbounded", |b| {
        b.iter(|| {
            let _ = rt.send(pid, mailbox::Message::User(payload.clone()));
        })
    });
}

fn bench_actor_throughput_bounded(c: &mut Criterion) {
    let rt = Arc::new(Runtime::new());
    let capacity = 1000;

    // Spawn a sink actor
    let pid = rt.spawn_actor_bounded(
        |mut rx| async move {
            while let Some(_msg) = rx.recv().await {
                // sink
            }
        },
        capacity,
    );

    let payload = Bytes::from_static(&[0u8; 64]);

    c.bench_function("actor_send_throughput_bounded", |b| {
        b.iter(|| {
            // We use black_box to ensure the result isn't optimized away
            let _ = black_box(rt.send(pid, mailbox::Message::User(payload.clone())));
        })
    });
}

criterion_group!(
    benches,
    bench_pid_allocation,
    bench_actor_throughput_unbounded,
    bench_actor_throughput_bounded
);
criterion_main!(benches);
