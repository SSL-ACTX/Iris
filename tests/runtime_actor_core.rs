use iris::{core::telemetry::HealthStatus, mailbox, Runtime};
use tokio::time::{sleep, timeout, Duration};

#[tokio::test]
async fn test_load_shedding_prevents_spawn() {
    let rt = Runtime::new();

    // 1. Initially disabled
    let pid = rt.spawn_observed_handler(10);
    assert_ne!(pid, 0);
    rt.stop(pid);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(rt.telemetry().get_actor_count(), 0);

    // 2. Enable load shedding with low capacity
    rt.set_system_capacity(1);
    rt.set_load_shedding(true);

    let pid1 = rt.spawn_observed_handler(10);
    assert_ne!(pid1, 0, "First spawn should succeed");

    // Second spawn should fail (rejected)
    let pid2 = rt.spawn_observed_handler(10);
    assert_eq!(pid2, 0, "Second spawn should be rejected by load shedding");

    // 3. Stop first and try again
    rt.stop(pid1);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        rt.telemetry().get_actor_count(),
        0,
        "Actor count should be 0 after stop"
    );

    let pid3 = rt.spawn_observed_handler(10);
    assert_ne!(pid3, 0, "Spawn should succeed again after freeing capacity");

    rt.stop(pid3);
}

#[tokio::test]
async fn test_load_shedding_deactivation() {
    let rt = Runtime::new();
    rt.set_system_capacity(0); // Everything rejected if enabled
    rt.set_load_shedding(true);

    let pid = rt.spawn_observed_handler(10);
    assert_eq!(pid, 0);

    rt.set_load_shedding(false);
    let pid2 = rt.spawn_observed_handler(10);
    assert_ne!(
        pid2, 0,
        "Spawn should succeed when load shedding is disabled"
    );
    rt.stop(pid2);
}

#[tokio::test]
async fn test_core_call_response() {
    let rt = Runtime::new();

    let rt_clone = rt.clone();
    // Spawn an actor that responds to Request
    let pid = rt.spawn_handler_with_budget(
        move |msg| {
            let rt_inner = rt_clone.clone();
            async move {
                match msg {
                    mailbox::Message::Request { reply_to, payload } => {
                        // Return "pong" if "ping"
                        if &payload[..] == b"ping" {
                            let _ =
                                rt_inner.send_user(reply_to, bytes::Bytes::from_static(b"pong"));
                        }
                    }
                    _ => {}
                }
            }
        },
        10,
    );

    let res = rt
        .call(
            pid,
            bytes::Bytes::from_static(b"ping"),
            Duration::from_secs(1),
        )
        .await;
    assert_eq!(res.unwrap(), b"pong"[..]);
}

#[tokio::test]
async fn test_core_health_introspection() {
    let rt = Runtime::new();
    let pid = rt.spawn_observed_handler(10);

    // Default health
    let info = rt.actor_info(pid).unwrap();
    assert_eq!(info.get("health").unwrap(), "Ready");

    // Set to Busy
    rt.set_actor_health(pid, HealthStatus::Busy);
    let info = rt.actor_info(pid).unwrap();
    assert_eq!(info.get("health").unwrap(), "Busy");

    // Set to Degraded
    rt.set_actor_health(pid, HealthStatus::Degraded);
    let info = rt.actor_info(pid).unwrap();
    assert_eq!(info.get("health").unwrap(), "Degraded");
}

#[tokio::test]
async fn test_core_metrics_throughput() {
    let rt = Runtime::new();
    let pid = rt.spawn_observed_handler(10);

    for _ in 0..10 {
        rt.send_user(pid, bytes::Bytes::from_static(b"data"))
            .unwrap();
    }

    // Await processing
    tokio::time::sleep(Duration::from_millis(100)).await;

    let tel = rt.telemetry();
    assert_eq!(tel.get_messages_sent(), 10);
    assert_eq!(tel.get_messages_received(), 10);
}

#[tokio::test]
async fn bounded_spawn_rejects_overflow() {
    let rt = Runtime::new();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();
    let handler = move |mut mailbox: mailbox::MailboxReceiver| {
        let tx = tx.clone();
        let start_rx = start_rx;
        async move {
            let _ = start_rx.await;
            if let Some(msg) = mailbox.recv().await {
                let _ = tx.send(msg);
            }
        }
    };

    let pid = rt.spawn_actor_bounded(handler, 1);
    assert!(rt
        .send(pid, mailbox::Message::User(b"x".to_vec().into()))
        .is_ok());
    assert!(rt
        .send(pid, mailbox::Message::User(b"y".to_vec().into()))
        .is_err());

    let _ = start_tx.send(());

    sleep(Duration::from_millis(50)).await;
    let got = rx.try_recv().unwrap();
    assert_eq!(got, mailbox::Message::User(b"x".to_vec().into()));
}

#[tokio::test]
async fn send_user_fast_path_roundtrip_and_missing_pid() {
    let rt = Runtime::new();

    let (tx, mut recv_rx) = tokio::sync::mpsc::unbounded_channel();
    let pid = rt.spawn_handler_with_budget(
        move |msg| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(msg);
            }
        },
        32,
    );

    assert!(rt
        .send_user(pid, bytes::Bytes::from_static(b"hello"))
        .is_ok());

    let got = recv_rx.recv().await.expect("message should be received");
    match got {
        mailbox::Message::User(b) => assert_eq!(b.as_ref(), b"hello"),
        _ => panic!("expected user message"),
    }

    rt.stop(pid);
    tokio::time::sleep(Duration::from_millis(20)).await;

    let payload = bytes::Bytes::from_static(b"payload");
    let err = rt
        .send_user(pid, payload.clone())
        .expect_err("send should fail for stopped pid");
    assert_eq!(err, payload);
}

#[tokio::test]
async fn send_user_shared_roundtrip_and_missing_pid() {
    let rt = Runtime::new();

    let (tx, mut recv_rx) = tokio::sync::mpsc::unbounded_channel();
    let pid = rt.spawn_handler_with_budget(
        move |msg| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(msg);
            }
        },
        32,
    );

    let payload: std::sync::Arc<[u8]> = std::sync::Arc::from(&b"hello-shared"[..]);
    assert!(rt.send_user_shared(pid, payload.clone()).is_ok());

    let got = recv_rx.recv().await.expect("message should be received");
    match got {
        mailbox::Message::User(b) => assert_eq!(b.as_ref(), b"hello-shared"),
        _ => panic!("expected user message"),
    }

    rt.stop(pid);
    tokio::time::sleep(Duration::from_millis(20)).await;

    let err = rt
        .send_user_shared(pid, payload.clone())
        .expect_err("send should fail for stopped pid");
    assert_eq!(err.as_ref(), payload.as_ref());
}

#[tokio::test]
async fn send_user_static_roundtrip_and_missing_pid() {
    let rt = Runtime::new();

    let (tx, mut recv_rx) = tokio::sync::mpsc::unbounded_channel();
    let pid = rt.spawn_handler_with_budget(
        move |msg| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(msg);
            }
        },
        32,
    );

    static PAYLOAD: &[u8] = b"hello-static";
    assert!(rt.send_user_static(pid, PAYLOAD).is_ok());

    let got = recv_rx.recv().await.expect("message should be received");
    match got {
        mailbox::Message::User(b) => assert_eq!(b.as_ref(), PAYLOAD),
        _ => panic!("expected user message"),
    }

    rt.stop(pid);
    tokio::time::sleep(Duration::from_millis(20)).await;

    let err = rt
        .send_user_static(pid, PAYLOAD)
        .expect_err("send should fail for stopped pid");
    assert_eq!(err, PAYLOAD);
}

#[tokio::test]
async fn overflow_policy_block_waits_until_capacity_then_succeeds() {
    let rt = Runtime::new();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();

    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let tx = tx.clone();
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                for _ in 0..2 {
                    if let Some(msg) = mailbox.recv().await {
                        let _ = tx.send(msg);
                    }
                }
            }
        },
        1,
    );

    rt.set_overflow_policy(pid, mailbox::OverflowPolicy::Block);

    assert!(rt
        .send(pid, mailbox::Message::User(b"b1".to_vec().into()))
        .is_ok());

    let rt_send = rt.clone();
    let mut send_task = tokio::task::spawn_blocking(move || {
        rt_send
            .send(pid, mailbox::Message::User(b"b2".to_vec().into()))
            .is_ok()
    });

    assert!(
        timeout(Duration::from_millis(30), &mut send_task)
            .await
            .is_err(),
        "block policy should wait while mailbox is full"
    );

    let _ = start_tx.send(());

    let send_ok = timeout(Duration::from_secs(1), send_task)
        .await
        .expect("blocked send should complete")
        .expect("send task should join");
    assert!(send_ok, "block policy send should report success");

    let first = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("first message receive")
        .expect("first message exists");
    let second = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("second message receive")
        .expect("second message exists");

    assert_eq!(first, mailbox::Message::User(b"b1".to_vec().into()));
    assert_eq!(second, mailbox::Message::User(b"b2".to_vec().into()));
}

#[tokio::test]
async fn overflow_policy_spill_forwards_copy_and_unblocks_when_primary_stops() {
    let rt = Runtime::new();

    let (fallback_tx, mut fallback_rx) = tokio::sync::mpsc::unbounded_channel();
    let (_start_tx, start_rx) = tokio::sync::oneshot::channel::<()>();

    let primary = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                while mailbox.recv().await.is_some() {}
            }
        },
        1,
    );

    let fallback = rt.spawn_actor(move |mut mailbox: mailbox::MailboxReceiver| {
        let fallback_tx = fallback_tx.clone();
        async move {
            if let Some(msg) = mailbox.recv().await {
                let _ = fallback_tx.send(msg);
            }
        }
    });

    rt.set_overflow_policy(primary, mailbox::OverflowPolicy::Spill(fallback));

    assert!(rt
        .send(primary, mailbox::Message::User(b"p1".to_vec().into()))
        .is_ok());

    let rt_send = rt.clone();
    let send_task = tokio::task::spawn_blocking(move || {
        rt_send
            .send(primary, mailbox::Message::User(b"p2".to_vec().into()))
            .is_ok()
    });

    let spilled = timeout(Duration::from_secs(2), fallback_rx.recv())
        .await
        .expect("spill should reach fallback promptly")
        .expect("fallback should receive spill copy");
    assert_eq!(spilled, mailbox::Message::User(b"p2".to_vec().into()));

    rt.stop(primary);

    let send_ok = timeout(Duration::from_secs(2), send_task)
        .await
        .expect("spill send should unblock when primary stops")
        .expect("send task should join");
    assert!(!send_ok, "spill send should fail once primary is stopped");
}

#[tokio::test]
async fn overflow_policy_redirect_to_self_returns_err() {
    let rt = Runtime::new();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();
    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                while mailbox.recv().await.is_some() {}
            }
        },
        1,
    );

    rt.set_overflow_policy(pid, mailbox::OverflowPolicy::Redirect(pid));
    assert!(rt
        .send(pid, mailbox::Message::User(b"m1".to_vec().into()))
        .is_ok());

    assert!(rt
        .send(pid, mailbox::Message::User(b"m2".to_vec().into()))
        .is_err());

    let _ = start_tx.send(());
}

#[tokio::test]
async fn mailbox_backpressure_signals_based_on_capacity() {
    let rt = Runtime::new();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();

    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                while mailbox.recv().await.is_some() {}
            }
        },
        5,
    );

    let mut sent = 0;
    for i in 1..=10 {
        let payload = format!("msg{}", i);
        if rt
            .send(pid, mailbox::Message::User(payload.into_bytes().into()))
            .is_ok()
        {
            sent += 1;
        } else {
            break;
        }
    }

    assert!(
        sent >= 4,
        "expected at least 4 messages to be accepted, got {}",
        sent
    );

    let level = rt
        .mailbox_backpressure(pid)
        .expect("backpressure should be available");
    assert!(matches!(
        level,
        mailbox::BackpressureLevel::High | mailbox::BackpressureLevel::Critical
    ));

    if sent >= 5 {
        assert_eq!(level, mailbox::BackpressureLevel::Critical);
    } else {
        assert_eq!(level, mailbox::BackpressureLevel::High);
    }

    let _ = start_tx.send(());
}

#[tokio::test]
async fn mailbox_backpressure_send_user_path_updates_level() {
    let rt = Runtime::new();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();

    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                while mailbox.recv().await.is_some() {}
            }
        },
        4,
    );

    for i in 0..3 {
        let payload = bytes::Bytes::from(format!("u{}", i));
        assert!(rt.send_user(pid, payload).is_ok());
    }
    assert_eq!(
        rt.mailbox_backpressure(pid),
        Some(mailbox::BackpressureLevel::High)
    );

    assert!(rt.send_user(pid, bytes::Bytes::from_static(b"u3")).is_ok());
    assert_eq!(
        rt.mailbox_backpressure(pid),
        Some(mailbox::BackpressureLevel::Critical)
    );

    let _ = start_tx.send(());
}

#[tokio::test]
async fn mailbox_backpressure_recovers_to_normal_after_drain() {
    let rt = Runtime::new();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();

    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                while mailbox.recv().await.is_some() {}
            }
        },
        3,
    );

    assert!(rt
        .send(pid, mailbox::Message::User(b"d1".to_vec().into()))
        .is_ok());
    assert!(rt
        .send(pid, mailbox::Message::User(b"d2".to_vec().into()))
        .is_ok());
    assert!(rt
        .send(pid, mailbox::Message::User(b"d3".to_vec().into()))
        .is_ok());
    assert_eq!(
        rt.mailbox_backpressure(pid),
        Some(mailbox::BackpressureLevel::Critical)
    );

    let _ = start_tx.send(());

    timeout(Duration::from_secs(1), async {
        loop {
            if rt.mailbox_size(pid) == Some(0) {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("mailbox should drain");

    assert_eq!(
        rt.mailbox_backpressure(pid),
        Some(mailbox::BackpressureLevel::Normal)
    );
}

#[tokio::test]
async fn mailbox_backpressure_unknown_pid_is_none() {
    let rt = Runtime::new();
    assert_eq!(rt.mailbox_backpressure(u64::MAX), None);
}

#[tokio::test]
async fn send_with_backpressure_returns_live_level() {
    let rt = Runtime::new();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();

    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                while mailbox.recv().await.is_some() {}
            }
        },
        5,
    );

    let l1 = rt
        .send_with_backpressure(pid, mailbox::Message::User(b"a".to_vec().into()))
        .expect("first send should succeed");
    assert_eq!(l1, mailbox::BackpressureLevel::Normal);

    let l2 = rt
        .send_with_backpressure(pid, mailbox::Message::User(b"b".to_vec().into()))
        .expect("second send should succeed");
    assert_eq!(l2, mailbox::BackpressureLevel::Normal);

    let l3 = rt
        .send_with_backpressure(pid, mailbox::Message::User(b"c".to_vec().into()))
        .expect("third send should succeed");
    assert_eq!(l3, mailbox::BackpressureLevel::Normal);

    let l4 = rt
        .send_with_backpressure(pid, mailbox::Message::User(b"d".to_vec().into()))
        .expect("fourth send should succeed");
    assert_eq!(l4, mailbox::BackpressureLevel::High);

    let l5 = rt
        .send_with_backpressure(pid, mailbox::Message::User(b"e".to_vec().into()))
        .expect("fifth send should succeed");
    assert_eq!(l5, mailbox::BackpressureLevel::Critical);

    let _ = start_tx.send(());
}

#[tokio::test]
async fn send_with_backpressure_respects_hysteresis_state() {
    let rt = Runtime::new();
    let (ready_tx, mut ready_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let (drain_tx, drain_rx) = tokio::sync::oneshot::channel();

    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let ready_tx = ready_tx.clone();
            let drain_rx = drain_rx;
            async move {
                let _ = drain_rx.await;
                for _ in 0..2 {
                    if mailbox.recv().await.is_some() {
                        let _ = ready_tx.send(());
                    }
                }
                std::future::pending::<()>().await;
            }
        },
        10,
    );

    for i in 0..9 {
        let payload = format!("h{}", i).into_bytes();
        assert!(rt.send(pid, mailbox::Message::User(payload.into())).is_ok());
    }

    assert_eq!(
        rt.mailbox_backpressure(pid),
        Some(mailbox::BackpressureLevel::Critical)
    );

    let _ = drain_tx.send(());
    timeout(Duration::from_secs(1), async {
        let _ = ready_rx.recv().await;
        let _ = ready_rx.recv().await;
    })
    .await
    .expect("actor should drain two messages");

    let level = rt
        .send_with_backpressure(pid, mailbox::Message::User(b"rehydrate".to_vec().into()))
        .expect("send should succeed");

    assert_eq!(
        level,
        mailbox::BackpressureLevel::Critical,
        "hysteresis should keep critical at 80% when previous state was critical"
    );
}

#[tokio::test]
async fn send_user_with_backpressure_unknown_pid_returns_err() {
    let rt = Runtime::new();
    let payload = bytes::Bytes::from_static(b"missing");
    let err = rt
        .send_user_with_backpressure(u64::MAX, payload.clone())
        .expect_err("missing pid should return original payload");
    assert_eq!(err, payload);
}

#[tokio::test]
async fn bounded_mailbox_state_is_gone_after_actor_exit() {
    let rt = Runtime::new();

    let pid = rt.spawn_actor_bounded(|_mailbox: mailbox::MailboxReceiver| async move {}, 2);

    timeout(Duration::from_secs(1), async {
        loop {
            if !rt.is_alive(pid) {
                break;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("actor should exit promptly");

    assert_eq!(rt.mailbox_size(pid), None);
    assert_eq!(rt.mailbox_backpressure(pid), None);
}

#[tokio::test]
async fn bounded_mailbox_state_is_gone_after_stop() {
    let rt = Runtime::new();
    let (start_tx, start_rx) = tokio::sync::oneshot::channel();

    let pid = rt.spawn_actor_bounded(
        move |mut mailbox: mailbox::MailboxReceiver| {
            let start_rx = start_rx;
            async move {
                let _ = start_rx.await;
                while mailbox.recv().await.is_some() {}
            }
        },
        2,
    );

    rt.stop(pid);
    let _ = start_tx.send(());

    timeout(Duration::from_secs(1), async {
        loop {
            if !rt.is_alive(pid) {
                break;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("stopped actor should exit promptly");

    assert_eq!(rt.mailbox_size(pid), None);
    assert_eq!(rt.mailbox_backpressure(pid), None);
}

#[tokio::test]
async fn virtual_actor_activates_on_first_send() {
    let rt = Runtime::new();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let pid = rt.spawn_virtual_handler_with_budget(
        move |msg| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(msg);
            }
        },
        16,
        None,
    );

    assert!(
        rt.mailbox_size(pid).is_none(),
        "virtual actor should be inactive initially"
    );

    assert!(rt
        .send(pid, mailbox::Message::User(b"lazy".to_vec().into()))
        .is_ok());

    let got = timeout(Duration::from_secs(1), rx.recv())
        .await
        .expect("virtual handler should receive message")
        .expect("message must exist");
    assert_eq!(got, mailbox::Message::User(b"lazy".to_vec().into()));
    assert!(rt.is_alive(pid));
}

#[tokio::test]
async fn virtual_actor_idle_timeout_deactivates_actor() {
    let rt = Runtime::new();

    let pid = rt.spawn_virtual_handler_with_budget(
        move |_msg| async move {},
        8,
        Some(Duration::from_millis(50)),
    );

    assert!(rt
        .send(pid, mailbox::Message::User(b"ping".to_vec().into()))
        .is_ok());

    timeout(Duration::from_secs(1), async {
        loop {
            if !rt.is_alive(pid) {
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("virtual actor should stop after idle timeout");
}

#[tokio::test]
async fn stop_unactivated_virtual_actor_deallocates_pid() {
    let rt = Runtime::new();

    let pid = rt.spawn_virtual_handler_with_budget(move |_msg| async move {}, 8, None);
    assert!(rt.is_alive(pid));

    rt.stop(pid);
    sleep(Duration::from_millis(20)).await;

    assert!(!rt.is_alive(pid));
}

#[tokio::test]
async fn behavior_version_increments_on_hot_swap() {
    let rt = Runtime::new();
    let pid = rt.spawn_observed_handler(8);

    assert_eq!(rt.behavior_version(pid), 1);

    rt.hot_swap(pid, 0xA11CE);
    sleep(Duration::from_millis(20)).await;

    assert_eq!(rt.behavior_version(pid), 2);
}

#[tokio::test]
async fn rollback_behavior_replays_previous_handler_ptr() {
    let rt = Runtime::new();
    let pid = rt.spawn_observed_handler(8);

    rt.hot_swap(pid, 0xBEEF);
    rt.hot_swap(pid, 0xCAFE);
    sleep(Duration::from_millis(20)).await;

    assert_eq!(rt.behavior_version(pid), 3);
    let rolled = rt
        .rollback_behavior(pid, 1)
        .expect("rollback should succeed");
    assert_eq!(rolled, 2);
    assert_eq!(rt.behavior_version(pid), 2);

    let swaps = timeout(Duration::from_secs(1), async {
        loop {
            let msgs = rt
                .get_observed_messages(pid)
                .expect("observed actor should still exist");
            let mut swaps = Vec::new();
            for msg in msgs {
                if let mailbox::Message::System(mailbox::SystemMessage::HotSwap(ptr)) = msg {
                    swaps.push(ptr);
                }
            }
            if swaps.len() >= 3 {
                break swaps;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("timed out waiting for hot swap replay messages");

    assert_eq!(swaps, vec![0xBEEF, 0xCAFE, 0xBEEF]);
}

#[tokio::test]
async fn rollback_behavior_requires_enough_history() {
    let rt = Runtime::new();
    let pid = rt.spawn_observed_handler(8);

    rt.hot_swap(pid, 0xBEEF);
    sleep(Duration::from_millis(20)).await;

    let err = rt
        .rollback_behavior(pid, 1)
        .expect_err("rollback should fail");
    assert!(err.contains("history"));
}

#[tokio::test]
async fn path_registry_roundtrip_and_listing() {
    let rt = Runtime::new();
    let pid1 = rt.spawn_observed_handler(8);
    let pid2 = rt.spawn_observed_handler(8);

    rt.register_path("/svc/auth/main".to_string(), pid1);
    rt.register_path("/svc/auth/shadow".to_string(), pid2);

    assert_eq!(rt.whereis_path("/svc/auth/main"), Some(pid1));

    let all = rt.list_children("/svc");
    assert!(all
        .iter()
        .any(|(p, id)| p == "/svc/auth/main" && *id == pid1));
    assert!(all
        .iter()
        .any(|(p, id)| p == "/svc/auth/shadow" && *id == pid2));

    let direct = rt.list_children_direct("/svc/auth");
    assert_eq!(direct.len(), 2);

    rt.unregister_path("/svc/auth/main");
    assert_eq!(rt.whereis_path("/svc/auth/main"), None);
}

#[tokio::test]
async fn path_supervisor_tracks_children() {
    let rt = Runtime::new();
    rt.create_path_supervisor("/svc/workers");

    let pid = rt.spawn_observed_handler(8);
    rt.path_supervisor_watch("/svc/workers", pid);

    let kids = rt.path_supervisor_children("/svc/workers");
    assert!(kids.contains(&pid));

    rt.stop(pid);
    sleep(Duration::from_millis(20)).await;

    rt.remove_path_supervisor("/svc/workers");
    assert!(rt.path_supervisor_children("/svc/workers").is_empty());
}
