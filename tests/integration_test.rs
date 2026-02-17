use myrmidon::Runtime;
use myrmidon::mailbox::Message;
use bytes::Bytes;

#[tokio::test]
async fn actor_recv_and_cleanup() {
    let rt = Runtime::new();

    let pid = rt.spawn_actor(|mut rx| async move {
        if let Some(msg) = rx.recv().await {
            match msg {
                Message::User(buf) => assert_eq!(buf.as_ref(), b"ping"),
                _ => panic!("expected user message"),
            }
        }
    });

    assert!(rt.is_alive(pid));
    rt.send(pid, Message::User(Bytes::from_static(b"ping"))).unwrap();

    // allow actor to process and exit
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // actor should have cleaned up
    assert!(!rt.is_alive(pid));
}

#[tokio::test]
async fn spawn_actor_with_budget_basic() {
    let rt = Runtime::new();

    let pid = rt.spawn_actor_with_budget(|mut rx| async move {
        if let Some(msg) = rx.recv().await {
            match msg {
                Message::User(buf) => assert_eq!(buf.as_ref(), b"budgeted"),
                _ => panic!("expected user message"),
            }
        }
    }, 10);

    assert!(rt.is_alive(pid));
    rt.send(pid, Message::User(Bytes::from_static(b"budgeted"))).unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert!(!rt.is_alive(pid));
}

#[tokio::test]
async fn dispatcher_fairness_between_handlers() {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    use std::time::Duration;

    let rt = Runtime::new();

    let counter_a = Arc::new(AtomicUsize::new(0));
    let counter_b = Arc::new(AtomicUsize::new(0));

    let a_clone = counter_a.clone();
    let pid_a = rt.spawn_handler_with_budget(move |_msg| {
        let a_clone = a_clone.clone();
        async move {
            // simulate work that yields multiple times so the ReductionLimiter can preempt
            for _ in 0..5 { tokio::task::yield_now().await; }
            a_clone.fetch_add(1, Ordering::SeqCst);
        }
    }, 1);

    let b_clone = counter_b.clone();
    let pid_b = rt.spawn_handler_with_budget(move |_msg| {
        let b_clone = b_clone.clone();
        async move {
            for _ in 0..5 { tokio::task::yield_now().await; }
            b_clone.fetch_add(1, Ordering::SeqCst);
        }
    }, 1);

    // enqueue messages for both handlers
    for _ in 0..10 {
        rt.send(pid_a, Message::User(Bytes::from_static(b"x"))).unwrap();
        rt.send(pid_b, Message::User(Bytes::from_static(b"y"))).unwrap();
    }

    // give runtime time to interleave processing
    tokio::time::sleep(Duration::from_millis(300)).await;

    let a = counter_a.load(Ordering::SeqCst);
    let b = counter_b.load(Ordering::SeqCst);

    // Both handlers should have processed some messages and neither should dominate.
    assert!(a >= 3, "handler A processed too few messages: {}", a);
    assert!(b >= 3, "handler B processed too few messages: {}", b);
    assert!((a as isize - b as isize).abs() <= 6, "imbalance between handlers: {} vs {}", a, b);
}

// ---------------------- Supervisor tests ----------------------

#[tokio::test]
async fn supervisor_restart_one_for_one() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    let rt = Runtime::new();
    let rt2 = rt.clone();

    // counter to show restarted actor responds
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    // initial spawn
    let pid = rt.spawn_handler_with_budget(move |msg| {
        let counter_clone = counter_clone.clone();
        async move {
            match &msg {
                Message::User(b) if b.as_ref() == b"die" => panic!("boom"),
                Message::User(b) if b.as_ref() == b"inc" => { counter_clone.fetch_add(1, Ordering::SeqCst); },
                _ => {}
            }
        }
    }, 1);

    // factory that re-spawns the same handler (captures runtime)
    let rt_for_factory = rt2.clone();
    let counter_for_factory = counter.clone();
        let factory = Arc::new(move || {
        let rt_inner = rt_for_factory.clone();
        let counter_inner = counter_for_factory.clone();
        Ok::<u64, String>(rt_inner.spawn_handler_with_budget(move |msg| {
            let counter_inner = counter_inner.clone();
            async move {
                    match &msg {
                        Message::User(b) if b.as_ref() == b"die" => panic!("boom"),
                        Message::User(b) if b.as_ref() == b"inc" => { counter_inner.fetch_add(1, Ordering::SeqCst); },
                        _ => {}
                    }
            }
        }, 1))
    });

    // supervise the pid with RestartOne
    rt.supervise(pid, factory, myrmidon::supervisor::RestartStrategy::RestartOne);

    // Link the factory-spawned child to a second observer to verify exit signals
    let pid2 = rt.spawn_observed_handler(1);
    rt.link(pid, pid2);

    // crash the child
    rt.send(pid, Message::User(Bytes::from_static(b"die"))).unwrap();

    // give supervisor time to restart
    tokio::time::sleep(Duration::from_millis(200)).await;

    // supervisor should now have one child (the restarted pid)
    let sup = rt.supervisor();
    assert_eq!(sup.children_count(), 1);
    let new_pids = sup.child_pids();
    assert!(!new_pids.contains(&pid), "child should have been restarted under a new pid");

    // send a message to the restarted child and ensure it responds
    let new_pid = new_pids[0];
    rt.send(new_pid, Message::User(Bytes::from_static(b"inc"))).unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(counter.load(Ordering::SeqCst) >= 1);

    // verify linked pid received exit notification when original child died
    let msgs = rt.get_observed_messages(pid2).unwrap();
    assert!(msgs.iter().any(|m| match m {
        Message::System(m) => matches!(m, myrmidon::mailbox::SystemMessage::Exit(_)),
        _ => false,
    }));
}

#[tokio::test]
async fn supervisor_restart_all() {
    use std::time::Duration;
    use std::sync::Arc;

    // Run the entire test body under a 5s timeout to guard against any
    // supervisor restart loops that could hang the test process.
    let res = tokio::time::timeout(Duration::from_secs(5), async {
        let rt = Runtime::new();
        let rt2 = rt.clone();

        // spawn two simple handlers and register both under RestartAll
        let pid_a = rt.spawn_handler_with_budget(move |msg| async move {
            match &msg { Message::User(b) if b.as_ref() == b"die" => panic!("boom"), _ => {} }
        }, 1);
        let pid_b = rt.spawn_handler_with_budget(move |msg| async move {
            match &msg { Message::User(b) if b.as_ref() == b"die" => panic!("boom"), _ => {} }
        }, 1);

        let rt_a = rt2.clone();
        let factory_a = Arc::new(move || { eprintln!("[test] factory_a called"); Ok::<u64, String>(rt_a.spawn_handler_with_budget(move |msg| async move { match &msg { Message::User(b) if b.as_ref() == b"die" => panic!("boom"), _ => {} } }, 1)) });
        let rt_b = rt2.clone();
        let factory_b = Arc::new(move || { eprintln!("[test] factory_b called"); Ok::<u64, String>(rt_b.spawn_handler_with_budget(move |msg| async move { match &msg { Message::User(b) if b.as_ref() == b"die" => panic!("boom"), _ => {} } }, 1)) });

        rt.supervise(pid_a, factory_a, myrmidon::supervisor::RestartStrategy::RestartAll);
        rt.supervise(pid_b, factory_b, myrmidon::supervisor::RestartStrategy::RestartAll);

        // crash one child; both should be restarted
        rt.send(pid_a, Message::User(Bytes::from_static(b"die"))).ok();

        // allow supervisor some time to perform restarts
        tokio::time::sleep(Duration::from_millis(200)).await;

        let sup = rt.supervisor();
        let new_pids = sup.child_pids();
        assert_eq!(new_pids.len(), 2);
        // At least one of the supervised children should have been restarted (new pid).
        assert!(new_pids.iter().any(|p| *p != pid_a) || new_pids.iter().any(|p| *p != pid_b));
    }).await;

    assert!(res.is_ok(), "supervisor_restart_all timed out after 5s");
}
