use bytes::Bytes;
use iris::mailbox::{bounded_channel, channel, BackpressureLevel, Message, SystemMessage};

#[tokio::test]
async fn send_and_recv() {
    let (tx, mut rx) = channel();
    tx.send(Message::User(Bytes::from_static(b"hello")))
        .unwrap();
    let got = rx.recv().await.expect("should receive");
    match got {
        Message::User(buf) => assert_eq!(buf.as_ref(), b"hello"),
        _ => panic!("expected user message"),
    }
}

#[tokio::test]
async fn bounded_mailbox_drop_new() {
    let (tx, mut rx) = bounded_channel(2);
    tx.send(Message::User(Bytes::from_static(b"m1"))).unwrap();
    tx.send(Message::User(Bytes::from_static(b"m2"))).unwrap();
    assert!(tx.send(Message::User(Bytes::from_static(b"m3"))).is_err());

    let first = rx.recv().await.expect("first");
    let second = rx.recv().await.expect("second");
    match first {
        Message::User(b) => assert_eq!(b.as_ref(), b"m1"),
        _ => panic!(),
    }
    match second {
        Message::User(b) => assert_eq!(b.as_ref(), b"m2"),
        _ => panic!(),
    }
}

#[tokio::test]
async fn selective_receive_defers_and_preserves_order() {
    let (tx, mut rx) = channel();

    tx.send(Message::User(Bytes::from_static(b"m1"))).unwrap();
    tx.send(Message::User(Bytes::from_static(b"target")))
        .unwrap();
    tx.send(Message::User(Bytes::from_static(b"m3"))).unwrap();

    let got = rx
        .selective_recv(|m| match m {
            Message::User(b) => b.as_ref() == b"target",
            _ => false,
        })
        .await
        .expect("should find target");

    match got {
        Message::User(b) => assert_eq!(b.as_ref(), b"target"),
        _ => panic!("expected user message"),
    }

    let first = rx.recv().await.expect("first deferred");
    let second = rx.recv().await.expect("second deferred");

    match first {
        Message::User(b) => assert_eq!(b.as_ref(), b"m1"),
        _ => panic!("expected user message"),
    }

    match second {
        Message::User(b) => assert_eq!(b.as_ref(), b"m3"),
        _ => panic!("expected user message"),
    }
}

#[tokio::test]
async fn drop_old_system_message_discards_oldest_user_message() {
    let (tx, mut rx) = bounded_channel(2);

    tx.send(Message::User(Bytes::from_static(b"m1"))).unwrap();
    tx.send(Message::User(Bytes::from_static(b"m2"))).unwrap();
    tx.send_system(SystemMessage::DropOld).unwrap();

    let got = rx.recv().await.expect("message after drop-old");
    match got {
        Message::User(b) => assert_eq!(b.as_ref(), b"m2"),
        _ => panic!("expected user message"),
    }
}

#[tokio::test]
async fn bounded_channel_capacity_matches_request() {
    let (tx, _rx) = bounded_channel(5);

    for i in 0..5 {
        let data = vec![b'a' + (i as u8)];
        assert!(tx
            .send(Message::User(Bytes::copy_from_slice(&data)))
            .is_ok());
    }

    assert!(tx
        .send(Message::User(Bytes::from_static(b"overflow")))
        .is_err());
}

#[tokio::test]
async fn backpressure_hysteresis_prevents_flapping() {
    let (tx, mut rx) = bounded_channel(10);

    for i in 0..9 {
        let payload = vec![b'a' + (i as u8)];
        tx.send(Message::User(Bytes::copy_from_slice(&payload)))
            .unwrap();
    }

    let level = tx.backpressure_level_with_hysteresis(Some(10), BackpressureLevel::Normal);
    assert_eq!(level, BackpressureLevel::Critical);

    let _ = rx.recv().await;
    let level = tx.backpressure_level_with_hysteresis(Some(10), BackpressureLevel::Critical);
    assert_eq!(level, BackpressureLevel::Critical);

    let _ = rx.recv().await;
    let level = tx.backpressure_level_with_hysteresis(Some(10), BackpressureLevel::Critical);
    assert_eq!(level, BackpressureLevel::High);
}

#[tokio::test]
async fn len_counter_reflects_queue_size() {
    let (tx, mut rx) = bounded_channel(3);
    assert_eq!(tx.len(), 0);
    tx.send(Message::User(Bytes::from_static(b"a"))).unwrap();
    assert_eq!(tx.len(), 1);
    tx.send(Message::User(Bytes::from_static(b"b"))).unwrap();
    assert_eq!(tx.len(), 2);
    // receive one and check counter decreases
    let _ = rx.recv().await.expect("recv");
    // len() should reflect the updated value
    assert!(tx.len() <= 1);
}
