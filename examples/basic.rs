use myrmidon::Runtime;

#[tokio::main]
async fn main() {
    let rt = Runtime::new();

    let pid = rt.spawn_actor(|mut rx| async move {
        if let Some(msg) = rx.recv().await {
            println!("actor got: {:?}", msg);
        }
    });

    rt.send(pid, myrmidon::mailbox::Message::User(bytes::Bytes::from_static(b"hello from example"))).unwrap();

    // give the actor a moment to run
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
}
