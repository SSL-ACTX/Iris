// src/node.rs
#![cfg(feature = "node")]

use crate::mailbox::{Message, SystemMessage};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{JsFunction, JsObject, JsUnknown, Result};
use napi_derive::napi;
use std::sync::Arc;
use tokio::sync::Mutex;

/// --- System Message Wrapper ---

#[napi(object)]
#[derive(Clone)]
pub struct JsSystemMessage {
    pub type_name: String,
    pub target_pid: Option<i64>,
    pub reason: Option<String>,
    pub metadata: Option<String>,
}

// Helper to convert Rust Message -> JS
fn message_to_js(env: &Env, msg: Message) -> Result<JsUnknown> {
    match msg {
        Message::User(bytes) => {
            let buf = env.create_buffer_with_data(bytes.to_vec())?.into_unknown();
            Ok(buf)
        }
        Message::System(sys) => {
            let (type_name, target_pid) = match sys {
                SystemMessage::Exit(info) => ("EXIT", Some(info.from as i64), match info.reason {
                    crate::mailbox::ExitReason::Normal => Some("normal".to_string()),
                    crate::mailbox::ExitReason::Panic => Some("panic".to_string()),
                    crate::mailbox::ExitReason::Timeout => Some("timeout".to_string()),
                    crate::mailbox::ExitReason::Killed => Some("killed".to_string()),
                    crate::mailbox::ExitReason::Oom => Some("oom".to_string()),
                    crate::mailbox::ExitReason::Other(ref s) => Some(s.clone()),
                }, info.metadata.clone()),
                SystemMessage::HotSwap(_) => ("HOT_SWAP", None),
                SystemMessage::Ping => ("PING", None),
                SystemMessage::Pong => ("PONG", None),
            };

            let mut obj = env.create_object()?;
            obj.set_named_property("typeName", type_name)?;
            if let Some(pid) = target_pid {
                obj.set_named_property("targetPid", pid)?;
            }
            if let Some(r) = reason {
                obj.set_named_property("reason", r)?;
            }
            if let Some(m) = metadata {
                obj.set_named_property("metadata", m)?;
            }

            Ok(obj.into_unknown())
        }
    }
}

/// --- Mailbox Wrapper ---

#[napi]
#[derive(Clone)]
pub struct JsMailbox {
    inner: Arc<Mutex<crate::mailbox::MailboxReceiver>>,
}

#[napi]
impl JsMailbox {
    /// Receive the next message (Async).
    #[napi]
    pub async fn recv(&self, timeout_sec: Option<f64>) -> Result<Option<WrappedMessage>> {
        let rx = self.inner.clone();

        let fut = async move {
            let mut guard = rx.lock().await;
            guard.recv().await
        };

        let result = if let Some(sec) = timeout_sec {
            match tokio::time::timeout(std::time::Duration::from_secs_f64(sec), fut).await {
                Ok(val) => val,
                Err(_) => return Ok(None),
            }
        } else {
            fut.await
        };

        match result {
            Some(msg) => Ok(Some(WrappedMessage::from(msg))),
            None => Ok(None),
        }
    }
}

#[napi(object)]
pub struct WrappedMessage {
    pub data: Option<Buffer>,
    pub system: Option<JsSystemMessage>,
}

impl From<Message> for WrappedMessage {
    fn from(msg: Message) -> Self {
        match msg {
            Message::User(b) => WrappedMessage {
                data: Some(b.to_vec().into()),
                system: None,
            },
            Message::System(sys) => {
                let (type_name, target_pid, reason, metadata) = match sys {
                    SystemMessage::Exit(info) => (
                        "EXIT".to_string(),
                        Some(info.from as i64),
                        match info.reason {
                            crate::mailbox::ExitReason::Normal => Some("normal".to_string()),
                            crate::mailbox::ExitReason::Panic => Some("panic".to_string()),
                            crate::mailbox::ExitReason::Timeout => Some("timeout".to_string()),
                            crate::mailbox::ExitReason::Killed => Some("killed".to_string()),
                            crate::mailbox::ExitReason::Oom => Some("oom".to_string()),
                            crate::mailbox::ExitReason::Other(ref s) => Some(s.clone()),
                        },
                        info.metadata.clone(),
                    ),
                    SystemMessage::HotSwap(_) => ("HOT_SWAP".to_string(), None, None, None),
                    SystemMessage::Ping => ("PING".to_string(), None, None, None),
                    SystemMessage::Pong => ("PONG".to_string(), None, None, None),
                };
                WrappedMessage {
                    data: None,
                    system: Some(JsSystemMessage {
                        type_name,
                        target_pid,
                        reason,
                        metadata,
                    }),
                }
            }
        }
    }
}

/// --- Runtime Wrapper ---

#[napi]
pub struct NodeRuntime {
    inner: Arc<crate::Runtime>,
}

#[napi]
impl NodeRuntime {
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(crate::Runtime::new()),
        }
    }

    // --- Core Spawning & Messaging ---

    #[napi]
    pub fn spawn(&self, handler: JsFunction, budget: Option<u32>) -> Result<i64> {
        let budget = budget.unwrap_or(100) as usize;

        // Create TSFN for the handler
        let tsfn: ThreadsafeFunction<Message, ErrorStrategy::Fatal> = handler
            .create_threadsafe_function(0, |ctx| {
                let msg = ctx.value;
                let js_val = message_to_js(&ctx.env, msg)?;
                Ok(vec![js_val])
            })?;

        // Shared behaviors for Hot Swap (RWLocked TSFN)
        // We wrap TSFN in a Send-safe container manually because TSFN is Send, but raw pointers are not.
        let behavior = Arc::new(parking_lot::RwLock::new(tsfn));

        let pid = self.inner.spawn_handler_with_budget(
            move |msg| {
                let b = behavior.clone();
                async move {
                    // Check for Hot Swap system message
                    if let Message::System(SystemMessage::HotSwap(ptr)) = msg {
                        unsafe {
                            // Reconstruct TSFN from raw pointer.
                            // We must ensure this pointer was created via Box::into_raw(Box::new(tsfn))
                            let new_tsfn_box = Box::from_raw(
                                ptr as *mut ThreadsafeFunction<Message, ErrorStrategy::Fatal>,
                            );
                            let mut guard = b.write();
                            *guard = *new_tsfn_box;
                        }
                        return;
                    }

                    // Normal execution
                    let guard = b.read();
                    guard.call(msg, ThreadsafeFunctionCallMode::NonBlocking);
                }
            },
            budget,
        );

        Ok(pid as i64)
    }

    #[napi]
    pub fn spawn_with_mailbox(&self, handler: JsFunction, budget: Option<u32>) -> Result<i64> {
        let budget = budget.unwrap_or(100) as usize;

        let tsfn: ThreadsafeFunction<JsMailbox, ErrorStrategy::Fatal> =
            handler.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

        let pid = self.inner.spawn_actor_with_budget(
            move |rx| async move {
                let mailbox = JsMailbox {
                    inner: Arc::new(Mutex::new(rx)),
                };
                tsfn.call(mailbox.clone(), ThreadsafeFunctionCallMode::NonBlocking);

                // Keep actor alive until mailbox is dropped by JS
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    if Arc::strong_count(&mailbox.inner) <= 1 {
                        break;
                    }
                }
            },
            budget,
        );

        Ok(pid as i64)
    }

    /// Spawn a handler backed by a bounded mailbox capacity. The JS callback
    /// receives messages just like `spawn` and will be invoked each time a
    /// user payload arrives. If the queue is full, additional sends will fail.
    #[napi]
    pub fn spawn_bounded(&self, handler: JsFunction, capacity: u32, budget: Option<u32>) -> Result<i64> {
        let budget = budget.unwrap_or(100) as usize;
        let cap = capacity as usize;

        // create TSFN binder like in `spawn`
        let tsfn: ThreadsafeFunction<Message, ErrorStrategy::Fatal> = handler
            .create_threadsafe_function(0, |ctx| {
                let msg = ctx.value;
                let js_val = message_to_js(&ctx.env, msg)?;
                Ok(vec![js_val])
            })?;

        let behavior = Arc::new(parking_lot::RwLock::new(tsfn));

        let pid = self.inner.spawn_actor_with_budget_bounded(
            move |mut rx| {
                let behavior = behavior.clone();
                async move {
                    while let Some(msg) = rx.recv().await {
                        // handle hot-swap at JS level
                        if let Message::System(SystemMessage::HotSwap(ptr)) = msg {
                            unsafe {
                                let new_tsfn_box = Box::from_raw(
                                    ptr as *mut ThreadsafeFunction<Message, ErrorStrategy::Fatal>,
                                );
                                let mut guard = behavior.write();
                                *guard = *new_tsfn_box;
                            }
                            continue;
                        }
                        let guard = behavior.read();
                        guard.call(msg, ThreadsafeFunctionCallMode::NonBlocking);
                    }
                }
            },
            budget,
            cap,
        );

        Ok(pid as i64)
    }

    #[napi]
    pub fn spawn_child(&self, parent: i64, handler: JsFunction, budget: Option<u32>) -> Result<i64> {
        let budget = budget.unwrap_or(100) as usize;

        let tsfn: ThreadsafeFunction<Message, ErrorStrategy::Fatal> = handler
            .create_threadsafe_function(0, |ctx| {
                let msg = ctx.value;
                let js_val = message_to_js(&ctx.env, msg)?;
                Ok(vec![js_val])
            })?;

        let behavior = Arc::new(parking_lot::RwLock::new(tsfn));
        let handler_wrapper = move |msg: crate::mailbox::Message| {
            let b = behavior.clone();
            async move {
                match msg {
                    crate::mailbox::Message::System(crate::mailbox::SystemMessage::HotSwap(ptr)) => {
                        unsafe {
                            let new_tsfn_box = Box::from_raw(
                                ptr as *mut ThreadsafeFunction<Message, ErrorStrategy::Fatal>,
                            );
                            let mut guard = b.write();
                            *guard = *new_tsfn_box;
                        }
                    }
                    crate::mailbox::Message::User(bytes) => {
                        let guard = b.read();
                        guard.call(msg, ThreadsafeFunctionCallMode::NonBlocking);
                    }
                    _ => {}
                }
            }
        };

        let pid = self
            .inner
            .spawn_child_with_budget(parent as u64, handler_wrapper, budget);
        Ok(pid as i64)
    }

    #[napi]
    pub fn spawn_child_with_mailbox(&self, parent: i64, handler: JsFunction, budget: Option<u32>) -> Result<i64> {
        let budget = budget.unwrap_or(100) as usize;

        let tsfn: ThreadsafeFunction<JsMailbox, ErrorStrategy::Fatal> =
            handler.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;

        let pid = self
            .inner
            .spawn_child_with_budget(parent as u64, move |rx| async move {
                let mailbox = JsMailbox {
                    inner: Arc::new(Mutex::new(rx)),
                };
                tsfn.call(mailbox.clone(), ThreadsafeFunctionCallMode::NonBlocking);

                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    if Arc::strong_count(&mailbox.inner) <= 1 {
                        break;
                    }
                }
            },
            budget);

        Ok(pid as i64)
    }

    #[napi]
    pub fn send(&self, pid: i64, data: Buffer) -> Result<bool> {
        let msg = crate::mailbox::Message::User(bytes::Bytes::from(data.to_vec()));
        Ok(self.inner.send(pid as u64, msg).is_ok())
    }

    #[napi]
    pub fn send_named(&self, name: String, data: Buffer) -> Result<bool> {
        let msg = crate::mailbox::Message::User(bytes::Bytes::from(data.to_vec()));
        Ok(self.inner.send_named(&name, msg).is_ok())
    }

    // --- Registry & Discovery ---

    #[napi]
    pub fn register(&self, name: String, pid: i64) {
        self.inner.register(name, pid as u64);
    }

    #[napi]
    pub fn resolve(&self, name: String) -> Option<i64> {
        self.inner.resolve(&name).map(|p| p as i64)
    }

    #[napi]
    pub async fn resolve_remote(&self, addr: String, name: String) -> Option<i64> {
        self.inner
            .resolve_remote_async(addr, name)
            .await
            .map(|p| p as i64)
    }

    #[napi]
    pub fn listen(&self, addr: String) {
        self.inner.listen(addr);
    }

    #[napi]
    pub fn send_remote(&self, addr: String, pid: i64, data: Buffer) {
        self.inner
            .send_remote(addr, pid as u64, bytes::Bytes::from(data.to_vec()));
    }

    #[napi]
    pub fn monitor_remote(&self, addr: String, pid: i64) {
        self.inner.monitor_remote(addr, pid as u64);
    }

    #[napi]
    pub async fn is_node_up(&self, addr: String) -> bool {
        tokio::net::TcpStream::connect(addr).await.is_ok()
    }

    // --- Lifecycle & Cluster Management ---

    #[napi]
    pub fn stop(&self, pid: i64) {
        self.inner.stop(pid as u64);
    }

    #[napi]
    pub async fn join(&self, pid: i64) {
        // Blocks the async future (yielding to JS event loop) until actor dies
        let rt = self.inner.clone();
        let pid_u64 = pid as u64;
        let wait_loop = async move {
            while rt.is_alive(pid_u64) {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        };
        wait_loop.await;
    }

    #[napi]
    pub fn is_alive(&self, pid: i64) -> bool {
        self.inner.is_alive(pid as u64)
    }

    #[napi]
    pub fn children_count(&self) -> u32 {
        self.inner.supervisor().children_count() as u32
    }

    #[napi]
    pub fn child_pids(&self) -> Vec<i64> {
        self.inner
            .supervisor()
            .child_pids()
            .into_iter()
            .map(|p| p as i64)
            .collect()
    }

    // --- Hot Swap ---

    #[napi]
    pub fn hot_swap(&self, pid: i64, new_handler: JsFunction) -> Result<()> {
        // Create a TSFN for the new handler
        let tsfn: ThreadsafeFunction<Message, ErrorStrategy::Fatal> = new_handler
            .create_threadsafe_function(0, |ctx| {
                let msg = ctx.value;
                let js_val = message_to_js(&ctx.env, msg)?;
                Ok(vec![js_val])
            })?;

        // Convert TSFN to a raw pointer to pass through the mailbox
        let ptr = Box::into_raw(Box::new(tsfn));

        // Send the HotSwap system message
        self.inner.hot_swap(pid as u64, ptr as usize);
        Ok(())
    }

    // --- Observation ---

    #[napi]
    pub fn spawn_observed_handler(&self, budget: Option<u32>) -> i64 {
        let b = budget.unwrap_or(100) as usize;
        self.inner.spawn_observed_handler(b) as i64
    }

    #[napi]
    pub fn get_messages(&self, pid: i64) -> Result<Vec<WrappedMessage>> {
        if let Some(msgs) = self.inner.get_observed_messages(pid as u64) {
            let wrapped = msgs.into_iter().map(WrappedMessage::from).collect();
            Ok(wrapped)
        } else {
            Ok(Vec::new())
        }
    }

    // --- Supervision ---

    #[napi]
    pub fn watch(&self, pid: i64, strategy: String) -> Result<()> {
        use crate::supervisor::ChildSpec;
        use crate::supervisor::RestartStrategy;

        let strat = match strategy.to_lowercase().as_str() {
            "restartone" | "restart_one" | "one" => RestartStrategy::RestartOne,
            "restartall" | "restart_all" | "all" => RestartStrategy::RestartAll,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "Invalid strategy".to_owned(),
                ))
            }
        };

        let pid_u64 = pid as u64;
        let spec = ChildSpec {
            factory: Arc::new(move || Ok(pid_u64)),
            strategy: strat,
        };
        self.inner.supervisor().add_child(pid_u64, spec);
        Ok(())
    }
}
