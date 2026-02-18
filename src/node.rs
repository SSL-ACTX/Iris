// src/node.rs
#![cfg(feature = "node")]

use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{JsBuffer, JsFunction, JsUnknown, Result};
use napi_derive::napi;
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::mailbox::{Message, SystemMessage};

/// --- System Message Wrapper ---

#[napi]
#[derive(Clone)]
pub struct JsSystemMessage {
    pub type_name: String,
    pub target_pid: Option<i64>,
}

fn message_to_js(env: &Env, msg: Message) -> Result<JsUnknown> {
    match msg {
        Message::User(bytes) => {
            let buf = env.create_buffer_with_data(bytes.to_vec())?.into_unknown();
            Ok(buf)
        }
        Message::System(sys) => {
            let (type_name, target_pid) = match sys {
                SystemMessage::Exit(pid) => ("EXIT".to_string(), Some(pid as i64)),
                SystemMessage::HotSwap(_) => ("HOT_SWAP".to_string(), None),
                SystemMessage::Ping => ("PING".to_string(), None),
                SystemMessage::Pong => ("PONG".to_string(), None),
            };
            let obj = JsSystemMessage { type_name, target_pid };
            env.to_js_value(&obj)
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
                 let (type_name, target_pid) = match sys {
                    SystemMessage::Exit(pid) => ("EXIT".to_string(), Some(pid as i64)),
                    SystemMessage::HotSwap(_) => ("HOT_SWAP".to_string(), None),
                    SystemMessage::Ping => ("PING".to_string(), None),
                    SystemMessage::Pong => ("PONG".to_string(), None),
                };
                WrappedMessage {
                    data: None,
                    system: Some(JsSystemMessage { type_name, target_pid }),
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
        Self { inner: Arc::new(crate::Runtime::new()) }
    }

    #[napi]
    pub fn spawn(&self, handler: JsFunction, budget: Option<u32>) -> Result<i64> {
        let budget = budget.unwrap_or(100) as usize;
        
        let tsfn: ThreadsafeFunction<Message, ErrorStrategy::Fatal> = handler
            .create_threadsafe_function(0, |ctx| {
                let msg = ctx.value;
                let js_val = message_to_js(&ctx.env, msg)?;
                Ok(vec![js_val])
            })?;

        let pid = self.inner.spawn_handler_with_budget(move |msg| {
            let tsfn = tsfn.clone();
            async move {
                tsfn.call(msg, ThreadsafeFunctionCallMode::NonBlocking);
            }
        }, budget);

        Ok(pid as i64)
    }

    #[napi]
    pub fn spawn_with_mailbox(&self, handler: JsFunction, budget: Option<u32>) -> Result<i64> {
         let budget = budget.unwrap_or(100) as usize;
         
         let tsfn: ThreadsafeFunction<JsMailbox, ErrorStrategy::Fatal> = handler
            .create_threadsafe_function(0, |ctx| {
                Ok(vec![ctx.value])
            })?;

        let pid = self.inner.spawn_actor_with_budget(move |rx| async move {
            let mailbox = JsMailbox { inner: Arc::new(Mutex::new(rx)) };
            
            tsfn.call(mailbox.clone(), ThreadsafeFunctionCallMode::NonBlocking);
            
            // Keep the Rust actor alive as long as the mailbox is held by JS
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if Arc::strong_count(&mailbox.inner) <= 1 {
                    break;
                }
            }
        }, budget);

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
        self.inner.resolve_remote_async(addr, name).await.map(|p| p as i64)
    }

    #[napi]
    pub fn listen(&self, addr: String) {
        self.inner.listen(addr);
    }

    #[napi]
    pub fn send_remote(&self, addr: String, pid: i64, data: Buffer) {
        self.inner.send_remote(addr, pid as u64, bytes::Bytes::from(data.to_vec()));
    }
    
    #[napi]
    pub fn monitor_remote(&self, addr: String, pid: i64) {
        self.inner.monitor_remote(addr, pid as u64);
    }

    #[napi]
    pub fn stop(&self, pid: i64) {
        self.inner.stop(pid as u64);
    }
    
    #[napi]
    pub async fn check_node_up(&self, addr: String) -> bool {
         tokio::net::TcpStream::connect(addr).await.is_ok()
    }
}