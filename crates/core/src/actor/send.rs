// src/core/send.rs
use crate::mailbox;
use crate::pid::Pid;
use crate::Runtime;
use std::sync::atomic::Ordering;
use std::time::Duration;

impl Runtime {
    /// Schedule a one-shot message to be sent after `delay_ms` milliseconds.
    /// Returns a timer id that can be used to cancel the pending send.
    pub fn send_after(&self, pid: Pid, delay_ms: u64, msg: mailbox::Message) -> u64 {
        let id = self.timer_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        self.timers.lock().unwrap().insert(id, tx);

        let rt_clone = self.clone();
        crate::RUNTIME.spawn(async move {
            let sleep = tokio::time::sleep(std::time::Duration::from_millis(delay_ms));
            tokio::select! {
                _ = sleep => {
                    let _ = rt_clone.send(pid, msg);
                }
                _ = rx => {
                    // cancelled
                }
            }
            let _ = rt_clone.timers.lock().unwrap().remove(&id);
        });

        id
    }

    /// Schedule a repeating interval that sends `msg` every `interval_ms` milliseconds.
    /// Returns a timer id that can be used to cancel the interval.
    pub fn send_interval(&self, pid: Pid, interval_ms: u64, msg: mailbox::Message) -> u64 {
        let id = self.timer_counter.fetch_add(1, Ordering::SeqCst) + 1;
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        self.timers.lock().unwrap().insert(id, tx);

        let rt_clone = self.clone();
        crate::RUNTIME.spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(interval_ms));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let _ = rt_clone.send(pid, msg.clone());
                    }
                    _ = &mut rx => {
                        break;
                    }
                }
            }
            let _ = rt_clone.timers.lock().unwrap().remove(&id);
        });

        id
    }

    /// Cancel a scheduled timer/interval. Returns true if a timer was cancelled.
    pub fn cancel_timer(&self, timer_id: u64) -> bool {
        if let Some(tx) = self.timers.lock().unwrap().remove(&timer_id) {
            let _ = tx.send(());
            true
        } else {
            false
        }
    }

    pub fn send(&self, pid: Pid, msg: mailbox::Message) -> Result<(), mailbox::Message> {
        let _ = self.ensure_virtual_actor_active(pid);

        // apply overflow policy if bounded
        if let Some(cap) = self.bounded_capacity.get(&pid) {
            let size = self.mailbox_size(pid).unwrap_or(0);
            if size >= *cap {
                if let Some(pol) = self.overflow_policy.get(&pid) {
                    match pol.value() {
                        mailbox::OverflowPolicy::DropNew => return Err(msg),
                        mailbox::OverflowPolicy::DropOld => {
                            // tell receiver to drop oldest user message
                            if let Some(sender) = self.mailboxes.0.get(&pid) {
                                let _ = sender.send_system(mailbox::SystemMessage::DropOld);
                            }
                            if !self.wait_for_mailbox_capacity(pid, *cap) {
                                return Err(msg);
                            }
                        }
                        mailbox::OverflowPolicy::Block => {
                            if !self.wait_for_mailbox_capacity(pid, *cap) {
                                return Err(msg);
                            }
                        }
                        mailbox::OverflowPolicy::Redirect(target) => {
                            if *target == pid {
                                return Err(msg);
                            }
                            // forward to target and consider message handled
                            return self.send(*target, msg);
                        }
                        mailbox::OverflowPolicy::Spill(target) => {
                            if *target != pid {
                                // best-effort spill copy; primary send result determines return value
                                let _ = self.send(*target, msg.clone());
                            }
                            if !self.wait_for_mailbox_capacity(pid, *cap) {
                                return Err(msg);
                            }
                        }
                    }
                } else {
                    return Err(msg);
                }
            }
        }
        let result = if let Some(sender) = self.mailboxes.0.get(&pid) {
            let len = match &msg {
                mailbox::Message::User(b) => b.len(),
                _ => 0,
            };
            let res = sender.send(msg);
            if res.is_ok() {
                self.telemetry
                    .0
                    .log_event(crate::telemetry::TelemetryEvent::MessageSent {
                        from: 0, // Unknown from send API
                        to: pid,
                        len,
                    });
                self.update_backpressure_after_enqueue(pid, sender.value());
            }
            res
        } else {
            Err(msg)
        };

        result
    }

    /// Capability-checked send.
    pub fn send_link(
        &self,
        link: &crate::actor::capability::Link,
        msg: mailbox::Message,
    ) -> Result<(), mailbox::Message> {
        use crate::actor::capability::Capability;
        match &msg {
            mailbox::Message::User(_) | mailbox::Message::Request { .. } => {
                if !link.has_capability(Capability::Send) {
                    return Err(msg);
                }
            }
            _ => {
                if !link.has_capability(Capability::Signal) {
                    return Err(msg);
                }
            }
        }

        self.send(link.target(), msg)
    }

    /// Send with immediate backpressure feedback for bounded mailboxes.
    ///
    /// This avoids an additional map lookup by computing pressure on the same
    /// enqueue path and returning the resulting level to the caller.
    pub fn send_with_backpressure(
        &self,
        pid: Pid,
        msg: mailbox::Message,
    ) -> Result<mailbox::BackpressureLevel, mailbox::Message> {
        self.send(pid, msg)?;
        Ok(self.current_backpressure_state(pid))
    }

    /// Send user bytes with a fast path that avoids wrapping in `Message` at callsite.
    /// Internal helper that sends raw bytes to an actor mailbox.
    ///
    /// This exists purely for performance; the public `send` API already
    /// routes through this path automatically.  It is **not** exposed to the
    /// Python bindings and is hidden from generated documentation.
    #[doc(hidden)]
    /// Send a request to an actor and await a response.
    /// Spawns a temporary observer actor to receive the reply.
    pub async fn call(
        &self,
        pid: Pid,
        payload: bytes::Bytes,
        timeout: Duration,
    ) -> Result<bytes::Bytes, String> {
        let reply_pid = self.spawn_observed_handler(1);

        let req = mailbox::Message::Request {
            reply_to: reply_pid,
            payload,
        };

        if self.send(pid, req).is_err() {
            self.stop(reply_pid);
            return Err("send failed".to_string());
        }

        let op = async {
            loop {
                if let Some(mailbox::Message::User(b)) =
                    self.take_observed_message_matching(reply_pid, |_| true)
                {
                    return Ok(b);
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        };

        let res = match tokio::time::timeout(timeout, op).await {
            Ok(val) => val,
            Err(_) => Err("timeout".to_string()),
        };

        self.stop(reply_pid);
        res
    }

    pub async fn call_link(
        &self,
        link: &crate::actor::capability::Link,
        payload: bytes::Bytes,
        timeout: Duration,
    ) -> Result<bytes::Bytes, String> {
        if !link.has_capability(crate::actor::capability::Capability::Send) {
            return Err("missing Send capability".to_string());
        }
        self.call(link.target(), payload, timeout).await
    }

    pub fn send_user(&self, pid: Pid, bytes: bytes::Bytes) -> Result<(), bytes::Bytes> {
        let _ = self.ensure_virtual_actor_active(pid);

        // emulate same overflow-policy logic but with raw bytes
        if let Some(cap) = self.bounded_capacity.get(&pid) {
            let size = self.mailbox_size(pid).unwrap_or(0);
            if size >= *cap {
                if let Some(pol) = self.overflow_policy.get(&pid) {
                    match pol.value() {
                        mailbox::OverflowPolicy::DropNew => return Err(bytes),
                        mailbox::OverflowPolicy::DropOld => {
                            if let Some(sender) = self.mailboxes.0.get(&pid) {
                                let _ = sender.send_system(mailbox::SystemMessage::DropOld);
                            }
                            if !self.wait_for_mailbox_capacity(pid, *cap) {
                                return Err(bytes);
                            }
                        }
                        mailbox::OverflowPolicy::Block => {
                            if !self.wait_for_mailbox_capacity(pid, *cap) {
                                return Err(bytes);
                            }
                        }
                        mailbox::OverflowPolicy::Redirect(target) => {
                            if *target == pid {
                                return Err(bytes);
                            }
                            return self.send_user(*target, bytes);
                        }
                        mailbox::OverflowPolicy::Spill(target) => {
                            if *target != pid {
                                let _ = self.send_user(*target, bytes.clone());
                            }
                            if !self.wait_for_mailbox_capacity(pid, *cap) {
                                return Err(bytes);
                            }
                        }
                    }
                } else {
                    return Err(bytes);
                }
            }
        }
        let result = if let Some(sender) = self.mailboxes.0.get(&pid) {
            let len = bytes.len();
            let res = sender.send_user_bytes(bytes);
            if res.is_ok() {
                self.telemetry
                    .0
                    .log_event(crate::telemetry::TelemetryEvent::MessageSent {
                        from: 0,
                        to: pid,
                        len,
                    });
                self.update_backpressure_after_enqueue(pid, sender.value());
            }
            res
        } else {
            Err(bytes)
        };

        result
    }

    /// Send user bytes with immediate backpressure feedback for bounded mailboxes.
    pub fn send_user_with_backpressure(
        &self,
        pid: Pid,
        bytes: bytes::Bytes,
    ) -> Result<mailbox::BackpressureLevel, bytes::Bytes> {
        self.send_user(pid, bytes)?;
        Ok(self.current_backpressure_state(pid))
    }

    /// Send a batch of user payloads and return the number accepted.
    ///
    /// For unbounded mailboxes this uses a single sender lookup for the whole
    /// batch. Bounded mailboxes fall back to per-message `send_user` to preserve
    /// overflow policy semantics.
    #[doc(hidden)]
    pub fn send_user_many(&self, pid: Pid, payloads: Vec<bytes::Bytes>) -> usize {
        let _ = self.ensure_virtual_actor_active(pid);

        if self.bounded_capacity.contains_key(&pid) {
            let mut accepted = 0usize;
            for payload in payloads {
                if self.send_user(pid, payload).is_ok() {
                    accepted += 1;
                }
            }
            return accepted;
        }

        let Some(sender) = self.mailboxes.0.get(&pid) else {
            return 0;
        };

        let mut accepted = 0usize;
        for payload in payloads {
            let len = payload.len();
            if sender.send_user_bytes(payload).is_ok() {
                self.telemetry
                    .0
                    .log_event(crate::telemetry::TelemetryEvent::MessageSent {
                        from: 0,
                        to: pid,
                        len,
                    });
                accepted += 1;
            } else {
                break;
            }
        }
        accepted
    }

    /// Send shared bytes from Arc-backed storage.
    ///
    /// This uses owner-backed `Bytes` construction, avoiding payload copy.
    pub fn send_user_shared(
        &self,
        pid: Pid,
        bytes: std::sync::Arc<[u8]>,
    ) -> Result<(), std::sync::Arc<[u8]>> {
        let payload = bytes::Bytes::from_owner(bytes.clone());
        self.send_user(pid, payload).map_err(|_| bytes)
    }

    /// Send shared bytes with immediate backpressure feedback.
    pub fn send_user_shared_with_backpressure(
        &self,
        pid: Pid,
        bytes: std::sync::Arc<[u8]>,
    ) -> Result<mailbox::BackpressureLevel, std::sync::Arc<[u8]>> {
        self.send_user_shared(pid, bytes)?;
        Ok(self
            .mailbox_backpressure(pid)
            .unwrap_or(mailbox::BackpressureLevel::Normal))
    }

    /// Send a static byte slice with zero allocation at callsite.
    pub fn send_user_static(&self, pid: Pid, bytes: &'static [u8]) -> Result<(), &'static [u8]> {
        self.send_user(pid, bytes::Bytes::from_static(bytes))
            .map_err(|_| bytes)
    }

    /// Send static bytes with immediate backpressure feedback.
    pub fn send_user_static_with_backpressure(
        &self,
        pid: Pid,
        bytes: &'static [u8],
    ) -> Result<mailbox::BackpressureLevel, &'static [u8]> {
        self.send_user_static(pid, bytes)?;
        Ok(self
            .mailbox_backpressure(pid)
            .unwrap_or(mailbox::BackpressureLevel::Normal))
    }

    /// Set overflow policy for an existing bounded mailbox.
    pub fn set_overflow_policy(&self, pid: Pid, policy: mailbox::OverflowPolicy) {
        self.overflow_policy.insert(pid, policy);
    }

    /// Return the number of queued user messages for the actor with `pid`.
    pub fn mailbox_size(&self, pid: Pid) -> Option<usize> {
        self.mailboxes.0.get(&pid).map(|s| s.len())
    }

    pub fn mailbox_backpressure(&self, pid: Pid) -> Option<mailbox::BackpressureLevel> {
        self.mailboxes.0.get(&pid).map(|sender| {
            let cap = self.bounded_capacity.get(&pid).map(|entry| *entry.value());
            sender.backpressure_level(cap)
        })
    }

    pub(crate) fn wait_for_mailbox_capacity(&self, pid: Pid, cap: usize) -> bool {
        let mut spins = 0usize;
        while self.mailbox_size(pid).unwrap_or(0) >= cap {
            if !self.mailboxes.0.contains_key(&pid) {
                return false;
            }

            spins += 1;
            if spins <= 64 {
                std::hint::spin_loop();
            } else if spins <= 512 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }

        true
    }

    pub(crate) fn update_backpressure_after_enqueue(
        &self,
        pid: Pid,
        sender: &mailbox::MailboxSender,
    ) -> Option<mailbox::BackpressureLevel> {
        // Unbounded mailboxes are always considered Normal, skip map churn.
        let cap = self
            .bounded_capacity
            .get(&pid)
            .map(|entry| *entry.value())?;

        let prev = self
            .backpressure_state
            .get(&pid)
            .map(|entry| *entry.value())
            .unwrap_or(mailbox::BackpressureLevel::Normal);
        let level = sender.backpressure_level_with_hysteresis(Some(cap), prev);
        self.emit_backpressure_signal(pid, level);
        Some(level)
    }

    pub(crate) fn emit_backpressure_signal(&self, pid: Pid, level: mailbox::BackpressureLevel) {
        let existing = self
            .backpressure_state
            .get(&pid)
            .map(|entry| *entry.value())
            .unwrap_or(mailbox::BackpressureLevel::Normal);

        if existing != level {
            self.backpressure_state.insert(pid, level);
        }
    }

    pub(crate) fn current_backpressure_state(&self, pid: Pid) -> mailbox::BackpressureLevel {
        self.backpressure_state
            .get(&pid)
            .map(|entry| *entry.value())
            .unwrap_or(mailbox::BackpressureLevel::Normal)
    }
}
