// src/core/spawn.rs
use super::types::{ErasedMessageHandler, VirtualActorSpec};
use super::Runtime;
use crate::mailbox;
use crate::pid::Pid;
use std::future::Future;
use std::sync::{Arc, Mutex};
use tokio::time::Duration;

impl Runtime {
    pub fn is_alive(&self, pid: Pid) -> bool {
        let slab = self.slab.lock().unwrap();
        slab.is_valid(pid)
    }

    pub fn spawn_actor<H, Fut>(&self, handler: H) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        if self.is_load_shedding_active() {
            tracing::warn!("[Iris] Load shedding active, rejecting actor spawn");
            return 0; // Return invalid PID
        }
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        super::RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(handler(rx));
            let res = actor_handle.await;

            // Determine exit reason and metadata
            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            Runtime::finalize_actor_exit(
                &mailboxes2,
                &supervisor2,
                &slab2,
                &path_supervisors2,
                &rt_exit_clone,
                pid,
                reason,
                meta,
            );
        });

        pid
    }

    /// Bounded mailbox variant of spawn_actor.
    pub fn spawn_actor_bounded<H, Fut>(&self, handler: H, capacity: usize) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        if self.is_load_shedding_active() {
            tracing::warn!("[Iris] Load shedding active, rejecting actor spawn");
            return 0;
        }
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::bounded_channel(capacity);
        self.mailboxes.insert(pid, tx.clone());
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);
        // track capacity and default policy
        self.bounded_capacity.insert(pid, capacity);
        self.overflow_policy
            .insert(pid, mailbox::OverflowPolicy::DropNew);

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        super::RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(handler(rx));
            let res = actor_handle.await;

            // Determine exit reason and metadata
            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            Runtime::finalize_actor_exit(
                &mailboxes2,
                &supervisor2,
                &slab2,
                &path_supervisors2,
                &rt_exit_clone,
                pid,
                reason,
                meta,
            );
        });

        pid
    }

    /// Bounded variant of spawn_actor_with_budget.
    pub fn spawn_actor_with_budget_bounded<H, Fut>(
        &self,
        handler: H,
        _budget: usize,
        capacity: usize,
    ) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        if self.is_load_shedding_active() {
            tracing::warn!("[Iris] Load shedding active, rejecting actor spawn");
            return 0;
        }
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::bounded_channel(capacity);
        self.mailboxes.insert(pid, tx.clone());
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);
        // track capacity and default overflow policy
        self.bounded_capacity.insert(pid, capacity);
        self.overflow_policy
            .insert(pid, mailbox::OverflowPolicy::DropNew);

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();
        let fut = handler(rx);

        super::RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(fut);
            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            Runtime::finalize_actor_exit(
                &mailboxes2,
                &supervisor2,
                &slab2,
                &path_supervisors2,
                &rt_exit_clone,
                pid,
                reason,
                meta,
            );
        });

        pid
    }

    pub fn spawn_actor_with_budget<H, Fut>(&self, handler: H, _budget: usize) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        if self.is_load_shedding_active() {
            tracing::warn!("[Iris] Load shedding active, rejecting actor spawn");
            return 0;
        }
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));

        let mailboxes2 = self.mailboxes.clone();
        let supervisor2 = self.supervisor.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();
        let fut = handler(rx);

        super::RUNTIME.spawn(async move {
            let actor_handle = tokio::spawn(fut);
            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            Runtime::finalize_actor_exit(
                &mailboxes2,
                &supervisor2,
                &slab2,
                &path_supervisors2,
                &rt_exit_clone,
                pid,
                reason,
                meta,
            );
        });

        pid
    }

    pub fn spawn_handler_with_budget<H, Fut>(&self, handler: H, budget: usize) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        if self.is_load_shedding_active() {
            tracing::warn!("[Iris] Load shedding active, rejecting actor spawn");
            return 0;
        }
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));

        let handler = std::sync::Arc::new(handler);
        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        super::RUNTIME.spawn(async move {
            let h_loop = handler.clone();
            #[cfg(feature = "vortex")]
            let rt_vortex_clone = rt_exit_clone.clone();

            #[cfg(feature = "vortex")]
            let mut vortex_engine = rt_exit_clone.get_vortex_engine().unwrap_or_default();

            let rt_inner = rt_exit_clone.clone();
            let actor_handle = tokio::spawn(async move {
                let mut processed = 0usize;
                #[cfg(feature = "vortex")]
                let mut dynamic_budget = budget.max(1);
                #[cfg(not(feature = "vortex"))]
                let dynamic_budget = budget.max(1);
                while let Some(first_msg) = rx.recv().await {
                    rt_inner
                        .telemetry()
                        .log_event(crate::core::telemetry::TelemetryEvent::MessageReceived { pid });
                    #[cfg(feature = "vortex")]
                    if rt_vortex_clone
                        .is_vortex_watchdog_enabled()
                        .unwrap_or(false)
                    {
                        tokio::task::yield_now().await;
                    }

                    #[cfg(feature = "vortex")]
                    let mut saw_suspend_in_cycle = false;
                    #[cfg(not(feature = "vortex"))]
                    let _saw_suspend_in_cycle = false;

                    #[cfg(feature = "vortex")]
                    let enable_genetic_budgeting = rt_vortex_clone
                        .is_vortex_genetic_budgeting_enabled()
                        .unwrap_or(false);
                    #[cfg(not(feature = "vortex"))]
                    let _enable_genetic_budgeting = false;

                    #[cfg(feature = "vortex")]
                    {
                        if vortex_engine.preempt_tick().is_err() {
                            saw_suspend_in_cycle = true;
                            let (suspend_count, total_count) = rt_vortex_clone
                                .get_vortex_genetic_history(pid)
                                .unwrap_or((0, 0));
                            rt_vortex_clone.vortex_genetic_history.insert(
                                pid,
                                (
                                    suspend_count.saturating_add(1),
                                    total_count.saturating_add(1),
                                ),
                            );
                            rt_vortex_clone.auto_checkpoint_and_replay_on_suspend(pid, budget);
                            vortex_engine.detach_stalled_thread();
                            vortex_engine.replenish_budget(budget);
                            tokio::task::yield_now().await;
                            vortex_engine.reclaim_thread();
                            if enable_genetic_budgeting {
                                let (low, high) = rt_vortex_clone
                                    .get_vortex_genetic_thresholds()
                                    .unwrap_or((0.4, 0.7));
                                dynamic_budget = crate::core::types::next_dynamic_budget(
                                    dynamic_budget,
                                    budget,
                                    saw_suspend_in_cycle,
                                    0.0,
                                    low,
                                    high,
                                );
                            }
                            continue;
                        }
                    }

                    let h = h_loop.clone();
                    (h)(first_msg).await;
                    processed += 1;

                    while processed < dynamic_budget {
                        match rx.try_recv() {
                            Some(next_msg) => {
                                rt_inner.telemetry().log_event(
                                    crate::core::telemetry::TelemetryEvent::MessageReceived { pid },
                                );
                                #[cfg(feature = "vortex")]
                                {
                                    if vortex_engine.preempt_tick().is_err() {
                                        saw_suspend_in_cycle = true;
                                        rt_vortex_clone
                                            .auto_checkpoint_and_replay_on_suspend(pid, budget);
                                        vortex_engine.detach_stalled_thread();
                                        vortex_engine.replenish_budget(budget);
                                        tokio::task::yield_now().await;
                                        vortex_engine.reclaim_thread();
                                        if enable_genetic_budgeting {
                                            let (low, high) = rt_vortex_clone
                                                .get_vortex_genetic_thresholds()
                                                .unwrap_or((0.4, 0.7));
                                            dynamic_budget =
                                                crate::core::types::next_dynamic_budget(
                                                    dynamic_budget,
                                                    budget,
                                                    saw_suspend_in_cycle,
                                                    0.0,
                                                    low,
                                                    high,
                                                );
                                        }
                                        break;
                                    }
                                }

                                let h = h_loop.clone();
                                (h)(next_msg).await;
                                processed += 1;
                            }
                            None => break,
                        }
                    }

                    #[cfg(feature = "vortex")]
                    {
                        let (suspend_count, total_count) = rt_vortex_clone
                            .get_vortex_genetic_history(pid)
                            .unwrap_or((0, 0));
                        let total_count = total_count.saturating_add(1);
                        let suspend_count = suspend_count + (saw_suspend_in_cycle as usize);
                        let suspend_rate = if total_count == 0 {
                            0.0
                        } else {
                            (suspend_count as f64) / (total_count as f64)
                        };

                        rt_vortex_clone
                            .vortex_genetic_history
                            .insert(pid, (suspend_count, total_count));

                        if enable_genetic_budgeting {
                            let (low, high) = rt_vortex_clone
                                .get_vortex_genetic_thresholds()
                                .unwrap_or((0.4, 0.7));

                            dynamic_budget = crate::core::types::next_dynamic_budget(
                                dynamic_budget,
                                budget,
                                saw_suspend_in_cycle,
                                suspend_rate,
                                low,
                                high,
                            );
                        }
                    }

                    if processed >= dynamic_budget {
                        processed = 0;
                        tokio::task::yield_now().await;
                    }
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            Runtime::finalize_actor_exit(
                &mailboxes2,
                &supervisor2,
                &slab2,
                &path_supervisors2,
                &rt_exit_clone,
                pid,
                reason,
                meta,
            );
        });

        pid
    }

    /// Spawn a new message-handler actor tied to `parent`.
    pub fn spawn_child_handler_with_budget<H, Fut>(
        &self,
        parent: Pid,
        handler: H,
        budget: usize,
    ) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_handler_with_budget(handler, budget);
        self.parent_of.insert(pid, parent);
        self.children_by_parent.entry(parent).or_default().push(pid);
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    pub fn spawn_observed_handler(&self, _budget: usize) -> Pid {
        if self.is_load_shedding_active() {
            tracing::warn!("[Iris] Load shedding active, rejecting actor spawn");
            return 0;
        }
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));
        self.backpressure_state
            .insert(pid, mailbox::BackpressureLevel::Normal);
        let vec = Arc::new(Mutex::new(Vec::new()));
        self.observers.insert(pid, vec.clone());

        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        super::RUNTIME.spawn(async move {
            let v_clone = vec.clone();
            let rt_inner = rt_exit_clone.clone();
            let actor_handle = tokio::spawn(async move {
                while let Some(msg) = rx.recv().await {
                    rt_inner
                        .telemetry()
                        .log_event(crate::core::telemetry::TelemetryEvent::MessageReceived { pid });
                    {
                        let mut guard = v_clone.lock().unwrap();
                        guard.push(msg);
                    }

                    while let Some(next_msg) = rx.try_recv() {
                        rt_inner.telemetry().log_event(
                            crate::core::telemetry::TelemetryEvent::MessageReceived { pid },
                        );
                        let mut guard = v_clone.lock().unwrap();
                        guard.push(next_msg);
                    }

                    tokio::task::yield_now().await;
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            Runtime::finalize_actor_exit(
                &mailboxes2,
                &supervisor2,
                &slab2,
                &path_supervisors2,
                &rt_exit_clone,
                pid,
                reason,
                meta,
            );
        });

        pid
    }

    /// Reserve a virtual/lazy actor PID. The actor is activated on first send.
    ///
    /// `idle_timeout` controls auto-shutdown after inactivity once activated.
    pub fn spawn_virtual_handler_with_budget<H, Fut>(
        &self,
        handler: H,
        budget: usize,
        idle_timeout: Option<Duration>,
    ) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        if self.is_load_shedding_active() {
            tracing::warn!("[Iris] Load shedding active, rejecting actor spawn");
            return 0;
        }
        let mut slab = self.slab.lock().unwrap();
        let pid = slab.allocate();
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));

        let erased: ErasedMessageHandler =
            Arc::new(move |msg: mailbox::Message| Box::pin(handler(msg)));

        self.virtual_specs.insert(
            pid,
            VirtualActorSpec {
                handler: erased,
                budget,
                idle_timeout,
            },
        );
        self.virtual_activate_locks
            .insert(pid, Arc::new(Mutex::new(())));

        pid
    }

    /// Reserve a virtual/lazy actor with default budget and no idle timeout.
    pub fn spawn_virtual_handler<H, Fut>(&self, handler: H) -> Pid
    where
        H: Fn(mailbox::Message) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.spawn_virtual_handler_with_budget(handler, 100, None)
    }

    pub(crate) fn ensure_virtual_actor_active(&self, pid: Pid) -> bool {
        if self.mailboxes.contains_key(&pid) {
            return true;
        }

        let lock = if let Some(lock_entry) = self.virtual_activate_locks.get(&pid) {
            lock_entry.clone()
        } else {
            return false;
        };

        let _guard = lock.lock().unwrap();

        if self.mailboxes.contains_key(&pid) {
            return true;
        }

        let spec = if let Some((_, spec)) = self.virtual_specs.remove(&pid) {
            spec
        } else {
            return self.mailboxes.contains_key(&pid);
        };
        self.virtual_activate_locks.remove(&pid);

        let (tx, mut rx) = mailbox::channel();
        self.mailboxes.insert(pid, tx.clone());
        self.telemetry
            .log_event(crate::core::telemetry::TelemetryEvent::ActorSpawned {
                pid,
                path: "".to_string(),
            });
        #[cfg(feature = "vortex")]
        self.vortex_genetic_history.insert(pid, (0, 0));

        let handler = spec.handler.clone();
        let budget = spec.budget;
        let idle_timeout = spec.idle_timeout;

        let supervisor2 = self.supervisor.clone();
        let mailboxes2 = self.mailboxes.clone();
        let slab2 = self.slab.clone();
        let path_supervisors2 = self.path_supervisors.clone();
        let rt_exit_clone = self.clone();

        super::RUNTIME.spawn(async move {
            let h_loop = handler.clone();
            let rt_inner = rt_exit_clone.clone();
            let actor_handle = tokio::spawn(async move {
                let mut processed = 0usize;
                loop {
                    let first_msg = if let Some(idle) = idle_timeout {
                        match tokio::time::timeout(idle, rx.recv()).await {
                            Ok(maybe) => maybe,
                            Err(_) => break,
                        }
                    } else {
                        rx.recv().await
                    };

                    let Some(first_msg) = first_msg else {
                        break;
                    };

                    rt_inner
                        .telemetry()
                        .log_event(crate::core::telemetry::TelemetryEvent::MessageReceived { pid });

                    let h = h_loop.clone();
                    (h)(first_msg).await;
                    processed += 1;

                    while processed < budget {
                        match rx.try_recv() {
                            Some(next_msg) => {
                                rt_inner.telemetry().log_event(
                                    crate::core::telemetry::TelemetryEvent::MessageReceived { pid },
                                );
                                let h = h_loop.clone();
                                (h)(next_msg).await;
                                processed += 1;
                            }
                            None => break,
                        }
                    }

                    if processed >= budget {
                        processed = 0;
                        tokio::task::yield_now().await;
                    }
                }
            });

            let res = actor_handle.await;

            let (reason, meta) = match res {
                Ok(_) => (crate::mailbox::ExitReason::Normal, None),
                Err(e) => {
                    if e.is_panic() {
                        (
                            crate::mailbox::ExitReason::Panic,
                            Some(format!("join_error: {:?}", e)),
                        )
                    } else {
                        (
                            crate::mailbox::ExitReason::Other("join_error".to_string()),
                            Some(format!("join_error: {:?}", e)),
                        )
                    }
                }
            };

            Runtime::finalize_actor_exit(
                &mailboxes2,
                &supervisor2,
                &slab2,
                &path_supervisors2,
                &rt_exit_clone,
                pid,
                reason,
                meta,
            );
        });

        true
    }

    /// Spawn an actor whose lifetime is tied to `parent`.
    /// When the parent PID exits (normal or crash) the child will be
    /// automatically stopped as well.  The returned PID behaves just like
    /// one created with `spawn_actor`.
    pub fn spawn_child<H, Fut>(&self, parent: Pid, handler: H) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_actor(handler);
        // record relationships for structured concurrency
        self.parent_of.insert(pid, parent);
        self.children_by_parent.entry(parent).or_default().push(pid);
        // if parent is already dead, immediately stop the child
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    /// Same as `spawn_child` but accepts a budget for cooperative scheduling.
    pub fn spawn_child_with_budget<H, Fut>(&self, parent: Pid, handler: H, budget: usize) -> Pid
    where
        H: FnOnce(mailbox::MailboxReceiver) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let pid = self.spawn_actor_with_budget(handler, budget);
        self.parent_of.insert(pid, parent);
        self.children_by_parent.entry(parent).or_default().push(pid);
        if !self.is_alive(parent) {
            self.stop(pid);
        }
        pid
    }

    pub fn get_observed_messages(&self, pid: Pid) -> Option<Vec<mailbox::Message>> {
        self.observers
            .get(&pid)
            .map(|entry| entry.value().lock().unwrap().clone())
    }

    /// Remove and return a single observed message matching the predicate.
    /// Used by FFI helpers to implement selective receive for observed actors.
    pub fn take_observed_message_matching<F>(
        &self,
        pid: Pid,
        matcher: F,
    ) -> Option<mailbox::Message>
    where
        F: FnMut(&mailbox::Message) -> bool,
    {
        if let Some(entry) = self.observers.get(&pid) {
            let mut guard = entry.value().lock().unwrap();
            if let Some(pos) = guard.iter().position(matcher) {
                return Some(guard.remove(pos));
            }
        }
        None
    }
}
