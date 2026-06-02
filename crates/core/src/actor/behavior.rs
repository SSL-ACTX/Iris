// src/core/behavior.rs
use crate::mailbox;
use crate::pid::Pid;
use crate::types::MAX_BEHAVIOR_HISTORY;
use crate::Runtime;

impl Runtime {
    /// Send a Hot Swap signal to the actor.
    pub fn hot_swap(&self, pid: Pid, handler_ptr: usize) {
        if let Some(sender) = self.mailboxes.0.get(&pid) {
            if sender
                .send_system(mailbox::SystemMessage::HotSwap(handler_ptr))
                .is_ok()
            {
                if let Some(mut ver) = self.behavior_versions.get_mut(&pid) {
                    *ver += 1;
                } else {
                    // initial behavior is version 1; first successful swap -> v2
                    self.behavior_versions.insert(pid, 2);
                }

                let mut history = self.behavior_history.entry(pid).or_default();
                history.push(handler_ptr);
                if history.len() > MAX_BEHAVIOR_HISTORY {
                    let overflow = history.len() - MAX_BEHAVIOR_HISTORY;
                    history.drain(0..overflow);
                }
            }
        }
    }

    /// Return current behavior version for an actor.
    pub fn behavior_version(&self, pid: Pid) -> u64 {
        self.behavior_versions
            .get(&pid)
            .map(|entry| *entry.value())
            .unwrap_or(1)
    }

    /// Roll back behavior by `steps` previously hot-swapped versions.
    ///
    /// Returns the new behavior version on success.
    pub fn rollback_behavior(&self, pid: Pid, steps: usize) -> Result<u64, String> {
        if steps == 0 {
            return Ok(self.behavior_version(pid));
        }

        let target_ptr = {
            let mut history = self
                .behavior_history
                .get_mut(&pid)
                .ok_or_else(|| "rollback failed: no behavior history".to_string())?;

            if history.len() <= steps {
                return Err("rollback failed: insufficient behavior history".to_string());
            }

            let target_idx = history.len() - 1 - steps;
            let target_ptr = history[target_idx];
            history.truncate(target_idx + 1);
            target_ptr
        };

        let sender = self
            .mailboxes
            .0
            .get(&pid)
            .ok_or_else(|| "rollback failed: pid not found".to_string())?;

        sender
            .send_system(mailbox::SystemMessage::HotSwap(target_ptr))
            .map_err(|_| "rollback failed: could not send hot swap".to_string())?;

        let next = self
            .behavior_version(pid)
            .saturating_sub(steps as u64)
            .max(1);
        self.behavior_versions.insert(pid, next);
        Ok(next)
    }
}
