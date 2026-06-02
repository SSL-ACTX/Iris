//! PID slab allocator (generational IDs) - Optimized with Atomics and Lock-free Free List

use crossbeam_queue::SegQueue;
use std::sync::atomic::{AtomicU32, Ordering};

/// Public PID type (u64 so it is FFI-friendly later).
pub type Pid = u64;

#[derive(Debug)]
#[repr(align(64))]
struct Entry {
    /// generation: u31 | occupied: u1
    state: AtomicU32,
}

/// Thread-safe slab allocator that produces generational `Pid` values.
///
/// PID layout: [ generation: u32 | index: u32 ]
pub struct SlabAllocator {
    entries: Box<[Entry]>,
    free_list: SegQueue<u32>,
    /// High-water mark for when free_list is empty
    next_index: AtomicU32,
    capacity: u32,
}

impl SlabAllocator {
    /// Create a slab allocator with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            entries.push(Entry {
                state: AtomicU32::new(1 << 1), // generation 1, occupied false (bit 0 = 0)
            });
        }
        Self {
            entries: entries.into_boxed_slice(),
            free_list: SegQueue::new(),
            next_index: AtomicU32::new(0),
            capacity: capacity as u32,
        }
    }

    /// Create an empty slab allocator with a default large capacity.
    pub fn new() -> Self {
        Self::with_capacity(1_000_000) // Default 1M capacity to avoid resizing
    }
}

impl Default for SlabAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SlabAllocator {
    /// Allocate a new slot and return its `Pid`.
    pub fn allocate(&self) -> Pid {
        if let Some(idx) = self.free_list.pop() {
            let entry = &self.entries[idx as usize];
            // Set occupied bit (bit 0)
            let old_state = entry.state.fetch_or(1, Ordering::SeqCst);
            let gen = old_state >> 1;
            return make_pid(idx as usize, gen);
        }

        let idx = self.next_index.fetch_add(1, Ordering::SeqCst);
        if idx >= self.capacity {
            // Fallback or panic? For Iris as-is, we return 0 as invalid.
            return 0;
        }

        let entry = &self.entries[idx as usize];
        entry.state.fetch_or(1, Ordering::SeqCst);
        make_pid(idx as usize, 1)
    }

    /// Deallocate a `Pid`. Returns `true` if the PID was valid and freed.
    pub fn deallocate(&self, pid: Pid) -> bool {
        let (idx, gen) = split_pid(pid);
        if idx >= self.entries.len() {
            return false;
        }
        let entry = &self.entries[idx];

        let old_state = entry.state.load(Ordering::SeqCst);
        let current_gen = old_state >> 1;
        let occupied = (old_state & 1) == 1;

        if !occupied || current_gen != gen {
            return false;
        }

        // Increment generation and clear occupied bit
        let next_gen = current_gen.wrapping_add(1).max(1);
        let new_state = (next_gen << 1) | 0;

        if entry
            .state
            .compare_exchange(old_state, new_state, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.free_list.push(idx as u32);
            true
        } else {
            // Contention or already deallocated
            false
        }
    }

    /// Check whether the pid currently refers to a live allocated slot.
    pub fn is_valid(&self, pid: Pid) -> bool {
        let (idx, gen) = split_pid(pid);
        if idx >= self.entries.len() {
            return false;
        }
        let state = self.entries[idx].state.load(Ordering::Relaxed);
        (state & 1) == 1 && (state >> 1) == gen
    }
}

fn make_pid(index: usize, generation: u32) -> Pid {
    ((generation as Pid) << 32) | (index as Pid)
}

fn split_pid(pid: Pid) -> (usize, u32) {
    ((pid & 0xffff_ffff) as usize, (pid >> 32) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_and_free() {
        let s = SlabAllocator::with_capacity(100);
        let a = s.allocate();
        let b = s.allocate();
        assert_ne!(a, b);
        assert!(s.is_valid(a));
        assert!(s.deallocate(a));
        assert!(!s.is_valid(a));
        // re-allocate should return a different generational PID
        let c = s.allocate();
        assert!(s.is_valid(c));
        assert_ne!(c, a);
    }
}
