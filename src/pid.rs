//! PID slab allocator (generational IDs)

/// Public PID type (u64 so it is FFI-friendly later).
pub type Pid = u64;

#[derive(Debug, Clone)]
struct Entry {
    generation: u32,
    occupied: bool,
}

/// Small slab allocator that produces generational `Pid` values.
///
/// PID layout: [ generation: u32 | index: u32 ]
pub struct SlabAllocator {
    entries: Vec<Entry>,
    free_list: Vec<usize>,
}

impl SlabAllocator {
    /// Create an empty slab allocator.
    pub fn new() -> Self {
        Self { entries: Vec::new(), free_list: Vec::new() }
    }

    /// Allocate a new slot and return its `Pid`.
    pub fn allocate(&mut self) -> Pid {
        if let Some(idx) = self.free_list.pop() {
            let entry = &mut self.entries[idx];
            entry.occupied = true;
            make_pid(idx, entry.generation)
        } else {
            let idx = self.entries.len();
            self.entries.push(Entry { generation: 1, occupied: true });
            make_pid(idx, 1)
        }
    }

    /// Deallocate a `Pid`. Returns `true` if the PID was valid and freed.
    pub fn deallocate(&mut self, pid: Pid) -> bool {
        let (idx, gen) = split_pid(pid);
        if idx >= self.entries.len() {
            return false;
        }
        let entry = &mut self.entries[idx];
        if !entry.occupied || entry.generation != gen {
            return false;
        }
        entry.occupied = false;
        // bump generation so stale PIDs are easily detected
        entry.generation = entry.generation.wrapping_add(1);
        self.free_list.push(idx);
        true
    }

    /// Check whether the pid currently refers to a live allocated slot.
    pub fn is_valid(&self, pid: Pid) -> bool {
        let (idx, gen) = split_pid(pid);
        idx < self.entries.len() && self.entries[idx].occupied && self.entries[idx].generation == gen
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
        let mut s = SlabAllocator::new();
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
