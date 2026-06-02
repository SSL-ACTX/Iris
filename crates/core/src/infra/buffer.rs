use bytes::BytesMut;
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub type BufferId = u64;

/// A high-performance buffer registry with zero-copy pooling.
pub struct BufferRegistry {
    /// Maps BufferId to a pre-allocated or active buffer.
    active: DashMap<BufferId, BytesMut>,
    /// Pool of reusable buffers to minimize allocations.
    pool: SegQueue<BytesMut>,
    next: AtomicU64,
}

impl BufferRegistry {
    pub fn new() -> Self {
        let registry = BufferRegistry {
            active: DashMap::new(),
            pool: SegQueue::new(),
            next: AtomicU64::new(1),
        };

        // Pre-allocate some buffers (Preload RAM)
        for _ in 0..1024 {
            registry.pool.push(BytesMut::with_capacity(4096));
        }

        registry
    }

    pub fn allocate(&self, size: usize) -> BufferId {
        let id = self.next.fetch_add(1, Ordering::SeqCst);

        // Try to get from pool if size fits, else allocate new
        let mut buf = if let Some(mut b) = self.pool.pop() {
            if b.capacity() < size {
                BytesMut::with_capacity(size)
            } else {
                b.clear();
                b.resize(size, 0);
                b
            }
        } else {
            BytesMut::with_capacity(size)
        };

        if buf.len() < size {
            buf.resize(size, 0);
        }

        self.active.insert(id, buf);
        id
    }

    pub fn ptr_len(&self, id: BufferId) -> Option<(*mut u8, usize)> {
        self.active
            .get_mut(&id)
            .map(|mut v| (v.as_mut_ptr(), v.len()))
    }

    /// Take ownership of the buffer, removing it from the registry.
    pub fn take(&self, id: BufferId) -> Option<Vec<u8>> {
        self.active.remove(&id).map(|(_, v)| v.to_vec())
    }

    /// Take as Bytes (Zero-Copy)
    pub fn take_bytes(&self, id: BufferId) -> Option<bytes::Bytes> {
        self.active.remove(&id).map(|(_, v)| v.freeze())
    }

    /// Free buffer and return it to the pool.
    pub fn free(&self, id: BufferId) {
        if let Some((_, buf)) = self.active.remove(&id) {
            self.pool.push(buf);
        }
    }
}

impl Default for BufferRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn global_registry() -> &'static BufferRegistry {
    use once_cell::sync::Lazy;
    static REG: Lazy<BufferRegistry> = Lazy::new(BufferRegistry::new);
    &REG
}
