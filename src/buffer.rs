use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

pub type BufferId = u64;

pub struct BufferRegistry {
    inner: Arc<Mutex<HashMap<BufferId, Vec<u8>>>>,
    next: AtomicU64,
}

impl BufferRegistry {
    pub fn new() -> Self {
        BufferRegistry { inner: Arc::new(Mutex::new(HashMap::new())), next: AtomicU64::new(1) }
    }

    pub fn allocate(&self, size: usize) -> BufferId {
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let mut map = self.inner.lock().unwrap();
        map.insert(id, vec![0u8; size]);
        id
    }

    pub fn ptr_len(&self, id: BufferId) -> Option<(*mut u8, usize)> {
        let map = self.inner.lock().unwrap();
        map.get(&id).map(|v| (v.as_ptr() as *mut u8, v.len()))
    }

    /// Take ownership of the buffer, removing it from the registry.
    pub fn take(&self, id: BufferId) -> Option<Vec<u8>> {
        let mut map = self.inner.lock().unwrap();
        map.remove(&id)
    }

    /// Free buffer without taking it.
    pub fn free(&self, id: BufferId) {
        let mut map = self.inner.lock().unwrap();
        map.remove(&id);
    }
}

pub fn global_registry() -> &'static BufferRegistry {
    use once_cell::sync::Lazy;
    static REG: Lazy<BufferRegistry> = Lazy::new(|| BufferRegistry::new());
    &REG
}
