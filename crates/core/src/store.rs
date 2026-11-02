//! The content-addressed object store and the reply cache.

use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The lowercase hexadecimal SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// A byte-bounded map that evicts the oldest entries first.
struct Bounded<V> {
    entries: HashMap<String, (V, usize)>,
    order: VecDeque<String>,
    bytes: usize,
    limit: usize,
}

impl<V: Clone> Bounded<V> {
    fn new(limit: usize) -> Self {
        Bounded {
            entries: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            limit,
        }
    }

    fn get(&self, key: &str) -> Option<V> {
        self.entries.get(key).map(|(v, _)| v.clone())
    }

    fn insert(&mut self, key: String, value: V, size: usize) {
        if size > self.limit || self.entries.contains_key(&key) {
            return;
        }
        while self.bytes + size > self.limit {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some((_, n)) = self.entries.remove(&old) {
                self.bytes -= n;
            }
        }
        self.bytes += size;
        self.order.push_back(key.clone());
        self.entries.insert(key, (value, size));
    }
}

/// Objects by SHA-256: in memory, or in a directory.
pub struct Store {
    memory: Mutex<Bounded<Arc<Vec<u8>>>>,
    dir: Option<PathBuf>,
}

impl Store {
    /// A store in memory, at most `limit` bytes.
    pub fn memory(limit: usize) -> Store {
        Store {
            memory: Mutex::new(Bounded::new(limit)),
            dir: None,
        }
    }

    /// A store in `dir`, with an in-memory front of at most `limit` bytes.
    pub fn directory(dir: PathBuf, limit: usize) -> std::io::Result<Store> {
        std::fs::create_dir_all(&dir)?;
        Ok(Store {
            memory: Mutex::new(Bounded::new(limit)),
            dir: Some(dir),
        })
    }

    fn path(&self, hex: &str) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|d| d.join("sha256").join(&hex[..2]).join(hex))
    }

    pub fn get(&self, hex: &str) -> Option<Arc<Vec<u8>>> {
        if let Some(bytes) = self.memory.lock().expect("store lock").get(hex) {
            return Some(bytes);
        }
        let bytes = Arc::new(std::fs::read(self.path(hex)?).ok()?);
        // A disk object may be corrupt or replaced. Serve only verified bytes.
        if sha256_hex(&bytes) != hex {
            return None;
        }
        self.memory
            .lock()
            .expect("store lock")
            .insert(hex.to_owned(), bytes.clone(), bytes.len());
        Some(bytes)
    }

    pub fn contains(&self, hex: &str) -> bool {
        self.memory
            .lock()
            .expect("store lock")
            .entries
            .contains_key(hex)
            || self.path(hex).is_some_and(|p| p.exists())
    }

    /// Store `bytes` under their hash. Returns the hash and whether they were new.
    pub fn put(&self, bytes: Vec<u8>) -> std::io::Result<(String, bool)> {
        let hex = sha256_hex(&bytes);
        let new = !self.contains(&hex);
        if new && let Some(path) = self.path(&hex) {
            let parent = path.parent().expect("object paths have a parent");
            std::fs::create_dir_all(parent)?;
            let tmp = parent.join(format!(".{hex}.{}", std::process::id()));
            std::fs::write(&tmp, &bytes)?;
            std::fs::rename(&tmp, &path)?;
        }
        let size = bytes.len();
        self.memory
            .lock()
            .expect("store lock")
            .insert(hex.clone(), Arc::new(bytes), size);
        Ok((hex, new))
    }
}

/// Reply parts: names and Arrow IPC streams, in protocol order.
pub type Parts = Arc<Vec<(String, Vec<u8>)>>;

/// Reply parts by cache key, at most a byte limit.
pub struct ReplyCache {
    entries: Mutex<Bounded<Parts>>,
}

impl ReplyCache {
    pub fn new(limit: usize) -> ReplyCache {
        ReplyCache {
            entries: Mutex::new(Bounded::new(limit)),
        }
    }

    pub fn get(&self, key: &str) -> Option<Parts> {
        self.entries.lock().expect("cache lock").get(key)
    }

    pub fn insert(&self, key: String, parts: Parts) {
        let size = parts.iter().map(|(n, b)| n.len() + b.len()).sum();
        self.entries
            .lock()
            .expect("cache lock")
            .insert(key, parts, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_evicts_oldest() {
        let store = Store::memory(10);
        let (a, new) = store.put(vec![1; 6]).unwrap();
        assert!(new);
        let (b, _) = store.put(vec![2; 6]).unwrap();
        assert!(store.get(&a).is_none());
        assert_eq!(store.get(&b).unwrap().len(), 6);
    }

    #[test]
    fn directory_store_survives_memory_eviction() {
        let dir = std::env::temp_dir().join(format!("fpb-store-{}", std::process::id()));
        let store = Store::directory(dir.clone(), 0).unwrap();
        let (hex, _) = store.put(b"abc".to_vec()).unwrap();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(store.get(&hex).unwrap().as_slice(), b"abc");
        let (_, new) = store.put(b"abc".to_vec()).unwrap();
        assert!(!new);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
