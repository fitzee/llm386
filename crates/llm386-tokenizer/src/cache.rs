//! `CachingTokenizer` — LRU cache wrapper for any [`Tokenizer`].

use std::num::NonZeroUsize;

use llm386_core::{ContentHash, TokenCount, Tokenizer, TokenizerError, TokenizerId};
use lru::LruCache;
use parking_lot::Mutex;

/// Wraps any [`Tokenizer`] with an LRU cache keyed by content hash.
///
/// Useful when the same bytes are tokenized repeatedly (e.g. the same
/// system prompt across many calls) — `count()` becomes a blake3 hash
/// + a hashmap lookup on cache hit.
pub struct CachingTokenizer<T: Tokenizer> {
    inner: T,
    cache: Mutex<LruCache<ContentHash, TokenCount>>,
}

impl<T: Tokenizer> CachingTokenizer<T> {
    pub fn new(inner: T, capacity: NonZeroUsize) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Build a cache with the given capacity, panicking if zero.
    #[must_use]
    pub fn with_capacity(inner: T, capacity: usize) -> Self {
        let nz = NonZeroUsize::new(capacity)
            .expect("CachingTokenizer capacity must be greater than zero");
        Self::new(inner, nz)
    }

    /// Drop all cached counts.
    pub fn clear(&self) {
        self.cache.lock().clear();
    }

    /// Borrow the wrapped tokenizer.
    pub fn inner(&self) -> &T {
        &self.inner
    }
}

impl<T: Tokenizer + std::fmt::Debug> std::fmt::Debug for CachingTokenizer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachingTokenizer")
            .field("inner", &self.inner)
            .field("cached_entries", &self.cache.lock().len())
            .finish()
    }
}

impl<T: Tokenizer> Tokenizer for CachingTokenizer<T> {
    fn id(&self) -> &TokenizerId {
        self.inner.id()
    }

    fn count(&self, bytes: &[u8]) -> Result<TokenCount, TokenizerError> {
        let key = ContentHash::of(bytes);
        if let Some(hit) = self.cache.lock().get(&key).copied() {
            return Ok(hit);
        }
        let count = self.inner.count(bytes)?;
        self.cache.lock().put(key, count);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingTokenizer {
        id: TokenizerId,
        calls: Arc<AtomicUsize>,
    }

    impl Tokenizer for CountingTokenizer {
        fn id(&self) -> &TokenizerId {
            &self.id
        }

        fn count(&self, bytes: &[u8]) -> Result<TokenCount, TokenizerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(TokenCount(u32::try_from(bytes.len()).unwrap_or(u32::MAX)))
        }
    }

    #[test]
    fn cache_hits_on_repeat() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = CountingTokenizer {
            id: TokenizerId::new("counting"),
            calls: calls.clone(),
        };
        let cached = CachingTokenizer::with_capacity(inner, 8);
        cached.count(b"abc").unwrap();
        cached.count(b"abc").unwrap();
        cached.count(b"abc").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_distinguishes_different_inputs() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = CountingTokenizer {
            id: TokenizerId::new("counting"),
            calls: calls.clone(),
        };
        let cached = CachingTokenizer::with_capacity(inner, 8);
        cached.count(b"abc").unwrap();
        cached.count(b"def").unwrap();
        cached.count(b"abc").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn clear_drops_cached_entries() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner = CountingTokenizer {
            id: TokenizerId::new("counting"),
            calls: calls.clone(),
        };
        let cached = CachingTokenizer::with_capacity(inner, 8);
        cached.count(b"abc").unwrap();
        cached.clear();
        cached.count(b"abc").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    #[should_panic(expected = "capacity must be greater than zero")]
    fn zero_capacity_panics() {
        let inner = CountingTokenizer {
            id: TokenizerId::new("counting"),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let _ = CachingTokenizer::with_capacity(inner, 0);
    }
}
