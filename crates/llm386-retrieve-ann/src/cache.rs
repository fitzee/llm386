//! `EmbeddingCache` — LMDB-backed persistent cache for ANN retrievers.

use std::fmt;
use std::path::Path;

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions};
use llm386_core::ContentHash;
use thiserror::Error;

/// Default LMDB map size for the embeddings env. 4 GiB is enough
/// for ~1M cached 1536-dim float vectors at single precision.
const DEFAULT_MAP_SIZE: usize = 4 * 1024 * 1024 * 1024;

/// Persistent embedding cache, keyed by
/// `(embedder_name, ContentHash)`. Different embedders produce
/// different vector spaces, so the embedder name is part of the
/// key — vectors from one model never collide with another.
///
/// Values are stored as `<dim:u32_le><f32_le bytes…>`.
///
/// Cheap to clone (clones share the underlying [`Env`]).
#[derive(Clone)]
pub struct EmbeddingCache {
    env: Env,
    db: Database<Bytes, Bytes>,
}

impl EmbeddingCache {
    /// Open (or create) the cache LMDB env at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, EmbeddingCacheError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).map_err(EmbeddingCacheError::Io)?;
        // SAFETY: opening an LMDB env is unsafe per heed's API
        // because LMDB's mmap-based concurrency model is undefined
        // when the same env path is opened by multiple processes
        // simultaneously, or when the underlying files are mutated
        // externally. Within a single process this open is safe;
        // cross-process access is the caller's responsibility.
        #[allow(unsafe_code)]
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(DEFAULT_MAP_SIZE)
                .max_dbs(2)
                .open(path)?
        };
        let mut wtxn = env.write_txn()?;
        let db = env.create_database(&mut wtxn, Some("embeddings"))?;
        wtxn.commit()?;
        Ok(Self { env, db })
    }

    /// Look up the embedding for `(embedder, hash)`, if present.
    pub fn get(
        &self,
        embedder: &str,
        hash: &ContentHash,
    ) -> Result<Option<Vec<f32>>, EmbeddingCacheError> {
        let key = make_key(embedder, hash);
        let rtxn = self.env.read_txn()?;
        let raw = self.db.get(&rtxn, &key)?;
        let Some(raw) = raw else { return Ok(None) };
        Ok(Some(decode_vec(raw)?))
    }

    /// Store the embedding for `(embedder, hash)`, overwriting any
    /// previous entry.
    pub fn put(
        &self,
        embedder: &str,
        hash: &ContentHash,
        vec: &[f32],
    ) -> Result<(), EmbeddingCacheError> {
        let key = make_key(embedder, hash);
        let value = encode_vec(vec);
        let mut wtxn = self.env.write_txn()?;
        self.db.put(&mut wtxn, &key, &value)?;
        wtxn.commit()?;
        Ok(())
    }
}

impl fmt::Debug for EmbeddingCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddingCache").finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum EmbeddingCacheError {
    #[error("io error: {0}")]
    Io(#[source] std::io::Error),
    #[error("LMDB error: {0}")]
    Lmdb(#[from] heed::Error),
    #[error("malformed cached vector: {0}")]
    Decode(String),
}

fn make_key(embedder: &str, hash: &ContentHash) -> Vec<u8> {
    // Embedder name length-prefixed (1 byte) so different names
    // never accidentally collide. Names ≥ 256 bytes are rejected
    // upstream (caller never feeds them in practice).
    let name = embedder.as_bytes();
    let n = u8::try_from(name.len()).unwrap_or(u8::MAX);
    let mut out = Vec::with_capacity(1 + name.len() + 32);
    out.push(n);
    out.extend_from_slice(name);
    out.extend_from_slice(&hash.0);
    out
}

fn encode_vec(v: &[f32]) -> Vec<u8> {
    let dim = u32::try_from(v.len()).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(4 + v.len() * 4);
    out.extend_from_slice(&dim.to_le_bytes());
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn decode_vec(bytes: &[u8]) -> Result<Vec<f32>, EmbeddingCacheError> {
    if bytes.len() < 4 {
        return Err(EmbeddingCacheError::Decode("header < 4 bytes".into()));
    }
    let dim_arr: [u8; 4] = bytes[..4].try_into().expect("checked length");
    let dim = u32::from_le_bytes(dim_arr) as usize;
    let expected = 4 + dim * 4;
    if bytes.len() != expected {
        return Err(EmbeddingCacheError::Decode(format!(
            "payload {} bytes, expected {expected}",
            bytes.len(),
        )));
    }
    let mut out = Vec::with_capacity(dim);
    for chunk in bytes[4..].chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().expect("chunks_exact yields 4");
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_tmp() -> (EmbeddingCache, TempDir) {
        let dir = TempDir::new().unwrap();
        let cache = EmbeddingCache::open(dir.path()).unwrap();
        (cache, dir)
    }

    #[test]
    fn put_then_get_roundtrips() {
        let (cache, _dir) = open_tmp();
        let h = ContentHash::of(b"hello");
        let v = vec![1.0_f32, -2.0, 3.5, 4.25];
        cache.put("test-embedder", &h, &v).unwrap();
        let got = cache.get("test-embedder", &h).unwrap().unwrap();
        assert_eq!(got, v);
    }

    #[test]
    fn missing_returns_none() {
        let (cache, _dir) = open_tmp();
        let h = ContentHash::of(b"never inserted");
        assert!(cache.get("e", &h).unwrap().is_none());
    }

    #[test]
    fn embedder_name_isolates_entries() {
        let (cache, _dir) = open_tmp();
        let h = ContentHash::of(b"same content");
        cache.put("openai-small", &h, &[1.0]).unwrap();
        cache.put("openai-large", &h, &[2.0]).unwrap();
        assert_eq!(cache.get("openai-small", &h).unwrap().unwrap(), vec![1.0]);
        assert_eq!(cache.get("openai-large", &h).unwrap().unwrap(), vec![2.0]);
    }

    #[test]
    fn reopen_preserves_entries() {
        let dir = TempDir::new().unwrap();
        let h = ContentHash::of(b"persist");
        {
            let cache = EmbeddingCache::open(dir.path()).unwrap();
            cache.put("e", &h, &[1.0, 2.0, 3.0]).unwrap();
        }
        let cache = EmbeddingCache::open(dir.path()).unwrap();
        assert_eq!(cache.get("e", &h).unwrap().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn put_overwrites_existing_entry() {
        let (cache, _dir) = open_tmp();
        let h = ContentHash::of(b"x");
        cache.put("e", &h, &[1.0]).unwrap();
        cache.put("e", &h, &[2.0, 3.0]).unwrap();
        assert_eq!(cache.get("e", &h).unwrap().unwrap(), vec![2.0, 3.0]);
    }
}
