//! `LmdbStore` — `BlockStore` implementation backed by LMDB.

use std::path::Path;

use heed::types::{Bytes, Str};
use heed::{Database, Env, EnvOpenOptions};
use llm386_core::{BlockId, BlockStore, ContentHash, ContextBlock, SessionId, StoreError};
use thiserror::Error;
use tracing::{debug, instrument};

/// Schema version written to the `meta` table on first open.
///
/// Bump this whenever the on-disk layout changes incompatibly. Older
/// stores will refuse to open with the new code.
const CURRENT_SCHEMA: u32 = 1;

/// Default LMDB map size — a 64 GiB virtual reservation, not an
/// allocation. Adjust via [`StoreConfig::map_size`] if you expect to
/// exceed this on a single host.
const DEFAULT_MAP_SIZE: usize = 64 * 1024 * 1024 * 1024;

/// Default `max_dbs` budget — covers the four named DBs we open today
/// plus headroom for future indexes (kind, time, edges, summaries,
/// token counts, traces).
const DEFAULT_MAX_DBS: u32 = 16;

/// Configuration for opening an [`LmdbStore`].
#[derive(Clone, Copy, Debug)]
pub struct StoreConfig {
    pub map_size: usize,
    pub max_dbs: u32,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            map_size: DEFAULT_MAP_SIZE,
            max_dbs: DEFAULT_MAX_DBS,
        }
    }
}

/// LMDB-backed implementation of the `BlockStore` trait.
///
/// Cheap to clone (clones share the underlying [`Env`]).
#[derive(Clone)]
pub struct LmdbStore {
    env: Env,
    blocks_by_id: Database<Bytes, Bytes>,
    blocks_by_hash: Database<Bytes, Bytes>,
    blocks_by_session: Database<Bytes, Bytes>,
    #[allow(dead_code)] // reserved for future schema migrations
    meta: Database<Str, Bytes>,
}

impl std::fmt::Debug for LmdbStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LmdbStore")
            .field("schema", &CURRENT_SCHEMA)
            .finish_non_exhaustive()
    }
}

impl LmdbStore {
    /// Open (or create) an LMDB env at `path` and prepare the named
    /// databases.
    #[instrument(skip(config), fields(path = %path.as_ref().display()))]
    pub fn open(path: impl AsRef<Path>, config: StoreConfig) -> Result<Self, StoreOpenError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).map_err(StoreOpenError::Io)?;

        // SAFETY: `EnvOpenOptions::open` is marked unsafe because LMDB's
        // mmap-based concurrency model is undefined when the same
        // env path is opened by multiple processes simultaneously, or
        // when the underlying files are mutated externally. Within a
        // single process, opening an env path once via this function
        // is safe; the contract that callers do not open the same
        // path twice (and do not point at network filesystems) is
        // documented as part of the `LmdbStore::open` API.
        #[allow(unsafe_code)]
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(config.map_size)
                .max_dbs(config.max_dbs)
                .open(path)?
        };

        let mut wtxn = env.write_txn()?;
        let blocks_by_id = env.create_database(&mut wtxn, Some("blocks_by_id"))?;
        let blocks_by_hash = env.create_database(&mut wtxn, Some("blocks_by_hash"))?;
        let blocks_by_session = env.create_database(&mut wtxn, Some("blocks_by_session"))?;
        let meta: Database<Str, Bytes> = env.create_database(&mut wtxn, Some("meta"))?;

        if let Some(existing) = meta.get(&wtxn, "schema_version")? {
            let arr: [u8; 4] = existing
                .try_into()
                .map_err(|_| StoreOpenError::CorruptMeta("schema_version width".into()))?;
            let found = u32::from_be_bytes(arr);
            if found != CURRENT_SCHEMA {
                return Err(StoreOpenError::SchemaMismatch {
                    expected: CURRENT_SCHEMA,
                    found,
                });
            }
        } else {
            meta.put(&mut wtxn, "schema_version", &CURRENT_SCHEMA.to_be_bytes())?;
            debug!(schema = CURRENT_SCHEMA, "initialized fresh LMDB env");
        }
        wtxn.commit()?;

        Ok(Self {
            env,
            blocks_by_id,
            blocks_by_hash,
            blocks_by_session,
            meta,
        })
    }
}

impl BlockStore for LmdbStore {
    #[instrument(skip(self, block), fields(id = %block.id, kind = ?block.kind))]
    fn put(&self, session: SessionId, block: ContextBlock) -> Result<BlockId, StoreError> {
        let mut wtxn = self.env.write_txn().map_err(|e| heed_err(&e))?;

        // Dedup: if a block with the same content hash already exists,
        // return its id and just record the new session membership.
        if let Some(existing_id_bytes) = self
            .blocks_by_hash
            .get(&wtxn, &block.hash.0)
            .map_err(|e| heed_err(&e))?
        {
            let existing_id = decode_block_id(existing_id_bytes)?;
            let session_key = session_block_key(session, existing_id);
            self.blocks_by_session
                .put(&mut wtxn, &session_key, &[])
                .map_err(|e| heed_err(&e))?;
            wtxn.commit().map_err(|e| heed_err(&e))?;
            debug!(?existing_id, "deduped on content hash");
            return Ok(existing_id);
        }

        let id = block.id;
        let id_key = id.0.to_be_bytes();
        let value = postcard::to_allocvec(&block)
            .map_err(|e| StoreError::Backend(format!("postcard encode: {e}")))?;

        self.blocks_by_id
            .put(&mut wtxn, &id_key, &value)
            .map_err(|e| heed_err(&e))?;
        self.blocks_by_hash
            .put(&mut wtxn, &block.hash.0, &id_key)
            .map_err(|e| heed_err(&e))?;
        let session_key = session_block_key(session, id);
        self.blocks_by_session
            .put(&mut wtxn, &session_key, &[])
            .map_err(|e| heed_err(&e))?;
        wtxn.commit().map_err(|e| heed_err(&e))?;

        Ok(id)
    }

    fn get(&self, id: BlockId) -> Result<Option<ContextBlock>, StoreError> {
        let rtxn = self.env.read_txn().map_err(|e| heed_err(&e))?;
        let key = id.0.to_be_bytes();
        match self
            .blocks_by_id
            .get(&rtxn, &key)
            .map_err(|e| heed_err(&e))?
        {
            Some(bytes) => {
                let block: ContextBlock = postcard::from_bytes(bytes)
                    .map_err(|e| StoreError::Backend(format!("postcard decode: {e}")))?;
                Ok(Some(block))
            }
            None => Ok(None),
        }
    }

    fn list_session(&self, session: SessionId) -> Result<Vec<BlockId>, StoreError> {
        let rtxn = self.env.read_txn().map_err(|e| heed_err(&e))?;
        let prefix = session.0.to_be_bytes();
        let iter = self
            .blocks_by_session
            .prefix_iter(&rtxn, &prefix)
            .map_err(|e| heed_err(&e))?;
        let mut ids = Vec::new();
        for entry in iter {
            let (key, _) = entry.map_err(|e| heed_err(&e))?;
            if key.len() != 32 {
                return Err(StoreError::Backend(format!(
                    "session key width {}",
                    key.len()
                )));
            }
            let block_bytes: [u8; 16] = key[16..]
                .try_into()
                .map_err(|_| StoreError::Backend("session key suffix".into()))?;
            ids.push(BlockId(u128::from_be_bytes(block_bytes)));
        }
        Ok(ids)
    }

    fn lookup_hash(&self, hash: ContentHash) -> Result<Option<BlockId>, StoreError> {
        let rtxn = self.env.read_txn().map_err(|e| heed_err(&e))?;
        match self
            .blocks_by_hash
            .get(&rtxn, &hash.0)
            .map_err(|e| heed_err(&e))?
        {
            Some(bytes) => Ok(Some(decode_block_id(bytes)?)),
            None => Ok(None),
        }
    }
}

/// Errors that can occur while opening an [`LmdbStore`].
#[derive(Debug, Error)]
pub enum StoreOpenError {
    #[error("io error: {0}")]
    Io(#[source] std::io::Error),
    #[error("LMDB error: {0}")]
    Lmdb(#[from] heed::Error),
    #[error("on-disk schema version {found} does not match expected {expected}")]
    SchemaMismatch { expected: u32, found: u32 },
    #[error("meta table is corrupt: {0}")]
    CorruptMeta(String),
}

fn heed_err(e: &heed::Error) -> StoreError {
    StoreError::Backend(format!("LMDB: {e}"))
}

fn session_block_key(session: SessionId, block: BlockId) -> [u8; 32] {
    let mut buf = [0u8; 32];
    buf[..16].copy_from_slice(&session.0.to_be_bytes());
    buf[16..].copy_from_slice(&block.0.to_be_bytes());
    buf
}

fn decode_block_id(bytes: &[u8]) -> Result<BlockId, StoreError> {
    let arr: [u8; 16] = bytes
        .try_into()
        .map_err(|_| StoreError::Backend(format!("BlockId width {}", bytes.len())))?;
    Ok(BlockId(u128::from_be_bytes(arr)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm386_core::{BlockKind, Provenance, Timestamp, TokenCounts};
    use tempfile::TempDir;

    fn make_block(bytes: &[u8], kind: BlockKind, ts_ms: u64, rnd: u128) -> ContextBlock {
        ContextBlock {
            id: BlockId::from_parts(ts_ms, rnd),
            kind,
            bytes: bytes.to_vec(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(ts_ms),
            updated_at: Timestamp(ts_ms),
            provenance: Provenance::default(),
            hash: ContentHash::of(bytes),
        }
    }

    fn open_tmp() -> (LmdbStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = LmdbStore::open(dir.path(), StoreConfig::default()).unwrap();
        (store, dir)
    }

    #[test]
    fn put_then_get_roundtrips() {
        let (store, _dir) = open_tmp();
        let session = SessionId(1);
        let block = make_block(b"hello", BlockKind::UserMessage, 1_000, 42);
        let id = store.put(session, block.clone()).unwrap();
        let fetched = store.get(id).unwrap().unwrap();
        assert_eq!(fetched.bytes, block.bytes);
        assert_eq!(fetched.kind, block.kind);
        assert_eq!(fetched.hash, block.hash);
    }

    #[test]
    fn duplicate_content_returns_existing_id() {
        let (store, _dir) = open_tmp();
        let session = SessionId(1);
        let first = make_block(b"hello", BlockKind::UserMessage, 1_000, 42);
        let id1 = store.put(session, first).unwrap();
        // Same bytes, different proposed id — store must dedup.
        let dup = make_block(b"hello", BlockKind::UserMessage, 2_000, 99);
        let id2 = store.put(session, dup).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn list_session_returns_all_inserted_blocks() {
        let (store, _dir) = open_tmp();
        let session = SessionId(7);
        let a = make_block(b"a", BlockKind::UserMessage, 1, 1);
        let b = make_block(b"b", BlockKind::UserMessage, 2, 2);
        let c = make_block(b"c", BlockKind::UserMessage, 3, 3);
        let id_a = store.put(session, a).unwrap();
        let id_b = store.put(session, b).unwrap();
        let id_c = store.put(session, c).unwrap();
        let mut listed = store.list_session(session).unwrap();
        listed.sort();
        let mut expected = vec![id_a, id_b, id_c];
        expected.sort();
        assert_eq!(listed, expected);
    }

    #[test]
    fn list_session_isolates_per_session() {
        let (store, _dir) = open_tmp();
        let s1 = SessionId(1);
        let s2 = SessionId(2);
        let a = make_block(b"alpha", BlockKind::UserMessage, 1, 1);
        let b = make_block(b"beta", BlockKind::UserMessage, 2, 2);
        store.put(s1, a).unwrap();
        store.put(s2, b).unwrap();
        assert_eq!(store.list_session(s1).unwrap().len(), 1);
        assert_eq!(store.list_session(s2).unwrap().len(), 1);
    }

    #[test]
    fn lookup_hash_finds_inserted_block() {
        let (store, _dir) = open_tmp();
        let session = SessionId(1);
        let block = make_block(b"findme", BlockKind::Fact, 1_000, 42);
        let id = store.put(session, block.clone()).unwrap();
        assert_eq!(store.lookup_hash(block.hash).unwrap(), Some(id));
    }

    #[test]
    fn lookup_hash_returns_none_for_unknown() {
        let (store, _dir) = open_tmp();
        let unknown = ContentHash::of(b"never inserted");
        assert!(store.lookup_hash(unknown).unwrap().is_none());
    }

    #[test]
    fn get_unknown_id_is_none() {
        let (store, _dir) = open_tmp();
        let id = BlockId::from_parts(0, 0);
        assert!(store.get(id).unwrap().is_none());
    }

    #[test]
    fn reopen_preserves_data_and_schema() {
        let dir = TempDir::new().unwrap();
        let session = SessionId(1);
        let id = {
            let store = LmdbStore::open(dir.path(), StoreConfig::default()).unwrap();
            let block = make_block(b"persist me", BlockKind::Plan, 1_000, 42);
            store.put(session, block).unwrap()
        };
        let store = LmdbStore::open(dir.path(), StoreConfig::default()).unwrap();
        let fetched = store.get(id).unwrap().unwrap();
        assert_eq!(fetched.bytes, b"persist me".to_vec());
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config { cases: 24, ..proptest::test_runner::Config::default() })]

        /// Putting the same bytes twice with different proposed ids
        /// must collapse to a single stored block (content-hash dedup).
        #[test]
        fn dedup_invariant_same_bytes_same_id(
            bytes in proptest::collection::vec(proptest::prelude::any::<u8>(), 1..256),
            seed_a in proptest::prelude::any::<u128>(),
            seed_b in proptest::prelude::any::<u128>(),
        ) {
            let (store, _dir) = open_tmp();
            let session = SessionId(1);
            let a = make_block(&bytes, BlockKind::Fact, 0, seed_a);
            let b = make_block(&bytes, BlockKind::Fact, 0, seed_b);
            let id_a = store.put(session, a).unwrap();
            let id_b = store.put(session, b).unwrap();
            proptest::prop_assert_eq!(id_a, id_b);
        }
    }
}
