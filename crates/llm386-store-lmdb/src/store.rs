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

    fn list_sessions(&self) -> Result<Vec<SessionId>, StoreError> {
        // The blocks_by_session table is keyed by `(session, block)`
        // (32 bytes total). Walk all keys and collect the unique
        // 16-byte session prefixes.
        use std::collections::BTreeSet;
        let rtxn = self.env.read_txn().map_err(|e| heed_err(&e))?;
        let iter = self
            .blocks_by_session
            .iter(&rtxn)
            .map_err(|e| heed_err(&e))?;
        let mut seen: BTreeSet<u128> = BTreeSet::new();
        for entry in iter {
            let (key, _) = entry.map_err(|e| heed_err(&e))?;
            if key.len() < 16 {
                continue;
            }
            let arr: [u8; 16] = key[..16]
                .try_into()
                .map_err(|_| StoreError::Backend("session key prefix".into()))?;
            seen.insert(u128::from_be_bytes(arr));
        }
        Ok(seen.into_iter().map(SessionId).collect())
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

    #[instrument(skip(self), fields(id = %id))]
    fn delete(&self, id: BlockId) -> Result<bool, StoreError> {
        let mut wtxn = self.env.write_txn().map_err(|e| heed_err(&e))?;
        let id_key = id.0.to_be_bytes();

        // Look up the block's content hash so we can clean the
        // hash index too. If the block doesn't exist, we still want
        // to scrub any orphaned session pointers below.
        let hash = match self
            .blocks_by_id
            .get(&wtxn, &id_key)
            .map_err(|e| heed_err(&e))?
        {
            Some(bytes) => {
                let block: ContextBlock = postcard::from_bytes(bytes)
                    .map_err(|e| StoreError::Backend(format!("postcard decode: {e}")))?;
                Some(block.hash)
            }
            None => None,
        };

        // Collect every (session, id) entry that references this block.
        let session_keys: Vec<Vec<u8>> = {
            let iter = self
                .blocks_by_session
                .iter(&wtxn)
                .map_err(|e| heed_err(&e))?;
            let mut keys = Vec::new();
            for entry in iter {
                let (key, _) = entry.map_err(|e| heed_err(&e))?;
                if key.len() == 32 && key[16..] == id_key {
                    keys.push(key.to_vec());
                }
            }
            keys
        };

        let existed = hash.is_some() || !session_keys.is_empty();
        if !existed {
            // Nothing to do — short-circuit before opening any writes.
            return Ok(false);
        }

        for key in &session_keys {
            self.blocks_by_session
                .delete(&mut wtxn, key.as_slice())
                .map_err(|e| heed_err(&e))?;
        }
        if hash.is_some() {
            self.blocks_by_id
                .delete(&mut wtxn, &id_key)
                .map_err(|e| heed_err(&e))?;
        }
        if let Some(hash) = hash {
            self.blocks_by_hash
                .delete(&mut wtxn, &hash.0)
                .map_err(|e| heed_err(&e))?;
        }
        wtxn.commit().map_err(|e| heed_err(&e))?;
        debug!(
            ?id,
            deleted_session_refs = session_keys.len(),
            "block deleted"
        );
        Ok(true)
    }

    #[instrument(skip(self), fields(session = %session))]
    fn purge_session(&self, session: SessionId) -> Result<usize, StoreError> {
        let mut wtxn = self.env.write_txn().map_err(|e| heed_err(&e))?;
        let prefix = session.0.to_be_bytes();

        // Step 1: collect every block id this session references.
        let block_ids: Vec<BlockId> = {
            let iter = self
                .blocks_by_session
                .prefix_iter(&wtxn, &prefix)
                .map_err(|e| heed_err(&e))?;
            let mut ids = Vec::new();
            for entry in iter {
                let (key, _) = entry.map_err(|e| heed_err(&e))?;
                if key.len() != 32 {
                    continue;
                }
                let block_bytes: [u8; 16] = key[16..]
                    .try_into()
                    .map_err(|_| StoreError::Backend("session key suffix".into()))?;
                ids.push(BlockId(u128::from_be_bytes(block_bytes)));
            }
            ids
        };
        let count = block_ids.len();
        if count == 0 {
            return Ok(0);
        }

        // Step 2: drop this session's references.
        for id in &block_ids {
            let key = session_block_key(session, *id);
            self.blocks_by_session
                .delete(&mut wtxn, &key)
                .map_err(|e| heed_err(&e))?;
        }

        // Step 3: for each id, scan blocks_by_session for any
        // remaining reference. If none, delete the block content
        // and its hash entry too.
        for id in block_ids {
            let id_key = id.0.to_be_bytes();
            let still_referenced = {
                let iter = self
                    .blocks_by_session
                    .iter(&wtxn)
                    .map_err(|e| heed_err(&e))?;
                let mut found = false;
                for entry in iter {
                    let (key, _) = entry.map_err(|e| heed_err(&e))?;
                    if key.len() == 32 && key[16..] == id_key {
                        found = true;
                        break;
                    }
                }
                found
            };
            if !still_referenced
                && let Some(bytes) = self
                    .blocks_by_id
                    .get(&wtxn, &id_key)
                    .map_err(|e| heed_err(&e))?
            {
                let block: ContextBlock = postcard::from_bytes(bytes)
                    .map_err(|e| StoreError::Backend(format!("postcard decode: {e}")))?;
                self.blocks_by_id
                    .delete(&mut wtxn, &id_key)
                    .map_err(|e| heed_err(&e))?;
                self.blocks_by_hash
                    .delete(&mut wtxn, &block.hash.0)
                    .map_err(|e| heed_err(&e))?;
            }
        }

        wtxn.commit().map_err(|e| heed_err(&e))?;
        Ok(count)
    }
}

impl LmdbStore {
    /// Read-only integrity check. Walks every block in the primary
    /// table, recomputes its content hash, and verifies the hash
    /// index and session index are consistent.
    pub fn verify(&self) -> Result<VerifyReport, StoreError> {
        use std::collections::HashSet;
        let rtxn = self.env.read_txn().map_err(|e| heed_err(&e))?;
        let mut report = VerifyReport::default();
        let mut all_ids: HashSet<BlockId> = HashSet::new();

        let iter = self.blocks_by_id.iter(&rtxn).map_err(|e| heed_err(&e))?;
        for entry in iter {
            let (key, value) = entry.map_err(|e| heed_err(&e))?;
            let id_bytes: [u8; 16] = key
                .try_into()
                .map_err(|_| StoreError::Backend(format!("id width {}", key.len())))?;
            let id = BlockId(u128::from_be_bytes(id_bytes));
            let block: ContextBlock = postcard::from_bytes(value)
                .map_err(|e| StoreError::Backend(format!("postcard decode: {e}")))?;
            all_ids.insert(id);
            report.blocks_checked += 1;

            // Hash sanity.
            let computed = ContentHash::of(&block.bytes);
            if computed != block.hash {
                report.hash_mismatches.push(id);
            }

            // Hash index entry must exist and point back at this id.
            match self
                .blocks_by_hash
                .get(&rtxn, &block.hash.0)
                .map_err(|e| heed_err(&e))?
            {
                None => report.missing_from_hash_index.push(id),
                Some(bytes) => {
                    let pointed = decode_block_id(bytes)?;
                    if pointed != id {
                        report.hash_index_misroutes.push(id);
                    }
                }
            }
        }

        // Sweep blocks_by_session for orphaned entries.
        let mut ids_with_session: HashSet<BlockId> = HashSet::new();
        let iter = self
            .blocks_by_session
            .iter(&rtxn)
            .map_err(|e| heed_err(&e))?;
        for entry in iter {
            let (key, _) = entry.map_err(|e| heed_err(&e))?;
            if key.len() != 32 {
                continue;
            }
            let id_bytes: [u8; 16] = key[16..]
                .try_into()
                .map_err(|_| StoreError::Backend("session key suffix".into()))?;
            let id = BlockId(u128::from_be_bytes(id_bytes));
            if all_ids.contains(&id) {
                ids_with_session.insert(id);
            } else {
                report.orphan_session_entries += 1;
            }
        }

        for id in &all_ids {
            if !ids_with_session.contains(id) {
                report.blocks_with_no_session.push(*id);
            }
        }

        Ok(report)
    }

    /// Rebuilds derivable indexes (`blocks_by_hash`) from
    /// `blocks_by_id` and removes orphan session entries that
    /// reference missing blocks. Blocks whose stored hash doesn't
    /// match their computed hash are left alone and reported — they
    /// indicate real corruption that needs human review.
    pub fn repair(&self) -> Result<RepairReport, StoreError> {
        use std::collections::HashSet;
        let mut report = RepairReport::default();
        let mut wtxn = self.env.write_txn().map_err(|e| heed_err(&e))?;

        // Step 1: enumerate every real block.
        let mut blocks: Vec<(BlockId, ContentHash)> = Vec::new();
        {
            let iter = self.blocks_by_id.iter(&wtxn).map_err(|e| heed_err(&e))?;
            for entry in iter {
                let (key, value) = entry.map_err(|e| heed_err(&e))?;
                let id_bytes: [u8; 16] = key
                    .try_into()
                    .map_err(|_| StoreError::Backend(format!("id width {}", key.len())))?;
                let id = BlockId(u128::from_be_bytes(id_bytes));
                let block: ContextBlock = postcard::from_bytes(value)
                    .map_err(|e| StoreError::Backend(format!("postcard decode: {e}")))?;
                let computed = ContentHash::of(&block.bytes);
                if computed != block.hash {
                    report.hash_mismatches_quarantined.push(id);
                }
                blocks.push((id, block.hash));
            }
        }

        // Step 2: rebuild blocks_by_hash from scratch.
        self.blocks_by_hash
            .clear(&mut wtxn)
            .map_err(|e| heed_err(&e))?;
        for (id, hash) in &blocks {
            let id_key = id.0.to_be_bytes();
            self.blocks_by_hash
                .put(&mut wtxn, &hash.0, &id_key)
                .map_err(|e| heed_err(&e))?;
            report.hash_entries_written += 1;
        }
        report.hash_index_rebuilt = true;

        // Step 3: scrub orphan session entries.
        let real_ids: HashSet<BlockId> = blocks.iter().map(|(id, _)| *id).collect();
        let to_remove: Vec<Vec<u8>> = {
            let iter = self
                .blocks_by_session
                .iter(&wtxn)
                .map_err(|e| heed_err(&e))?;
            let mut v = Vec::new();
            for entry in iter {
                let (key, _) = entry.map_err(|e| heed_err(&e))?;
                if key.len() != 32 {
                    continue;
                }
                let id_bytes: [u8; 16] = key[16..]
                    .try_into()
                    .map_err(|_| StoreError::Backend("session key suffix".into()))?;
                let id = BlockId(u128::from_be_bytes(id_bytes));
                if !real_ids.contains(&id) {
                    v.push(key.to_vec());
                }
            }
            v
        };
        for key in &to_remove {
            self.blocks_by_session
                .delete(&mut wtxn, key.as_slice())
                .map_err(|e| heed_err(&e))?;
        }
        report.orphan_session_entries_removed = to_remove.len();

        // Step 4: report blocks with zero session refs (can't auto-fix).
        let mut ids_with_session: HashSet<BlockId> = HashSet::new();
        {
            let iter = self
                .blocks_by_session
                .iter(&wtxn)
                .map_err(|e| heed_err(&e))?;
            for entry in iter {
                let (key, _) = entry.map_err(|e| heed_err(&e))?;
                if key.len() != 32 {
                    continue;
                }
                let id_bytes: [u8; 16] = key[16..]
                    .try_into()
                    .map_err(|_| StoreError::Backend("session key suffix".into()))?;
                ids_with_session.insert(BlockId(u128::from_be_bytes(id_bytes)));
            }
        }
        for id in real_ids {
            if !ids_with_session.contains(&id) {
                report.blocks_with_no_session.push(id);
            }
        }

        wtxn.commit().map_err(|e| heed_err(&e))?;
        Ok(report)
    }
}

/// Result of [`LmdbStore::verify`].
#[derive(Debug, Default)]
pub struct VerifyReport {
    pub blocks_checked: usize,
    pub hash_mismatches: Vec<BlockId>,
    pub missing_from_hash_index: Vec<BlockId>,
    pub hash_index_misroutes: Vec<BlockId>,
    pub orphan_session_entries: usize,
    pub blocks_with_no_session: Vec<BlockId>,
}

impl VerifyReport {
    /// True when every check passed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.hash_mismatches.is_empty()
            && self.missing_from_hash_index.is_empty()
            && self.hash_index_misroutes.is_empty()
            && self.orphan_session_entries == 0
            && self.blocks_with_no_session.is_empty()
    }
}

/// Result of [`LmdbStore::repair`].
#[derive(Debug, Default)]
pub struct RepairReport {
    pub hash_index_rebuilt: bool,
    pub hash_entries_written: usize,
    pub orphan_session_entries_removed: usize,
    /// Blocks whose stored hash doesn't match their bytes. These
    /// are left in the store untouched; human review needed.
    pub hash_mismatches_quarantined: Vec<BlockId>,
    /// Blocks with no remaining session reference after repair.
    /// Not removed automatically.
    pub blocks_with_no_session: Vec<BlockId>,
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
    fn list_sessions_returns_unique_sorted_ids() {
        let (store, _dir) = open_tmp();
        let s_a = SessionId(7);
        let s_b = SessionId(3);
        let s_c = SessionId(11);
        store
            .put(s_a, make_block(b"x", BlockKind::Fact, 1, 1))
            .unwrap();
        store
            .put(s_a, make_block(b"y", BlockKind::Fact, 2, 2))
            .unwrap();
        store
            .put(s_b, make_block(b"z", BlockKind::Fact, 3, 3))
            .unwrap();
        store
            .put(s_c, make_block(b"w", BlockKind::Fact, 4, 4))
            .unwrap();
        let sessions = store.list_sessions().unwrap();
        assert_eq!(sessions, vec![SessionId(3), SessionId(7), SessionId(11)]);
    }

    #[test]
    fn list_sessions_empty_when_no_blocks() {
        let (store, _dir) = open_tmp();
        assert!(store.list_sessions().unwrap().is_empty());
    }

    #[test]
    fn delete_removes_block_from_all_indexes() {
        let (store, _dir) = open_tmp();
        let session = SessionId(1);
        let block = make_block(b"to-be-deleted", BlockKind::Fact, 1, 1);
        let hash = block.hash;
        let id = store.put(session, block).unwrap();

        assert!(store.get(id).unwrap().is_some());
        assert_eq!(store.lookup_hash(hash).unwrap(), Some(id));
        assert_eq!(store.list_session(session).unwrap(), vec![id]);

        let deleted = store.delete(id).unwrap();
        assert!(deleted);
        assert!(store.get(id).unwrap().is_none());
        assert_eq!(store.lookup_hash(hash).unwrap(), None);
        assert!(store.list_session(session).unwrap().is_empty());
    }

    #[test]
    fn delete_returns_false_for_unknown_block() {
        let (store, _dir) = open_tmp();
        let bogus = BlockId::from_parts(99, 99);
        assert!(!store.delete(bogus).unwrap());
    }

    #[test]
    fn delete_scrubs_block_from_every_session_referencing_it() {
        let (store, _dir) = open_tmp();
        let s1 = SessionId(1);
        let s2 = SessionId(2);
        let block = make_block(b"shared", BlockKind::Fact, 1, 1);
        let id_a = store.put(s1, block.clone()).unwrap();
        // Same content → dedup → same id under a different session.
        let id_b = store.put(s2, block).unwrap();
        assert_eq!(id_a, id_b);

        store.delete(id_a).unwrap();
        assert!(store.list_session(s1).unwrap().is_empty());
        assert!(store.list_session(s2).unwrap().is_empty());
    }

    #[test]
    fn purge_session_removes_blocks_unique_to_that_session() {
        let (store, _dir) = open_tmp();
        let session = SessionId(7);
        store
            .put(session, make_block(b"a", BlockKind::Fact, 1, 1))
            .unwrap();
        store
            .put(session, make_block(b"b", BlockKind::Fact, 2, 2))
            .unwrap();
        store
            .put(session, make_block(b"c", BlockKind::Fact, 3, 3))
            .unwrap();

        let purged = store.purge_session(session).unwrap();
        assert_eq!(purged, 3);
        assert!(store.list_session(session).unwrap().is_empty());
        assert!(store.list_sessions().unwrap().is_empty());
    }

    #[test]
    fn purge_session_keeps_blocks_referenced_by_other_sessions() {
        let (store, _dir) = open_tmp();
        let s1 = SessionId(1);
        let s2 = SessionId(2);
        // Same content → both sessions point at the same id.
        let id = store
            .put(s1, make_block(b"shared", BlockKind::Fact, 1, 1))
            .unwrap();
        let id_b = store
            .put(s2, make_block(b"shared", BlockKind::Fact, 2, 2))
            .unwrap();
        assert_eq!(id, id_b);
        // s1-only block.
        let _solo = store
            .put(s1, make_block(b"solo", BlockKind::Fact, 3, 3))
            .unwrap();

        store.purge_session(s1).unwrap();
        // s1 is gone.
        assert!(store.list_session(s1).unwrap().is_empty());
        // s2 still references the shared block, and the block content survives.
        assert_eq!(store.list_session(s2).unwrap(), vec![id]);
        assert!(store.get(id).unwrap().is_some());
    }

    #[test]
    fn verify_clean_store_reports_zero_errors() {
        let (store, _dir) = open_tmp();
        let session = SessionId(1);
        store
            .put(session, make_block(b"a", BlockKind::Fact, 1, 1))
            .unwrap();
        store
            .put(session, make_block(b"b", BlockKind::Fact, 2, 2))
            .unwrap();
        let report = store.verify().unwrap();
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.blocks_checked, 2);
    }

    #[test]
    fn repair_rebuilds_hash_index_from_primary_table() {
        let (store, _dir) = open_tmp();
        let session = SessionId(1);
        store
            .put(session, make_block(b"a", BlockKind::Fact, 1, 1))
            .unwrap();
        store
            .put(session, make_block(b"b", BlockKind::Fact, 2, 2))
            .unwrap();
        let report = store.repair().unwrap();
        assert!(report.hash_index_rebuilt);
        assert_eq!(report.hash_entries_written, 2);
        assert_eq!(report.orphan_session_entries_removed, 0);
        // Verify reports clean afterwards.
        assert!(store.verify().unwrap().is_clean());
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
