//! `BlockStore` — persistent storage of context blocks.

use thiserror::Error;

use crate::block::ContextBlock;
use crate::ids::{BlockId, ContentHash, SessionId};

/// Persistent block storage.
///
/// Implementations must enforce content-hash dedup: putting bytes
/// already present in the store returns the existing `BlockId`
/// instead of creating a new one.
pub trait BlockStore: Send + Sync {
    /// Insert a block, returning its assigned id.
    ///
    /// Implementations must dedup by `block.hash`; if a block with
    /// the same hash already exists, return the existing id.
    fn put(&self, session: SessionId, block: ContextBlock) -> Result<BlockId, StoreError>;

    /// Fetch a block by id.
    fn get(&self, id: BlockId) -> Result<Option<ContextBlock>, StoreError>;

    /// All block ids belonging to a session.
    fn list_session(&self, session: SessionId) -> Result<Vec<BlockId>, StoreError>;

    /// Every distinct session id that has at least one block in the
    /// store, sorted ascending. Default impl returns an empty vec —
    /// stores that can't enumerate sessions cheaply may leave it.
    fn list_sessions(&self) -> Result<Vec<SessionId>, StoreError> {
        Ok(vec![])
    }

    /// Look up a block id by its content hash.
    fn lookup_hash(&self, hash: ContentHash) -> Result<Option<BlockId>, StoreError>;

    /// Delete a block entirely: from the primary table, the
    /// content-hash index, and every session that referenced it.
    /// Returns `true` if the block existed, `false` otherwise.
    ///
    /// Default impl errors out so existing stores that don't yet
    /// support deletion fail loudly rather than silently no-op.
    fn delete(&self, _id: BlockId) -> Result<bool, StoreError> {
        Err(StoreError::Backend(
            "delete not implemented for this BlockStore".into(),
        ))
    }

    /// Remove every block belonging to `session`. Blocks that no
    /// other session references are also removed from the primary
    /// table and the content-hash index. Returns the number of
    /// blocks affected (whether or not they were dedup'd elsewhere).
    fn purge_session(&self, _session: SessionId) -> Result<usize, StoreError> {
        Err(StoreError::Backend(
            "purge_session not implemented for this BlockStore".into(),
        ))
    }
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("block not found: {0}")]
    NotFound(BlockId),
    #[error("storage backend error: {0}")]
    Backend(String),
}
