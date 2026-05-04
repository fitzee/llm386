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

    /// Look up a block id by its content hash.
    fn lookup_hash(&self, hash: ContentHash) -> Result<Option<BlockId>, StoreError>;
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("block not found: {0}")]
    NotFound(BlockId),
    #[error("storage backend error: {0}")]
    Backend(String),
}
