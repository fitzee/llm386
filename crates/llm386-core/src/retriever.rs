//! `Retriever` — pluggable block retrieval.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{BlockId, SessionId};

/// A candidate block surfaced by a retriever.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct RetrievalCandidate {
    pub block_id: BlockId,
    pub score: f32,
    /// Name of the retriever that produced this candidate.
    pub source: String,
}

/// Pluggable retrieval strategy.
///
/// Multiple `Retriever` implementations are typically run in parallel
/// and their results merged + scored by the pager.
pub trait Retriever: Send + Sync {
    /// Stable identifier for this retriever — used in
    /// [`RetrievalCandidate::source`] and in traces. Must be a
    /// `'static` string since retriever names are identifiers, not
    /// values.
    fn name(&self) -> &'static str;

    fn retrieve(
        &self,
        session: SessionId,
        task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError>;
}

#[derive(Debug, Error)]
pub enum RetrievalError {
    #[error("retrieval failed: {0}")]
    Failed(String),
}
