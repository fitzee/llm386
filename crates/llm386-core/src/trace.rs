//! `TraceSink` — observability and replay storage.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{CallId, ContentHash, SessionId, Timestamp, TokenCount};
use crate::page::PagePlan;

/// Record of a single page+pack invocation.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    pub call_id: CallId,
    pub session: SessionId,
    pub model: String,
    pub plan: PagePlan,
    pub prompt_tokens: TokenCount,
    pub prompt_hash: ContentHash,
    pub started_at: Timestamp,
    pub duration_ms: u32,
}

pub trait TraceSink: Send + Sync {
    fn record(&self, trace: TraceRecord) -> Result<(), TraceError>;
    fn fetch(&self, call_id: CallId) -> Result<Option<TraceRecord>, TraceError>;
}

#[derive(Debug, Error)]
pub enum TraceError {
    #[error("trace sink failed: {0}")]
    Failed(String),
}
