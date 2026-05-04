//! `Summarizer` trait — the seam between core and `llm386-compress`.
//!
//! A summarizer collapses a slice of [`ContextBlock`]s into a single
//! string suitable to ingest as a `BlockKind::Summary` block. The
//! trait deliberately stays simple: implementations may be pure (the
//! shipped truncating summarizer), or may delegate to an LLM call,
//! an embedding-aware reducer, etc.

use thiserror::Error;

use crate::block::ContextBlock;

pub trait Summarizer: Send + Sync {
    /// Stable identifier for this summarizer (e.g. `"truncating"`,
    /// `"llm-anthropic"`). Surfaces in trace records and `Provenance`.
    fn name(&self) -> &'static str;

    /// Produce a summary string from `blocks`. The summary is owned
    /// — callers typically wrap it in a new `ContextBlock` of kind
    /// [`BlockKind::Summary`] with the originals listed in
    /// `Provenance::parents`.
    ///
    /// [`BlockKind::Summary`]: crate::BlockKind::Summary
    fn summarize(&self, blocks: &[ContextBlock]) -> Result<String, SummarizerError>;
}

#[derive(Debug, Error)]
pub enum SummarizerError {
    #[error("summarizer failed: {0}")]
    Failed(String),
}
