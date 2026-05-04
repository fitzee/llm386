//! `TraceSink` — observability and replay storage.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{CallId, ContentHash, SessionId, Timestamp, TokenCount};
use crate::page::PagePlan;

/// Record of a single page+pack invocation.
///
/// Older sinks recorded only what was known at pack time (the plan,
/// the prompt, the timing). The output-side fields are populated
/// after the model returns and the agent loop calls
/// [`TraceSink::update_output`]. Existing records on disk that pre-date
/// these fields deserialize with the defaults via `#[serde(default)]`.
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
    /// Specific model build behind the [`TraceRecord::model`] alias.
    /// E.g. `model = "gpt-4o"` and `model_version = "gpt-4o-2024-08-06"`.
    /// Defaults to an empty string for records written before this
    /// field existed.
    #[serde(default)]
    pub model_version: String,
    /// Tokenizer used to compute `prompt_tokens` and `output_tokens`.
    /// Lets a replay distinguish "same prompt text under a different
    /// tokenizer" from a real change. Defaults to empty for legacy
    /// records.
    #[serde(default)]
    pub tokenizer_version: String,
    /// Model output text, if recorded. `None` until the agent loop
    /// patches it in via [`TraceSink::update_output`].
    #[serde(default)]
    pub output: Option<String>,
    /// Token count of `output` under `tokenizer_version`. Set together
    /// with `output`.
    #[serde(default)]
    pub output_tokens: Option<TokenCount>,
}

pub trait TraceSink: Send + Sync {
    fn record(&self, trace: TraceRecord) -> Result<(), TraceError>;
    fn fetch(&self, call_id: CallId) -> Result<Option<TraceRecord>, TraceError>;

    /// Patch the model-side fields of an already-recorded trace. The
    /// default impl fetches, mutates, and re-records so any sink that
    /// implements `record` + `fetch` works out of the box.
    fn update_output(
        &self,
        call_id: CallId,
        output: String,
        output_tokens: TokenCount,
    ) -> Result<(), TraceError> {
        let mut record = self
            .fetch(call_id)?
            .ok_or_else(|| TraceError::Failed(format!("no trace for {call_id}")))?;
        record.output = Some(output);
        record.output_tokens = Some(output_tokens);
        self.record(record)
    }
}

#[derive(Debug, Error)]
pub enum TraceError {
    #[error("trace sink failed: {0}")]
    Failed(String),
}
