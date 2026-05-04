//! `NoopSummarizer` — emits a one-line placeholder.

use llm386_core::{ContextBlock, Summarizer, SummarizerError};

/// A summarizer that returns a single-line placeholder. Use as a
/// default sentinel until a real summarizer is plugged in.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopSummarizer;

impl Summarizer for NoopSummarizer {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn summarize(&self, blocks: &[ContextBlock]) -> Result<String, SummarizerError> {
        Ok(format!("[summary of {} block(s) elided]", blocks.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm386_core::{BlockId, BlockKind, ContentHash, Provenance, Timestamp, TokenCounts};

    fn block(bytes: &[u8]) -> ContextBlock {
        ContextBlock {
            id: BlockId::from_parts(0, 0),
            kind: BlockKind::Fact,
            bytes: bytes.to_vec(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(0),
            updated_at: Timestamp(0),
            provenance: Provenance::default(),
            hash: ContentHash::of(bytes),
        }
    }

    #[test]
    fn empty_input_yields_zero_count_placeholder() {
        let s = NoopSummarizer.summarize(&[]).unwrap();
        assert!(s.contains("0 block"));
    }

    #[test]
    fn nonempty_input_includes_count() {
        let s = NoopSummarizer
            .summarize(&[block(b"a"), block(b"b")])
            .unwrap();
        assert!(s.contains("2 block"));
    }
}
