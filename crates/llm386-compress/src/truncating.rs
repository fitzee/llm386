//! `TruncatingSummarizer` — first-N-chars-per-block bullet list.

use llm386_core::{BlockKind, ContextBlock, Summarizer, SummarizerError};

/// Emits a bullet list summarizing each input block by its kind and
/// the first `max_chars_per_block` characters of its body.
///
/// Deterministic and free — no LLM calls, no network. Useful as a
/// quick reduction step or as a baseline against richer summarizers.
#[derive(Clone, Copy, Debug)]
pub struct TruncatingSummarizer {
    max_chars_per_block: usize,
}

impl TruncatingSummarizer {
    /// Build a summarizer that keeps at most `max_chars_per_block`
    /// characters per block (default 80 via [`Self::default`]).
    #[must_use]
    pub const fn new(max_chars_per_block: usize) -> Self {
        Self {
            max_chars_per_block,
        }
    }
}

impl Default for TruncatingSummarizer {
    fn default() -> Self {
        Self::new(80)
    }
}

impl Summarizer for TruncatingSummarizer {
    fn name(&self) -> &'static str {
        "truncating"
    }

    fn summarize(&self, blocks: &[ContextBlock]) -> Result<String, SummarizerError> {
        let mut out = String::new();
        for block in blocks {
            let text = std::str::from_utf8(&block.bytes)
                .map_err(|e| SummarizerError::Failed(format!("non-utf8 block: {e}")))?;
            // Collapse internal whitespace runs so multi-line blocks
            // render as a clean single bullet.
            let oneline: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
            let truncated: String = if oneline.chars().count() > self.max_chars_per_block {
                let head: String = oneline.chars().take(self.max_chars_per_block).collect();
                format!("{head}…")
            } else {
                oneline
            };
            out.push_str("- ");
            out.push_str(kind_label(block.kind));
            out.push_str(": ");
            out.push_str(&truncated);
            out.push('\n');
        }
        Ok(out)
    }
}

const fn kind_label(kind: BlockKind) -> &'static str {
    match kind {
        BlockKind::System => "system",
        BlockKind::UserMessage => "user",
        BlockKind::AssistantMessage => "assistant",
        BlockKind::ToolResult => "tool",
        BlockKind::Summary => "summary",
        BlockKind::Fact => "fact",
        BlockKind::DocumentChunk => "document",
        BlockKind::Plan => "plan",
        BlockKind::State => "state",
        BlockKind::Trace => "trace",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm386_core::{BlockId, ContentHash, Provenance, Timestamp, TokenCounts};

    fn block(bytes: &[u8], kind: BlockKind) -> ContextBlock {
        ContextBlock {
            id: BlockId::from_parts(0, 0),
            kind,
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
    fn empty_input_yields_empty_summary() {
        let s = TruncatingSummarizer::default().summarize(&[]).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn renders_bullet_per_block() {
        let s = TruncatingSummarizer::default()
            .summarize(&[
                block(b"hello there friend", BlockKind::UserMessage),
                block(b"hi back", BlockKind::AssistantMessage),
            ])
            .unwrap();
        assert!(s.contains("- user: hello there friend"));
        assert!(s.contains("- assistant: hi back"));
    }

    #[test]
    fn truncates_long_blocks_with_ellipsis() {
        let long = "x".repeat(200);
        let s = TruncatingSummarizer::new(20)
            .summarize(&[block(long.as_bytes(), BlockKind::Fact)])
            .unwrap();
        assert!(s.contains("…"));
        // 20 keep + 1 ellipsis + leading "- fact: " prefix + trailing newline.
        assert!(s.lines().next().unwrap().chars().count() <= 20 + 1 + "- fact: ".len() + 1);
    }

    #[test]
    fn collapses_multiline_blocks_into_one_line() {
        let s = TruncatingSummarizer::default()
            .summarize(&[block(b"line one\nline two\nline three", BlockKind::Fact)])
            .unwrap();
        assert!(s.contains("- fact: line one line two line three"));
        assert_eq!(s.lines().count(), 1);
    }

    #[test]
    fn rejects_non_utf8_input() {
        let bad = block(&[0xff, 0xfe, 0xfd], BlockKind::Fact);
        let err = TruncatingSummarizer::default()
            .summarize(&[bad])
            .unwrap_err();
        assert!(matches!(err, SummarizerError::Failed(_)));
    }
}
