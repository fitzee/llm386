//! `ContextBlock` — the atomic unit of memory in LLM386.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::{BlockId, ContentHash, Timestamp, TokenCount};
use crate::packed::SectionKind;
use crate::tokenizer::TokenizerId;

/// The atomic unit of memory.
///
/// Blocks are conceptually immutable: once written and assigned an
/// `id`, the `bytes` and `hash` never change. Logical updates are
/// modeled as new blocks with a `parents` link in [`Provenance`].
/// `updated_at` tracks when *derived* state (token counts, indexes)
/// was last refreshed — never the bytes themselves.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct ContextBlock {
    pub id: BlockId,
    pub kind: BlockKind,
    pub bytes: Vec<u8>,
    pub token_counts: TokenCounts,
    pub priority: f32,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub provenance: Provenance,
    pub hash: ContentHash,
}

/// Kind / role of a context block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum BlockKind {
    System,
    UserMessage,
    AssistantMessage,
    ToolResult,
    Summary,
    Fact,
    DocumentChunk,
    Plan,
    State,
    Trace,
}

impl BlockKind {
    /// Map this block kind to the [`SectionKind`] the pager and
    /// packer use by default. Both crates share this mapping so
    /// budget allocation and rendering agree on which section a
    /// block belongs to.
    #[must_use]
    pub const fn default_section(self) -> SectionKind {
        match self {
            Self::System => SectionKind::System,
            Self::State => SectionKind::State,
            Self::Plan => SectionKind::Plan,
            Self::Summary | Self::Fact => SectionKind::Retrieved,
            Self::ToolResult => SectionKind::Tools,
            Self::UserMessage | Self::AssistantMessage => SectionKind::Recent,
            Self::DocumentChunk | Self::Trace => SectionKind::Background,
        }
    }
}

/// Where a block came from and how it relates to others.
#[derive(Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub struct Provenance {
    /// Optional identifier for the originating document, tool call,
    /// message, etc.
    pub source: Option<String>,
    /// Parent block ids — used to model summarization, supersession,
    /// and other derivations.
    pub parents: Vec<BlockId>,
    /// Free-form labels for filtering.
    pub labels: Vec<String>,
}

/// Map of [`TokenizerId`] → [`TokenCount`] precomputed for a block.
///
/// Storage is a `BTreeMap` so serialized output is deterministic.
#[derive(Clone, PartialEq, Eq, Default, Debug, Serialize, Deserialize)]
pub struct TokenCounts {
    counts: BTreeMap<TokenizerId, TokenCount>,
}

impl TokenCounts {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, tokenizer: TokenizerId, count: TokenCount) {
        self.counts.insert(tokenizer, count);
    }

    #[must_use]
    pub fn get(&self, tokenizer: &TokenizerId) -> Option<TokenCount> {
        self.counts.get(tokenizer).copied()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TokenizerId, TokenCount)> {
        self.counts.iter().map(|(k, v)| (k, *v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_insert_and_get() {
        let mut tc = TokenCounts::new();
        let id = TokenizerId::new("cl100k_base");
        tc.insert(id.clone(), TokenCount(42));
        assert_eq!(tc.get(&id), Some(TokenCount(42)));
        assert_eq!(tc.len(), 1);
        assert!(!tc.is_empty());
    }

    #[test]
    fn token_counts_get_missing_is_none() {
        let tc = TokenCounts::new();
        assert!(tc.get(&TokenizerId::new("nope")).is_none());
        assert!(tc.is_empty());
    }
}
