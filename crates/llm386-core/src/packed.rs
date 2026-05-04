//! `PackedPrompt` — the packer's output: a deterministic prompt and
//! the per-block manifest behind it.

use serde::{Deserialize, Serialize};

use crate::ids::{BlockId, TokenCount};

/// A single block as it appears in the packed prompt.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PackedBlock {
    pub block_id: BlockId,
    pub section: SectionKind,
    pub tokens: TokenCount,
    pub score: f32,
}

/// Section / slot a block occupies in the rendered prompt.
///
/// Variant order matches the canonical packer section order — this
/// is also the natural `Ord` for the type so map iterations land in
/// canonical order without a separate `SECTION_ORDER` table.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum SectionKind {
    /// System / hard constraints.
    System,
    /// Current task statement.
    Task,
    /// Active state of the agent.
    State,
    /// Current plan.
    Plan,
    /// Relevant retrieved memory.
    Retrieved,
    /// Tool results.
    Tools,
    /// Recent transcript.
    Recent,
    /// Optional background context.
    Background,
    /// Anti-overflow headroom — intentionally left unfilled.
    Slack,
}

/// The final prompt sent to the model.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PackedPrompt {
    pub model: String,
    pub input_tokens: TokenCount,
    pub blocks: Vec<PackedBlock>,
    pub rendered: String,
}
