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
#[serde(rename_all = "lowercase")]
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

/// Role of a single message in a chat-formatted prompt.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One role-tagged message in a chat-formatted prompt.
///
/// `section` carries the [`SectionKind`] the message originated
/// from. Used by provider adapters to set prompt-cache markers
/// (Anthropic `cache_control`, Gemini `CachedContent`) on the
/// stable prefix of the prompt — see [`ChatPrompt::cache_boundary`].
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub section: Option<SectionKind>,
}

/// Chat-formatted equivalent of [`PackedPrompt`] for chat-API models
/// (OpenAI, Anthropic, etc.).
///
/// `messages` is the role-tagged sequence to send. `input_tokens`
/// is the total tokenized cost across every message content (the
/// packer guarantees `input_tokens <= ModelProfile::input_budget`).
///
/// `cache_boundary`, when `Some(n)`, is the index of the last
/// message in the stable prefix — i.e. `messages[0..=n]` are the
/// portion of the prompt that does not change between turns and is
/// a candidate for provider prompt caching. Computed by the packer
/// from its `stable_sections` config. `None` means no message
/// qualifies as part of the stable prefix.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ChatPrompt {
    pub model: String,
    pub input_tokens: TokenCount,
    pub messages: Vec<ChatMessage>,
    pub cache_boundary: Option<usize>,
}
