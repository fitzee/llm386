//! `llm386-compress` — block summarization and structured reductions.
//!
//! Concrete `Summarizer` implementations live here; the trait itself
//! is defined in `llm386-core`. Two cheap, deterministic summarizers
//! ship today:
//!
//! - [`NoopSummarizer`] — emits a placeholder string. Useful as a
//!   default while richer summarizers are wired up.
//! - [`TruncatingSummarizer`] — emits a bullet list with the first N
//!   characters of each block. Deterministic, no network calls.
//!
//! LLM-driven summarization is intentionally out of scope here — it
//! belongs alongside whichever Anthropic / OpenAI client a downstream
//! application already uses, behind the same `Summarizer` trait.

#![doc(html_root_url = "https://docs.rs/llm386-compress/0.1.0")]

mod noop;
mod truncating;

pub use noop::NoopSummarizer;
pub use truncating::TruncatingSummarizer;
