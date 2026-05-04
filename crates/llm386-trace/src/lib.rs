//! `llm386-trace` — observability and replay storage for LLM386.
//!
//! Provides [`LmdbTraceSink`], a `TraceSink` implementation that
//! persists [`TraceRecord`]s to LMDB so a page+pack invocation can
//! be inspected or replayed after the fact.

#![doc(html_root_url = "https://docs.rs/llm386-trace/1.0.0-alpha")]

mod lmdb;

pub use lmdb::{LmdbTraceSink, TraceOpenError};
