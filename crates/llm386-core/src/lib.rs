//! `llm386-core` — core domain types and trait seams for the LLM386 runtime.
//!
//! See `CLAUDE.md` at the workspace root for the architectural overview.
//! This crate defines the data model and the trait seams used by
//! storage, retrieval, paging, packing, and tracing implementations in
//! their respective sibling crates.

#![doc(html_root_url = "https://docs.rs/llm386-core/0.1.0")]

mod block;
mod error;
mod ids;
mod model;
mod packed;
mod packer;
mod page;
mod pager;
mod retriever;
mod store;
mod tokenizer;
mod trace;

pub use block::{BlockKind, ContextBlock, Provenance, TokenCounts};
pub use error::LlmError;
pub use ids::{BlockId, CallId, ContentHash, SessionId, Timestamp, TokenCount};
pub use model::{ModelProfile, ModelRegistry, default_profiles, default_registry};
pub use packed::{ChatMessage, ChatPrompt, ChatRole, PackedBlock, PackedPrompt, SectionKind};
pub use packer::{Packer, PackerError};
pub use page::{OmissionReason, OmittedBlock, PagePlan, PageRequest};
pub use pager::{Pager, PagerError};
pub use retriever::{RetrievalCandidate, RetrievalError, Retriever};
pub use store::{BlockStore, StoreError};
pub use tokenizer::{Tokenizer, TokenizerError, TokenizerId};
pub use trace::{TraceError, TraceRecord, TraceSink};
