//! `llm386-core` — core domain types and trait seams for the LLM386 runtime.
//!
//! See the workspace-level `README.md` for the architectural overview.
//! This crate defines the data model and the trait seams used by
//! storage, retrieval, paging, packing, and tracing implementations in
//! their respective sibling crates.

#![doc(html_root_url = "https://docs.rs/llm386-core/0.1.0")]

mod block;
mod edge;
mod embedder;
mod error;
mod ids;
mod model;
mod packed;
mod packer;
mod page;
mod pager;
mod reducer;
mod retriever;
mod store;
mod summarizer;
mod tokenizer;
mod trace;

pub use block::{BlockKind, ContextBlock, Provenance, TokenCounts};
pub use edge::{Edge, EdgeKind};
pub use embedder::{Embedder, EmbedderError};
pub use error::LlmError;
pub use ids::{BlockId, CallId, ContentHash, SessionId, Timestamp, TokenCount};
pub use model::{ModelProfile, ModelRegistry, default_profiles, default_registry};
pub use packed::{ChatMessage, ChatPrompt, ChatRole, PackedBlock, PackedPrompt, SectionKind};
pub use packer::{Packer, PackerError};
pub use page::{OmissionReason, OmittedBlock, PagePlan, PageRequest, Selection, SelectionReason};
pub use pager::{Pager, PagerError};
pub use reducer::{Reducer, ReducerError, Reduction};
pub use retriever::{RetrievalCandidate, RetrievalError, Retriever};
pub use store::{BlockStore, StoreError};
pub use summarizer::{Summarizer, SummarizerError};
pub use tokenizer::{Tokenizer, TokenizerError, TokenizerId};
pub use trace::{TraceError, TraceRecord, TraceSink};
