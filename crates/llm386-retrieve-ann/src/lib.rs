//! `llm386-retrieve-ann` — embedding-based retrievers for LLM386.
//!
//! Two pieces:
//!
//! - [`LinearAnnRetriever`] — brute-force cosine-similarity search
//!   over an in-memory `BlockId → Vec<f32>` index. Embeddings are
//!   computed lazily on the first `retrieve` call and cached. Fine
//!   for thousands of blocks; for larger sessions wire in an HNSW
//!   backend.
//! - [`OpenAiEmbedder`] — calls the OpenAI `/v1/embeddings`
//!   endpoint via blocking `reqwest`. Reads `OPENAI_API_KEY` via
//!   [`OpenAiEmbedder::from_env`].
//!
//! Both can be used independently — bring your own [`Embedder`] if
//! you don't want the OpenAI dep, or bring your own retriever if
//! you want HNSW. The trait surface is the same.

#![doc(html_root_url = "https://docs.rs/llm386-retrieve-ann/1.0.0-alpha")]

mod cache;
mod hnsw;
mod linear;
mod openai;

pub use cache::{EmbeddingCache, EmbeddingCacheError};
pub use hnsw::HnswAnnRetriever;
pub use linear::LinearAnnRetriever;
pub use openai::OpenAiEmbedder;
