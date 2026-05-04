//! `Embedder` trait — the seam between core and `llm386-retrieve-ann`.
//!
//! An embedder turns text (a UTF-8 byte slice) into a fixed-length
//! vector of `f32`. The exact dimensionality is a property of the
//! embedder; consumers (e.g. the ANN retriever) treat all vectors
//! produced by one embedder as compatible and never mix embedders
//! within a single index.

use thiserror::Error;

pub trait Embedder: Send + Sync {
    /// Stable identifier (e.g. `"openai-text-embedding-3-small"`)
    /// used in traces and to guard against mixing embedders.
    fn name(&self) -> &'static str;

    /// Output vector dimensionality. Must be identical for every
    /// `embed` call from this embedder.
    fn dimensions(&self) -> usize;

    /// Compute an embedding for the given text. The returned vector
    /// length must equal [`dimensions`].
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;

    /// Batch embedding — defaults to calling [`embed`] for each
    /// input. Adapters with a real batch endpoint (OpenAI, Cohere)
    /// should override this for efficiency.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

#[derive(Debug, Error)]
pub enum EmbedderError {
    #[error("embedder failed: {0}")]
    Failed(String),
}
