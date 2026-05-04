//! `OpenAiEmbedder` — calls the OpenAI `/v1/embeddings` endpoint.

use std::fmt;
use std::time::Duration;

use llm386_core::{Embedder, EmbedderError};
use serde::{Deserialize, Serialize};

const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_DIMS: usize = 1_536; // text-embedding-3-small native size
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// Calls OpenAI's `/v1/embeddings` endpoint via a blocking
/// `reqwest` client.
///
/// Defaults: model `text-embedding-3-small` (1536-dim, cheapest,
/// good-enough for most cases). Override via [`Self::with_model`]
/// and [`Self::with_dimensions`] together — they must match the
/// model the OpenAI side uses or [`embed`] will return vectors of
/// the wrong length.
pub struct OpenAiEmbedder {
    api_key: String,
    model: String,
    api_base: String,
    dimensions: usize,
    client: reqwest::blocking::Client,
}

impl OpenAiEmbedder {
    /// Build with an explicit API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .expect("reqwest client should build with default settings");
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.into(),
            api_base: DEFAULT_API_BASE.into(),
            dimensions: DEFAULT_DIMS,
            client,
        }
    }

    /// Build using the `OPENAI_API_KEY` env var. Errors if unset
    /// or empty.
    pub fn from_env() -> Result<Self, EmbedderError> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| EmbedderError::Failed("OPENAI_API_KEY env var not set".into()))?;
        if key.is_empty() {
            return Err(EmbedderError::Failed(
                "OPENAI_API_KEY env var is empty".into(),
            ));
        }
        Ok(Self::new(key))
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the expected output dimensionality. Must match what the
    /// configured model produces; `text-embedding-3-small` is 1536,
    /// `text-embedding-3-large` is 3072.
    #[must_use]
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    #[must_use]
    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }

    fn post_embeddings(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let req = EmbeddingsRequest {
            model: &self.model,
            input: inputs,
        };
        let url = format!("{}/embeddings", self.api_base);
        let resp: EmbeddingsResponse = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .map_err(|e| EmbedderError::Failed(format!("request: {e}")))?
            .error_for_status()
            .map_err(|e| EmbedderError::Failed(format!("status: {e}")))?
            .json()
            .map_err(|e| EmbedderError::Failed(format!("parse: {e}")))?;

        // Re-order data by index so caller order is preserved.
        let mut data = resp.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
    }
}

impl fmt::Debug for OpenAiEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Don't leak the API key.
        f.debug_struct("OpenAiEmbedder")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("api_base", &self.api_base)
            .finish_non_exhaustive()
    }
}

impl Embedder for OpenAiEmbedder {
    fn name(&self) -> &'static str {
        // We can't return the model dynamically as &'static str, so
        // identify the *adapter* family. Traces include the model id
        // separately when needed.
        "openai-embeddings"
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let mut vecs = self.post_embeddings(&[text])?;
        vecs.pop()
            .ok_or_else(|| EmbedderError::Failed("empty response".into()))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.post_embeddings(texts)
    }
}

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    index: u32,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_handles_unset_or_set() {
        match OpenAiEmbedder::from_env() {
            Ok(_) => assert!(std::env::var("OPENAI_API_KEY").is_ok()),
            Err(EmbedderError::Failed(msg)) => {
                assert!(msg.contains("OPENAI_API_KEY"), "unexpected: {msg}");
            }
        }
    }

    #[test]
    fn debug_does_not_leak_api_key() {
        let e = OpenAiEmbedder::new("sk-secret-key-must-not-leak");
        let dbg = format!("{e:?}");
        assert!(!dbg.contains("sk-secret-key-must-not-leak"));
    }

    #[test]
    fn defaults_are_sensible() {
        let e = OpenAiEmbedder::new("dummy");
        assert_eq!(e.name(), "openai-embeddings");
        assert_eq!(e.dimensions(), 1_536);
    }

    #[test]
    fn empty_batch_does_not_call_api() {
        let e = OpenAiEmbedder::new("dummy");
        // No network call possible — just verify the short-circuit.
        let out = e.embed_batch(&[]).unwrap();
        assert!(out.is_empty());
    }
}
