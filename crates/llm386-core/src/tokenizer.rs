//! `Tokenizer` trait — count tokens in arbitrary bytes for a specific
//! tokenizer family.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::TokenCount;

/// String-keyed identifier for a tokenizer family
/// (e.g. `"cl100k_base"`, `"o200k_base"`, `"llama-3"`).
///
/// `#[serde(transparent)]` so it round-trips as a plain string in
/// JSON / TOML / etc. — `tokenizer = "cl100k_base"` is the wire form.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokenizerId(String);

impl TokenizerId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TokenizerId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for TokenizerId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl fmt::Display for TokenizerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Counts tokens in arbitrary bytes for a specific tokenizer family.
///
/// Implementations live in `llm386-tokenizer`. `count` must be
/// deterministic and pure for the given input.
pub trait Tokenizer: Send + Sync {
    fn id(&self) -> &TokenizerId;
    fn count(&self, bytes: &[u8]) -> Result<TokenCount, TokenizerError>;
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("tokenizer encoding failed: {0}")]
    EncodingFailed(String),
}
