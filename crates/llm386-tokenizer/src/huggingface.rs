//! Adapter for the `tokenizers` crate (HuggingFace tokenizer
//! family — Llama, Qwen, Mistral, etc.).
//!
//! Use this when you need an exact (not approximated) token count
//! for a non-OpenAI model. Construction takes a `tokenizer.json`
//! file from the model's HuggingFace repo:
//!
//! ```ignore
//! use std::sync::Arc;
//! use llm386_core::Tokenizer;
//! use llm386_tokenizer::HfTokenizer;
//!
//! let llama = HfTokenizer::from_file("models/llama-3/tokenizer.json", "llama-3")?;
//! let n = llama.count(b"hello world")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::fmt;
use std::path::Path;

use llm386_core::{TokenCount, Tokenizer, TokenizerError, TokenizerId};

/// HuggingFace `tokenizers`-backed [`Tokenizer`].
///
/// Inputs must be valid UTF-8 (the underlying tokenizer requires
/// `&str`); binary blobs return [`TokenizerError::EncodingFailed`].
/// Special-token markers in user content are treated as plain text
/// (no special-token handling) — matching how the model's serving
/// stack typically counts user input.
pub struct HfTokenizer {
    id: TokenizerId,
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    /// Build from an already-loaded `tokenizers::Tokenizer`. Use the
    /// `from_file` / `from_bytes` helpers for the common cases.
    pub fn new(id: TokenizerId, inner: tokenizers::Tokenizer) -> Self {
        Self { id, inner }
    }

    /// Load a tokenizer from a `tokenizer.json` file on disk.
    ///
    /// `id` is the [`TokenizerId`] this tokenizer registers under
    /// (e.g. `TokenizerId::new("llama-3")`); model profiles
    /// reference the tokenizer by this name.
    pub fn from_file(
        path: impl AsRef<Path>,
        id: impl Into<TokenizerId>,
    ) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| TokenizerError::EncodingFailed(format!("from_file: {e}")))?;
        Ok(Self::new(id.into(), inner))
    }

    /// Load a tokenizer from a `tokenizer.json` byte buffer (e.g.
    /// embedded via `include_bytes!`).
    pub fn from_bytes(bytes: &[u8], id: impl Into<TokenizerId>) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| TokenizerError::EncodingFailed(format!("from_bytes: {e}")))?;
        Ok(Self::new(id.into(), inner))
    }
}

impl fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Underlying Tokenizer doesn't impl Debug usefully; elide.
        f.debug_struct("HfTokenizer")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Tokenizer for HfTokenizer {
    fn id(&self) -> &TokenizerId {
        &self.id
    }

    fn count(&self, bytes: &[u8]) -> Result<TokenCount, TokenizerError> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| TokenizerError::EncodingFailed(format!("invalid UTF-8: {e}")))?;
        // Encode without adding special tokens — see module-level
        // doc for the rationale.
        let encoding = self
            .inner
            .encode(s, false)
            .map_err(|e| TokenizerError::EncodingFailed(format!("encode: {e}")))?;
        let n = encoding.get_ids().len();
        Ok(TokenCount(u32::try_from(n).unwrap_or(u32::MAX)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal hand-rolled WordLevel tokenizer.json — keeps the
    /// test self-contained (no fixture files, no network).
    const MINIMAL_WORDLEVEL: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "hello": 0, "world": 1, "[UNK]": 2 },
    "unk_token": "[UNK]"
  }
}"#;

    fn make_tokenizer() -> HfTokenizer {
        HfTokenizer::from_bytes(
            MINIMAL_WORDLEVEL.as_bytes(),
            TokenizerId::new("test-wordlevel"),
        )
        .expect("minimal tokenizer should parse")
    }

    #[test]
    fn from_bytes_loads_minimal_tokenizer() {
        let t = make_tokenizer();
        assert_eq!(t.id().as_str(), "test-wordlevel");
    }

    #[test]
    fn counts_whitespace_words() {
        let t = make_tokenizer();
        assert_eq!(t.count(b"hello world").unwrap(), TokenCount(2));
        assert_eq!(t.count(b"hello").unwrap(), TokenCount(1));
        assert_eq!(t.count(b"").unwrap(), TokenCount(0));
    }

    #[test]
    fn unknown_words_route_to_unk_token() {
        let t = make_tokenizer();
        // "unknown" is not in vocab → 1 [UNK] token.
        assert_eq!(t.count(b"unknown").unwrap(), TokenCount(1));
        // Two unknown words → 2 [UNK] tokens.
        assert_eq!(t.count(b"unknown gibberish").unwrap(), TokenCount(2));
    }

    #[test]
    fn invalid_utf8_returns_error() {
        let t = make_tokenizer();
        assert!(t.count(&[0xff, 0xfe, 0xfd]).is_err());
    }

    #[test]
    fn malformed_json_returns_error() {
        assert!(HfTokenizer::from_bytes(b"not json", TokenizerId::new("x")).is_err());
    }
}
