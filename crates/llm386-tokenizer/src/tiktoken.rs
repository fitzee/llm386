//! Adapter for the `tiktoken-rs` crate (OpenAI BPE tokenizers).

use std::fmt;

use llm386_core::{TokenCount, Tokenizer, TokenizerError, TokenizerId};

/// `tiktoken-rs`-backed [`Tokenizer`].
///
/// Inputs must be valid UTF-8; binary blobs return
/// [`TokenizerError::EncodingFailed`]. Counts use `encode_ordinary`
/// so that special-token markers in user content (e.g. literal
/// `<|endoftext|>`) are treated as plain text rather than control
/// tokens — matching how the OpenAI API counts user-supplied input.
pub struct TiktokenTokenizer {
    id: TokenizerId,
    inner: tiktoken_rs::CoreBPE,
}

impl TiktokenTokenizer {
    pub fn new(id: TokenizerId, inner: tiktoken_rs::CoreBPE) -> Self {
        Self { id, inner }
    }
}

impl fmt::Debug for TiktokenTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // CoreBPE doesn't impl Debug; intentionally elided.
        f.debug_struct("TiktokenTokenizer")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Tokenizer for TiktokenTokenizer {
    fn id(&self) -> &TokenizerId {
        &self.id
    }

    fn count(&self, bytes: &[u8]) -> Result<TokenCount, TokenizerError> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| TokenizerError::EncodingFailed(format!("invalid UTF-8: {e}")))?;
        let n = self.inner.encode_ordinary(s).len();
        Ok(TokenCount(u32::try_from(n).unwrap_or(u32::MAX)))
    }
}

/// `cl100k_base` — used by GPT-4 and GPT-3.5-turbo.
pub fn cl100k_base() -> Result<TiktokenTokenizer, TokenizerError> {
    let bpe = tiktoken_rs::cl100k_base()
        .map_err(|e| TokenizerError::EncodingFailed(format!("cl100k_base init: {e}")))?;
    Ok(TiktokenTokenizer::new(TokenizerId::new("cl100k_base"), bpe))
}

/// `o200k_base` — used by GPT-4o and the o-series.
pub fn o200k_base() -> Result<TiktokenTokenizer, TokenizerError> {
    let bpe = tiktoken_rs::o200k_base()
        .map_err(|e| TokenizerError::EncodingFailed(format!("o200k_base init: {e}")))?;
    Ok(TiktokenTokenizer::new(TokenizerId::new("o200k_base"), bpe))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cl100k_counts_known_string() {
        let t = cl100k_base().unwrap();
        // "hello world" is 2 cl100k tokens: "hello" + " world".
        assert_eq!(t.count(b"hello world").unwrap(), TokenCount(2));
        assert_eq!(t.id().as_str(), "cl100k_base");
    }

    #[test]
    fn empty_string_is_zero_tokens() {
        let t = cl100k_base().unwrap();
        assert_eq!(t.count(b"").unwrap(), TokenCount(0));
    }

    #[test]
    fn invalid_utf8_returns_error() {
        let t = cl100k_base().unwrap();
        assert!(t.count(&[0xff, 0xfe, 0xfd]).is_err());
    }

    #[test]
    fn o200k_counts_independently_from_cl100k() {
        let cl = cl100k_base().unwrap();
        let o = o200k_base().unwrap();
        // Both should at least tokenize the same input non-zero.
        let s = b"The quick brown fox jumps over the lazy dog.";
        assert!(cl.count(s).unwrap().0 > 0);
        assert!(o.count(s).unwrap().0 > 0);
        assert_eq!(o.id().as_str(), "o200k_base");
    }
}
