//! `llm386-tokenizer` — tokenizer adapters and registry for LLM386.
//!
//! Concrete `Tokenizer` implementations live here; the trait itself
//! is defined in `llm386-core`.

#![doc(html_root_url = "https://docs.rs/llm386-tokenizer/0.1.0")]

mod cache;
mod huggingface;
mod registry;
mod tiktoken;

use std::sync::Arc;

use llm386_core::TokenizerError;

pub use cache::CachingTokenizer;
pub use huggingface::HfTokenizer;
pub use registry::{RegistryError, TokenizerRegistry};
pub use tiktoken::{TiktokenTokenizer, cl100k_base, o200k_base};

/// Build a registry preloaded with the common OpenAI BPE tokenizers
/// (`cl100k_base`, `o200k_base`).
///
/// Anthropic and Llama tokenizers are not yet shipped — model
/// profiles for those families should reference `cl100k_base` as a
/// rough approximation and bump `safety_margin_tokens` accordingly
/// until dedicated adapters land.
pub fn default_registry() -> Result<TokenizerRegistry, TokenizerError> {
    let mut reg = TokenizerRegistry::new();
    reg.register(Arc::new(cl100k_base()?));
    reg.register(Arc::new(o200k_base()?));
    Ok(reg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm386_core::TokenizerId;

    #[test]
    fn default_registry_has_common_tokenizers() {
        let reg = default_registry().unwrap();
        assert!(reg.get(&TokenizerId::new("cl100k_base")).is_some());
        assert!(reg.get(&TokenizerId::new("o200k_base")).is_some());
        assert_eq!(reg.len(), 2);
    }
}
