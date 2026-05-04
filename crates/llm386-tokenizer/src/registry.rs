//! `TokenizerRegistry` — name-based lookup of tokenizer implementations.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use llm386_core::{Tokenizer, TokenizerId};
use thiserror::Error;

/// Maps a [`TokenizerId`] to a concrete [`Tokenizer`] implementation.
///
/// Cheap to clone (each entry is an `Arc`).
#[derive(Default, Clone)]
pub struct TokenizerRegistry {
    entries: HashMap<TokenizerId, Arc<dyn Tokenizer>>,
}

impl TokenizerRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tokenizer under its own [`Tokenizer::id`].
    pub fn register(&mut self, tokenizer: Arc<dyn Tokenizer>) {
        let id = tokenizer.id().clone();
        self.entries.insert(id, tokenizer);
    }

    /// Look up a tokenizer by id.
    #[must_use]
    pub fn get(&self, id: &TokenizerId) -> Option<Arc<dyn Tokenizer>> {
        self.entries.get(id).cloned()
    }

    /// Look up a tokenizer or return [`RegistryError::NotFound`].
    pub fn require(&self, id: &TokenizerId) -> Result<Arc<dyn Tokenizer>, RegistryError> {
        self.get(id)
            .ok_or_else(|| RegistryError::NotFound(id.clone()))
    }

    /// Iterate over the registered ids.
    pub fn ids(&self) -> impl Iterator<Item = &TokenizerId> {
        self.entries.keys()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Debug for TokenizerRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenizerRegistry")
            .field("ids", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("tokenizer not found: {0}")]
    NotFound(TokenizerId),
}
