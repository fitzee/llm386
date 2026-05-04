//! [`IdentityReducer`] — a no-op reducer.

use llm386_core::{ContextBlock, Reducer, ReducerError, Reduction};

/// Reducer that never changes anything. Returned `Reduction` is
/// always empty; mostly useful as a placeholder before a real
/// reducer is wired in, and in tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityReducer;

impl Reducer for IdentityReducer {
    fn name(&self) -> &'static str {
        "identity"
    }

    fn reduce(
        &self,
        _state: Option<&ContextBlock>,
        _output: &str,
    ) -> Result<Reduction, ReducerError> {
        Ok(Reduction::empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_returns_empty_reduction() {
        let r = IdentityReducer;
        let red = r.reduce(None, "anything").unwrap();
        assert!(red.is_empty());
    }
}
