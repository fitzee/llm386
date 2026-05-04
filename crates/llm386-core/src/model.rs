//! `ModelProfile` — model-specific context constraints and budgets.

use serde::{Deserialize, Serialize};

use crate::ids::TokenCount;
use crate::tokenizer::TokenizerId;

/// Constraints and capabilities of a target model.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub max_context_tokens: u32,
    pub reserved_output_tokens: u32,
    pub safety_margin_tokens: u32,
    pub tokenizer: TokenizerId,
    pub supports_system_role: bool,
    pub supports_tools: bool,
}

impl ModelProfile {
    /// Effective input budget after subtracting reserved output and
    /// the safety margin. Saturates at zero if the profile is
    /// misconfigured (sum of reservations ≥ context window).
    #[must_use]
    pub const fn input_budget(&self) -> TokenCount {
        let avail = self
            .max_context_tokens
            .saturating_sub(self.reserved_output_tokens)
            .saturating_sub(self.safety_margin_tokens);
        TokenCount(avail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(max: u32, reserved: u32, margin: u32) -> ModelProfile {
        ModelProfile {
            name: "test".to_string(),
            max_context_tokens: max,
            reserved_output_tokens: reserved,
            safety_margin_tokens: margin,
            tokenizer: TokenizerId::new("test"),
            supports_system_role: true,
            supports_tools: true,
        }
    }

    #[test]
    fn input_budget_subtracts_output_and_margin() {
        let p = profile(128_000, 4_000, 1_000);
        assert_eq!(p.input_budget(), TokenCount(123_000));
    }

    #[test]
    fn input_budget_saturates_at_zero_when_misconfigured() {
        let p = profile(1_000, 4_000, 0);
        assert_eq!(p.input_budget(), TokenCount(0));
    }
}
