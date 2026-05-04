//! `ModelProfile` and `ModelRegistry` — model-specific context
//! constraints, budgets, and a name-keyed lookup for them.

use std::collections::HashMap;

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

/// Name-keyed lookup of [`ModelProfile`]s.
///
/// Cloning is cheap-ish (clones the inner `HashMap`).
#[derive(Default, Clone, Debug)]
pub struct ModelRegistry {
    by_name: HashMap<String, ModelProfile>,
}

impl ModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a profile under its own [`ModelProfile::name`].
    pub fn register(&mut self, profile: ModelProfile) {
        self.by_name.insert(profile.name.clone(), profile);
    }

    /// Look up a profile by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ModelProfile> {
        self.by_name.get(name)
    }

    /// Iterate over registered profile names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    /// Iterate over all registered profiles.
    pub fn profiles(&self) -> impl Iterator<Item = &ModelProfile> {
        self.by_name.values()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// Built-in model profiles for the OpenAI, Anthropic, Llama, and Qwen
/// families.
///
/// Anthropic profiles reference `cl100k_base` as a tokenizer
/// approximation (Anthropic does not publish an exact public
/// tokenizer) and bump `safety_margin_tokens` accordingly. Llama and
/// Qwen profiles reference tokenizer ids that are not yet shipped by
/// `llm386-tokenizer`; using these profiles before those adapters
/// land will fail at the lookup site, not silently miscount.
///
/// These numbers are starting points; tune per workload.
#[must_use]
pub fn default_profiles() -> Vec<ModelProfile> {
    let openai = TokenizerId::new("o200k_base");
    let anthropic = TokenizerId::new("cl100k_base");
    let llama = TokenizerId::new("llama-3");
    let qwen = TokenizerId::new("qwen-2.5");

    vec![
        // OpenAI
        ModelProfile {
            name: "gpt-4.1".to_string(),
            max_context_tokens: 1_048_576,
            reserved_output_tokens: 32_768,
            safety_margin_tokens: 256,
            tokenizer: openai.clone(),
            supports_system_role: true,
            supports_tools: true,
        },
        ModelProfile {
            name: "gpt-4o".to_string(),
            max_context_tokens: 128_000,
            reserved_output_tokens: 16_384,
            safety_margin_tokens: 256,
            tokenizer: openai.clone(),
            supports_system_role: true,
            supports_tools: true,
        },
        ModelProfile {
            name: "gpt-4o-mini".to_string(),
            max_context_tokens: 128_000,
            reserved_output_tokens: 16_384,
            safety_margin_tokens: 256,
            tokenizer: openai.clone(),
            supports_system_role: true,
            supports_tools: true,
        },
        ModelProfile {
            name: "o1".to_string(),
            max_context_tokens: 200_000,
            reserved_output_tokens: 100_000,
            safety_margin_tokens: 256,
            tokenizer: openai.clone(),
            supports_system_role: false,
            supports_tools: true,
        },
        ModelProfile {
            name: "o3".to_string(),
            max_context_tokens: 200_000,
            reserved_output_tokens: 100_000,
            safety_margin_tokens: 256,
            tokenizer: openai,
            supports_system_role: true,
            supports_tools: true,
        },
        // Anthropic — cl100k_base is an approximation; bumped margin.
        ModelProfile {
            name: "claude-opus-4-7".to_string(),
            max_context_tokens: 200_000,
            reserved_output_tokens: 8_192,
            safety_margin_tokens: 4_096,
            tokenizer: anthropic.clone(),
            supports_system_role: true,
            supports_tools: true,
        },
        ModelProfile {
            name: "claude-sonnet-4-6".to_string(),
            max_context_tokens: 200_000,
            reserved_output_tokens: 8_192,
            safety_margin_tokens: 4_096,
            tokenizer: anthropic.clone(),
            supports_system_role: true,
            supports_tools: true,
        },
        ModelProfile {
            name: "claude-haiku-4-5".to_string(),
            max_context_tokens: 200_000,
            reserved_output_tokens: 8_192,
            safety_margin_tokens: 4_096,
            tokenizer: anthropic,
            supports_system_role: true,
            supports_tools: true,
        },
        // Llama / Qwen — tokenizer adapters not yet shipped.
        ModelProfile {
            name: "llama-3.1-70b".to_string(),
            max_context_tokens: 128_000,
            reserved_output_tokens: 4_096,
            safety_margin_tokens: 512,
            tokenizer: llama,
            supports_system_role: true,
            supports_tools: true,
        },
        ModelProfile {
            name: "qwen-2.5-72b".to_string(),
            max_context_tokens: 128_000,
            reserved_output_tokens: 4_096,
            safety_margin_tokens: 512,
            tokenizer: qwen,
            supports_system_role: true,
            supports_tools: true,
        },
    ]
}

/// Build a [`ModelRegistry`] preloaded with [`default_profiles`].
#[must_use]
pub fn default_registry() -> ModelRegistry {
    let mut reg = ModelRegistry::new();
    for p in default_profiles() {
        reg.register(p);
    }
    reg
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

    #[test]
    fn registry_register_and_lookup() {
        let mut reg = ModelRegistry::new();
        reg.register(profile(1_000, 100, 10));
        assert!(reg.get("test").is_some());
        assert!(reg.get("nope").is_none());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn default_registry_includes_known_models() {
        let reg = default_registry();
        for name in [
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o3",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "llama-3.1-70b",
            "qwen-2.5-72b",
        ] {
            assert!(reg.get(name).is_some(), "missing built-in profile: {name}");
        }
    }

    #[test]
    fn all_default_profiles_have_positive_input_budget() {
        for p in default_profiles() {
            assert!(
                p.input_budget().0 > 0,
                "profile {} has non-positive input budget",
                p.name,
            );
        }
    }

    #[test]
    fn anthropic_profiles_use_bumped_safety_margin() {
        let reg = default_registry();
        for name in ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"] {
            let p = reg.get(name).unwrap();
            assert!(
                p.safety_margin_tokens >= 1024,
                "expected bumped margin on {name} (cl100k approximation)",
            );
        }
    }
}
