//! `llm386-compress-anthropic` — Anthropic Claude-backed
//! [`Summarizer`] for LLM386.
//!
//! Lives in its own crate (rather than under `llm386-compress`)
//! because it pulls in `reqwest` + `rustls-tls`. Downstream apps
//! that don't need LLM-driven summarization don't pay for them.
//!
//! ```ignore
//! use llm386_compress_anthropic::AnthropicSummarizer;
//! use llm386_core::Summarizer;
//!
//! let summarizer = AnthropicSummarizer::from_env()?
//!     .with_model("claude-haiku-4-5");
//! let summary = summarizer.summarize(&blocks)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc(html_root_url = "https://docs.rs/llm386-compress-anthropic/0.1.0")]

use std::fmt;
use std::fmt::Write as _;
use std::time::Duration;

use llm386_core::{ContextBlock, Summarizer, SummarizerError};
use serde::{Deserialize, Serialize};

const DEFAULT_API_BASE: &str = "https://api.anthropic.com/v1";
const DEFAULT_MODEL: &str = "claude-haiku-4-5";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 1_024;
const DEFAULT_TIMEOUT_SECS: u64 = 60;

const DEFAULT_INSTRUCTION: &str = "Summarize the following blocks concisely. Preserve essential facts and the overall conversational arc. Output only the summary itself with no preamble.";

/// Anthropic Claude-backed [`Summarizer`].
///
/// Uses the `/v1/messages` endpoint via a blocking `reqwest` client
/// (the `Summarizer` trait is sync). Reads the API key from the
/// `ANTHROPIC_API_KEY` env var when constructed via [`from_env`].
pub struct AnthropicSummarizer {
    api_key: String,
    model: String,
    api_base: String,
    max_tokens: u32,
    instruction: String,
    client: reqwest::blocking::Client,
}

impl AnthropicSummarizer {
    /// Build with an explicit API key. Defaults: model
    /// `claude-haiku-4-5`, `max_tokens` 1024, base
    /// `https://api.anthropic.com/v1`.
    pub fn new(api_key: impl Into<String>) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .expect("reqwest client should build with default settings");
        Self {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.into(),
            api_base: DEFAULT_API_BASE.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            instruction: DEFAULT_INSTRUCTION.into(),
            client,
        }
    }

    /// Build using the `ANTHROPIC_API_KEY` env var. Errors if the
    /// var is unset or empty.
    pub fn from_env() -> Result<Self, SummarizerError> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| SummarizerError::Failed("ANTHROPIC_API_KEY env var not set".into()))?;
        if key.is_empty() {
            return Err(SummarizerError::Failed("ANTHROPIC_API_KEY env var is empty".into()));
        }
        Ok(Self::new(key))
    }

    /// Override the model id (defaults to `claude-haiku-4-5`).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the response token cap (defaults to 1024).
    #[must_use]
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Override the API base URL (useful for proxies or testing).
    #[must_use]
    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }

    /// Override the instruction prepended to the blocks. The
    /// default tells Claude to be concise and preserve essentials.
    #[must_use]
    pub fn with_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = instruction.into();
        self
    }
}

impl fmt::Debug for AnthropicSummarizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Don't leak the API key in Debug output.
        f.debug_struct("AnthropicSummarizer")
            .field("model", &self.model)
            .field("api_base", &self.api_base)
            .field("max_tokens", &self.max_tokens)
            .finish_non_exhaustive()
    }
}

impl Summarizer for AnthropicSummarizer {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn summarize(&self, blocks: &[ContextBlock]) -> Result<String, SummarizerError> {
        if blocks.is_empty() {
            return Ok(String::new());
        }

        let mut user_msg = String::with_capacity(self.instruction.len() + 256);
        user_msg.push_str(&self.instruction);
        user_msg.push_str("\n\n");
        for (i, block) in blocks.iter().enumerate() {
            let text = std::str::from_utf8(&block.bytes)
                .map_err(|e| SummarizerError::Failed(format!("non-utf8 block: {e}")))?;
            let _ = writeln!(user_msg, "--- block {} ({:?}) ---", i + 1, block.kind);
            user_msg.push_str(text);
            user_msg.push_str("\n\n");
        }

        let req = MessagesRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            messages: vec![Message {
                role: "user",
                content: &user_msg,
            }],
        };

        let url = format!("{}/messages", self.api_base);
        let resp: MessagesResponse = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&req)
            .send()
            .map_err(|e| SummarizerError::Failed(format!("request: {e}")))?
            .error_for_status()
            .map_err(|e| SummarizerError::Failed(format!("status: {e}")))?
            .json()
            .map_err(|e| SummarizerError::Failed(format!("parse: {e}")))?;

        let summary: String = resp
            .content
            .into_iter()
            .filter(|c| c.kind == "text")
            .map(|c| c.text)
            .collect();
        if summary.is_empty() {
            return Err(SummarizerError::Failed(
                "no text content in response".into(),
            ));
        }
        Ok(summary)
    }
}

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<Message<'a>>,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_env_errors_when_key_missing() {
        // We can't safely manipulate env vars from a parallel test
        // process under Rust 2024 — instead, verify that whatever
        // the current state is, the error message *would* point at
        // ANTHROPIC_API_KEY when the var is missing. Rather than
        // mutating shared env, just check the live result: if the
        // var is set in the host env, the constructor succeeds; if
        // not, it errors with the expected message.
        match AnthropicSummarizer::from_env() {
            Ok(_) => {
                assert!(std::env::var("ANTHROPIC_API_KEY").is_ok());
            }
            Err(SummarizerError::Failed(msg)) => {
                assert!(msg.contains("ANTHROPIC_API_KEY"), "unexpected error: {msg}");
            }
        }
    }

    #[test]
    fn debug_does_not_leak_api_key() {
        let s = AnthropicSummarizer::new("sk-secret-key-do-not-leak");
        let dbg = format!("{s:?}");
        assert!(!dbg.contains("sk-secret-key-do-not-leak"));
    }

    #[test]
    fn empty_input_returns_empty_string() {
        let s = AnthropicSummarizer::new("dummy-key");
        // No network call since blocks is empty.
        let out = s.summarize(&[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn name_is_anthropic() {
        let s = AnthropicSummarizer::new("dummy-key");
        assert_eq!(s.name(), "anthropic");
    }
}
