//! `llm386-packer` — deterministic prompt construction for LLM386.
//!
//! Renders a `PagePlan` into a single prompt string with section
//! headers in the canonical order, verifying that the result fits
//! within the model's input budget.

#![doc(html_root_url = "https://docs.rs/llm386-packer/1.0.0-alpha")]

mod simple;

pub use simple::{CacheOptions, PackerOptions, SimplePacker};
