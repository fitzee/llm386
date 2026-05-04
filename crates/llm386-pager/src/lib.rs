//! `llm386-pager` — working-set selection for LLM386.
//!
//! Picks the subset of stored blocks that fits within a model's input
//! budget for a given session and task. The first cut is a recency-
//! weighted greedy pager; section budgets and richer scoring will
//! land in follow-on phases.

#![doc(html_root_url = "https://docs.rs/llm386-pager/0.1.0")]

mod greedy;

pub use greedy::{GreedyPager, ScoringPolicy};
