//! `Reducer` — turn model output into state and event blocks.
//!
//! The runtime's core invariant is that the model is never the source
//! of truth. State lives outside the model and is updated via an
//! explicit reduction step:
//!
//! ```text
//! state(t+1) = reduce(state(t), output)
//! ```
//!
//! A `Reducer` parses model output (free text, JSON, tool calls) and
//! returns a [`Reduction`]: an optional new `State` block, any new
//! blocks the agent should commit (facts, plans, summaries derived
//! from the output), and any new typed [`Edge`]s tying them to
//! existing blocks.
//!
//! Reducers must be deterministic on `(state, output)` so a recorded
//! trace can be replayed by re-running the same reducer against the
//! same inputs.

use thiserror::Error;

use crate::block::ContextBlock;
use crate::edge::Edge;

/// Outcome of applying a reducer to one model response.
#[derive(Clone, Default, PartialEq, Debug)]
pub struct Reduction {
    /// Replacement `State` block, or `None` to leave state untouched.
    pub next_state: Option<ContextBlock>,
    /// Newly derived blocks (facts, summaries, plans, ...) that the
    /// agent should `put` into the store.
    pub new_blocks: Vec<ContextBlock>,
    /// Typed edges to commit alongside `new_blocks`. Edges may
    /// reference both freshly-created blocks and existing ones.
    pub new_edges: Vec<Edge>,
}

impl Reduction {
    /// Convenience for "nothing changed".
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// True when the reducer would not change anything in the store.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.next_state.is_none() && self.new_blocks.is_empty() && self.new_edges.is_empty()
    }
}

/// Applies model output to the previous state to produce the next.
///
/// Implementations MUST be pure for a given `(state, output)` so that
/// a recorded trace is replayable.
pub trait Reducer: Send + Sync {
    /// Stable name used for trace records and config selection.
    fn name(&self) -> &'static str;

    /// Compute a [`Reduction`] from the previous state and the latest
    /// model output.
    fn reduce(
        &self,
        state: Option<&ContextBlock>,
        output: &str,
    ) -> Result<Reduction, ReducerError>;
}

#[derive(Debug, Error)]
pub enum ReducerError {
    #[error("reducer `{name}` failed: {message}")]
    Failed { name: String, message: String },
    #[error("reducer `{name}` rejected output: {message}")]
    InvalidOutput { name: String, message: String },
}

impl ReducerError {
    pub fn failed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Failed { name: name.into(), message: message.into() }
    }

    pub fn invalid(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidOutput { name: name.into(), message: message.into() }
    }
}
