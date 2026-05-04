//! `llm386-reduce` — reference [`Reducer`] implementations.
//!
//! Reducers are the explicit step that turns model output into
//! committed state changes. The runtime treats the model as a
//! stateless function; the reducer is what closes the loop and
//! produces the next state and event blocks.
//!
//! Three impls ship today:
//!
//! - [`IdentityReducer`] — never changes anything; useful as a
//!   placeholder or as a no-op in tests.
//! - [`AppendOutputReducer`] — stores the raw model output as an
//!   `AssistantMessage` block, optionally parented to the previous
//!   state block via a [`Parent`](EdgeKind::Parent) edge. The most
//!   minimal "actually useful" reducer.
//! - [`JsonEventsReducer`] — parses output as a structured envelope
//!   and emits typed event blocks (facts, plans) plus an optional
//!   replacement state block. Useful when the model is prompted to
//!   produce JSON.
//!
//! All three are deterministic on `(state, output)`.

#![doc(html_root_url = "https://docs.rs/llm386-reduce/1.0.0-alpha")]

mod identity;
mod json_events;
mod tools;

pub use identity::IdentityReducer;
pub use json_events::{Event, JsonEnvelope, JsonEventsReducer};
pub use tools::AppendOutputReducer;
