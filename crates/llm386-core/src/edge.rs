//! Typed edges between [`ContextBlock`](crate::ContextBlock)s.
//!
//! Edges are how the runtime models *why one block depends on another*
//! beyond the per-block `Provenance.parents` field — which only
//! captures lineage. Edges add typed relationships (supports,
//! contradicts, derived-from, tool-invocation) that the pager can use
//! for dependency-aware inclusion and that downstream tooling can
//! reason over.
//!
//! Edges are directed: `from` is the *referencing* block, `to` is the
//! *referenced* block. For a "child message → its parent" edge, the
//! child is `from` and the parent is `to`.

use serde::{Deserialize, Serialize};

use crate::ids::BlockId;

/// A directed, typed link between two blocks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub from: BlockId,
    pub to: BlockId,
    pub kind: EdgeKind,
}

/// What the edge means.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Reply / continuation lineage. Mirrors `Provenance.parents` but
    /// makes the relationship explicit and queryable.
    Parent,
    /// `from` was produced by reducing or summarizing `to`.
    DerivedFrom,
    /// `from` cites `to` as evidence.
    Supports,
    /// `from` contradicts `to`.
    Contradicts,
    /// `from` is an assistant message that invoked `to` as a tool
    /// call and consumed its result.
    ToolInvocation,
}
