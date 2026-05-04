//! `PageRequest` / `PagePlan` — the pager's input and output.

use serde::{Deserialize, Serialize};

use crate::ids::{BlockId, SessionId, TokenCount};
use crate::model::ModelProfile;

/// Request to the pager — what to assemble context for.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct PageRequest {
    pub session_id: SessionId,
    pub task: String,
    pub model: ModelProfile,
    /// Block ids that must appear in the plan.
    pub required_blocks: Vec<BlockId>,
}

/// The pager's decision: which blocks were selected, which were
/// omitted, and the aggregate token estimate.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PagePlan {
    pub selected: Vec<BlockId>,
    pub omitted: Vec<OmittedBlock>,
    pub estimated_tokens: TokenCount,
}

/// A candidate block that the pager considered but did not include.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct OmittedBlock {
    pub block_id: BlockId,
    pub reason: OmissionReason,
    pub score: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum OmissionReason {
    /// Section budget was full when the block was considered.
    Budget,
    /// Block was older than the configured staleness threshold.
    Stale,
    /// Block was redundant with another already in the plan.
    Redundant,
    /// Block scored below the configured threshold.
    LowScore,
    /// Block was filtered out by kind / labels.
    FilteredByKind,
    /// Block depended on another block that was not included.
    DependencyMissing,
}
