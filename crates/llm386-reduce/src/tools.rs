//! [`AppendOutputReducer`] — minimal "actually useful" reducer.

use llm386_core::{
    BlockId, BlockKind, ContentHash, ContextBlock, Edge, EdgeKind, Provenance, Reducer,
    ReducerError, Reduction, Timestamp, TokenCounts,
};

/// Stores the raw model output as a fresh `AssistantMessage` block.
/// When a previous state block exists and `link_to_state` is on, the
/// new block also gets a [`Parent`](EdgeKind::Parent) edge pointing
/// at the prior state.
///
/// Block ids are derived deterministically from `now_ms` and the
/// content hash so re-running the same reduction yields the same
/// id (and content-hash dedup will collapse it in the store).
#[derive(Clone, Copy, Debug)]
pub struct AppendOutputReducer {
    pub now_ms: u64,
    pub link_to_state: bool,
}

impl AppendOutputReducer {
    #[must_use]
    pub fn new(now_ms: u64) -> Self {
        Self { now_ms, link_to_state: true }
    }
}

impl Reducer for AppendOutputReducer {
    fn name(&self) -> &'static str {
        "append-output"
    }

    fn reduce(
        &self,
        state: Option<&ContextBlock>,
        output: &str,
    ) -> Result<Reduction, ReducerError> {
        if output.is_empty() {
            return Ok(Reduction::empty());
        }
        let bytes = output.as_bytes().to_vec();
        let hash = ContentHash::of(&bytes);
        // Stable id from (now_ms, hash) — same input → same id.
        let block_id = block_id_from(self.now_ms, &hash);
        let provenance = match state {
            Some(s) if self.link_to_state => Provenance {
                source: Some("model-output".into()),
                parents: vec![s.id],
                labels: vec![],
            },
            _ => Provenance {
                source: Some("model-output".into()),
                parents: vec![],
                labels: vec![],
            },
        };
        let block = ContextBlock {
            id: block_id,
            kind: BlockKind::AssistantMessage,
            bytes,
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(self.now_ms),
            updated_at: Timestamp(self.now_ms),
            provenance,
            hash,
        };
        let edges = match state {
            Some(s) if self.link_to_state => vec![Edge {
                from: block_id,
                to: s.id,
                kind: EdgeKind::Parent,
            }],
            _ => vec![],
        };
        Ok(Reduction {
            next_state: None,
            new_blocks: vec![block],
            new_edges: edges,
        })
    }
}

fn block_id_from(now_ms: u64, hash: &ContentHash) -> BlockId {
    // Pack the high 64 bits with `now_ms` (so ids sort by recency)
    // and the low 64 with a stable hash slice.
    let mut low = [0u8; 8];
    low.copy_from_slice(&hash.0[..8]);
    BlockId::from_parts(now_ms, u128::from(u64::from_be_bytes(low)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> ContextBlock {
        let bytes = b"prior state".to_vec();
        let hash = ContentHash::of(&bytes);
        ContextBlock {
            id: BlockId::from_parts(100, 1),
            kind: BlockKind::State,
            bytes,
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(100),
            updated_at: Timestamp(100),
            provenance: Provenance::default(),
            hash,
        }
    }

    #[test]
    fn empty_output_is_empty_reduction() {
        let r = AppendOutputReducer::new(200);
        let red = r.reduce(None, "").unwrap();
        assert!(red.is_empty());
    }

    #[test]
    fn appends_assistant_block_with_state_parent() {
        let state = make_state();
        let r = AppendOutputReducer::new(200);
        let red = r.reduce(Some(&state), "hello there").unwrap();
        assert_eq!(red.new_blocks.len(), 1);
        let new_block = &red.new_blocks[0];
        assert_eq!(new_block.kind, BlockKind::AssistantMessage);
        assert_eq!(new_block.bytes, b"hello there".to_vec());
        assert_eq!(new_block.provenance.parents, vec![state.id]);
        assert_eq!(red.new_edges.len(), 1);
        let edge = red.new_edges[0];
        assert_eq!(edge.from, new_block.id);
        assert_eq!(edge.to, state.id);
        assert_eq!(edge.kind, EdgeKind::Parent);
    }

    #[test]
    fn deterministic_on_same_inputs() {
        let r = AppendOutputReducer::new(200);
        let a = r.reduce(None, "same input").unwrap();
        let b = r.reduce(None, "same input").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn skip_state_link_when_disabled() {
        let state = make_state();
        let mut r = AppendOutputReducer::new(200);
        r.link_to_state = false;
        let red = r.reduce(Some(&state), "hello").unwrap();
        assert!(red.new_edges.is_empty());
        assert!(red.new_blocks[0].provenance.parents.is_empty());
    }
}
