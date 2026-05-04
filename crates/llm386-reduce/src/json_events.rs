//! [`JsonEventsReducer`] — parse a JSON envelope of typed events.

use llm386_core::{
    BlockId, BlockKind, ContentHash, ContextBlock, Provenance, Reducer, ReducerError,
    Reduction, Timestamp, TokenCounts,
};
use serde::{Deserialize, Serialize};

/// Envelope the model is expected to produce when paired with this
/// reducer. The `state` field, if present, fully replaces the
/// previous `State` block. `events` are appended as new blocks.
///
/// Example output:
///
/// ```json
/// {
///   "state": "user is debugging the auth flow",
///   "events": [
///     {"kind": "fact",        "body": "session tokens persist for 30d"},
///     {"kind": "plan",        "body": "1. read middleware.rs\n2. check token expiry"}
///   ]
/// }
/// ```
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct JsonEnvelope {
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub events: Vec<Event>,
}

/// One event from a [`JsonEnvelope`].
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Event {
    pub kind: EventKind,
    pub body: String,
}

/// Allowed event kinds in a [`JsonEnvelope`]. Mapped 1:1 to
/// [`BlockKind`] when committed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EventKind {
    Fact,
    Plan,
    Summary,
    DocumentChunk,
}

impl From<EventKind> for BlockKind {
    fn from(k: EventKind) -> Self {
        match k {
            EventKind::Fact => Self::Fact,
            EventKind::Plan => Self::Plan,
            EventKind::Summary => Self::Summary,
            EventKind::DocumentChunk => Self::DocumentChunk,
        }
    }
}

/// Reducer that expects model output to be a [`JsonEnvelope`]. Each
/// event becomes a new block of the corresponding kind; a non-empty
/// `state` field becomes a replacement `State` block.
#[derive(Clone, Copy, Debug)]
pub struct JsonEventsReducer {
    pub now_ms: u64,
}

impl JsonEventsReducer {
    #[must_use]
    pub fn new(now_ms: u64) -> Self {
        Self { now_ms }
    }
}

impl Reducer for JsonEventsReducer {
    fn name(&self) -> &'static str {
        "json-events"
    }

    fn reduce(
        &self,
        state: Option<&ContextBlock>,
        output: &str,
    ) -> Result<Reduction, ReducerError> {
        let envelope: JsonEnvelope = serde_json::from_str(output.trim())
            .map_err(|e| ReducerError::invalid("json-events", format!("parse: {e}")))?;

        let mut new_blocks = Vec::with_capacity(envelope.events.len());
        let parents = state.map(|s| vec![s.id]).unwrap_or_default();

        let next_state = envelope.state.as_deref().filter(|s| !s.is_empty()).map(|body| {
            make_block(BlockKind::State, body, &parents, self.now_ms)
        });

        for ev in envelope.events {
            if ev.body.is_empty() {
                continue;
            }
            new_blocks.push(make_block(ev.kind.into(), &ev.body, &parents, self.now_ms));
        }

        Ok(Reduction {
            next_state,
            new_blocks,
            new_edges: vec![],
        })
    }
}

fn make_block(kind: BlockKind, body: &str, parents: &[BlockId], now_ms: u64) -> ContextBlock {
    let bytes = body.as_bytes().to_vec();
    let hash = ContentHash::of(&bytes);
    let mut low = [0u8; 8];
    // Mix kind into the low half so two events with identical bodies
    // but different kinds get distinct ids. (Content-hash dedup in
    // the store also keys on the bytes, not the kind, so this is
    // belt-and-braces for the in-memory result.)
    low[..8].copy_from_slice(&hash.0[..8]);
    low[0] ^= match kind {
        BlockKind::State => 0xFE,
        BlockKind::Fact => 0x01,
        BlockKind::Plan => 0x02,
        BlockKind::Summary => 0x03,
        BlockKind::DocumentChunk => 0x04,
        _ => 0x00,
    };
    let block_id =
        BlockId::from_parts(now_ms, u128::from(u64::from_be_bytes(low)));
    ContextBlock {
        id: block_id,
        kind,
        bytes,
        token_counts: TokenCounts::new(),
        priority: 0.0,
        created_at: Timestamp(now_ms),
        updated_at: Timestamp(now_ms),
        provenance: Provenance {
            source: Some("model-output".into()),
            parents: parents.to_vec(),
            labels: vec![],
        },
        hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_envelope_into_blocks_and_state() {
        let r = JsonEventsReducer::new(500);
        let body = r#"{
            "state": "user is configuring oauth",
            "events": [
                {"kind": "fact", "body": "callback url must be https"},
                {"kind": "plan", "body": "1. set redirect uri\n2. enable refresh token"}
            ]
        }"#;
        let red = r.reduce(None, body).unwrap();
        let next = red.next_state.unwrap();
        assert_eq!(next.kind, BlockKind::State);
        assert_eq!(next.bytes, b"user is configuring oauth".to_vec());
        assert_eq!(red.new_blocks.len(), 2);
        assert_eq!(red.new_blocks[0].kind, BlockKind::Fact);
        assert_eq!(red.new_blocks[1].kind, BlockKind::Plan);
    }

    #[test]
    fn missing_state_is_none() {
        let r = JsonEventsReducer::new(500);
        let red = r.reduce(None, r#"{"events": []}"#).unwrap();
        assert!(red.next_state.is_none());
        assert!(red.new_blocks.is_empty());
    }

    #[test]
    fn invalid_json_returns_invalid_output() {
        let r = JsonEventsReducer::new(500);
        let err = r.reduce(None, "not json").unwrap_err();
        assert!(matches!(err, ReducerError::InvalidOutput { .. }));
    }

    #[test]
    fn deterministic_on_same_inputs() {
        let r = JsonEventsReducer::new(500);
        let body = r#"{"events":[{"kind":"fact","body":"x"}]}"#;
        let a = r.reduce(None, body).unwrap();
        let b = r.reduce(None, body).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn parents_point_at_previous_state() {
        let prior = ContextBlock {
            id: BlockId::from_parts(100, 7),
            kind: BlockKind::State,
            bytes: b"old state".to_vec(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(100),
            updated_at: Timestamp(100),
            provenance: Provenance::default(),
            hash: ContentHash::of(b"old state"),
        };
        let r = JsonEventsReducer::new(500);
        let red = r
            .reduce(
                Some(&prior),
                r#"{"state":"new state","events":[{"kind":"fact","body":"x"}]}"#,
            )
            .unwrap();
        assert_eq!(red.next_state.as_ref().unwrap().provenance.parents, vec![prior.id]);
        assert_eq!(red.new_blocks[0].provenance.parents, vec![prior.id]);
    }
}
