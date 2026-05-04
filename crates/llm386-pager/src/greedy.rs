//! `GreedyPager` — recency-weighted greedy block selection.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use llm386_core::{
    BlockId, BlockStore, ContextBlock, OmissionReason, OmittedBlock, PagePlan, PageRequest, Pager,
    PagerError, TokenCount, Tokenizer,
};
use tracing::instrument;

/// Weights for the linear score function.
///
/// All weights should be non-negative. Defaults are tuned for "show
/// the most recent stuff that fits, biased toward higher-priority
/// blocks."
#[derive(Clone, Copy, Debug)]
pub struct ScoringPolicy {
    pub recency_weight: f32,
    pub priority_weight: f32,
}

impl Default for ScoringPolicy {
    fn default() -> Self {
        Self {
            recency_weight: 1.0,
            priority_weight: 0.5,
        }
    }
}

/// Recency-weighted greedy [`Pager`].
///
/// Pipeline:
/// 1. Resolve required blocks (must exist in the store).
/// 2. List the rest of the session's blocks; score by normalized
///    recency + priority; sort by score-per-token descending.
/// 3. Greedy-fill the remaining input budget. Blocks that don't fit
///    land in [`PagePlan::omitted`] with [`OmissionReason::Budget`].
///
/// Section budgets, multi-retriever fan-in, and redundancy detection
/// are deliberately deferred to later phases — this pager is the
/// minimum needed to assemble a budget-respecting prompt.
pub struct GreedyPager<S: BlockStore> {
    store: Arc<S>,
    tokenizer: Arc<dyn Tokenizer>,
    scoring: ScoringPolicy,
}

impl<S: BlockStore> GreedyPager<S> {
    pub fn new(store: Arc<S>, tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self {
            store,
            tokenizer,
            scoring: ScoringPolicy::default(),
        }
    }

    #[must_use]
    pub fn with_scoring(mut self, scoring: ScoringPolicy) -> Self {
        self.scoring = scoring;
        self
    }
}

impl<S: BlockStore> fmt::Debug for GreedyPager<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GreedyPager")
            .field("tokenizer", &self.tokenizer.id())
            .field("scoring", &self.scoring)
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static> Pager for GreedyPager<S> {
    #[instrument(
        skip(self, request),
        fields(session = %request.session_id, model = %request.model.name),
    )]
    fn page(&self, request: PageRequest) -> Result<PagePlan, PagerError> {
        if self.tokenizer.id() != &request.model.tokenizer {
            return Err(PagerError::TokenizerMismatch {
                pager: self.tokenizer.id().clone(),
                model: request.model.tokenizer.clone(),
            });
        }

        let budget = request.model.input_budget();

        // Step 1: required blocks (always selected, validated upfront).
        let mut selected: Vec<BlockId> = Vec::with_capacity(request.required_blocks.len());
        let mut used = TokenCount::ZERO;
        let mut required_set: HashSet<BlockId> =
            HashSet::with_capacity(request.required_blocks.len());
        for &id in &request.required_blocks {
            let block = self
                .store
                .get(id)?
                .ok_or(PagerError::RequiredBlockMissing(id))?;
            let tokens = self.tokens_for(&block)?;
            if used.saturating_add(tokens).0 > budget.0 {
                return Err(PagerError::RequiredOverBudget);
            }
            used = used.saturating_add(tokens);
            selected.push(id);
            required_set.insert(id);
        }

        // Step 2: build candidate list from the rest of the session.
        let session_ids = self.store.list_session(request.session_id)?;
        let mut candidates: Vec<Candidate> = Vec::with_capacity(session_ids.len());
        let mut min_ts = u64::MAX;
        let mut max_ts = u64::MIN;
        let mut blocks: Vec<ContextBlock> = Vec::with_capacity(session_ids.len());
        for id in session_ids {
            if required_set.contains(&id) {
                continue;
            }
            if let Some(block) = self.store.get(id)? {
                let ts = block.id.timestamp_ms();
                min_ts = min_ts.min(ts);
                max_ts = max_ts.max(ts);
                blocks.push(block);
            }
        }
        for block in blocks {
            let tokens = self.tokens_for(&block)?;
            let score = self.score_for(&block, min_ts, max_ts);
            candidates.push(Candidate {
                id: block.id,
                tokens,
                score,
            });
        }

        // Sort by score-per-token descending; stable tiebreakers so
        // the same input always produces the same plan.
        candidates.sort_by(|a, b| {
            let a_eff = score_per_token(a);
            let b_eff = score_per_token(b);
            b_eff
                .partial_cmp(&a_eff)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal))
                .then_with(|| b.id.cmp(&a.id))
        });

        // Step 3: greedy fill.
        let mut omitted: Vec<OmittedBlock> = Vec::new();
        for cand in candidates {
            if used.saturating_add(cand.tokens).0 <= budget.0 {
                used = used.saturating_add(cand.tokens);
                selected.push(cand.id);
            } else {
                omitted.push(OmittedBlock {
                    block_id: cand.id,
                    reason: OmissionReason::Budget,
                    score: cand.score,
                });
            }
        }

        Ok(PagePlan {
            selected,
            omitted,
            estimated_tokens: used,
        })
    }
}

impl<S: BlockStore> GreedyPager<S> {
    fn tokens_for(&self, block: &ContextBlock) -> Result<TokenCount, PagerError> {
        // Prefer the precomputed count for this tokenizer if present;
        // fall back to live tokenization on the bytes.
        if let Some(n) = block.token_counts.get(self.tokenizer.id()) {
            return Ok(n);
        }
        Ok(self.tokenizer.count(&block.bytes)?)
    }

    #[allow(clippy::cast_precision_loss)]
    fn score_for(&self, block: &ContextBlock, min_ts: u64, max_ts: u64) -> f32 {
        let span = max_ts.saturating_sub(min_ts).max(1) as f32;
        let recency = (block.id.timestamp_ms().saturating_sub(min_ts) as f32) / span;
        let priority = block.priority.clamp(0.0, 1.0);
        self.scoring.recency_weight * recency + self.scoring.priority_weight * priority
    }
}

#[allow(clippy::cast_precision_loss)]
fn score_per_token(c: &Candidate) -> f32 {
    c.score / (c.tokens.0.max(1) as f32)
}

struct Candidate {
    id: BlockId,
    tokens: TokenCount,
    score: f32,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm386_core::{
        BlockKind, ContentHash, ContextBlock, ModelProfile, PageRequest, Provenance, SessionId,
        Timestamp, TokenCounts, Tokenizer, TokenizerId,
    };
    use llm386_store_lmdb::{LmdbStore, StoreConfig};
    use llm386_tokenizer::cl100k_base;
    use tempfile::TempDir;

    use super::*;

    fn setup() -> (Arc<LmdbStore>, TempDir, Arc<dyn Tokenizer>) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LmdbStore::open(dir.path(), StoreConfig::default()).unwrap());
        let tok: Arc<dyn Tokenizer> = Arc::new(cl100k_base().unwrap());
        (store, dir, tok)
    }

    fn block(bytes: &[u8], kind: BlockKind, ts_ms: u64, rnd: u128) -> ContextBlock {
        ContextBlock {
            id: BlockId::from_parts(ts_ms, rnd),
            kind,
            bytes: bytes.to_vec(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(ts_ms),
            updated_at: Timestamp(ts_ms),
            provenance: Provenance::default(),
            hash: ContentHash::of(bytes),
        }
    }

    fn profile(max: u32, reserved: u32) -> ModelProfile {
        ModelProfile {
            name: "test".into(),
            max_context_tokens: max,
            reserved_output_tokens: reserved,
            safety_margin_tokens: 0,
            tokenizer: TokenizerId::new("cl100k_base"),
            supports_system_role: true,
            supports_tools: true,
        }
    }

    #[test]
    fn empty_session_returns_empty_plan() {
        let (store, _dir, tok) = setup();
        let pager = GreedyPager::new(store, tok);
        let plan = pager
            .page(PageRequest {
                session_id: SessionId(1),
                task: "anything".into(),
                model: profile(1_000, 100),
                required_blocks: vec![],
            })
            .unwrap();
        assert!(plan.selected.is_empty());
        assert!(plan.omitted.is_empty());
        assert_eq!(plan.estimated_tokens, TokenCount::ZERO);
    }

    #[test]
    fn required_block_missing_returns_error() {
        let (store, _dir, tok) = setup();
        let pager = GreedyPager::new(store, tok);
        let bogus = BlockId::from_parts(1, 1);
        let err = pager
            .page(PageRequest {
                session_id: SessionId(1),
                task: "x".into(),
                model: profile(1_000, 100),
                required_blocks: vec![bogus],
            })
            .unwrap_err();
        assert!(matches!(err, PagerError::RequiredBlockMissing(id) if id == bogus));
    }

    #[test]
    fn fills_session_blocks_within_budget() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let id_a = store
            .put(session, block(b"alpha", BlockKind::UserMessage, 1, 1))
            .unwrap();
        let id_b = store
            .put(session, block(b"beta", BlockKind::UserMessage, 2, 2))
            .unwrap();
        let id_c = store
            .put(session, block(b"gamma", BlockKind::UserMessage, 3, 3))
            .unwrap();
        let pager = GreedyPager::new(store, tok);
        let plan = pager
            .page(PageRequest {
                session_id: session,
                task: "fill".into(),
                model: profile(1_000, 100),
                required_blocks: vec![],
            })
            .unwrap();
        assert_eq!(plan.selected.len(), 3);
        let mut sorted = plan.selected.clone();
        sorted.sort();
        let mut expected = vec![id_a, id_b, id_c];
        expected.sort();
        assert_eq!(sorted, expected);
        assert!(plan.omitted.is_empty());
    }

    #[test]
    fn never_exceeds_budget() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        for i in 0..50_u64 {
            let bytes = format!("block number {i}");
            store
                .put(
                    session,
                    block(bytes.as_bytes(), BlockKind::UserMessage, i, u128::from(i)),
                )
                .unwrap();
        }
        // Tiny budget — most blocks should be omitted.
        let pager = GreedyPager::new(store, tok);
        let plan = pager
            .page(PageRequest {
                session_id: session,
                task: "x".into(),
                model: profile(20, 0),
                required_blocks: vec![],
            })
            .unwrap();
        assert!(plan.estimated_tokens.0 <= 20);
        assert!(!plan.omitted.is_empty(), "expected some omissions");
    }

    #[test]
    fn required_blocks_count_against_budget() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let big = "lorem ipsum ".repeat(200);
        let huge_id = store
            .put(session, block(big.as_bytes(), BlockKind::UserMessage, 1, 1))
            .unwrap();
        let pager = GreedyPager::new(store, tok);
        // Budget too small to hold the required block.
        let err = pager
            .page(PageRequest {
                session_id: session,
                task: "x".into(),
                model: profile(5, 0),
                required_blocks: vec![huge_id],
            })
            .unwrap_err();
        assert!(matches!(err, PagerError::RequiredOverBudget));
    }

    #[test]
    fn tokenizer_mismatch_is_caught() {
        let (store, _dir, tok) = setup();
        let pager = GreedyPager::new(store, tok);
        let bad_profile = ModelProfile {
            tokenizer: TokenizerId::new("o200k_base"),
            ..profile(1_000, 100)
        };
        let err = pager
            .page(PageRequest {
                session_id: SessionId(1),
                task: "x".into(),
                model: bad_profile,
                required_blocks: vec![],
            })
            .unwrap_err();
        assert!(matches!(err, PagerError::TokenizerMismatch { .. }));
    }

    #[test]
    fn precomputed_token_count_is_used_when_present() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let mut b = block(b"hello world", BlockKind::UserMessage, 1, 1);
        // Lie about the count: claim 999 tokens to force an overrun
        // we can detect.
        b.token_counts
            .insert(TokenizerId::new("cl100k_base"), TokenCount(999));
        let id = store.put(session, b).unwrap();
        let pager = GreedyPager::new(store, tok);
        // Budget = 100 < 999, so the block should be omitted.
        let plan = pager
            .page(PageRequest {
                session_id: session,
                task: "x".into(),
                model: profile(100, 0),
                required_blocks: vec![],
            })
            .unwrap();
        assert!(plan.selected.is_empty());
        assert_eq!(plan.omitted.len(), 1);
        assert_eq!(plan.omitted[0].block_id, id);
    }
}
