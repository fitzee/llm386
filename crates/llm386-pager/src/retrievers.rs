//! Retriever implementations.
//!
//! Each retriever surfaces a `Vec<RetrievalCandidate>` for a given
//! session + task. The pager fans out across multiple retrievers
//! and merges their results by `BlockId` (max score wins), so
//! retrievers compose: e.g. a default `RecencyRetriever` plus an
//! opt-in `LexicalRetriever` plus an explicit `PinnedRetriever`
//! tend to give a good first cut for chat-style workloads.
//!
//! Convention: scores are in `[0.0, 1.0]`. Implementations that can
//! produce unbounded scores should clamp before returning.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use llm386_core::{BlockId, BlockStore, RetrievalCandidate, RetrievalError, Retriever, SessionId};

/// Returns every block in the session with a flat baseline score.
///
/// Useful as a "give me everything, let downstream rank" retriever
/// — the pager's per-section greedy fill still applies budgets and
/// scoring on top.
pub struct SessionRetriever<S: BlockStore> {
    store: Arc<S>,
    /// Score assigned to every returned candidate. `0.0` is fine if
    /// you want this retriever to act purely as a candidate source
    /// while delegating scoring to another retriever.
    score: f32,
}

impl<S: BlockStore> SessionRetriever<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store, score: 0.5 }
    }

    #[must_use]
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score.clamp(0.0, 1.0);
        self
    }
}

impl<S: BlockStore> fmt::Debug for SessionRetriever<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionRetriever")
            .field("score", &self.score)
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static> Retriever for SessionRetriever<S> {
    fn name(&self) -> &'static str {
        "session"
    }

    fn retrieve(
        &self,
        session: SessionId,
        _task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        let mut ids = self
            .store
            .list_session(session)
            .map_err(|e| store_err(&e))?;
        if ids.len() > limit {
            ids.truncate(limit);
        }
        Ok(ids
            .into_iter()
            .map(|id| RetrievalCandidate {
                block_id: id,
                score: self.score,
                source: "session".into(),
            })
            .collect())
    }
}

/// Scores every session block by normalized recency (most-recent
/// block → 1.0, oldest → 0.0). Uses the `BlockId`'s embedded
/// timestamp; no extra storage required.
pub struct RecencyRetriever<S: BlockStore> {
    store: Arc<S>,
}

impl<S: BlockStore> RecencyRetriever<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }
}

impl<S: BlockStore> fmt::Debug for RecencyRetriever<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecencyRetriever").finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static> Retriever for RecencyRetriever<S> {
    fn name(&self) -> &'static str {
        "recency"
    }

    #[allow(clippy::cast_precision_loss)]
    fn retrieve(
        &self,
        session: SessionId,
        _task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        let ids = self
            .store
            .list_session(session)
            .map_err(|e| store_err(&e))?;
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let mut min_ts = u64::MAX;
        let mut max_ts = u64::MIN;
        for id in &ids {
            let ts = id.timestamp_ms();
            min_ts = min_ts.min(ts);
            max_ts = max_ts.max(ts);
        }
        let span = max_ts.saturating_sub(min_ts).max(1) as f32;

        let mut cands: Vec<RetrievalCandidate> = ids
            .into_iter()
            .map(|id| {
                let recency = (id.timestamp_ms().saturating_sub(min_ts) as f32) / span;
                RetrievalCandidate {
                    block_id: id,
                    score: recency.clamp(0.0, 1.0),
                    source: "recency".into(),
                }
            })
            .collect();
        cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if cands.len() > limit {
            cands.truncate(limit);
        }
        Ok(cands)
    }
}

/// Token-overlap retriever — splits the task on whitespace, lower-
/// cases the result, drops short stop-style words, then scores each
/// block by the fraction of its words that appear in the task set.
///
/// Scope: cheap and fully in-process. For real lexical search use
/// a BM25 / FTS index when one ships.
pub struct LexicalRetriever<S: BlockStore> {
    store: Arc<S>,
    min_word_len: usize,
}

impl<S: BlockStore> LexicalRetriever<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            min_word_len: 3,
        }
    }

    /// Override the minimum token length (default 3).
    #[must_use]
    pub fn with_min_word_len(mut self, n: usize) -> Self {
        self.min_word_len = n;
        self
    }

    fn tokenize(&self, s: &str) -> HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= self.min_word_len)
            .map(str::to_lowercase)
            .collect()
    }
}

impl<S: BlockStore> fmt::Debug for LexicalRetriever<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LexicalRetriever")
            .field("min_word_len", &self.min_word_len)
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static> Retriever for LexicalRetriever<S> {
    fn name(&self) -> &'static str {
        "lexical"
    }

    #[allow(clippy::cast_precision_loss)]
    fn retrieve(
        &self,
        session: SessionId,
        task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        let task_tokens = self.tokenize(task);
        if task_tokens.is_empty() {
            return Ok(vec![]);
        }

        let ids = self
            .store
            .list_session(session)
            .map_err(|e| store_err(&e))?;
        let mut cands: Vec<RetrievalCandidate> = Vec::new();
        for id in ids {
            let Some(block) = self.store.get(id).map_err(|e| store_err(&e))? else {
                continue;
            };
            let Ok(text) = std::str::from_utf8(&block.bytes) else {
                continue;
            };
            let block_tokens = self.tokenize(text);
            if block_tokens.is_empty() {
                continue;
            }
            let overlap = task_tokens.intersection(&block_tokens).count();
            if overlap == 0 {
                continue;
            }
            let score = (overlap as f32) / (block_tokens.len() as f32);
            cands.push(RetrievalCandidate {
                block_id: id,
                score: score.clamp(0.0, 1.0),
                source: "lexical".into(),
            });
        }
        cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if cands.len() > limit {
            cands.truncate(limit);
        }
        Ok(cands)
    }
}

/// BM25-scored lexical retriever. Treats each block as a document
/// and the task string as the query, scoring documents by the
/// classic Okapi BM25 formula:
///
/// ```text
/// score(D, Q) = Σ_t∈Q  IDF(t) · ( tf(t,D) · (k1+1) )
///                              ─────────────────────
///                              tf(t,D) + k1·(1 - b + b·|D|/avgdl)
/// ```
///
/// IDF uses the smoothed form `ln((N - df + 0.5) / (df + 0.5) + 1)`
/// so it is always ≥ 0. The final candidate score is clamped into
/// `[0, 1]` by dividing by the per-call max raw score (so scales
/// stay comparable across retrievers).
///
/// Defaults: `k1 = 1.2`, `b = 0.75` — standard BM25 starting point.
pub struct Bm25Retriever<S: BlockStore> {
    store: Arc<S>,
    k1: f32,
    b: f32,
    min_word_len: usize,
}

impl<S: BlockStore> Bm25Retriever<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self {
            store,
            k1: 1.2,
            b: 0.75,
            min_word_len: 2,
        }
    }

    /// Override the BM25 term-frequency saturation parameter.
    #[must_use]
    pub fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    /// Override the BM25 length-normalization parameter.
    #[must_use]
    pub fn with_b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }

    /// Override the minimum query / document token length (default 2).
    #[must_use]
    pub fn with_min_word_len(mut self, n: usize) -> Self {
        self.min_word_len = n;
        self
    }

    fn tokenize(&self, s: &str) -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= self.min_word_len)
            .map(str::to_lowercase)
            .collect()
    }
}

impl<S: BlockStore> fmt::Debug for Bm25Retriever<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bm25Retriever")
            .field("k1", &self.k1)
            .field("b", &self.b)
            .field("min_word_len", &self.min_word_len)
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static> Retriever for Bm25Retriever<S> {
    fn name(&self) -> &'static str {
        "bm25"
    }

    #[allow(clippy::cast_precision_loss)]
    fn retrieve(
        &self,
        session: SessionId,
        task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        let query: HashSet<String> = self.tokenize(task).into_iter().collect();
        if query.is_empty() {
            return Ok(vec![]);
        }

        // Pass 1: tokenize every document, collect lengths and term
        // frequencies. We materialize all documents; for very large
        // sessions this would want a posting list, but at the scale
        // we target (thousands of blocks) this is cheap.
        let ids = self
            .store
            .list_session(session)
            .map_err(|e| store_err(&e))?;
        let mut docs: Vec<(BlockId, std::collections::HashMap<String, u32>, u32)> =
            Vec::with_capacity(ids.len());
        for id in ids {
            let Some(block) = self.store.get(id).map_err(|e| store_err(&e))? else {
                continue;
            };
            let Ok(text) = std::str::from_utf8(&block.bytes) else {
                continue;
            };
            let toks = self.tokenize(text);
            if toks.is_empty() {
                continue;
            }
            let len = u32::try_from(toks.len()).unwrap_or(u32::MAX);
            let mut tf: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
            for tok in toks {
                if query.contains(&tok) {
                    *tf.entry(tok).or_insert(0) += 1;
                }
            }
            docs.push((id, tf, len));
        }

        if docs.is_empty() {
            return Ok(vec![]);
        }
        let n = docs.len() as f32;
        let avgdl = docs
            .iter()
            .map(|(_, _, l)| f32::from(u16::try_from(*l).unwrap_or(u16::MAX)))
            .sum::<f32>()
            / n;

        // Document frequency per query term.
        let mut df: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
        for (_, tf, _) in &docs {
            for term in tf.keys() {
                *df.entry(term.as_str()).or_insert(0) += 1;
            }
        }

        // Score every document.
        let mut scored: Vec<(BlockId, f32)> = Vec::with_capacity(docs.len());
        let mut max_raw = 0.0_f32;
        for (id, tf, dl) in &docs {
            let mut score = 0.0_f32;
            for (term, freq) in tf {
                let f = *freq as f32;
                let df_t = *df.get(term.as_str()).unwrap_or(&0) as f32;
                let idf = ((n - df_t + 0.5) / (df_t + 0.5) + 1.0).ln();
                let denom = f + self.k1 * (1.0 - self.b + self.b * (*dl as f32) / avgdl);
                score += idf * (f * (self.k1 + 1.0)) / denom.max(f32::EPSILON);
            }
            if score > max_raw {
                max_raw = score;
            }
            if score > 0.0 {
                scored.push((*id, score));
            }
        }

        // Normalize to [0, 1] using the per-call max so the scores
        // mix sensibly with other retrievers in the pager merge.
        let cands: Vec<RetrievalCandidate> = if max_raw > 0.0 {
            scored
                .into_iter()
                .map(|(id, raw)| RetrievalCandidate {
                    block_id: id,
                    score: (raw / max_raw).clamp(0.0, 1.0),
                    source: "bm25".into(),
                })
                .collect()
        } else {
            vec![]
        };

        let mut cands = cands;
        cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if cands.len() > limit {
            cands.truncate(limit);
        }
        Ok(cands)
    }
}

/// Always returns a fixed list of block ids with score `1.0`.
///
/// Different from `PageRequest::required_blocks` in that pinned
/// blocks still go through normal budgeting — they may be dropped
/// if their section is full. Use `required_blocks` for must-include.
#[derive(Clone, Debug)]
pub struct PinnedRetriever {
    pinned: Vec<BlockId>,
}

impl PinnedRetriever {
    #[must_use]
    pub fn new(pinned: Vec<BlockId>) -> Self {
        Self { pinned }
    }
}

impl Retriever for PinnedRetriever {
    fn name(&self) -> &'static str {
        "pinned"
    }

    fn retrieve(
        &self,
        _session: SessionId,
        _task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        Ok(self
            .pinned
            .iter()
            .take(limit)
            .map(|&id| RetrievalCandidate {
                block_id: id,
                score: 1.0,
                source: "pinned".into(),
            })
            .collect())
    }
}

fn store_err(e: &llm386_core::StoreError) -> RetrievalError {
    RetrievalError::Failed(format!("store: {e}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm386_core::{
        BlockId, BlockKind, ContentHash, ContextBlock, Provenance, SessionId, Timestamp,
        TokenCounts,
    };
    use llm386_store_lmdb::{LmdbStore, StoreConfig};
    use tempfile::TempDir;

    use super::*;

    fn open_tmp() -> (Arc<LmdbStore>, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LmdbStore::open(dir.path(), StoreConfig::default()).unwrap());
        (store, dir)
    }

    fn put(
        store: &LmdbStore,
        session: SessionId,
        bytes: &[u8],
        ts: u64,
        kind: BlockKind,
    ) -> BlockId {
        store
            .put(
                session,
                ContextBlock {
                    id: BlockId::from_parts(ts, u128::from(ts)),
                    kind,
                    bytes: bytes.to_vec(),
                    token_counts: TokenCounts::new(),
                    priority: 0.0,
                    created_at: Timestamp(ts),
                    updated_at: Timestamp(ts),
                    provenance: Provenance::default(),
                    hash: ContentHash::of(bytes),
                },
            )
            .unwrap()
    }

    #[test]
    fn session_retriever_returns_all_session_blocks() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        put(&store, s, b"a", 1, BlockKind::Fact);
        put(&store, s, b"b", 2, BlockKind::Fact);
        let r = SessionRetriever::new(store);
        let cands = r.retrieve(s, "irrelevant", usize::MAX).unwrap();
        assert_eq!(cands.len(), 2);
        assert!(cands.iter().all(|c| c.source == "session"));
    }

    #[test]
    fn session_retriever_respects_limit() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        for i in 0..10_u64 {
            put(
                &store,
                s,
                format!("b{i}").as_bytes(),
                i + 1,
                BlockKind::Fact,
            );
        }
        let r = SessionRetriever::new(store);
        let cands = r.retrieve(s, "x", 3).unwrap();
        assert_eq!(cands.len(), 3);
    }

    #[test]
    fn recency_retriever_orders_newest_first() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        let _old = put(&store, s, b"old", 1_000, BlockKind::Fact);
        let mid = put(&store, s, b"mid", 5_000, BlockKind::Fact);
        let new = put(&store, s, b"new", 9_000, BlockKind::Fact);
        let r = RecencyRetriever::new(store);
        let cands = r.retrieve(s, "x", usize::MAX).unwrap();
        assert_eq!(cands.len(), 3);
        assert_eq!(cands[0].block_id, new);
        assert_eq!(cands[1].block_id, mid);
        assert!((cands[0].score - 1.0).abs() < f32::EPSILON);
        assert!(cands[2].score.abs() < f32::EPSILON);
    }

    #[test]
    fn lexical_retriever_matches_overlapping_words() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        let canberra = put(
            &store,
            s,
            b"canberra is the capital of australia",
            1,
            BlockKind::Fact,
        );
        let _other = put(&store, s, b"the moon is far away", 2, BlockKind::Fact);
        let r = LexicalRetriever::new(store);
        let cands = r
            .retrieve(s, "what is the capital of australia", usize::MAX)
            .unwrap();
        assert!(!cands.is_empty());
        assert_eq!(cands[0].block_id, canberra);
        assert!(cands.iter().all(|c| c.source == "lexical"));
    }

    #[test]
    fn lexical_retriever_returns_empty_for_empty_task() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        put(&store, s, b"some content", 1, BlockKind::Fact);
        let r = LexicalRetriever::new(store);
        assert!(r.retrieve(s, "", usize::MAX).unwrap().is_empty());
    }

    #[test]
    fn bm25_ranks_more_relevant_block_higher() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        let canberra = put(
            &store,
            s,
            b"canberra is the capital of australia",
            1,
            BlockKind::Fact,
        );
        let _moon = put(
            &store,
            s,
            b"the moon is a satellite of earth",
            2,
            BlockKind::Fact,
        );
        let _empty = put(&store, s, b"nothing useful here", 3, BlockKind::Fact);
        let r = Bm25Retriever::new(store);
        let cands = r
            .retrieve(s, "what is the capital of australia", usize::MAX)
            .unwrap();
        assert!(!cands.is_empty());
        // First-ranked should be the canberra block (best query overlap).
        assert_eq!(cands[0].block_id, canberra);
        // All scores must be in [0, 1] (normalized).
        assert!(cands.iter().all(|c| (0.0..=1.0).contains(&c.score)));
        assert!(cands.iter().all(|c| c.source == "bm25"));
    }

    #[test]
    fn bm25_returns_empty_for_empty_query() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        put(&store, s, b"some text", 1, BlockKind::Fact);
        let r = Bm25Retriever::new(store);
        assert!(r.retrieve(s, "", usize::MAX).unwrap().is_empty());
    }

    #[test]
    fn bm25_skips_blocks_with_no_query_overlap() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        let _miss = put(
            &store,
            s,
            b"completely unrelated text here",
            1,
            BlockKind::Fact,
        );
        let hit = put(
            &store,
            s,
            b"australia and canberra are mentioned",
            2,
            BlockKind::Fact,
        );
        let cands = Bm25Retriever::new(store)
            .retrieve(s, "australia", usize::MAX)
            .unwrap();
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].block_id, hit);
    }

    #[test]
    fn bm25_respects_limit() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        for i in 0..10_u64 {
            put(
                &store,
                s,
                format!("relevant relevant block {i}").as_bytes(),
                i + 1,
                BlockKind::Fact,
            );
        }
        let cands = Bm25Retriever::new(store)
            .retrieve(s, "relevant", 3)
            .unwrap();
        assert_eq!(cands.len(), 3);
    }

    #[test]
    fn pinned_retriever_returns_configured_ids() {
        let id_a = BlockId::from_parts(1, 1);
        let id_b = BlockId::from_parts(2, 2);
        let r = PinnedRetriever::new(vec![id_a, id_b]);
        let cands = r.retrieve(SessionId(0), "x", usize::MAX).unwrap();
        assert_eq!(cands.len(), 2);
        assert!(cands.iter().all(|c| (c.score - 1.0).abs() < f32::EPSILON));
        assert!(cands.iter().all(|c| c.source == "pinned"));
    }
}
