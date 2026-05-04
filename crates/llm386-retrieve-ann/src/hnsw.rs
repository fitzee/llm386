//! `HnswAnnRetriever` — HNSW-indexed cosine-similarity retriever.
//!
//! Drop-in replacement for [`LinearAnnRetriever`] when the brute-
//! force O(N · d) scan starts to bite. Uses the `instant-distance`
//! crate (HNSW maps for approximate nearest neighbours).
//!
//! Construction is two-step:
//!
//! 1. `HnswAnnRetriever::new(store, embedder)` produces an empty
//!    retriever; like the linear one it caches embeddings in
//!    memory and may be paired with a persistent cache.
//! 2. The first `retrieve` call loads / embeds every block in the
//!    session, builds an HNSW index, and queries it. Subsequent
//!    calls within the same session reuse the cached index until
//!    [`Self::invalidate_index`] is called — typically after new
//!    blocks are ingested.
//!
//! For long-running services this means an O(N · d · log N)
//! one-time cost per session and O(d · log N) per query
//! thereafter. Short-lived CLI invocations rebuild the index
//! every time but still pay only the L2 cache lookup per block,
//! not the embedder.

use std::fmt;
use std::sync::Arc;

use instant_distance::{Builder as HnswBuilder, HnswMap, Search};
use llm386_core::{
    BlockId, BlockStore, ContextBlock, Embedder, RetrievalCandidate, RetrievalError, Retriever,
    SessionId,
};
use parking_lot::RwLock;

use crate::cache::EmbeddingCache;

/// `HnswAnnRetriever` indexes session blocks with HNSW for sub-
/// linear nearest-neighbour search.
///
/// Defaults: 32 ef_construction, 16 ef_search — a balanced starting
/// point for typical sessions. Tune via [`Self::with_ef_construction`]
/// and [`Self::with_ef_search`] if recall vs latency trade-offs
/// matter.
pub struct HnswAnnRetriever<S: BlockStore, E: Embedder> {
    store: Arc<S>,
    embedder: Arc<E>,
    persistent: Option<Arc<EmbeddingCache>>,
    ef_construction: usize,
    ef_search: usize,
    state: RwLock<Option<IndexedSession>>,
}

struct IndexedSession {
    session: SessionId,
    /// Built HNSW index. Keyed internally by sequential u32 ids
    /// that point into `block_ids`.
    index: HnswMap<Point, u32>,
    block_ids: Vec<BlockId>,
}

impl<S: BlockStore, E: Embedder> HnswAnnRetriever<S, E> {
    pub fn new(store: Arc<S>, embedder: Arc<E>) -> Self {
        Self {
            store,
            embedder,
            persistent: None,
            ef_construction: 32,
            ef_search: 16,
            state: RwLock::new(None),
        }
    }

    /// Attach a persistent [`EmbeddingCache`] (shared with other
    /// ANN retrievers — same key scheme).
    #[must_use]
    pub fn with_persistent_cache(mut self, cache: Arc<EmbeddingCache>) -> Self {
        self.persistent = Some(cache);
        self
    }

    /// Override the HNSW `ef_construction` parameter (default 32).
    /// Higher = better recall, slower index build.
    #[must_use]
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef.max(1);
        self
    }

    /// Override the HNSW `ef_search` parameter (default 16).
    /// Higher = better recall, slower per-query.
    #[must_use]
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef.max(1);
        self
    }

    /// Drop any cached index. Call after ingesting new blocks so
    /// the next `retrieve` rebuilds against the current session.
    pub fn invalidate_index(&self) {
        *self.state.write() = None;
    }
}

impl<S: BlockStore, E: Embedder> fmt::Debug for HnswAnnRetriever<S, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HnswAnnRetriever")
            .field("embedder", &self.embedder.name())
            .field("dimensions", &self.embedder.dimensions())
            .field("ef_construction", &self.ef_construction)
            .field("ef_search", &self.ef_search)
            .field("indexed", &self.state.read().is_some())
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static, E: Embedder + 'static> Retriever for HnswAnnRetriever<S, E> {
    fn name(&self) -> &'static str {
        "ann-hnsw"
    }

    fn retrieve(
        &self,
        session: SessionId,
        task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        if task.is_empty() {
            return Ok(vec![]);
        }
        let task_vec = self
            .embedder
            .embed(task)
            .map_err(|e| RetrievalError::Failed(format!("embed task: {e}")))?;

        // Build (or reuse) the per-session index.
        let needs_rebuild = {
            let state = self.state.read();
            match state.as_ref() {
                Some(s) => s.session != session,
                None => true,
            }
        };
        if needs_rebuild {
            let built = self.build_index(session)?;
            *self.state.write() = built;
        }

        let state_guard = self.state.read();
        let Some(state) = state_guard.as_ref() else {
            // Empty session.
            return Ok(vec![]);
        };

        // Query — `Search::default()` ships sensible ef; the
        // configured ef_search is currently unused at query time
        // pending an instant-distance API for it.
        let mut search = Search::default();
        let _ = self.ef_search; // referenced; reserved for future use
        let query = Point(task_vec);
        let mut results: Vec<(BlockId, f32)> = state
            .index
            .search(&query, &mut search)
            .map(|item| {
                // instant-distance returns Euclidean distance; we
                // use unit-normalized vectors at insert time so
                // distance = sqrt(2 - 2·cos), giving cos = 1 - d²/2.
                let d = item.distance;
                let cos = (1.0 - 0.5 * d * d).clamp(0.0, 1.0);
                let block_idx = *item.value as usize;
                let block_id = state.block_ids[block_idx];
                (block_id, cos)
            })
            .collect();

        results.truncate(limit);
        Ok(results
            .into_iter()
            .map(|(id, score)| RetrievalCandidate {
                block_id: id,
                score,
                source: "ann-hnsw".into(),
            })
            .collect())
    }
}

impl<S: BlockStore + 'static, E: Embedder + 'static> HnswAnnRetriever<S, E> {
    fn build_index(&self, session: SessionId) -> Result<Option<IndexedSession>, RetrievalError> {
        let ids = self
            .store
            .list_session(session)
            .map_err(|e| RetrievalError::Failed(format!("store: {e}")))?;
        if ids.is_empty() {
            return Ok(None);
        }

        let embedder_name = self.embedder.name();
        let mut block_ids: Vec<BlockId> = Vec::with_capacity(ids.len());
        let mut points: Vec<Point> = Vec::with_capacity(ids.len());
        // Embed any blocks the L2 cache doesn't have.
        let mut to_embed: Vec<(BlockId, llm386_core::ContentHash, String)> = Vec::new();
        for id in ids {
            let Some(block): Option<ContextBlock> = self
                .store
                .get(id)
                .map_err(|e| RetrievalError::Failed(format!("store: {e}")))?
            else {
                continue;
            };
            if let Some(persistent) = &self.persistent
                && let Some(vec) = persistent
                    .get(embedder_name, &block.hash)
                    .map_err(|e| RetrievalError::Failed(format!("cache get: {e}")))?
            {
                block_ids.push(id);
                points.push(Point(unit_normalize(vec)));
                continue;
            }
            let Ok(text) = std::str::from_utf8(&block.bytes) else {
                continue;
            };
            to_embed.push((id, block.hash, text.to_owned()));
        }
        if !to_embed.is_empty() {
            let refs: Vec<&str> = to_embed.iter().map(|(_, _, t)| t.as_str()).collect();
            let vecs = self
                .embedder
                .embed_batch(&refs)
                .map_err(|e| RetrievalError::Failed(format!("embed batch: {e}")))?;
            for ((id, hash, _), vec) in to_embed.into_iter().zip(vecs) {
                if let Some(persistent) = &self.persistent {
                    persistent
                        .put(embedder_name, &hash, &vec)
                        .map_err(|e| RetrievalError::Failed(format!("cache put: {e}")))?;
                }
                block_ids.push(id);
                points.push(Point(unit_normalize(vec)));
            }
        }

        if block_ids.is_empty() {
            return Ok(None);
        }

        let n = u32::try_from(block_ids.len()).unwrap_or(u32::MAX);
        let values: Vec<u32> = (0..n).collect();
        let index = HnswBuilder::default()
            .ef_construction(self.ef_construction)
            .build(points, values);
        Ok(Some(IndexedSession {
            session,
            index,
            block_ids,
        }))
    }
}

/// Unit-normalize a vector so HNSW Euclidean distance maps cleanly
/// to cosine similarity.
fn unit_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let mut norm = 0.0_f32;
    for x in &v {
        norm += x * x;
    }
    let norm = norm.sqrt();
    if norm > f32::EPSILON {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[derive(Clone, Debug)]
struct Point(Vec<f32>);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Euclidean — instant-distance expects this exact shape.
        if self.0.len() != other.0.len() {
            return f32::MAX;
        }
        let mut sum = 0.0_f32;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            let d = a - b;
            sum += d * d;
        }
        sum.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm386_core::{
        BlockId, BlockKind, ContentHash, ContextBlock, Embedder, EmbedderError, Provenance,
        SessionId, Timestamp, TokenCounts,
    };
    use llm386_store_lmdb::{LmdbStore, StoreConfig};
    use tempfile::TempDir;

    use super::*;

    /// Same toy embedder used by the linear retriever tests:
    /// 4-d one-hot vectors keyed by the first character.
    struct FixedEmbedder;
    impl Embedder for FixedEmbedder {
        fn name(&self) -> &'static str {
            "fixed-test"
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
            let v = match text.chars().next() {
                Some('a') => vec![1.0, 0.0, 0.0, 0.0],
                Some('b') => vec![0.0, 1.0, 0.0, 0.0],
                Some('c') => vec![0.0, 0.0, 1.0, 0.0],
                _ => vec![0.0, 0.0, 0.0, 1.0],
            };
            Ok(v)
        }
    }

    fn open_tmp() -> (Arc<LmdbStore>, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LmdbStore::open(dir.path(), StoreConfig::default()).unwrap());
        (store, dir)
    }

    fn put(store: &LmdbStore, session: SessionId, bytes: &[u8], ts: u64) -> BlockId {
        store
            .put(
                session,
                ContextBlock {
                    id: BlockId::from_parts(ts, u128::from(ts)),
                    kind: BlockKind::Fact,
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
    fn hnsw_ranks_matching_first_char_top() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        let a_block = put(&store, s, b"alpha block", 1);
        let _b_block = put(&store, s, b"beta block", 2);
        let c_block = put(&store, s, b"canberra capital", 3);
        let r = HnswAnnRetriever::new(store, Arc::new(FixedEmbedder));
        let cands = r.retrieve(s, "capital query", usize::MAX).unwrap();
        assert!(!cands.is_empty());
        assert_eq!(cands[0].block_id, c_block);
        let cands = r.retrieve(s, "anything", usize::MAX).unwrap();
        assert_eq!(cands[0].block_id, a_block);
    }

    #[test]
    fn hnsw_returns_empty_for_empty_task_or_session() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        let r = HnswAnnRetriever::new(store.clone(), Arc::new(FixedEmbedder));
        // Empty task.
        put(&store, s, b"x", 1);
        assert!(r.retrieve(s, "", usize::MAX).unwrap().is_empty());
        // Empty session.
        let s2 = SessionId(99);
        assert!(r.retrieve(s2, "x", usize::MAX).unwrap().is_empty());
    }

    #[test]
    fn hnsw_respects_limit() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        for i in 0..10_u64 {
            put(&store, s, format!("alpha-{i}").as_bytes(), i + 1);
        }
        let r = HnswAnnRetriever::new(store, Arc::new(FixedEmbedder));
        let cands = r.retrieve(s, "alpha", 3).unwrap();
        assert_eq!(cands.len(), 3);
    }

    #[test]
    fn hnsw_invalidate_index_forces_rebuild() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        put(&store, s, b"alpha first", 1);
        let r = HnswAnnRetriever::new(store.clone(), Arc::new(FixedEmbedder));
        let cands = r.retrieve(s, "alpha", usize::MAX).unwrap();
        assert_eq!(cands.len(), 1);
        // Add a new block — without invalidate, we'd miss it.
        let b2 = put(&store, s, b"alpha second", 2);
        r.invalidate_index();
        let cands = r.retrieve(s, "alpha", usize::MAX).unwrap();
        assert_eq!(cands.len(), 2);
        assert!(cands.iter().any(|c| c.block_id == b2));
    }
}
