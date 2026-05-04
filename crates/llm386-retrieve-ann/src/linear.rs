//! `LinearAnnRetriever` — brute-force cosine-similarity retriever.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use llm386_core::{BlockStore, Embedder, RetrievalCandidate, RetrievalError, Retriever, SessionId};
use parking_lot::RwLock;

/// In-memory cosine-similarity ANN retriever.
///
/// On each `retrieve` call: embeds the task, ensures every session
/// block has an embedding cached (computing on demand via the
/// [`Embedder`]), then ranks blocks by cosine similarity against
/// the task vector. Cache lives for the lifetime of the retriever
/// — long-running services keep it warm; short-lived CLI
/// invocations re-embed every time.
///
/// O(N · d) per call where N is the session size and d is the
/// embedding dimensionality. Fine for thousands of blocks; for
/// larger sessions, swap in an HNSW-indexed retriever.
pub struct LinearAnnRetriever<S: BlockStore, E: Embedder> {
    store: Arc<S>,
    embedder: Arc<E>,
    cache: RwLock<HashMap<llm386_core::BlockId, Vec<f32>>>,
}

impl<S: BlockStore, E: Embedder> LinearAnnRetriever<S, E> {
    pub fn new(store: Arc<S>, embedder: Arc<E>) -> Self {
        Self {
            store,
            embedder,
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Drop every cached embedding. Use after re-indexing or when
    /// the embedder model changes.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Number of cached embeddings.
    #[must_use]
    pub fn cached_len(&self) -> usize {
        self.cache.read().len()
    }
}

impl<S: BlockStore, E: Embedder> fmt::Debug for LinearAnnRetriever<S, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearAnnRetriever")
            .field("embedder", &self.embedder.name())
            .field("dimensions", &self.embedder.dimensions())
            .field("cached", &self.cached_len())
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static, E: Embedder + 'static> Retriever for LinearAnnRetriever<S, E> {
    fn name(&self) -> &'static str {
        "ann-linear"
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

        let ids = self
            .store
            .list_session(session)
            .map_err(|e| RetrievalError::Failed(format!("store: {e}")))?;
        if ids.is_empty() {
            return Ok(vec![]);
        }

        // Figure out which blocks need embedding.
        let needs_embed: Vec<llm386_core::BlockId> = {
            let cache = self.cache.read();
            ids.iter()
                .copied()
                .filter(|id| !cache.contains_key(id))
                .collect()
        };

        // Load + embed missing blocks in one batch call (the
        // embedder may have a more efficient batch path).
        if !needs_embed.is_empty() {
            let mut texts: Vec<String> = Vec::with_capacity(needs_embed.len());
            let mut ok_ids: Vec<llm386_core::BlockId> = Vec::with_capacity(needs_embed.len());
            for &id in &needs_embed {
                let block = self
                    .store
                    .get(id)
                    .map_err(|e| RetrievalError::Failed(format!("store: {e}")))?;
                let Some(block) = block else {
                    continue;
                };
                let Ok(text) = std::str::from_utf8(&block.bytes) else {
                    continue;
                };
                texts.push(text.to_owned());
                ok_ids.push(id);
            }
            if !texts.is_empty() {
                let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                let vecs = self
                    .embedder
                    .embed_batch(&refs)
                    .map_err(|e| RetrievalError::Failed(format!("embed batch: {e}")))?;
                let mut cache = self.cache.write();
                for (id, vec) in ok_ids.into_iter().zip(vecs) {
                    cache.insert(id, vec);
                }
            }
        }

        // Score every block we have an embedding for.
        let cache = self.cache.read();
        let mut scored: Vec<(llm386_core::BlockId, f32)> = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(vec) = cache.get(&id) {
                let sim = cosine(&task_vec, vec);
                if sim > 0.0 {
                    scored.push((id, sim));
                }
            }
        }
        drop(cache);

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.len() > limit {
            scored.truncate(limit);
        }
        Ok(scored
            .into_iter()
            .map(|(id, score)| RetrievalCandidate {
                block_id: id,
                score: score.clamp(0.0, 1.0),
                source: "ann-linear".into(),
            })
            .collect())
    }
}

#[allow(clippy::cast_precision_loss)]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }
    (dot / denom).clamp(0.0, 1.0)
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

    /// Toy embedder: returns a fixed 4-d vector picked by the first
    /// character of the input. Used to drive cosine ordering tests
    /// deterministically without any network.
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
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!(cosine(&[], &[]).abs() < f32::EPSILON);
        // Mismatched dims short-circuit to zero.
        assert!(cosine(&[1.0], &[1.0, 0.0]).abs() < f32::EPSILON);
    }

    #[test]
    fn linear_ann_ranks_matching_first_char_top() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        let a_block = put(&store, s, b"alpha block", 1);
        let b_block = put(&store, s, b"beta block", 2);
        let c_block = put(&store, s, b"canberra capital", 3);
        let r = LinearAnnRetriever::new(store, Arc::new(FixedEmbedder));

        // Task starting with 'c' should match the canberra block.
        let cands = r.retrieve(s, "capital query", usize::MAX).unwrap();
        assert!(!cands.is_empty());
        assert_eq!(cands[0].block_id, c_block);

        // Now task starting with 'a' → alpha.
        let cands = r.retrieve(s, "anything goes", usize::MAX).unwrap();
        assert_eq!(cands[0].block_id, a_block);

        // 'b' → beta.
        let cands = r.retrieve(s, "blocks here", usize::MAX).unwrap();
        assert_eq!(cands[0].block_id, b_block);
    }

    #[test]
    fn linear_ann_caches_embeddings_after_first_call() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        for i in 0..5_u64 {
            put(&store, s, format!("block-{i}").as_bytes(), i + 1);
        }
        let r = LinearAnnRetriever::new(store, Arc::new(FixedEmbedder));
        assert_eq!(r.cached_len(), 0);
        r.retrieve(s, "task", usize::MAX).unwrap();
        assert_eq!(r.cached_len(), 5);
        r.clear_cache();
        assert_eq!(r.cached_len(), 0);
    }

    #[test]
    fn linear_ann_returns_empty_for_empty_task() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        put(&store, s, b"any", 1);
        let r = LinearAnnRetriever::new(store, Arc::new(FixedEmbedder));
        assert!(r.retrieve(s, "", usize::MAX).unwrap().is_empty());
    }

    #[test]
    fn linear_ann_respects_limit() {
        let (store, _dir) = open_tmp();
        let s = SessionId(1);
        for i in 0..10_u64 {
            // All blocks start with 'a' → all match the same task vec.
            put(&store, s, format!("alpha-{i}").as_bytes(), i + 1);
        }
        let r = LinearAnnRetriever::new(store, Arc::new(FixedEmbedder));
        let cands = r.retrieve(s, "alpha query", 3).unwrap();
        assert_eq!(cands.len(), 3);
    }
}
