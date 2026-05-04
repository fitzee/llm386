//! `LmdbTraceSink` — LMDB-backed [`TraceSink`].

use std::fmt;
use std::path::Path;

use heed::types::{Bytes, Str};
use heed::{Database, Env, EnvOpenOptions};
use llm386_core::{CallId, TraceError, TraceRecord, TraceSink};
use thiserror::Error;
use tracing::{debug, instrument};

const CURRENT_SCHEMA: u32 = 1;

/// Default LMDB map size for trace storage. Smaller than the block
/// store's because traces are dense, predictable records.
const DEFAULT_MAP_SIZE: usize = 4 * 1024 * 1024 * 1024;

/// `max_dbs` budget — covers `trace_by_call` + `meta` plus headroom
/// for per-session or per-day indexes added later.
const DEFAULT_MAX_DBS: u32 = 4;

/// LMDB-backed [`TraceSink`].
///
/// Cheap to clone (clones share the underlying [`Env`]).
#[derive(Clone)]
pub struct LmdbTraceSink {
    env: Env,
    trace_by_call: Database<Bytes, Bytes>,
    #[allow(dead_code)] // reserved for future schema migrations
    meta: Database<Str, Bytes>,
}

impl fmt::Debug for LmdbTraceSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LmdbTraceSink")
            .field("schema", &CURRENT_SCHEMA)
            .finish_non_exhaustive()
    }
}

impl LmdbTraceSink {
    /// Open (or create) an LMDB env at `path` and prepare the trace
    /// databases.
    #[instrument(fields(path = %path.as_ref().display()))]
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TraceOpenError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).map_err(TraceOpenError::Io)?;

        // SAFETY: opening an LMDB env is unsafe per heed's API
        // because LMDB's mmap-based concurrency model is undefined
        // when the same env path is opened by multiple processes
        // simultaneously, or when the underlying files are mutated
        // externally. Within a single process this open is safe; we
        // document the cross-process / network-fs constraints on the
        // `LmdbTraceSink::open` API.
        #[allow(unsafe_code)]
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(DEFAULT_MAP_SIZE)
                .max_dbs(DEFAULT_MAX_DBS)
                .open(path)?
        };

        let mut wtxn = env.write_txn()?;
        let trace_by_call = env.create_database(&mut wtxn, Some("trace_by_call"))?;
        let meta: Database<Str, Bytes> = env.create_database(&mut wtxn, Some("meta"))?;

        if let Some(existing) = meta.get(&wtxn, "schema_version")? {
            let arr: [u8; 4] = existing
                .try_into()
                .map_err(|_| TraceOpenError::CorruptMeta("schema_version width".into()))?;
            let found = u32::from_be_bytes(arr);
            if found != CURRENT_SCHEMA {
                return Err(TraceOpenError::SchemaMismatch {
                    expected: CURRENT_SCHEMA,
                    found,
                });
            }
        } else {
            meta.put(&mut wtxn, "schema_version", &CURRENT_SCHEMA.to_be_bytes())?;
            debug!(schema = CURRENT_SCHEMA, "initialized fresh trace env");
        }
        wtxn.commit()?;

        Ok(Self {
            env,
            trace_by_call,
            meta,
        })
    }
}

impl TraceSink for LmdbTraceSink {
    #[instrument(skip(self, trace), fields(call_id = %trace.call_id))]
    fn record(&self, trace: TraceRecord) -> Result<(), TraceError> {
        let key = trace.call_id.0.to_be_bytes();
        let value = postcard::to_allocvec(&trace)
            .map_err(|e| TraceError::Failed(format!("postcard encode: {e}")))?;
        let mut wtxn = self.env.write_txn().map_err(|e| trace_err(&e))?;
        self.trace_by_call
            .put(&mut wtxn, &key, &value)
            .map_err(|e| trace_err(&e))?;
        wtxn.commit().map_err(|e| trace_err(&e))?;
        Ok(())
    }

    fn fetch(&self, call_id: CallId) -> Result<Option<TraceRecord>, TraceError> {
        let rtxn = self.env.read_txn().map_err(|e| trace_err(&e))?;
        let key = call_id.0.to_be_bytes();
        match self
            .trace_by_call
            .get(&rtxn, &key)
            .map_err(|e| trace_err(&e))?
        {
            Some(bytes) => {
                let trace: TraceRecord = postcard::from_bytes(bytes)
                    .map_err(|e| TraceError::Failed(format!("postcard decode: {e}")))?;
                Ok(Some(trace))
            }
            None => Ok(None),
        }
    }
}

/// Errors that can occur while opening an [`LmdbTraceSink`].
#[derive(Debug, Error)]
pub enum TraceOpenError {
    #[error("io error: {0}")]
    Io(#[source] std::io::Error),
    #[error("LMDB error: {0}")]
    Lmdb(#[from] heed::Error),
    #[error("on-disk trace schema version {found} does not match expected {expected}")]
    SchemaMismatch { expected: u32, found: u32 },
    #[error("trace meta table is corrupt: {0}")]
    CorruptMeta(String),
}

fn trace_err(e: &heed::Error) -> TraceError {
    TraceError::Failed(format!("LMDB: {e}"))
}

#[cfg(test)]
mod tests {
    use llm386_core::{
        ContentHash, PagePlan, SessionId, Timestamp, TokenCount, TraceRecord, TraceSink,
    };
    use tempfile::TempDir;

    use super::*;

    fn fake_record(call_id: u128) -> TraceRecord {
        TraceRecord {
            call_id: CallId(call_id),
            session: SessionId(1),
            model: "gpt-4o".into(),
            plan: PagePlan {
                selected: vec![],
                selections: vec![],
                omitted: vec![],
                estimated_tokens: TokenCount(0),
            },
            prompt_tokens: TokenCount(0),
            prompt_hash: ContentHash::of(b""),
            started_at: Timestamp(1_000),
            duration_ms: 0,
            model_version: "gpt-4o-2024-08-06".into(),
            tokenizer_version: "o200k_base".into(),
            output: None,
            output_tokens: None,
        }
    }

    #[test]
    fn record_and_fetch_roundtrip() {
        let dir = TempDir::new().unwrap();
        let sink = LmdbTraceSink::open(dir.path()).unwrap();
        let rec = fake_record(42);
        sink.record(rec.clone()).unwrap();
        let fetched = sink.fetch(rec.call_id).unwrap().unwrap();
        assert_eq!(fetched.call_id, rec.call_id);
        assert_eq!(fetched.model, rec.model);
    }

    #[test]
    fn fetch_unknown_returns_none() {
        let dir = TempDir::new().unwrap();
        let sink = LmdbTraceSink::open(dir.path()).unwrap();
        assert!(sink.fetch(CallId(999)).unwrap().is_none());
    }

    #[test]
    fn reopen_preserves_records() {
        let dir = TempDir::new().unwrap();
        let id = {
            let sink = LmdbTraceSink::open(dir.path()).unwrap();
            let rec = fake_record(7);
            sink.record(rec.clone()).unwrap();
            rec.call_id
        };
        let sink = LmdbTraceSink::open(dir.path()).unwrap();
        assert!(sink.fetch(id).unwrap().is_some());
    }

    #[test]
    fn update_output_patches_in_model_response() {
        let dir = TempDir::new().unwrap();
        let sink = LmdbTraceSink::open(dir.path()).unwrap();
        let rec = fake_record(11);
        sink.record(rec.clone()).unwrap();
        sink.update_output(rec.call_id, "the answer is 42".into(), TokenCount(5))
            .unwrap();
        let patched = sink.fetch(rec.call_id).unwrap().unwrap();
        assert_eq!(patched.output.as_deref(), Some("the answer is 42"));
        assert_eq!(patched.output_tokens, Some(TokenCount(5)));
        // Untouched fields stay intact.
        assert_eq!(patched.model, rec.model);
        assert_eq!(patched.model_version, rec.model_version);
    }

    #[test]
    fn update_output_on_unknown_call_errors() {
        let dir = TempDir::new().unwrap();
        let sink = LmdbTraceSink::open(dir.path()).unwrap();
        let err = sink
            .update_output(CallId(404), "x".into(), TokenCount(1))
            .unwrap_err();
        assert!(matches!(err, llm386_core::TraceError::Failed(_)));
    }

    #[test]
    fn distinct_records_dont_collide() {
        let dir = TempDir::new().unwrap();
        let sink = LmdbTraceSink::open(dir.path()).unwrap();
        sink.record(fake_record(1)).unwrap();
        sink.record(fake_record(2)).unwrap();
        sink.record(fake_record(3)).unwrap();
        assert_eq!(sink.fetch(CallId(1)).unwrap().unwrap().call_id, CallId(1));
        assert_eq!(sink.fetch(CallId(2)).unwrap().unwrap().call_id, CallId(2));
        assert_eq!(sink.fetch(CallId(3)).unwrap().unwrap().call_id, CallId(3));
    }
}
