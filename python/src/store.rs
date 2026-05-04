//! `Store` PyO3 class — the main entry point for the Python SDK.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use llm386_compress::{NoopSummarizer, TruncatingSummarizer};
use llm386_compress_anthropic::AnthropicSummarizer;
use llm386_core::{
    BlockId, BlockKind, BlockStore, CallId, ContentHash, ContextBlock as RustBlock,
    ModelRegistry, Packer, PageRequest, Pager, Provenance, SessionId, Summarizer, Timestamp,
    TokenCounts, Tokenizer, TraceRecord as RustTraceRecord, TraceSink, default_registry,
};
use llm386_packer::SimplePacker;
use llm386_pager::GreedyPager;
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::{TokenizerRegistry, default_registry as default_tokenizers};
use llm386_trace::LmdbTraceSink;

use crate::config;
use crate::types::{ChatMessage, ContextBlock, PackResult, PagePlan, TraceRecord};

create_exception!(llm386, LLM386Error, PyException);

#[pyclass]
pub struct Store {
    inner: Arc<LmdbStore>,
    tokenizers: TokenizerRegistry,
    models: ModelRegistry,
    retriever_entries: Vec<config::RetrieverEntry>,
}

#[pymethods]
impl Store {
    /// Open (or create) an LMDB store at `path`. Idempotent.
    ///
    /// `profiles` optionally points at a TOML config file with the
    /// same schema the CLI accepts (`[[profile]]`,
    /// `[[hf_tokenizer]]`, `[[retriever]]`). User profiles override
    /// built-ins by name; user retriever entries replace the
    /// default RecencyRetriever stack.
    #[new]
    #[pyo3(signature = (path, *, profiles = None))]
    fn new(path: PathBuf, profiles: Option<PathBuf>) -> PyResult<Self> {
        let inner = Arc::new(
            LmdbStore::open(&path, StoreConfig::default())
                .map_err(|e| LLM386Error::new_err(format!("open store: {e}")))?,
        );
        let mut tokenizers = default_tokenizers()
            .map_err(|e| LLM386Error::new_err(format!("init tokenizers: {e}")))?;
        let mut models = default_registry();
        let retriever_entries = if let Some(cfg_path) = profiles {
            let parsed = config::parse(&cfg_path).map_err(LLM386Error::new_err)?;
            config::apply(parsed, &mut models, &mut tokenizers)
                .map_err(LLM386Error::new_err)?
        } else {
            Vec::new()
        };
        Ok(Self { inner, tokenizers, models, retriever_entries })
    }

    /// Insert a block. Returns the assigned BlockId (32-char hex).
    #[pyo3(signature = (session, kind, body, *, priority = 0.0))]
    fn put(
        &self,
        session: u128,
        kind: &str,
        body: &Bound<'_, PyAny>,
        priority: f32,
    ) -> PyResult<String> {
        let bytes = if let Ok(b) = body.cast::<PyBytes>() {
            b.as_bytes().to_vec()
        } else if let Ok(s) = body.cast::<PyString>() {
            s.extract::<String>()?.into_bytes()
        } else {
            return Err(PyTypeError::new_err("body must be bytes or str"));
        };
        let block_kind = parse_kind(kind)?;
        let session = SessionId(session);
        let id = new_block_id();
        let now = Timestamp(now_ms());
        let block = RustBlock {
            id,
            kind: block_kind,
            bytes: bytes.clone(),
            token_counts: TokenCounts::new(),
            priority,
            created_at: now,
            updated_at: now,
            provenance: Provenance::default(),
            hash: ContentHash::of(&bytes),
        };
        let stored = self
            .inner
            .put(session, block)
            .map_err(|e| LLM386Error::new_err(format!("put: {e}")))?;
        Ok(format!("{stored}"))
    }

    /// Fetch a block by id (32-char hex string).
    fn show(&self, block_id: &str) -> PyResult<ContextBlock> {
        let id = parse_block_id(block_id)?;
        let block = self
            .inner
            .get(id)
            .map_err(|e| LLM386Error::new_err(format!("get: {e}")))?
            .ok_or_else(|| LLM386Error::new_err(format!("block not found: {block_id}")))?;
        Ok(ContextBlock::from_rust(block))
    }

    /// Every distinct session id with at least one block.
    fn list_sessions(&self) -> PyResult<Vec<String>> {
        let sessions = self
            .inner
            .list_sessions()
            .map_err(|e| LLM386Error::new_err(format!("list_sessions: {e}")))?;
        Ok(sessions.into_iter().map(|s| format!("{s}")).collect())
    }

    /// Delete a block entirely: from the primary table, the
    /// content-hash index, and every session that referenced it.
    /// Returns True if the block existed.
    fn delete(&self, block_id: &str) -> PyResult<bool> {
        let id = parse_block_id(block_id)?;
        self.inner.delete(id).map_err(|e| LLM386Error::new_err(format!("delete: {e}")))
    }

    /// Remove every block belonging to `session`. Returns the count
    /// of blocks affected. Blocks still referenced by other
    /// sessions are kept; ones with no remaining references are
    /// removed entirely (including from the content-hash index).
    fn purge_session(&self, session: u128) -> PyResult<usize> {
        self.inner
            .purge_session(SessionId(session))
            .map_err(|e| LLM386Error::new_err(format!("purge_session: {e}")))
    }

    /// Run the pager and return the resulting plan.
    fn page(&self, session: u128, model: &str, task: &str) -> PyResult<PagePlan> {
        let (profile, tokenizer) = self.profile_and_tokenizer(model)?;
        let mut pager = GreedyPager::new(self.inner.clone(), tokenizer);
        if let Some(retrievers) = config::build_retrievers(&self.retriever_entries, &self.inner)
            .map_err(LLM386Error::new_err)?
        {
            pager = pager.with_retrievers(retrievers);
        }
        let request = PageRequest {
            session_id: SessionId(session),
            task: task.to_string(),
            model: profile,
            required_blocks: vec![],
        };
        let plan = pager.page(request).map_err(|e| LLM386Error::new_err(format!("page: {e}")))?;
        Ok(PagePlan::from_rust(plan))
    }

    /// Run page+pack and return either a rendered prompt or a list
    /// of role-tagged chat messages, optionally recording a trace.
    #[pyo3(signature = (session, model, task, *, chat = false, trace = None))]
    fn pack(
        &self,
        session: u128,
        model: &str,
        task: &str,
        chat: bool,
        trace: Option<PathBuf>,
    ) -> PyResult<PackResult> {
        let (profile, tokenizer) = self.profile_and_tokenizer(model)?;
        let mut pager = GreedyPager::new(self.inner.clone(), tokenizer.clone());
        if let Some(retrievers) = config::build_retrievers(&self.retriever_entries, &self.inner)
            .map_err(LLM386Error::new_err)?
        {
            pager = pager.with_retrievers(retrievers);
        }
        let packer = SimplePacker::new(self.inner.clone(), tokenizer);
        let request = PageRequest {
            session_id: SessionId(session),
            task: task.to_string(),
            model: profile,
            required_blocks: vec![],
        };

        let started_at = Timestamp(now_ms());
        let started = Instant::now();
        let plan = pager
            .page(request.clone())
            .map_err(|e| LLM386Error::new_err(format!("page: {e}")))?;

        if chat {
            let chat_prompt = packer
                .pack_chat(&request, &plan)
                .map_err(|e| LLM386Error::new_err(format!("pack_chat: {e}")))?;
            let messages: Vec<ChatMessage> =
                chat_prompt.messages.into_iter().map(ChatMessage::from_rust).collect();
            let trace_id = if let Some(trace_path) = trace {
                Some(record_trace(
                    &trace_path,
                    SessionId(session),
                    &request.model.name,
                    &plan,
                    chat_prompt.input_tokens,
                    started_at,
                    u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
                )?)
            } else {
                None
            };
            Ok(PackResult { rendered: None, messages: Some(messages), trace_id })
        } else {
            let prompt = packer
                .pack(&request, &plan)
                .map_err(|e| LLM386Error::new_err(format!("pack: {e}")))?;
            let trace_id = if let Some(trace_path) = trace {
                Some(record_trace(
                    &trace_path,
                    SessionId(session),
                    &request.model.name,
                    &plan,
                    prompt.input_tokens,
                    started_at,
                    u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
                )?)
            } else {
                None
            };
            Ok(PackResult { rendered: Some(prompt.rendered), messages: None, trace_id })
        }
    }

    /// Summarize a session via the named summarizer.
    #[pyo3(signature = (
        session,
        *,
        summarizer = "truncating",
        max_chars = 80,
        last = None,
        store_summary = false,
        anthropic_model = None,
        anthropic_max_tokens = None,
    ))]
    fn summarize(
        &self,
        session: u128,
        summarizer: &str,
        max_chars: usize,
        last: Option<usize>,
        store_summary: bool,
        anthropic_model: Option<&str>,
        anthropic_max_tokens: Option<u32>,
    ) -> PyResult<String> {
        let session = SessionId(session);
        let mut ids = self
            .inner
            .list_session(session)
            .map_err(|e| LLM386Error::new_err(format!("list_session: {e}")))?;
        ids.sort();
        if let Some(n) = last {
            let from = ids.len().saturating_sub(n);
            ids.drain(0..from);
        }
        let mut blocks: Vec<RustBlock> = Vec::with_capacity(ids.len());
        for &id in &ids {
            if let Some(b) =
                self.inner.get(id).map_err(|e| LLM386Error::new_err(format!("get: {e}")))?
            {
                blocks.push(b);
            }
        }

        let (text, name) = match summarizer {
            "noop" => {
                let s = NoopSummarizer;
                (
                    s.summarize(&blocks)
                        .map_err(|e| LLM386Error::new_err(format!("summarize: {e}")))?,
                    s.name(),
                )
            }
            "truncating" => {
                let s = TruncatingSummarizer::new(max_chars);
                (
                    s.summarize(&blocks)
                        .map_err(|e| LLM386Error::new_err(format!("summarize: {e}")))?,
                    s.name(),
                )
            }
            "anthropic" => {
                let mut s = AnthropicSummarizer::from_env()
                    .map_err(|e| LLM386Error::new_err(format!("anthropic init: {e}")))?;
                if let Some(model) = anthropic_model {
                    s = s.with_model(model);
                }
                if let Some(n) = anthropic_max_tokens {
                    s = s.with_max_tokens(n);
                }
                (
                    s.summarize(&blocks)
                        .map_err(|e| LLM386Error::new_err(format!("summarize: {e}")))?,
                    s.name(),
                )
            }
            other => {
                return Err(LLM386Error::new_err(format!(
                    "unknown summarizer `{other}` (expected: noop | truncating | anthropic)",
                )));
            }
        };

        if store_summary {
            let bytes = text.clone().into_bytes();
            let now = Timestamp(now_ms());
            let id = new_block_id();
            let block = RustBlock {
                id,
                kind: BlockKind::Summary,
                bytes: bytes.clone(),
                token_counts: TokenCounts::new(),
                priority: 0.0,
                created_at: now,
                updated_at: now,
                provenance: Provenance {
                    source: Some(format!("summarize:{name}")),
                    parents: ids,
                    labels: vec![],
                },
                hash: ContentHash::of(&bytes),
            };
            self.inner
                .put(session, block)
                .map_err(|e| LLM386Error::new_err(format!("store summary: {e}")))?;
        }
        Ok(text)
    }
}

impl Store {
    fn profile_and_tokenizer(
        &self,
        model_name: &str,
    ) -> PyResult<(llm386_core::ModelProfile, Arc<dyn Tokenizer>)> {
        let profile = self
            .models
            .get(model_name)
            .ok_or_else(|| LLM386Error::new_err(format!("unknown model: {model_name}")))?
            .clone();
        let tokenizer = self.tokenizers.get(&profile.tokenizer).ok_or_else(|| {
            LLM386Error::new_err(format!(
                "no tokenizer adapter for {} (used by model {})",
                profile.tokenizer, profile.name,
            ))
        })?;
        Ok((profile, tokenizer))
    }
}

fn record_trace(
    path: &std::path::Path,
    session: SessionId,
    model: &str,
    plan: &llm386_core::PagePlan,
    prompt_tokens: llm386_core::TokenCount,
    started_at: Timestamp,
    duration_ms: u32,
) -> PyResult<String> {
    let sink = LmdbTraceSink::open(path)
        .map_err(|e| LLM386Error::new_err(format!("open trace: {e}")))?;
    let call_id = new_call_id();
    sink.record(RustTraceRecord {
        call_id,
        session,
        model: model.to_string(),
        plan: plan.clone(),
        prompt_tokens,
        prompt_hash: ContentHash::of(&[]),
        started_at,
        duration_ms,
    })
    .map_err(|e| LLM386Error::new_err(format!("record trace: {e}")))?;
    Ok(format!("{call_id}"))
}

/// Read-only wrapper around an LMDB trace store. Pair with
/// `Store.pack(trace="./traces")` to inspect a recorded trace.
#[pyclass]
pub struct Trace {
    sink: LmdbTraceSink,
}

#[pymethods]
impl Trace {
    /// Open (or create) a trace store at `path`.
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let sink = LmdbTraceSink::open(&path)
            .map_err(|e| LLM386Error::new_err(format!("open trace store: {e}")))?;
        Ok(Self { sink })
    }

    /// Fetch a trace record by call id.
    fn show(&self, call_id: &str) -> PyResult<TraceRecord> {
        let id = parse_call_id(call_id)?;
        let record = self
            .sink
            .fetch(id)
            .map_err(|e| LLM386Error::new_err(format!("fetch trace: {e}")))?
            .ok_or_else(|| LLM386Error::new_err(format!("no trace for {call_id}")))?;
        Ok(TraceRecord::from_rust(record))
    }
}

fn parse_call_id(s: &str) -> PyResult<CallId> {
    let n = u128::from_str_radix(s, 16)
        .map_err(|e| LLM386Error::new_err(format!("invalid call id `{s}`: {e}")))?;
    Ok(CallId(n))
}

fn parse_kind(s: &str) -> PyResult<BlockKind> {
    let kind = match s {
        "system" | "System" => BlockKind::System,
        "user-message" | "UserMessage" => BlockKind::UserMessage,
        "assistant-message" | "AssistantMessage" => BlockKind::AssistantMessage,
        "tool-result" | "ToolResult" => BlockKind::ToolResult,
        "summary" | "Summary" => BlockKind::Summary,
        "fact" | "Fact" => BlockKind::Fact,
        "document-chunk" | "DocumentChunk" => BlockKind::DocumentChunk,
        "plan" | "Plan" => BlockKind::Plan,
        "state" | "State" => BlockKind::State,
        "trace" | "Trace" => BlockKind::Trace,
        other => {
            return Err(LLM386Error::new_err(format!("unknown block kind: {other}")));
        }
    };
    Ok(kind)
}

fn parse_block_id(s: &str) -> PyResult<BlockId> {
    let n = u128::from_str_radix(s, 16)
        .map_err(|e| LLM386Error::new_err(format!("invalid block id `{s}`: {e}")))?;
    Ok(BlockId(n))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn new_block_id() -> BlockId {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("getrandom failed");
    BlockId::from_parts(now_ms(), u128::from_be_bytes(buf))
}

fn new_call_id() -> CallId {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("getrandom failed");
    CallId(u128::from_be_bytes(buf))
}
