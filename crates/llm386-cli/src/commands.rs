//! Subcommand handlers for `llm386`.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use llm386_compress::{NoopSummarizer, TruncatingSummarizer};
use llm386_compress_anthropic::AnthropicSummarizer;
use llm386_core::{
    BlockId, BlockKind, BlockStore, CallId, ContentHash, ContextBlock, ModelProfile, ModelRegistry,
    Packer, PageRequest, Pager, Provenance, Retriever, SessionId, Summarizer, Timestamp,
    TokenCounts, Tokenizer, TokenizerId, TraceRecord, TraceSink, default_registry,
};
use llm386_packer::SimplePacker;
use llm386_pager::{
    Bm25Retriever, GreedyPager, LexicalRetriever, RecencyRetriever, SessionRetriever,
};
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::{HfTokenizer, TokenizerRegistry, default_registry as tokenizer_registry};
use llm386_trace::LmdbTraceSink;
use serde::Deserialize;

use crate::cli::{Command, SummarizerArg, TraceSub};

const PROFILES_ENV: &str = "LLM386_PROFILES";

/// Bundle of registries the CLI hands off to every subcommand
/// handler. Built once at startup from defaults + (optional) user
/// config file. Retrievers can't be pre-built because they hold a
/// store reference — the CLI rebuilds them per-command from
/// `retriever_entries`.
pub(crate) struct LoadedConfig {
    pub models: ModelRegistry,
    pub tokenizers: TokenizerRegistry,
    pub retriever_entries: Vec<RetrieverEntry>,
    pub section_budgets: Option<llm386_pager::SectionBudgetTable>,
    pub packer_options: llm386_packer::PackerOptions,
}

/// Load the built-in registries, then merge in any user-supplied
/// `[[profile]]` and `[[hf_tokenizer]]` entries from
/// `--profiles <path>` (or, if absent, the `LLM386_PROFILES` env
/// var). User entries override built-ins with the same name.
pub(crate) fn load_config(flag_path: Option<&Path>) -> Result<LoadedConfig> {
    let mut models = default_registry();
    let mut tokenizers = tokenizer_registry().context("initializing default tokenizer registry")?;
    let mut retriever_entries: Vec<RetrieverEntry> = Vec::new();
    let mut section_budgets: Option<llm386_pager::SectionBudgetTable> = None;
    let mut packer_options = llm386_packer::PackerOptions::default();

    let path = flag_path
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os(PROFILES_ENV).map(std::path::PathBuf::from));
    if let Some(path) = path {
        let toml_str = std::fs::read_to_string(&path)
            .with_context(|| format!("reading config file at {}", path.display()))?;
        let parsed = parse_config_toml(&toml_str)
            .with_context(|| format!("parsing config file at {}", path.display()))?;
        for profile in parsed.profiles {
            models.register(profile);
        }
        for entry in parsed.hf_tokenizers {
            let tok = HfTokenizer::from_file(&entry.path, TokenizerId::new(&entry.name))
                .with_context(|| {
                    format!(
                        "loading huggingface tokenizer `{}` from {}",
                        entry.name,
                        entry.path.display(),
                    )
                })?;
            tokenizers.register(Arc::new(tok));
        }
        retriever_entries = parsed.retrievers;
        section_budgets = parsed.section_budgets;
        if let Some(p) = parsed.packer {
            packer_options = p;
        }
        if let Some(c) = parsed.cache {
            packer_options.cache = c;
        }
    }

    Ok(LoadedConfig {
        models,
        tokenizers,
        retriever_entries,
        section_budgets,
        packer_options,
    })
}

#[derive(Default)]
struct ParsedConfig {
    profiles: Vec<ModelProfile>,
    hf_tokenizers: Vec<HfTokenizerEntry>,
    retrievers: Vec<RetrieverEntry>,
    section_budgets: Option<llm386_pager::SectionBudgetTable>,
    packer: Option<llm386_packer::PackerOptions>,
    cache: Option<llm386_packer::CacheOptions>,
}

#[derive(Deserialize)]
struct ConfigFile {
    #[serde(default)]
    profile: Vec<ModelProfile>,
    #[serde(default)]
    hf_tokenizer: Vec<HfTokenizerEntry>,
    #[serde(default)]
    retriever: Vec<RetrieverEntry>,
    #[serde(default)]
    section_budgets: Option<SectionBudgetEntry>,
    #[serde(default)]
    packer: Option<PackerEntry>,
    #[serde(default)]
    cache: Option<CacheEntry>,
}

/// `[packer]` table — opt-in packer behavior knobs. Mirrors
/// [`llm386_packer::PackerOptions`].
#[derive(Default, Deserialize)]
struct PackerEntry {
    /// When `true`, prepend each rendered block with its `created_at`
    /// timestamp in ISO 8601 UTC, and emit a `Current time:` anchor
    /// at the start of the Task section.
    #[serde(default)]
    include_timestamps: bool,
}

impl PackerEntry {
    fn build(self) -> llm386_packer::PackerOptions {
        llm386_packer::PackerOptions {
            include_timestamps: self.include_timestamps,
            now_ms: None,
            cache: llm386_packer::CacheOptions::default(),
        }
    }
}

/// `[cache]` table — prompt-cache knobs for `pack_chat`.
///
/// `stable_sections` lists which sections are considered stable
/// across turns. Section names are lowercased SectionKind variants:
/// `"system"`, `"state"`, `"plan"`, `"retrieved"`, `"background"`.
/// Sections outside this list are emitted after the stable prefix
/// so they don't break the cache key.
#[derive(Default, Deserialize)]
struct CacheEntry {
    #[serde(default)]
    stable_sections: Option<Vec<String>>,
}

impl CacheEntry {
    fn build(self) -> Result<llm386_packer::CacheOptions> {
        use llm386_core::SectionKind;
        let mut out = llm386_packer::CacheOptions::default();
        if let Some(names) = self.stable_sections {
            let mut sections = Vec::with_capacity(names.len());
            for name in names {
                let s = match name.to_ascii_lowercase().as_str() {
                    "system" => SectionKind::System,
                    "state" => SectionKind::State,
                    "plan" => SectionKind::Plan,
                    "retrieved" => SectionKind::Retrieved,
                    "background" => SectionKind::Background,
                    other => {
                        return Err(anyhow!(
                            "[cache].stable_sections: unsupported section `{other}` — valid: system, state, plan, retrieved, background",
                        ));
                    }
                };
                sections.push(s);
            }
            out.stable_sections = sections;
        }
        Ok(out)
    }
}

/// `[section_budgets]` table — fractions of the variable budget per
/// section. Any field omitted defaults to 0.0 and that section gets no
/// allocation. Sums above 1.0 are normalized down at allocation time.
#[derive(Default, Deserialize)]
struct SectionBudgetEntry {
    #[serde(default)]
    state: Option<f32>,
    #[serde(default)]
    plan: Option<f32>,
    #[serde(default)]
    recent: Option<f32>,
    #[serde(default)]
    retrieved: Option<f32>,
    #[serde(default)]
    tools: Option<f32>,
    #[serde(default)]
    background: Option<f32>,
    #[serde(default)]
    slack: Option<f32>,
}

impl SectionBudgetEntry {
    fn build(self) -> llm386_pager::SectionBudgetTable {
        use llm386_core::SectionKind;
        let mut table = llm386_pager::SectionBudgetTable::empty();
        if let Some(v) = self.state {
            table.set(SectionKind::State, v);
        }
        if let Some(v) = self.plan {
            table.set(SectionKind::Plan, v);
        }
        if let Some(v) = self.recent {
            table.set(SectionKind::Recent, v);
        }
        if let Some(v) = self.retrieved {
            table.set(SectionKind::Retrieved, v);
        }
        if let Some(v) = self.tools {
            table.set(SectionKind::Tools, v);
        }
        if let Some(v) = self.background {
            table.set(SectionKind::Background, v);
        }
        if let Some(v) = self.slack {
            table.set(SectionKind::Slack, v);
        }
        table
    }
}

#[derive(Deserialize)]
struct HfTokenizerEntry {
    name: String,
    path: std::path::PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct RetrieverEntry {
    pub kind: String,
    /// RecencyRetriever: switch to exponential decay if set.
    #[serde(default)]
    pub half_life_secs: Option<f32>,
    /// LexicalRetriever / Bm25Retriever: minimum query/document
    /// token length.
    #[serde(default)]
    pub min_word_len: Option<usize>,
    /// Bm25Retriever: term-frequency saturation parameter.
    #[serde(default)]
    pub k1: Option<f32>,
    /// Bm25Retriever: length-normalization parameter.
    #[serde(default)]
    pub b: Option<f32>,
    /// SessionRetriever: flat baseline score for every block.
    #[serde(default)]
    pub score: Option<f32>,
}

fn parse_config_toml(s: &str) -> Result<ParsedConfig> {
    let parsed: ConfigFile = toml::from_str(s)?;
    let cache = parsed.cache.map(CacheEntry::build).transpose()?;
    Ok(ParsedConfig {
        profiles: parsed.profile,
        hf_tokenizers: parsed.hf_tokenizer,
        retrievers: parsed.retriever,
        section_budgets: parsed.section_budgets.map(SectionBudgetEntry::build),
        packer: parsed.packer.map(PackerEntry::build),
        cache,
    })
}

/// Materialize the retriever set declared in the config, bound to
/// `store`. Returns `None` when no `[[retriever]]` entries were
/// configured — callers fall back to the GreedyPager default
/// (RecencyRetriever).
fn build_retrievers(
    entries: &[RetrieverEntry],
    store: &Arc<LmdbStore>,
) -> Result<Option<Vec<Arc<dyn Retriever>>>> {
    if entries.is_empty() {
        return Ok(None);
    }
    let mut out: Vec<Arc<dyn Retriever>> = Vec::with_capacity(entries.len());
    for entry in entries {
        let r: Arc<dyn Retriever> = match entry.kind.as_str() {
            "recency" => {
                let mut r = RecencyRetriever::new(store.clone());
                if let Some(h) = entry.half_life_secs {
                    r = r.with_half_life(h);
                }
                Arc::new(r)
            }
            "lexical" => {
                let mut r = LexicalRetriever::new(store.clone());
                if let Some(n) = entry.min_word_len {
                    r = r.with_min_word_len(n);
                }
                Arc::new(r)
            }
            "bm25" => {
                let mut r = Bm25Retriever::new(store.clone());
                if let Some(k) = entry.k1 {
                    r = r.with_k1(k);
                }
                if let Some(b) = entry.b {
                    r = r.with_b(b);
                }
                if let Some(n) = entry.min_word_len {
                    r = r.with_min_word_len(n);
                }
                Arc::new(r)
            }
            "session" => {
                let mut r = SessionRetriever::new(store.clone());
                if let Some(s) = entry.score {
                    r = r.with_score(s);
                }
                Arc::new(r)
            }
            other => {
                return Err(anyhow!(
                    "unknown retriever kind `{other}`; expected one of: recency, lexical, bm25, session",
                ));
            }
        };
        out.push(r);
    }
    Ok(Some(out))
}

pub(crate) fn dispatch(command: Command, config: &LoadedConfig) -> Result<()> {
    match command {
        Command::Init { path } => init(&path),
        Command::Put {
            store,
            session,
            kind,
            priority,
            file,
        } => put(&store, SessionId(session), kind.into(), priority, &file),
        Command::ListModels => list_models(&config.models),
        Command::Page {
            store,
            session,
            model,
            task,
            json,
        } => page(&store, SessionId(session), &model, &task, json, config),
        Command::Pack {
            store,
            session,
            model,
            task,
            prompt_only,
            chat,
            timestamps,
            trace,
        } => pack(
            &store,
            SessionId(session),
            &model,
            &task,
            prompt_only,
            chat,
            timestamps,
            trace.as_deref(),
            config,
        ),
        Command::Trace(TraceSub::Show { store, call_id }) => trace_show(&store, CallId(call_id)),
        Command::Trace(TraceSub::Diff { store, prev, next }) => {
            trace_diff(&store, CallId(prev), CallId(next))
        }
        Command::ListSessions { store } => list_sessions(&store),
        Command::Verify { store } => verify(&store),
        Command::Repair { store, yes } => repair(&store, yes),
        Command::Purge {
            store,
            block,
            session,
            yes,
        } => purge(&store, block, session, yes),
        Command::Show { store, id, json } => show(&store, BlockId(id), json),
        Command::AddEdge {
            store,
            from,
            to,
            kind,
        } => add_edge(&store, BlockId(from), BlockId(to), kind.into()),
        Command::Edges {
            store,
            id,
            incoming,
        } => edges(&store, BlockId(id), incoming),
        Command::Summarize {
            store,
            session,
            summarizer,
            max_chars,
            last,
            store_summary,
            anthropic_model,
            anthropic_max_tokens,
        } => summarize(&SummarizeArgs {
            store_path: &store,
            session: SessionId(session),
            summarizer,
            max_chars,
            last,
            store_summary,
            anthropic_model: anthropic_model.as_deref(),
            anthropic_max_tokens,
        }),
    }
}

fn init(path: &Path) -> Result<()> {
    let _store = LmdbStore::open(path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", path.display()))?;
    println!("initialized store at {}", path.display());
    Ok(())
}

fn put(
    store_path: &Path,
    session: SessionId,
    kind: BlockKind,
    priority: f32,
    file: &Path,
) -> Result<()> {
    let bytes = if file == Path::new("-") {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .context("reading stdin")?;
        buf
    } else {
        std::fs::read(file).with_context(|| format!("reading {}", file.display()))?
    };

    let store = LmdbStore::open(store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", store_path.display()))?;

    let id = new_block_id();
    let now = Timestamp(now_ms());
    let block = ContextBlock {
        id,
        kind,
        bytes: bytes.clone(),
        token_counts: TokenCounts::new(),
        priority,
        created_at: now,
        updated_at: now,
        provenance: Provenance::default(),
        hash: ContentHash::of(&bytes),
    };
    let stored = store.put(session, block)?;
    println!("{stored}");
    Ok(())
}

#[allow(clippy::unnecessary_wraps)] // matches sibling-handler signatures.
fn list_models(reg: &ModelRegistry) -> Result<()> {
    let mut profiles: Vec<&ModelProfile> = reg.profiles().collect();
    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    println!(
        "{:<24}  {:>8}  {:>6}  {:>6}  {:<14}",
        "name", "ctx", "out", "margin", "tokenizer"
    );
    for p in profiles {
        println!(
            "{:<24}  {:>8}  {:>6}  {:>6}  {:<14}",
            p.name,
            p.max_context_tokens,
            p.reserved_output_tokens,
            p.safety_margin_tokens,
            p.tokenizer,
        );
    }
    Ok(())
}

fn page(
    store_path: &Path,
    session: SessionId,
    model_name: &str,
    task: &str,
    json: bool,
    config: &LoadedConfig,
) -> Result<()> {
    let (store, profile, tokenizer) = open_for_model(store_path, model_name, config)?;
    let mut pager = GreedyPager::new(store.clone(), tokenizer);
    if let Some(retrievers) = build_retrievers(&config.retriever_entries, &store)? {
        pager = pager.with_retrievers(retrievers);
    }
    if let Some(budgets) = &config.section_budgets {
        pager = pager.with_budgets(budgets.clone());
    }
    let plan = pager.page(PageRequest {
        session_id: session,
        task: task.to_string(),
        model: profile,
        required_blocks: vec![],
    })?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).context("serializing plan")?
        );
        return Ok(());
    }

    println!("selected ({}):", plan.selected.len());
    for id in &plan.selected {
        println!("  {id}");
    }
    println!("omitted ({}):", plan.omitted.len());
    for o in &plan.omitted {
        println!("  {} ({:?}, score={:.4})", o.block_id, o.reason, o.score);
    }
    println!("estimated_tokens: {}", plan.estimated_tokens);
    Ok(())
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)] // CLI flags map 1:1 to handler args; refactoring to a struct buys nothing here.
fn pack(
    store_path: &Path,
    session: SessionId,
    model_name: &str,
    task: &str,
    prompt_only: bool,
    chat: bool,
    timestamps_flag: bool,
    trace_path: Option<&Path>,
    config: &LoadedConfig,
) -> Result<()> {
    let (store, profile, tokenizer) = open_for_model(store_path, model_name, config)?;
    let mut pager = GreedyPager::new(store.clone(), tokenizer.clone());
    if let Some(retrievers) = build_retrievers(&config.retriever_entries, &store)? {
        pager = pager.with_retrievers(retrievers);
    }
    if let Some(budgets) = &config.section_budgets {
        pager = pager.with_budgets(budgets.clone());
    }
    let mut packer_options = config.packer_options.clone();
    if timestamps_flag {
        packer_options.include_timestamps = true;
    }
    let packer = SimplePacker::new(store, tokenizer).with_options(packer_options);

    let request = PageRequest {
        session_id: session,
        task: task.to_string(),
        model: profile,
        required_blocks: vec![],
    };

    let started_at = Timestamp(now_ms());
    let started = Instant::now();
    let plan = pager.page(request.clone())?;
    let prompt = packer.pack(&request, &plan)?;
    let duration_ms = u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX);

    let trace_id = if let Some(path) = trace_path {
        let sink = LmdbTraceSink::open(path)
            .with_context(|| format!("opening trace store at {}", path.display()))?;
        let call_id = new_call_id();
        sink.record(TraceRecord {
            call_id,
            session,
            model: request.model.name.clone(),
            plan: plan.clone(),
            prompt_tokens: prompt.input_tokens,
            prompt_hash: ContentHash::of(prompt.rendered.as_bytes()),
            started_at,
            duration_ms,
            model_version: request.model.name.clone(),
            tokenizer_version: request.model.tokenizer.as_str().to_string(),
            output: None,
            output_tokens: None,
        })?;
        Some(call_id)
    } else {
        None
    };

    if chat {
        // Re-render the same plan as role-tagged messages.
        let chat_prompt = packer.pack_chat(&request, &plan)?;
        eprintln!("# model:          {}", chat_prompt.model);
        eprintln!("# input_tokens:   {}", chat_prompt.input_tokens);
        eprintln!("# messages:       {}", chat_prompt.messages.len());
        match chat_prompt.cache_boundary {
            Some(n) => eprintln!("# cache_boundary: {n} (messages[0..={n}] cacheable)"),
            None => eprintln!("# cache_boundary: none"),
        }
        eprintln!("# duration_ms:    {duration_ms}");
        if let Some(id) = trace_id {
            eprintln!("# trace_id:       {id}");
        }
        eprintln!("---");
        let json = serde_json::to_string_pretty(&chat_prompt)
            .context("serializing chat prompt")?;
        println!("{json}");
    } else if prompt_only {
        print!("{}", prompt.rendered);
    } else {
        eprintln!("# model:         {}", prompt.model);
        eprintln!("# input_tokens:  {}", prompt.input_tokens);
        eprintln!("# blocks:        {}", prompt.blocks.len());
        eprintln!("# duration_ms:   {duration_ms}");
        if let Some(id) = trace_id {
            eprintln!("# trace_id:      {id}");
        }
        eprintln!("---");
        print!("{}", prompt.rendered);
    }
    Ok(())
}

struct SummarizeArgs<'a> {
    store_path: &'a Path,
    session: SessionId,
    summarizer: SummarizerArg,
    max_chars: usize,
    last: Option<usize>,
    store_summary: bool,
    anthropic_model: Option<&'a str>,
    anthropic_max_tokens: Option<u32>,
}

fn summarize(args: &SummarizeArgs<'_>) -> Result<()> {
    let store = LmdbStore::open(args.store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", args.store_path.display()))?;

    let mut ids = store.list_session(args.session)?;
    ids.sort(); // BlockId order is chronological.
    if let Some(n) = args.last {
        let from = ids.len().saturating_sub(n);
        ids.drain(0..from);
    }
    let mut blocks: Vec<ContextBlock> = Vec::with_capacity(ids.len());
    for &id in &ids {
        if let Some(b) = store.get(id)? {
            blocks.push(b);
        }
    }

    let (summary_text, summarizer_name) = match args.summarizer {
        SummarizerArg::Noop => {
            let s = NoopSummarizer;
            (s.summarize(&blocks)?, s.name())
        }
        SummarizerArg::Truncating => {
            let s = TruncatingSummarizer::new(args.max_chars);
            (s.summarize(&blocks)?, s.name())
        }
        SummarizerArg::Anthropic => {
            let mut s =
                AnthropicSummarizer::from_env().context("constructing AnthropicSummarizer")?;
            if let Some(model) = args.anthropic_model {
                s = s.with_model(model);
            }
            if let Some(n) = args.anthropic_max_tokens {
                s = s.with_max_tokens(n);
            }
            (s.summarize(&blocks)?, s.name())
        }
    };

    print!("{summary_text}");

    if args.store_summary {
        let bytes = summary_text.into_bytes();
        let now = Timestamp(now_ms());
        let id = new_block_id();
        let block = ContextBlock {
            id,
            kind: BlockKind::Summary,
            bytes: bytes.clone(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: now,
            updated_at: now,
            provenance: Provenance {
                source: Some(format!("summarize:{summarizer_name}")),
                parents: ids,
                labels: vec![],
            },
            hash: ContentHash::of(&bytes),
        };
        let stored = store.put(args.session, block)?;
        eprintln!("# summary stored: {stored}");
    }

    Ok(())
}

fn list_sessions(store_path: &Path) -> Result<()> {
    let store = LmdbStore::open(store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", store_path.display()))?;
    let sessions = store.list_sessions()?;
    for s in sessions {
        println!("{s}");
    }
    Ok(())
}

fn verify(store_path: &Path) -> Result<()> {
    let store = LmdbStore::open(store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", store_path.display()))?;
    let report = store.verify()?;
    println!("blocks checked:           {}", report.blocks_checked);
    println!("hash mismatches:          {}", report.hash_mismatches.len());
    println!(
        "missing from hash index:  {}",
        report.missing_from_hash_index.len()
    );
    println!(
        "hash index misroutes:     {}",
        report.hash_index_misroutes.len()
    );
    println!(
        "orphan session entries:   {}",
        report.orphan_session_entries
    );
    println!(
        "blocks with no session:   {}",
        report.blocks_with_no_session.len()
    );
    if !report.hash_mismatches.is_empty() {
        eprintln!("\nhash mismatches:");
        for id in &report.hash_mismatches {
            eprintln!("  {id}");
        }
    }
    if !report.missing_from_hash_index.is_empty() {
        eprintln!("\nmissing from hash index:");
        for id in &report.missing_from_hash_index {
            eprintln!("  {id}");
        }
    }
    if report.is_clean() {
        println!("\nOK");
        Ok(())
    } else {
        Err(anyhow!("integrity check failed"))
    }
}

fn repair(store_path: &Path, yes: bool) -> Result<()> {
    if !yes {
        return Err(anyhow!("destructive operation: pass --yes to confirm"));
    }
    let store = LmdbStore::open(store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", store_path.display()))?;
    let report = store.repair()?;
    println!(
        "hash index rebuilt:                  {}",
        report.hash_index_rebuilt
    );
    println!(
        "hash entries written:                {}",
        report.hash_entries_written
    );
    println!(
        "orphan session entries removed:      {}",
        report.orphan_session_entries_removed
    );
    println!(
        "blocks with no session (untouched):  {}",
        report.blocks_with_no_session.len()
    );
    println!(
        "hash mismatches quarantined:         {}",
        report.hash_mismatches_quarantined.len()
    );
    if !report.hash_mismatches_quarantined.is_empty() {
        eprintln!("\nhash mismatches (left as-is, need human review):");
        for id in &report.hash_mismatches_quarantined {
            eprintln!("  {id}");
        }
    }
    Ok(())
}

fn purge(store_path: &Path, block: Option<u128>, session: Option<u128>, yes: bool) -> Result<()> {
    if !yes {
        return Err(anyhow!("destructive operation: pass --yes to confirm"));
    }
    match (block, session) {
        (Some(_), Some(_)) | (None, None) => {
            Err(anyhow!("specify exactly one of --block or --session"))
        }
        (Some(id), None) => {
            let store = LmdbStore::open(store_path, StoreConfig::default())
                .with_context(|| format!("opening store at {}", store_path.display()))?;
            let deleted = store.delete(BlockId(id))?;
            if deleted {
                println!("deleted block {}", BlockId(id));
            } else {
                eprintln!("block not found: {}", BlockId(id));
            }
            Ok(())
        }
        (None, Some(sid)) => {
            let store = LmdbStore::open(store_path, StoreConfig::default())
                .with_context(|| format!("opening store at {}", store_path.display()))?;
            let count = store.purge_session(SessionId(sid))?;
            println!("purged {count} blocks from session {}", SessionId(sid));
            Ok(())
        }
    }
}

fn show(store_path: &Path, id: BlockId, json: bool) -> Result<()> {
    let store = LmdbStore::open(store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", store_path.display()))?;
    let block = store
        .get(id)?
        .ok_or_else(|| anyhow!("block not found: {id}"))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&block).context("serializing block")?
        );
        return Ok(());
    }

    println!("id:            {}", block.id);
    println!("kind:          {:?}", block.kind);
    println!("priority:      {:.4}", block.priority);
    println!("created_at:    {}", block.created_at.0);
    println!("updated_at:    {}", block.updated_at.0);
    println!("hash:          {:?}", block.hash);
    println!("body_bytes:    {}", block.bytes.len());

    if block.provenance.source.is_some()
        || !block.provenance.parents.is_empty()
        || !block.provenance.labels.is_empty()
    {
        println!("provenance:");
        if let Some(src) = &block.provenance.source {
            println!("  source:      {src}");
        }
        if !block.provenance.parents.is_empty() {
            println!("  parents ({}):", block.provenance.parents.len());
            for p in &block.provenance.parents {
                println!("    {p}");
            }
        }
        if !block.provenance.labels.is_empty() {
            println!("  labels:      {}", block.provenance.labels.join(", "));
        }
    }

    if !block.token_counts.is_empty() {
        println!("token_counts:");
        for (tid, count) in block.token_counts.iter() {
            println!("  {tid}: {count}");
        }
    }

    println!("---");
    if let Ok(text) = std::str::from_utf8(&block.bytes) {
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
    } else {
        // Binary body — hex dump first 256 bytes.
        for chunk in block.bytes.chunks(16).take(16) {
            for b in chunk {
                print!("{b:02x} ");
            }
            println!();
        }
        if block.bytes.len() > 256 {
            println!("... ({} more bytes)", block.bytes.len() - 256);
        }
    }
    Ok(())
}

fn trace_diff(store_path: &Path, prev: CallId, next: CallId) -> Result<()> {
    let sink = LmdbTraceSink::open(store_path)
        .with_context(|| format!("opening trace store at {}", store_path.display()))?;
    let prev_rec = sink.fetch(prev)?.ok_or_else(|| anyhow!("no trace for {prev}"))?;
    let next_rec = sink.fetch(next)?.ok_or_else(|| anyhow!("no trace for {next}"))?;

    let diff = llm386_diff::diff_traces(&prev_rec, &next_rec);
    println!("prev:    {prev}");
    println!("next:    {next}");
    println!("summary: {}", diff.summary());

    if !diff.added.is_empty() {
        println!("added ({}):", diff.added.len());
        for entry in &diff.added {
            println!(
                "  + {} ({:?})",
                entry.block_id,
                entry.reason_next.expect("added entries have a next reason"),
            );
        }
    }
    if !diff.removed.is_empty() {
        println!("removed ({}):", diff.removed.len());
        for entry in &diff.removed {
            println!(
                "  - {} ({:?})",
                entry.block_id,
                entry.reason_prev.expect("removed entries have a prev reason"),
            );
        }
    }
    let changed: Vec<_> = diff.kept.iter().filter(|e| e.reason_changed()).collect();
    if !changed.is_empty() {
        println!("reason changes ({}):", changed.len());
        for entry in changed {
            println!(
                "  ~ {} ({:?} -> {:?})",
                entry.block_id,
                entry.reason_prev.expect("kept entries have a prev reason"),
                entry.reason_next.expect("kept entries have a next reason"),
            );
        }
    }
    Ok(())
}

fn add_edge(
    store_path: &Path,
    from: BlockId,
    to: BlockId,
    kind: llm386_core::EdgeKind,
) -> Result<()> {
    let store = LmdbStore::open(store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", store_path.display()))?;
    store.put_edge(llm386_core::Edge { from, to, kind })?;
    println!("edge added: {from} --{kind:?}--> {to}");
    Ok(())
}

fn edges(store_path: &Path, id: BlockId, incoming: bool) -> Result<()> {
    let store = LmdbStore::open(store_path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", store_path.display()))?;
    let edges = if incoming {
        store.edges_to(id)?
    } else {
        store.edges_from(id)?
    };
    if edges.is_empty() {
        println!("no edges");
        return Ok(());
    }
    for edge in edges {
        println!("{} --{:?}--> {}", edge.from, edge.kind, edge.to);
    }
    Ok(())
}

fn trace_show(store_path: &Path, call_id: CallId) -> Result<()> {
    let sink = LmdbTraceSink::open(store_path)
        .with_context(|| format!("opening trace store at {}", store_path.display()))?;
    let trace = sink
        .fetch(call_id)?
        .ok_or_else(|| anyhow!("no trace for {call_id}"))?;

    println!("call_id:         {}", trace.call_id);
    println!("session:         {}", trace.session);
    println!("model:           {}", trace.model);
    println!("started_at_ms:   {}", trace.started_at.0);
    println!("duration_ms:     {}", trace.duration_ms);
    println!("prompt_tokens:   {}", trace.prompt_tokens);
    println!("prompt_hash:     {:?}", trace.prompt_hash);
    println!("estimated:       {}", trace.plan.estimated_tokens);
    println!("plan.selected ({}):", trace.plan.selected.len());
    for id in &trace.plan.selected {
        println!("  {id}");
    }
    println!("plan.omitted ({}):", trace.plan.omitted.len());
    for o in &trace.plan.omitted {
        println!("  {} ({:?}, score={:.4})", o.block_id, o.reason, o.score);
    }
    Ok(())
}

fn open_for_model(
    store_path: &Path,
    model_name: &str,
    config: &LoadedConfig,
) -> Result<(Arc<LmdbStore>, ModelProfile, Arc<dyn Tokenizer>)> {
    let store = Arc::new(
        LmdbStore::open(store_path, StoreConfig::default())
            .with_context(|| format!("opening store at {}", store_path.display()))?,
    );
    let profile = config
        .models
        .get(model_name)
        .ok_or_else(|| anyhow!("unknown model profile: {model_name}"))?
        .clone();
    let tokenizer = config.tokenizers.get(&profile.tokenizer).ok_or_else(|| {
        anyhow!(
            "no tokenizer adapter for {} (used by model {}); register one via [[hf_tokenizer]] in the config file",
            profile.tokenizer,
            profile.name,
        )
    })?;
    Ok((store, profile, tokenizer))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn new_block_id() -> BlockId {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("getrandom should not fail");
    BlockId::from_parts(now_ms(), u128::from_be_bytes(buf))
}

fn new_call_id() -> CallId {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("getrandom should not fail");
    CallId(u128::from_be_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_toml_basic_profile() {
        let s = r#"
[[profile]]
name = "my-model"
max_context_tokens = 64000
reserved_output_tokens = 8000
tokenizer = "o200k_base"
"#;
        let parsed = parse_config_toml(s).unwrap();
        assert_eq!(parsed.profiles.len(), 1);
        let p = &parsed.profiles[0];
        assert_eq!(p.name, "my-model");
        assert_eq!(p.max_context_tokens, 64_000);
        assert_eq!(p.reserved_output_tokens, 8_000);
        // Defaults applied.
        assert_eq!(p.safety_margin_tokens, 0);
        assert!(p.supports_system_role);
        assert!(p.supports_tools);
        assert_eq!(p.tokenizer.as_str(), "o200k_base");
    }

    #[test]
    fn parse_config_toml_explicit_fields() {
        let s = r#"
[[profile]]
name = "strict"
max_context_tokens = 1000
reserved_output_tokens = 100
safety_margin_tokens = 50
tokenizer = "cl100k_base"
supports_system_role = false
supports_tools = false
"#;
        let p = parse_config_toml(s)
            .unwrap()
            .profiles
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(p.safety_margin_tokens, 50);
        assert!(!p.supports_system_role);
        assert!(!p.supports_tools);
    }

    #[test]
    fn parse_config_toml_empty_file_yields_empty_vecs() {
        let parsed = parse_config_toml("").unwrap();
        assert!(parsed.profiles.is_empty());
        assert!(parsed.hf_tokenizers.is_empty());
    }

    #[test]
    fn parse_config_toml_rejects_profile_missing_required_field() {
        // No `tokenizer` field — should fail.
        let s = r#"
[[profile]]
name = "broken"
max_context_tokens = 100
reserved_output_tokens = 10
"#;
        assert!(parse_config_toml(s).is_err());
    }

    #[test]
    fn parse_config_toml_loads_retriever_entries() {
        let s = r#"
[[retriever]]
kind = "recency"
half_life_secs = 60.0

[[retriever]]
kind = "bm25"
k1 = 1.5
b = 0.5
min_word_len = 3

[[retriever]]
kind = "lexical"

[[retriever]]
kind = "session"
score = 0.25
"#;
        let parsed = parse_config_toml(s).unwrap();
        assert_eq!(parsed.retrievers.len(), 4);
        assert_eq!(parsed.retrievers[0].kind, "recency");
        assert_eq!(parsed.retrievers[0].half_life_secs, Some(60.0));
        assert_eq!(parsed.retrievers[1].kind, "bm25");
        assert_eq!(parsed.retrievers[1].k1, Some(1.5));
        assert_eq!(parsed.retrievers[2].kind, "lexical");
        assert_eq!(parsed.retrievers[3].kind, "session");
        assert_eq!(parsed.retrievers[3].score, Some(0.25));
    }

    #[test]
    fn parse_config_toml_loads_hf_tokenizer_entries() {
        let s = r#"
[[hf_tokenizer]]
name = "llama-3"
path = "/tmp/llama-3-tokenizer.json"

[[hf_tokenizer]]
name = "qwen-2.5"
path = "/tmp/qwen-2.5-tokenizer.json"
"#;
        let parsed = parse_config_toml(s).unwrap();
        assert_eq!(parsed.hf_tokenizers.len(), 2);
        assert_eq!(parsed.hf_tokenizers[0].name, "llama-3");
        assert_eq!(parsed.hf_tokenizers[1].name, "qwen-2.5");
    }
}
