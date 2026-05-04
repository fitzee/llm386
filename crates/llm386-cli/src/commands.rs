//! Subcommand handlers for `llm386`.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use llm386_core::{
    BlockId, BlockKind, BlockStore, CallId, ContentHash, ContextBlock, ModelProfile, Packer,
    PageRequest, Pager, Provenance, SessionId, Timestamp, TokenCounts, Tokenizer, TraceRecord,
    TraceSink, default_registry,
};
use llm386_packer::SimplePacker;
use llm386_pager::GreedyPager;
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::default_registry as tokenizer_registry;
use llm386_trace::LmdbTraceSink;

use crate::cli::{Cli, Command, TraceSub};

pub(crate) fn dispatch(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Init { path } => init(&path),
        Command::Put {
            store,
            session,
            kind,
            priority,
            file,
        } => put(&store, SessionId(session), kind.into(), priority, &file),
        Command::ListModels => list_models(),
        Command::Page {
            store,
            session,
            model,
            task,
        } => page(&store, SessionId(session), &model, &task),
        Command::Pack {
            store,
            session,
            model,
            task,
            prompt_only,
            trace,
        } => pack(
            &store,
            SessionId(session),
            &model,
            &task,
            prompt_only,
            trace.as_deref(),
        ),
        Command::Trace(TraceSub::Show { store, call_id }) => trace_show(&store, CallId(call_id)),
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
fn list_models() -> Result<()> {
    let reg = default_registry();
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

fn page(store_path: &Path, session: SessionId, model_name: &str, task: &str) -> Result<()> {
    let (store, profile, tokenizer) = open_for_model(store_path, model_name)?;
    let pager = GreedyPager::new(store, tokenizer);
    let plan = pager.page(PageRequest {
        session_id: session,
        task: task.to_string(),
        model: profile,
        required_blocks: vec![],
    })?;

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

fn pack(
    store_path: &Path,
    session: SessionId,
    model_name: &str,
    task: &str,
    prompt_only: bool,
    trace_path: Option<&Path>,
) -> Result<()> {
    let (store, profile, tokenizer) = open_for_model(store_path, model_name)?;
    let pager = GreedyPager::new(store.clone(), tokenizer.clone());
    let packer = SimplePacker::new(store, tokenizer);

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
        })?;
        Some(call_id)
    } else {
        None
    };

    if prompt_only {
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
) -> Result<(Arc<LmdbStore>, ModelProfile, Arc<dyn Tokenizer>)> {
    let store = Arc::new(
        LmdbStore::open(store_path, StoreConfig::default())
            .with_context(|| format!("opening store at {}", store_path.display()))?,
    );
    let profile = default_registry()
        .get(model_name)
        .ok_or_else(|| anyhow!("unknown model profile: {model_name}"))?
        .clone();
    let tokenizers = tokenizer_registry().context("initializing default tokenizer registry")?;
    let tokenizer = tokenizers.get(&profile.tokenizer).ok_or_else(|| {
        anyhow!(
            "no tokenizer adapter for {} (used by model {})",
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
