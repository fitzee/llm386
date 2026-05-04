//! Subcommand handlers for `llm386`.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use llm386_core::{
    BlockId, BlockKind, BlockStore, ContentHash, ContextBlock, ModelProfile, Packer, PageRequest,
    Pager, Provenance, SessionId, Timestamp, TokenCounts, Tokenizer, default_registry,
};
use llm386_packer::SimplePacker;
use llm386_pager::GreedyPager;
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::default_registry as tokenizer_registry;

use crate::cli::{Cli, Command};

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
        } => pack(&store, SessionId(session), &model, &task, prompt_only),
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
    let plan = pager.page(request.clone())?;
    let prompt = packer.pack(&request, &plan)?;

    if prompt_only {
        print!("{}", prompt.rendered);
    } else {
        eprintln!("# model:         {}", prompt.model);
        eprintln!("# input_tokens:  {}", prompt.input_tokens);
        eprintln!("# blocks:        {}", prompt.blocks.len());
        eprintln!("---");
        print!("{}", prompt.rendered);
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
