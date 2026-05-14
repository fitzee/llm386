//! TOML config loader for the Python `Store(path, profiles=...)`.
//!
//! Mirrors the schema the CLI uses (`[[profile]]`,
//! `[[hf_tokenizer]]`, `[[retriever]]`) so a single config file
//! works for both surfaces.

use std::path::Path;
use std::sync::Arc;

use llm386_core::{ModelProfile, ModelRegistry, Retriever, SectionKind, TokenizerId};
use llm386_packer::{CacheOptions, PackerOptions};
use llm386_pager::{
    Bm25Retriever, LexicalRetriever, RecencyRetriever, SectionBudgetTable, SessionRetriever,
};
use llm386_store_lmdb::LmdbStore;
use llm386_tokenizer::{HfTokenizer, TokenizerRegistry};
use serde::Deserialize;

#[derive(Default, Deserialize)]
pub(crate) struct ConfigFile {
    #[serde(default)]
    pub profile: Vec<ModelProfile>,
    #[serde(default)]
    pub hf_tokenizer: Vec<HfTokenizerEntry>,
    #[serde(default)]
    pub retriever: Vec<RetrieverEntry>,
    #[serde(default)]
    pub section_budgets: Option<SectionBudgetEntry>,
    #[serde(default)]
    pub packer: Option<PackerEntry>,
    #[serde(default)]
    pub cache: Option<CacheEntry>,
}

#[derive(Default, Deserialize)]
pub(crate) struct PackerEntry {
    #[serde(default)]
    include_timestamps: bool,
}

impl PackerEntry {
    pub(crate) fn build(self) -> PackerOptions {
        PackerOptions {
            include_timestamps: self.include_timestamps,
            now_ms: None,
            cache: CacheOptions::default(),
        }
    }
}

#[derive(Default, Deserialize)]
pub(crate) struct CacheEntry {
    #[serde(default)]
    stable_sections: Option<Vec<String>>,
}

impl CacheEntry {
    pub(crate) fn build(self) -> Result<CacheOptions, String> {
        let mut out = CacheOptions::default();
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
                        return Err(format!(
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

#[derive(Default, Deserialize)]
pub(crate) struct SectionBudgetEntry {
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
    pub(crate) fn build(self) -> SectionBudgetTable {
        let mut table = SectionBudgetTable::empty();
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
pub(crate) struct HfTokenizerEntry {
    pub name: String,
    pub path: std::path::PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct RetrieverEntry {
    pub kind: String,
    #[serde(default)]
    pub half_life_secs: Option<f32>,
    #[serde(default)]
    pub min_word_len: Option<usize>,
    #[serde(default)]
    pub k1: Option<f32>,
    #[serde(default)]
    pub b: Option<f32>,
    #[serde(default)]
    pub score: Option<f32>,
}

pub(crate) fn parse(path: &Path) -> Result<ConfigFile, String> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| format!("reading config file at {}: {e}", path.display()))?;
    toml::from_str(&s).map_err(|e| format!("parsing config file at {}: {e}", path.display()))
}

pub(crate) struct Applied {
    pub retrievers: Vec<RetrieverEntry>,
    pub section_budgets: Option<SectionBudgetTable>,
    pub packer_options: Option<PackerOptions>,
}

/// Apply parsed [[profile]] and [[hf_tokenizer]] entries to the
/// given registries, mutating in place. Returns the parsed
/// retriever entries (the caller materializes those per-call, since
/// they bind to a specific store) and the optional
/// [section_budgets] table.
pub(crate) fn apply(
    parsed: ConfigFile,
    models: &mut ModelRegistry,
    tokenizers: &mut TokenizerRegistry,
) -> Result<Applied, String> {
    for profile in parsed.profile {
        models.register(profile);
    }
    for entry in parsed.hf_tokenizer {
        let tok = HfTokenizer::from_file(&entry.path, TokenizerId::new(&entry.name)).map_err(
            |e| {
                format!(
                    "loading huggingface tokenizer `{}` from {}: {e}",
                    entry.name,
                    entry.path.display(),
                )
            },
        )?;
        tokenizers.register(Arc::new(tok));
    }
    let mut packer_options = parsed.packer.map(PackerEntry::build);
    if let Some(cache) = parsed.cache {
        let cache_opts = cache.build()?;
        let opts = packer_options.get_or_insert_with(PackerOptions::default);
        opts.cache = cache_opts;
    }
    Ok(Applied {
        retrievers: parsed.retriever,
        section_budgets: parsed.section_budgets.map(SectionBudgetEntry::build),
        packer_options,
    })
}

/// Materialize a Vec<Arc<dyn Retriever>> from the parsed entries,
/// bound to the given store. Returns None when no entries were
/// configured (caller falls back to the GreedyPager default).
pub(crate) fn build_retrievers(
    entries: &[RetrieverEntry],
    store: &Arc<LmdbStore>,
) -> Result<Option<Vec<Arc<dyn Retriever>>>, String> {
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
                return Err(format!(
                    "unknown retriever kind `{other}`; expected one of: recency, lexical, bm25, session",
                ));
            }
        };
        out.push(r);
    }
    Ok(Some(out))
}
