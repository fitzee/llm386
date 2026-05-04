//! TOML config loader for the Python `Store(path, profiles=...)`.
//!
//! Mirrors the schema the CLI uses (`[[profile]]`,
//! `[[hf_tokenizer]]`, `[[retriever]]`) so a single config file
//! works for both surfaces.

use std::path::Path;
use std::sync::Arc;

use llm386_core::{ModelProfile, ModelRegistry, Retriever, TokenizerId};
use llm386_pager::{Bm25Retriever, LexicalRetriever, RecencyRetriever, SessionRetriever};
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

/// Apply parsed [[profile]] and [[hf_tokenizer]] entries to the
/// given registries, mutating in place. Returns the parsed
/// retriever entries (the caller materializes those per-call,
/// since they bind to a specific store).
pub(crate) fn apply(
    parsed: ConfigFile,
    models: &mut ModelRegistry,
    tokenizers: &mut TokenizerRegistry,
) -> Result<Vec<RetrieverEntry>, String> {
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
    Ok(parsed.retriever)
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
