//! `SimplePacker` — string-rendering [`Packer`].
//!
//! Walks `PagePlan::selected`, classifies each block by its
//! [`BlockKind`] into one of nine canonical sections, renders the
//! sections in fixed order with `## <name>` headers, and verifies
//! that the rendered token count does not exceed
//! [`ModelProfile::input_budget`].
//!
//! Determinism: within each section, blocks render in `BlockId`
//! order (chronological). Identical inputs always produce a
//! byte-identical `rendered` string.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use llm386_core::{
    BlockKind, BlockStore, ContextBlock, PackedBlock, PackedPrompt, Packer, PackerError, PagePlan,
    PageRequest, SectionKind, StoreError, Tokenizer,
};
use tracing::instrument;

/// Canonical render order. `Slack` is intentional headroom and is
/// never emitted; `Task` is synthesized from `PageRequest::task`
/// rather than drawn from blocks.
const SECTION_ORDER: [SectionKind; 9] = [
    SectionKind::System,
    SectionKind::Task,
    SectionKind::State,
    SectionKind::Plan,
    SectionKind::Retrieved,
    SectionKind::Tools,
    SectionKind::Recent,
    SectionKind::Background,
    SectionKind::Slack,
];

/// String-rendering [`Packer`].
pub struct SimplePacker<S: BlockStore> {
    store: Arc<S>,
    tokenizer: Arc<dyn Tokenizer>,
}

impl<S: BlockStore> SimplePacker<S> {
    pub fn new(store: Arc<S>, tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self { store, tokenizer }
    }
}

impl<S: BlockStore> fmt::Debug for SimplePacker<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimplePacker")
            .field("tokenizer", &self.tokenizer.id())
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static> Packer for SimplePacker<S> {
    #[instrument(skip(self, request, plan), fields(model = %request.model.name))]
    fn pack(&self, request: &PageRequest, plan: &PagePlan) -> Result<PackedPrompt, PackerError> {
        // Load every selected block, grouped by section.
        let mut by_section: HashMap<SectionKind, Vec<ContextBlock>> = HashMap::new();
        for &id in &plan.selected {
            let block = self.store.get(id)?.ok_or_else(|| {
                PackerError::Storage(StoreError::Backend(format!(
                    "selected block {id} missing from store",
                )))
            })?;
            by_section
                .entry(section_for(block.kind))
                .or_default()
                .push(block);
        }
        for blocks in by_section.values_mut() {
            blocks.sort_by_key(|b| b.id);
        }

        let mut rendered = String::new();
        let mut packed_blocks: Vec<PackedBlock> = Vec::new();

        for &section in &SECTION_ORDER {
            match section {
                SectionKind::Slack => (), // Intentional headroom — never emitted.
                SectionKind::Task => {
                    if !request.task.is_empty() {
                        write_header(&mut rendered, section);
                        rendered.push_str(&request.task);
                        rendered.push_str("\n\n");
                    }
                }
                _ => {
                    let Some(blocks) = by_section.get(&section) else {
                        continue;
                    };
                    if blocks.is_empty() {
                        continue;
                    }
                    write_header(&mut rendered, section);
                    for block in blocks {
                        let body = std::str::from_utf8(&block.bytes).map_err(|e| {
                            PackerError::Storage(StoreError::Backend(format!(
                                "non-utf8 block {}: {e}",
                                block.id,
                            )))
                        })?;
                        rendered.push_str(body);
                        rendered.push_str("\n\n");

                        let tokens = self.tokenizer.count(&block.bytes)?;
                        packed_blocks.push(PackedBlock {
                            block_id: block.id,
                            section,
                            tokens,
                            score: 0.0,
                        });
                    }
                }
            }
        }

        let total = self.tokenizer.count(rendered.as_bytes())?;
        let budget = request.model.input_budget();
        if total.0 > budget.0 {
            return Err(PackerError::BudgetExceeded {
                tokens: total.0,
                budget: budget.0,
            });
        }

        Ok(PackedPrompt {
            model: request.model.name.clone(),
            input_tokens: total,
            blocks: packed_blocks,
            rendered,
        })
    }
}

fn write_header(buf: &mut String, section: SectionKind) {
    buf.push_str("## ");
    buf.push_str(section_label(section));
    buf.push_str("\n\n");
}

const fn section_for(kind: BlockKind) -> SectionKind {
    match kind {
        BlockKind::System => SectionKind::System,
        BlockKind::State => SectionKind::State,
        BlockKind::Plan => SectionKind::Plan,
        BlockKind::Summary | BlockKind::Fact => SectionKind::Retrieved,
        BlockKind::ToolResult => SectionKind::Tools,
        BlockKind::UserMessage | BlockKind::AssistantMessage => SectionKind::Recent,
        BlockKind::DocumentChunk | BlockKind::Trace => SectionKind::Background,
    }
}

const fn section_label(s: SectionKind) -> &'static str {
    match s {
        SectionKind::System => "System",
        SectionKind::Task => "Task",
        SectionKind::State => "State",
        SectionKind::Plan => "Plan",
        SectionKind::Retrieved => "Retrieved",
        SectionKind::Tools => "Tools",
        SectionKind::Recent => "Recent",
        SectionKind::Background => "Background",
        SectionKind::Slack => "Slack",
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm386_core::{
        BlockId, ContentHash, ContextBlock, ModelProfile, PagePlan, PageRequest, Pager, Provenance,
        SessionId, Timestamp, TokenCount, TokenCounts, Tokenizer, TokenizerId,
    };
    use llm386_store_lmdb::{LmdbStore, StoreConfig};
    use llm386_tokenizer::cl100k_base;
    use tempfile::TempDir;

    use super::*;

    fn setup() -> (Arc<LmdbStore>, TempDir, Arc<dyn Tokenizer>) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LmdbStore::open(dir.path(), StoreConfig::default()).unwrap());
        let tok: Arc<dyn Tokenizer> = Arc::new(cl100k_base().unwrap());
        (store, dir, tok)
    }

    fn make_block(bytes: &[u8], kind: BlockKind, ts_ms: u64, rnd: u128) -> ContextBlock {
        ContextBlock {
            id: BlockId::from_parts(ts_ms, rnd),
            kind,
            bytes: bytes.to_vec(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(ts_ms),
            updated_at: Timestamp(ts_ms),
            provenance: Provenance::default(),
            hash: ContentHash::of(bytes),
        }
    }

    fn profile(max: u32) -> ModelProfile {
        ModelProfile {
            name: "test".into(),
            max_context_tokens: max,
            reserved_output_tokens: 0,
            safety_margin_tokens: 0,
            tokenizer: TokenizerId::new("cl100k_base"),
            supports_system_role: true,
            supports_tools: true,
        }
    }

    #[test]
    fn empty_plan_renders_only_task() {
        let (store, _dir, tok) = setup();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: SessionId(1),
            task: "answer the user".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let prompt = packer.pack(&request, &plan).unwrap();
        assert!(prompt.rendered.contains("## Task"));
        assert!(prompt.rendered.contains("answer the user"));
        assert!(prompt.blocks.is_empty());
        assert!(prompt.input_tokens.0 > 0);
    }

    #[test]
    fn plan_renders_sections_in_canonical_order() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"SYS", BlockKind::System, 1, 1))
            .unwrap();
        let user_id = store
            .put(session, make_block(b"USR", BlockKind::UserMessage, 2, 2))
            .unwrap();
        let plan_id = store
            .put(session, make_block(b"PLN", BlockKind::Plan, 3, 3))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "TASKBODY".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            // Deliberately in wrong order — packer must canonicalize.
            selected: vec![plan_id, user_id, sys_id],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let prompt = packer.pack(&request, &plan).unwrap();
        let sys_pos = prompt.rendered.find("SYS").unwrap();
        let task_pos = prompt.rendered.find("TASKBODY").unwrap();
        let plan_pos = prompt.rendered.find("PLN").unwrap();
        let user_pos = prompt.rendered.find("USR").unwrap();
        // System < Task < Plan < Recent
        assert!(sys_pos < task_pos);
        assert!(task_pos < plan_pos);
        assert!(plan_pos < user_pos);
    }

    #[test]
    fn pack_is_deterministic() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let a = store
            .put(session, make_block(b"AAA", BlockKind::Fact, 1, 1))
            .unwrap();
        let b = store
            .put(session, make_block(b"BBB", BlockKind::Fact, 2, 2))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "t".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![b, a],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let p1 = packer.pack(&request, &plan).unwrap();
        let p2 = packer.pack(&request, &plan).unwrap();
        assert_eq!(p1.rendered, p2.rendered);
        assert_eq!(p1.input_tokens, p2.input_tokens);
    }

    #[test]
    fn budget_exceeded_returns_error() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        // Build a plan that's way larger than the budget.
        let big = "lorem ipsum dolor sit amet ".repeat(200);
        let id = store
            .put(session, make_block(big.as_bytes(), BlockKind::Fact, 1, 1))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "t".into(),
            model: profile(10), // tiny budget
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![id],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let err = packer.pack(&request, &plan).unwrap_err();
        assert!(matches!(err, PackerError::BudgetExceeded { .. }));
    }

    #[test]
    fn end_to_end_with_pager() {
        use llm386_pager::GreedyPager;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        store
            .put(session, make_block(b"system rule", BlockKind::System, 1, 1))
            .unwrap();
        store
            .put(
                session,
                make_block(b"hello there", BlockKind::UserMessage, 2, 2),
            )
            .unwrap();
        store
            .put(
                session,
                make_block(b"sure thing", BlockKind::AssistantMessage, 3, 3),
            )
            .unwrap();
        let pager = GreedyPager::new(store.clone(), tok.clone());
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "summarize the conversation".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = pager.page(request.clone()).unwrap();
        let prompt = packer.pack(&request, &plan).unwrap();
        assert!(prompt.rendered.contains("system rule"));
        assert!(prompt.rendered.contains("hello there"));
        assert!(prompt.rendered.contains("summarize the conversation"));
        assert!(prompt.input_tokens.0 <= request.model.input_budget().0);
    }
}
