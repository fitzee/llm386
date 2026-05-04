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

use llm386_core::BlockKind;
use llm386_core::{
    BlockStore, ChatMessage, ChatPrompt, ChatRole, ContextBlock, PackedBlock, PackedPrompt, Packer,
    PackerError, PagePlan, PageRequest, SectionKind, StoreError, TokenCount, Tokenizer,
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
                .entry(block.kind.default_section())
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

impl<S: BlockStore + 'static> SimplePacker<S> {
    /// Render the same plan as a sequence of role-tagged
    /// [`ChatMessage`]s, suitable for chat-completion APIs.
    ///
    /// Mapping:
    ///
    /// - System / State / Plan / Retrieved / Background → folded
    ///   into a single `system` message (or `user` if the model's
    ///   `supports_system_role` is false), with section headers
    ///   preserved so the model can tell them apart.
    /// - Recent (`UserMessage` / `AssistantMessage`) → individual
    ///   role-tagged messages in `BlockId` order.
    /// - Tool results → individual `tool` messages, emitted just
    ///   before the final task message.
    /// - Task → final `user` message containing `request.task`.
    ///
    /// As with [`pack`], the result is verified against
    /// `ModelProfile::input_budget`; over-budget plans return
    /// [`PackerError::BudgetExceeded`].
    #[instrument(skip(self, request, plan), fields(model = %request.model.name))]
    pub fn pack_chat(
        &self,
        request: &PageRequest,
        plan: &PagePlan,
    ) -> Result<ChatPrompt, PackerError> {
        let mut by_section: HashMap<SectionKind, Vec<ContextBlock>> = HashMap::new();
        for &id in &plan.selected {
            let block = self.store.get(id)?.ok_or_else(|| {
                PackerError::Storage(StoreError::Backend(format!(
                    "selected block {id} missing from store",
                )))
            })?;
            by_section
                .entry(block.kind.default_section())
                .or_default()
                .push(block);
        }
        for blocks in by_section.values_mut() {
            blocks.sort_by_key(|b| b.id);
        }

        let mut messages: Vec<ChatMessage> = Vec::new();

        // (a) System / State / Plan / Retrieved / Background — one
        //     consolidated context message.
        let mut context = String::new();
        for &section in &[
            SectionKind::System,
            SectionKind::State,
            SectionKind::Plan,
            SectionKind::Retrieved,
            SectionKind::Background,
        ] {
            let Some(blocks) = by_section.get(&section) else {
                continue;
            };
            if blocks.is_empty() {
                continue;
            }
            if !context.is_empty() {
                context.push_str("\n\n");
            }
            write_section(&mut context, section, blocks)?;
        }
        if !context.is_empty() {
            let role = if request.model.supports_system_role {
                ChatRole::System
            } else {
                ChatRole::User
            };
            messages.push(ChatMessage {
                role,
                content: context.trim_end().into(),
            });
        }

        // (b) Recent — preserve user/assistant alternation.
        if let Some(recent) = by_section.get(&SectionKind::Recent) {
            for block in recent {
                let body = block_text(block)?;
                let role = match block.kind {
                    BlockKind::AssistantMessage => ChatRole::Assistant,
                    _ => ChatRole::User,
                };
                messages.push(ChatMessage {
                    role,
                    content: body.into(),
                });
            }
        }

        // (c) Tools — emitted as tool-role messages.
        if let Some(tools) = by_section.get(&SectionKind::Tools) {
            for block in tools {
                let body = block_text(block)?;
                messages.push(ChatMessage {
                    role: ChatRole::Tool,
                    content: body.into(),
                });
            }
        }

        // (d) Task — final user message.
        if !request.task.is_empty() {
            messages.push(ChatMessage {
                role: ChatRole::User,
                content: request.task.clone(),
            });
        }

        // Total tokens across all message contents.
        let mut total = TokenCount::ZERO;
        for m in &messages {
            total = total.saturating_add(self.tokenizer.count(m.content.as_bytes())?);
        }
        let budget = request.model.input_budget();
        if total.0 > budget.0 {
            return Err(PackerError::BudgetExceeded {
                tokens: total.0,
                budget: budget.0,
            });
        }

        Ok(ChatPrompt {
            model: request.model.name.clone(),
            input_tokens: total,
            messages,
        })
    }
}

fn write_section(
    buf: &mut String,
    section: SectionKind,
    blocks: &[ContextBlock],
) -> Result<(), PackerError> {
    write_header(buf, section);
    for block in blocks {
        buf.push_str(block_text(block)?);
        buf.push_str("\n\n");
    }
    Ok(())
}

fn block_text(block: &ContextBlock) -> Result<&str, PackerError> {
    std::str::from_utf8(&block.bytes).map_err(|e| {
        PackerError::Storage(StoreError::Backend(format!(
            "non-utf8 block {}: {e}",
            block.id,
        )))
    })
}

fn write_header(buf: &mut String, section: SectionKind) {
    buf.push_str("## ");
    buf.push_str(section_label(section));
    buf.push_str("\n\n");
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

    #[test]
    fn pack_chat_emits_role_tagged_messages() {
        use llm386_core::ChatRole;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"be brief", BlockKind::System, 1, 1))
            .unwrap();
        let user_id = store
            .put(session, make_block(b"hello?", BlockKind::UserMessage, 2, 2))
            .unwrap();
        let asst_id = store
            .put(
                session,
                make_block(b"hi there", BlockKind::AssistantMessage, 3, 3),
            )
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "what's next?".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![sys_id, user_id, asst_id],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // System (with `be brief`), then user, assistant, then task user.
        let roles: Vec<ChatRole> = chat.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                ChatRole::System,
                ChatRole::User,
                ChatRole::Assistant,
                ChatRole::User
            ],
        );
        assert!(chat.messages[0].content.contains("be brief"));
        assert_eq!(chat.messages[1].content, "hello?");
        assert_eq!(chat.messages[2].content, "hi there");
        assert_eq!(chat.messages[3].content, "what's next?");
        assert!(chat.input_tokens.0 > 0);
    }

    #[test]
    fn pack_chat_folds_system_into_user_when_unsupported() {
        use llm386_core::ChatRole;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"be brief", BlockKind::System, 1, 1))
            .unwrap();
        let mut p = profile(1_000);
        p.supports_system_role = false;
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "go".into(),
            model: p,
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![sys_id],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // No system role; first message is user with the system content.
        assert!(chat.messages.iter().all(|m| m.role != ChatRole::System));
        assert_eq!(chat.messages[0].role, ChatRole::User);
        assert!(chat.messages[0].content.contains("be brief"));
    }

    #[test]
    fn pack_chat_emits_tool_role_for_tool_results() {
        use llm386_core::ChatRole;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let tool_id = store
            .put(
                session,
                make_block(b"{\"result\": 42}", BlockKind::ToolResult, 1, 1),
            )
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "x".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![tool_id],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        let tool_msg = chat
            .messages
            .iter()
            .find(|m| m.role == ChatRole::Tool)
            .unwrap();
        assert_eq!(tool_msg.content, "{\"result\": 42}");
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config { cases: 12, ..proptest::test_runner::Config::default() })]

        /// Pack must be byte-deterministic across two calls with the
        /// same store state, request, and plan — this is the property
        /// the trace layer relies on for replay.
        #[test]
        fn pack_is_byte_deterministic(
            n_blocks in 0u64..10,
            task_seed in 0u32..100,
        ) {
            use llm386_pager::GreedyPager;
            let (store, _dir, tok) = setup();
            let session = SessionId(1);
            for i in 0..n_blocks {
                let bytes = format!("p{task_seed}/{i} content");
                store
                    .put(session, make_block(bytes.as_bytes(), BlockKind::Fact, i, u128::from(i)))
                    .unwrap();
            }
            let pager = GreedyPager::new(store.clone(), tok.clone());
            let packer = SimplePacker::new(store, tok);
            let request = PageRequest {
                session_id: session,
                task: format!("task {task_seed}"),
                model: profile(10_000),
                required_blocks: vec![],
            };
            let plan = pager.page(request.clone()).unwrap();
            let a = packer.pack(&request, &plan).unwrap();
            let b = packer.pack(&request, &plan).unwrap();
            proptest::prop_assert_eq!(a.rendered, b.rendered);
            proptest::prop_assert_eq!(a.input_tokens, b.input_tokens);
        }

        /// If pack succeeds, its rendered prompt must fit within the
        /// model's input budget. (Covers the BudgetExceeded short-
        /// circuit and the surrounding accounting.)
        #[test]
        fn successful_pack_fits_budget(
            n_blocks in 0u64..10,
            budget in 80u32..2_000,
        ) {
            use llm386_pager::GreedyPager;
            let (store, _dir, tok) = setup();
            let session = SessionId(1);
            for i in 0..n_blocks {
                let bytes = format!("block {i} content");
                store
                    .put(session, make_block(bytes.as_bytes(), BlockKind::Fact, i, u128::from(i)))
                    .unwrap();
            }
            let pager = GreedyPager::new(store.clone(), tok.clone());
            let packer = SimplePacker::new(store, tok);
            let request = PageRequest {
                session_id: session,
                task: "task".into(),
                model: profile(budget),
                required_blocks: vec![],
            };
            let plan = pager.page(request.clone()).unwrap();
            if let Ok(prompt) = packer.pack(&request, &plan) {
                proptest::prop_assert!(
                    prompt.input_tokens.0 <= budget,
                    "input_tokens={} > budget={}",
                    prompt.input_tokens.0,
                    budget,
                );
            }
        }
    }
}
