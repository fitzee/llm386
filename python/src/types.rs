//! PyO3 wrapper types that cross the FFI boundary.

use pyo3::prelude::*;

use llm386_core::{
    ChatMessage as RustChatMessage, ChatRole, ContextBlock as RustBlock,
    ModelProfile as RustModelProfile, OmittedBlock as RustOmitted, PagePlan as RustPagePlan,
    Provenance as RustProvenance,
};

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct Provenance {
    pub source: Option<String>,
    pub parents: Vec<String>,
    pub labels: Vec<String>,
}

#[pymethods]
impl Provenance {
    fn __repr__(&self) -> String {
        format!(
            "Provenance(source={:?}, parents={:?}, labels={:?})",
            self.source, self.parents, self.labels,
        )
    }
}

impl Provenance {
    pub fn from_rust(p: RustProvenance) -> Self {
        Self {
            source: p.source,
            parents: p.parents.into_iter().map(|id| format!("{id}")).collect(),
            labels: p.labels,
        }
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct ContextBlock {
    pub id: String,
    pub kind: String,
    pub body: Vec<u8>,
    pub priority: f32,
    pub created_at: u64,
    pub updated_at: u64,
    pub hash: String,
    pub provenance: Provenance,
}

#[pymethods]
impl ContextBlock {
    fn __repr__(&self) -> String {
        format!(
            "ContextBlock(id={:?}, kind={:?}, body=<{} bytes>, priority={})",
            self.id,
            self.kind,
            self.body.len(),
            self.priority,
        )
    }
}

impl ContextBlock {
    pub fn from_rust(b: RustBlock) -> Self {
        let hash_hex = b.hash.0.iter().fold(String::with_capacity(64), |mut acc, byte| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{byte:02x}");
            acc
        });
        Self {
            id: format!("{}", b.id),
            kind: kind_to_str(b.kind).to_string(),
            body: b.bytes,
            priority: b.priority,
            created_at: b.created_at.0,
            updated_at: b.updated_at.0,
            hash: hash_hex,
            provenance: Provenance::from_rust(b.provenance),
        }
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct OmittedBlock {
    pub block_id: String,
    pub reason: String,
    pub score: f32,
}

#[pymethods]
impl OmittedBlock {
    fn __repr__(&self) -> String {
        format!(
            "OmittedBlock(block_id={:?}, reason={:?}, score={})",
            self.block_id, self.reason, self.score,
        )
    }
}

impl OmittedBlock {
    pub fn from_rust(o: RustOmitted) -> Self {
        Self {
            block_id: format!("{}", o.block_id),
            reason: format!("{:?}", o.reason),
            score: o.score,
        }
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PagePlan {
    pub selected: Vec<String>,
    pub omitted: Vec<OmittedBlock>,
    pub estimated_tokens: u32,
}

#[pymethods]
impl PagePlan {
    fn __repr__(&self) -> String {
        format!(
            "PagePlan(selected=[{} ids], omitted=[{} blocks], estimated_tokens={})",
            self.selected.len(),
            self.omitted.len(),
            self.estimated_tokens,
        )
    }
}

impl PagePlan {
    pub fn from_rust(plan: RustPagePlan) -> Self {
        Self {
            selected: plan.selected.into_iter().map(|id| format!("{id}")).collect(),
            omitted: plan.omitted.into_iter().map(OmittedBlock::from_rust).collect(),
            estimated_tokens: plan.estimated_tokens.0,
        }
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[pymethods]
impl ChatMessage {
    fn __repr__(&self) -> String {
        let preview: String = self.content.chars().take(40).collect();
        format!("ChatMessage(role={:?}, content={:?})", self.role, preview)
    }
}

impl ChatMessage {
    pub fn from_rust(m: RustChatMessage) -> Self {
        Self { role: chat_role_to_str(m.role).to_string(), content: m.content }
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PackResult {
    pub rendered: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub trace_id: Option<String>,
}

#[pymethods]
impl PackResult {
    fn __repr__(&self) -> String {
        let mode = if self.messages.is_some() { "chat" } else { "rendered" };
        format!(
            "PackResult(mode={mode:?}, trace_id={:?})",
            self.trace_id,
        )
    }
}

#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct ModelProfile {
    pub name: String,
    pub max_context_tokens: u32,
    pub reserved_output_tokens: u32,
    pub safety_margin_tokens: u32,
    pub tokenizer: String,
    pub supports_system_role: bool,
    pub supports_tools: bool,
}

#[pymethods]
impl ModelProfile {
    fn __repr__(&self) -> String {
        format!(
            "ModelProfile(name={:?}, ctx={}, out={}, tokenizer={:?})",
            self.name, self.max_context_tokens, self.reserved_output_tokens, self.tokenizer,
        )
    }
}

impl ModelProfile {
    pub fn from_rust(p: RustModelProfile) -> Self {
        Self {
            name: p.name,
            max_context_tokens: p.max_context_tokens,
            reserved_output_tokens: p.reserved_output_tokens,
            safety_margin_tokens: p.safety_margin_tokens,
            tokenizer: p.tokenizer.as_str().to_string(),
            supports_system_role: p.supports_system_role,
            supports_tools: p.supports_tools,
        }
    }
}

const fn kind_to_str(kind: llm386_core::BlockKind) -> &'static str {
    match kind {
        llm386_core::BlockKind::System => "System",
        llm386_core::BlockKind::UserMessage => "UserMessage",
        llm386_core::BlockKind::AssistantMessage => "AssistantMessage",
        llm386_core::BlockKind::ToolResult => "ToolResult",
        llm386_core::BlockKind::Summary => "Summary",
        llm386_core::BlockKind::Fact => "Fact",
        llm386_core::BlockKind::DocumentChunk => "DocumentChunk",
        llm386_core::BlockKind::Plan => "Plan",
        llm386_core::BlockKind::State => "State",
        llm386_core::BlockKind::Trace => "Trace",
    }
}

const fn chat_role_to_str(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}
