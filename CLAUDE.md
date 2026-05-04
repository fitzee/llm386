# CLAUDE.md

## Project

LLM386 is a Rust-based context virtualization runtime for LLM agents.

It treats the model as a stateless inference function and manages the external state required to make agents fast, reliable, and model-agnostic.

Core idea:

persistent memory + retrieval + paging + packing
→ bounded model-specific context window
→ stateless LLM call

The runtime should feel like EMM386 for LLM context: a fast pager that maps a larger external memory space into a smaller active working set.

---

## Primary goals

- Fast context assembly.
- Deterministic prompt packing.
- Model-agnostic context sizing.
- LMDB-backed persistent block storage.
- Explicit state management outside the model.
- Support for multiple retrieval strategies.
- Clean observability of what was injected and why.
- Rust-first implementation with tight control over memory, latency, and data layout.

---

## Non-goals

Do not build a chatbot UI.

Do not hide state inside prompts.

Do not rely on the model as the source of truth.

Do not make the initial version distributed.

Do not optimize for speculative agent magic. Optimize for inspectable, deterministic context management.

---

## Architecture

llm386
├── store       // LMDB-backed block and metadata storage
├── model       // model profiles, tokenizer configs, budgets
├── pager       // working-set selection
├── packer      // deterministic prompt construction
├── retrieve    // lexical/vector/graph retrieval adapters
├── compress    // summaries and structured reductions
├── trace       // observability and replay
└── api         // library/service interface

---

## Core abstractions

### ContextBlock

A context block is the atomic unit of memory.

pub struct ContextBlock {
    pub id: BlockId,
    pub kind: BlockKind,
    pub bytes: Vec<u8>,
    pub token_counts: TokenCounts,
    pub priority: f32,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub provenance: Provenance,
    pub hash: ContentHash,
}

Block kinds:

pub enum BlockKind {
    System,
    UserMessage,
    AssistantMessage,
    ToolResult,
    Summary,
    Fact,
    DocumentChunk,
    Plan,
    State,
    Trace,
}

---

### ModelProfile

Model-specific context constraints.

pub struct ModelProfile {
    pub name: String,
    pub max_context_tokens: usize,
    pub reserved_output_tokens: usize,
    pub safety_margin_tokens: usize,
    pub tokenizer: TokenizerId,
    pub supports_system_role: bool,
    pub supports_tools: bool,
}

Effective budget:

input_budget =
    max_context_tokens
    - reserved_output_tokens
    - safety_margin_tokens

---

### PackedPrompt

The final prompt sent to the model.

pub struct PackedPrompt {
    pub model: String,
    pub input_tokens: usize,
    pub blocks: Vec<PackedBlock>,
    pub rendered: String,
}

Every packed prompt must be traceable back to the source blocks.

---

## Memory model

Use a tiered working-set model:

PINNED  - always included
HOT     - likely included
WARM    - retrieved when relevant
COLD    - summarized or omitted
EVICTED - available only by explicit retrieval

The model never owns durable state.

State transitions are explicit:

LLM output
→ parsed event
→ validation
→ committed block/state update

---

## Storage

Use LMDB for the hot persistent store.

Suggested databases:

blocks_by_id
blocks_by_hash
blocks_by_session
blocks_by_kind
blocks_by_time
edges_by_block
summaries_by_scope
token_counts_by_model
trace_by_call

Use content hashes for dedupe.

Use precomputed token counts per tokenizer.

Avoid JSON in hot paths unless debugging.

Prefer compact binary encodings such as bincode, postcard, rkyv, or capnp.

---

## Retrieval

Retrieval should be pluggable.

Initial retrieval modes:

lexical
recency
pinned state
graph neighbors
manual block IDs

Later retrieval modes:

embedding ANN
hybrid search
reranking
external vector DB

Candidate scoring should include:

relevance
recency
authority
priority
dependency importance
token cost
redundancy penalty
staleness penalty

---

## Pager

The pager selects blocks for a specific task and model budget.

Inputs:

pub struct PageRequest {
    pub session_id: SessionId,
    pub task: String,
    pub model: ModelProfile,
    pub required_blocks: Vec<BlockId>,
}

Output:

pub struct PagePlan {
    pub selected: Vec<BlockId>,
    pub omitted: Vec<OmittedBlock>,
    pub estimated_tokens: usize,
}

The pager should explain why blocks were selected or omitted.

---

## Packer

The packer converts selected blocks into a deterministic prompt.

Preferred section order:

1. System / hard constraints
2. Current task
3. Active state
4. Current plan
5. Relevant retrieved memory
6. Tool results
7. Recent transcript
8. Optional background

Use section budgets. Do not simply sort all blocks by relevance.

Example section budget:

system:    fixed
task:      fixed
state:     15%
recent:    20%
retrieved: 45%
tools:     15%
slack:      5%

The packer must never silently exceed the model budget.

---

## Performance rules

- Pre-tokenize blocks.
- Cache token counts by tokenizer.
- Cache embeddings outside the hot path.
- Cache summaries outside the hot path.
- Use mmap-friendly reads.
- Avoid unnecessary allocation during packing.
- Avoid cloning large block payloads.
- Keep active session state hot in memory.
- Use blake3 for content hashing.
- Use tracing for observability.
- Benchmark paging and packing separately.

---

## Rust conventions

Use:

anyhow or thiserror
tracing
serde only at boundaries
tokio only where async is useful
criterion for benchmarks
proptest for invariants

Prefer clear domain types over raw strings.

Example:

pub struct BlockId(pub u128);
pub struct SessionId(pub u128);
pub struct TokenCount(pub usize);

Avoid global mutable state.

Separate traits from implementations.

Keep hot-path APIs explicit and allocation-aware.

---

## Initial milestone

Build a single-node embedded library.

Minimum viable flow:

create session
put blocks
define model profile
page context
pack prompt
record trace

No network service required initially.

---

## Suggested first crate layout

crates/
  llm386-core/
  llm386-store-lmdb/
  llm386-tokenizer/
  llm386-pager/
  llm386-packer/
  llm386-trace/
  llm386-cli/

---

## First CLI target

llm386 init ./memory
llm386 put --session demo --kind user-message ./input.txt
llm386 profile add gpt-4.1 --context 128000 --reserve-output 4000
llm386 page --session demo --model gpt-4.1 --task "answer the user"
llm386 pack --session demo --model gpt-4.1

---

## Design principle

The LLM should receive exactly the state it needs, no more and no less.

The runtime owns memory.

The model consumes a mapped working set.

That is LLM386.

