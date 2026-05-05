# Frequently asked questions

## Quick index

- What does the model see? → [How it works](#how-it-works)
- How fast is it? → [Performance and sizing](#performance-and-sizing)
- Will it fill the entire context window? → [Does the runtime pack only what's needed?](#does-the-runtime-pack-only-whats-needed-or-does-it-fill-the-models-context-window)
- How do I delete data? → [Data lifecycle](#data-lifecycle)
- How do sessions work? → [Sessions and multi-tenancy](#sessions-and-multi-tenancy)
- How does retrieval work? → [Retrieval and RAG](#retrieval-and-rag)
- How do tools integrate? → [How do MCP servers and tools work with this?](#how-do-mcp-servers-and-tools-work-with-this-do-tool-schemas-get-committed-to-memory-like-other-facts)
- How can things go wrong? → [Failure modes](#failure-modes)

## Full table of contents

- [Naming and motivation](#naming-and-motivation)
  - [The README mentions EMM386. What was it, and how is LLM386 similar?](#the-readme-mentions-emm386-what-was-it-and-how-is-llm386-similar)
- [How it works](#how-it-works)
  - [How does the context in LLM386 get exposed to the LLM model?](#how-does-the-context-in-llm386-get-exposed-to-the-llm-model)
- [Performance and sizing](#performance-and-sizing)
  - [How much latency does this add to my agent?](#how-much-latency-does-this-add-to-my-agent)
  - [How big can the memory store get?](#how-big-can-the-memory-store-get)
  - [Does the runtime pack only what's needed, or does it fill the model's context window?](#does-the-runtime-pack-only-whats-needed-or-does-it-fill-the-models-context-window)
- [Data lifecycle](#data-lifecycle)
  - [If it gets corrupted, can I rebuild it somehow?](#if-it-gets-corrupted-can-i-rebuild-it-somehow)
  - [Is it ever a good idea to purge memory?](#is-it-ever-a-good-idea-to-purge-memory)
  - [Legal/security asked me to remove customer data. How do I find and remove it?](#legalsecurity-asked-me-to-remove-customer-data-how-do-i-find-and-remove-it)
  - [How do I migrate "memories" from an existing memory subsystem?](#how-do-i-migrate-memories-from-an-existing-memory-subsystem)
- [Sessions and multi-tenancy](#sessions-and-multi-tenancy)
  - [Does it support multiple user sessions?](#does-it-support-multiple-user-sessions)
  - [A user has many sessions. How does memory span them?](#a-user-has-many-sessions-how-does-memory-span-them)
  - [Can I assert facts that are available to every session?](#can-i-assert-facts-that-are-available-to-every-session)
  - [Can multiple agents share the same memory store?](#can-multiple-agents-share-the-same-memory-store)
- [Retrieval and RAG](#retrieval-and-rag)
  - [Score normalization](#score-normalization)
  - [How does it work with RAG, and does it store links to blobs/documents?](#how-does-it-work-with-rag-and-does-it-store-links-to-blobsdocuments)
  - [How do I write a custom retriever for Pinecone, and wire it in?](#how-do-i-write-a-custom-retriever-for-pinecone-and-wire-it-in)
  - [How do MCP servers and tools work with this?](#how-do-mcp-servers-and-tools-work-with-this-do-tool-schemas-get-committed-to-memory-like-other-facts)
- [Failure modes](#failure-modes)
- [Comparison and adoption](#comparison-and-adoption)
  - [Why should I use this over other approaches?](#why-should-i-use-this-over-other-approaches)

---

## Naming and motivation

### The README mentions EMM386. What was it, and how is LLM386 similar?

EMM386 ("Expanded Memory Manager for the 386") was a DOS-era memory manager from the late 1980s and early 1990s. The constraint at the time: DOS programs could only directly address the first 640 KB of memory ("conventional memory"), even on machines with several megabytes installed. EMM386 used the 80386 CPU's address-translation hardware to page chunks of that larger memory through a small fixed window inside the 640 KB region. Programs that knew how to ask got effectively unlimited memory — through a peephole.

LLM386 does the same trick for LLM context windows.

| EMM386                                 | LLM386                                              |
|----------------------------------------|-----------------------------------------------------|
| Conventional memory: bounded (640 KB)  | Context window: bounded (32 K, 128 K, 1 M tokens)   |
| External memory: gigabytes available   | External memory: persistent block store             |
| Page frame: a small 64 KB window       | The model's prompt: a few thousand tokens at a time |
| EMS pages chunks in/out on demand      | Pager selects blocks for each call                  |
| Caller sees a single flat view         | Model sees a single flat prompt                     |
| Hardware does the address translation  | The pager + packer do the assembly                  |

In both cases the underlying constraint never moves — DOS still only sees 640 KB, the model still only gets its context window. The trick is making a much larger external memory available *through* that window, by paging only what's relevant for the current operation.

The 386 in the name is also a tip of the hat to the era. There is no special significance to the number beyond the homage.

---

## How it works

### How does the context in LLM386 get exposed to the LLM model?

**Core mental model.** LLM386 does not extend the model's context window. It constructs a bounded working set for each call by selecting, ordering, and rendering blocks from a larger persistent memory. Every model invocation is independent; continuity is achieved by recomputing the working set each time.

LLM386 never calls the model itself. It produces a `PackedPrompt` (or a `ChatPrompt`) and hands it back to your agent code, which then makes the actual API call. This is a deliberate boundary: the runtime owns context assembly, your code owns inference.

There are two output shapes, and which one you use depends on the API you're calling.

**Single-string mode (`pack`).** `Store.pack(...)` returns a `PackedPrompt` with a `rendered: String` field — one Markdown-shaped document with section headers in a fixed canonical order:

```
## System
You are a careful expert.

## Task
Explain Australia's history.

## State
<active state blocks>

## Plan
<plan blocks>

## Retrieved memory
<facts and document chunks the retrievers surfaced>

## Tools
<tool result blocks>

## Recent
<recent user/assistant turns>

## Background
<low-priority context>
```

The order is hardcoded and the tokenizer is the model's own, so the rendered string is byte-identical for the same inputs (this is what makes traces replay-cleanly). You hand this string to a completion-style API as a single prompt:

```python
result = store.pack(session=1, model="gpt-4o", task="...")
response = openai.completions.create(model="gpt-4o", prompt=result.rendered)
```

**Chat-message mode (`pack_chat`).** Most modern LLM APIs are chat-shaped, not completion-shaped. `Store.pack(..., chat=True)` returns a `ChatPrompt` with a `messages: List[ChatMessage]` field — a list of role-tagged messages ready to drop into a chat-completion API. The packer maps blocks to roles like this:

| `BlockKind`        | Chat role     |
|--------------------|---------------|
| `System`           | `system`      |
| `UserMessage`      | `user`        |
| `AssistantMessage` | `assistant`   |
| `ToolResult`       | `tool`        |
| Other kinds (`Fact`, `DocumentChunk`, `Plan`, `State`, `Summary`, ...) | `user`, with section headers preserved as Markdown inside the body |

The current task ends up as the final `user` message so it's positioned where the model expects "what to do next." If the model's `ModelProfile` has `supports_system_role = false` (some local models, some Anthropic configurations), the packer folds system content into the first `user` message instead — so you don't have to think about per-model quirks.

```python
result = store.pack(session=1, model="gpt-4o", task="...", chat=True)
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": m.role, "content": m.content} for m in result.messages],
)
```

**What the model never sees.** The rule is absolute: anything not serialized into the final prompt (string or messages) does not exist to the model. Store state ≠ model state. Concretely:

- *Block ids, hashes, provenance.* These live in the store for your benefit (replay, audit, deletion), not the model's. The packer renders only the bytes.
- *Edges.* Typed edges between blocks shape *which* blocks the pager picks, but the edges themselves aren't serialized into the prompt.
- *Tool schemas.* These belong in the API's `tools` parameter, not the prompt body. See ["How do MCP servers and tools work with this?"](#how-do-mcp-servers-and-tools-work-with-this-do-tool-schemas-get-committed-to-memory-like-other-facts) for why.

**Determinism guarantee.** Same blocks + same model profile + same task ⟹ same rendered string (or same chat message list) ⟹ same prompt hash in the trace. Determinism holds *only* if all four of the following hold:

- block ordering is stable (controlled by `BlockId` ordering inside each section);
- tokenizer version is identical (the `tokenizer_version` field on the trace catches drift);
- packer logic is unchanged (rebuilding the binary against a different `SimplePacker` breaks byte-equality);
- retriever outputs are deterministic (random-seeded ANN, time-of-day-keyed retrievers, or a network retriever returning paginated results in different orders all break this).

Under those conditions `llm386 trace diff` is meaningful and you can reproduce "what did the model see two weeks ago when it gave that answer?" exactly. Outside those conditions the trace still tells you what the model saw on that specific call, but a re-run is not guaranteed to match.

**A worked packing example.** Given:

- 2 system blocks
- 1 task
- 3 retrieved facts
- 2 recent messages

the packed prompt becomes:

```
## System
<system block 1>
<system block 2>

## Task
<task>

## Retrieved memory
<fact 1>
<fact 2>
<fact 3>

## Recent
<recent message 1>
<recent message 2>

Total tokens: 2413 / 8192
```

Empty sections are omitted. Token totals are reported in the manifest header that `llm386 pack` prints to stderr (and on the `PackedPrompt.input_tokens` field programmatically).

---

## Performance and sizing

### How much latency does this add to my agent?

Specific to your hardware and session size, but here is a baseline from the bundled benchmarks on a 2024 Apple Silicon laptop:

- Pager: 141 µs for 100 blocks, 1.4 ms for 1000 blocks (linear in N).
- Tokenizer (cl100k_base): 56 µs for 2.7 KB, 1.2 ms for 45 KB.
- LMDB put: low single-digit ms.
- LMDB get: sub-millisecond.

For a typical chat-style turn (50 to 100 blocks selected, a few KB of rendered prompt) end-to-end `pack` from the Rust library lands in the 5 to 10 ms range. The model API call itself dominates by orders of magnitude.

The Python SDK (PyO3 bindings) is in-process and runs at near-native speed. The previous CLI-shelling SDK added 30 to 50 ms per call from process startup; that path is no longer the default.

If you enable a network-backed summarizer (Anthropic) or embedder (OpenAI), those add their own latency on top of the runtime.

### How big can the memory store get?

The default LMDB `map_size` is 64 GiB. That is a virtual reservation, not an allocation, so the on-disk footprint only grows as you write. Concrete capacity depends on your average block size:

- Chat-style blocks (~200 bytes each): hundreds of millions of blocks.
- Document chunks (~2 KB each): tens of millions.
- Embeddings (1536-dim float32, ~6 KB each): roughly 10 million.

If you need more, pass a larger `map_size` to `StoreConfig`. LMDB's hard ceiling is your platform's address space (effectively unbounded on 64-bit hosts).

There is no built-in size readout yet. `du -sh ./store` is the easy answer.

### Does the runtime pack only what's needed, or does it fill the model's context window?

The pager filters by relevance, but treats your section budgets as ceilings — when there are enough relevant blocks, it fills them up to the ceiling. Filters that drop content automatically:

- **Off-topic blocks.** Retrievers score every block against the current task. Low-scoring blocks don't make the cut for variable sections regardless of available budget.
- **Old blocks.** Recency decay (exponential half-life) deweights stale content unless something else (BM25 hit, manual pin, edge dependency) lifts it.
- **Redundant blocks.** Jaccard overlap on word sets per section drops near-duplicates to `OmissionReason::Redundant` even when budget is available.
- **Low-priority blocks.** Anything near `priority = 0.0` stays out unless a retriever surfaces it.

The pager does not automatically shrink the prompt below what fits relevant content. With 500 turns of session history and a `Recent` budget of 20% of 95K tokens, the pager will pack roughly 19K of recent messages even when the current query would do fine with 3K. The relevance filter runs inside an allocation that defaults to "fill what you've allocated."

For tighter prompts the knobs are:

- **Tighten section fractions** in `llm386.toml` (`SectionBudgetTable`). Drop `Recent` from 0.20 to 0.05 when the task doesn't benefit from chat history.
- **Set `score_threshold`** in the scoring policy. Drops anything below it even when budget is available.
- **Lower `retriever.limit`.** If each retriever returns 50 candidates and only 5 are needed, cap them at 5.
- **Use the `Slack` section.** Reserved headroom that is never filled — set to 0.30 to leave 30% of the input budget hard-unused.

The trade-off matters. More context is slower (linear in input tokens for time-to-first-token, more once cache misses are involved), more expensive (input tokens cost money), and often produces worse output quality (lost-in-the-middle, attention dilution, instruction drift). For focused tasks — single-fact Q&A, structured extraction — a tight 2K prompt typically beats a 50K kitchen-sink one. Broad tasks (summarization, multi-document reasoning, code understanding) benefit from bigger windows. LLM386 does not pick for you. `llm386 trace diff` between a tight-budget run and a loose-budget run is the cheap way to find the right operating point for a given task.

---

## Data lifecycle

### If it gets corrupted, can I rebuild it somehow?

LMDB itself is crash-safe. It uses a B+ tree with copy-on-write writes, so a pulled power cord or a `kill -9` during a transaction leaves the store readable; the in-flight transaction is just rolled back.

Application-level corruption (your own code or a future schema migration writing the wrong bytes) is partially recoverable:

- The schema version stamped in the `meta` table prevents older code from opening a newer-format store.
- Each block carries its content hash, so corrupt block bodies are detectable.
- `blocks_by_hash` and `blocks_by_session` are indexes derived from `blocks_by_id`, so they can in principle be rebuilt from the primary table.

For corruption you can detect and fix yourself, the runtime ships two subcommands:

- `llm386 verify --store ./store` walks every block in the primary table, recomputes its content hash, and checks the hash and session indexes for consistency. Read-only. Returns a non-zero exit code on any inconsistency.
- `llm386 repair --store ./store --yes` rebuilds derivable state (the hash index) from the primary table and removes orphan session entries that point at missing blocks. Blocks whose stored hash doesn't match their bytes are left in place and reported — those need human review, not silent rewrite.

Beyond what those tools cover, the honest answer is: keep backups. Copying the store directory while no writer is active is a valid backup.

### Is it ever a good idea to purge memory?

Yes, in three cases:

1. **Compliance.** GDPR right-to-be-forgotten and similar regulations require deletion. The pager respects what is in the store, so removing blocks is the right answer.
2. **Hygiene.** Outdated facts (an old address, a stale API endpoint) keep getting retrieved if they stay in the store. Deleting them is more reliable than adding a "this is wrong, ignore it" note that the model will sometimes ignore back.
3. **Privacy boundaries.** In multi-tenant systems where one user's data should never reach another user's context.

Don't purge to "save tokens" or "fit the context window". The pager and section budgets already drop what doesn't fit; that's a runtime concern, not a storage concern. For long-running sessions, summarize old turns instead (the COLD-tier behavior in the pager is built for exactly this).

Blocks are immutable by design and the runtime doesn't expose a `delete` API today. Workarounds are covered in the next question.

### Legal/security asked me to remove customer data. How do I find and remove it?

Two-step process: find the offending blocks, then physically remove them.

**Finding the blocks.** Walk every block, check the body, record matches. From Python:

```python
from llm386 import Store

store = Store("./store")
needle = b"sensitive-substring"
hits = []

for session in store.list_sessions():
    # Page with a generous budget to surface every block in the session.
    plan = store.page(int(session, 16), model="gpt-4.1", task="")
    for block_id in plan.selected:
        block = store.show(block_id)
        if needle in block.body:
            hits.append((session, block_id, block.kind, block.created_at))

for session, bid, kind, ts in hits:
    print(f"{session}\t{bid}\t{kind}\t{ts}")
```

For larger stores, the same shape works directly against the Rust library and skips the per-call FFI.

**Removing the blocks.** Use the `purge` subcommand (or its Python equivalent). Both are destructive and require explicit confirmation:

```
llm386 purge --store ./store --block <block-id> --yes
llm386 purge --store ./store --session <session-id> --yes
```

```python
store.delete(block_id)             # returns True if anything was removed
store.purge_session(session_id)    # returns count of blocks affected
```

`delete` removes the block from the primary table, the content-hash index, and every session that referenced it. `purge_session` removes the entire session's references; blocks left orphaned by that (no other session points at them) are then dropped from the primary table and the hash index too. Both operations run in a single LMDB write transaction.

For audit trail, capture `(session, block_id, kind, hash, created_at)` for every hit before you delete, and store that in a separate compliance log. The content hash makes it easy to prove later that a specific block did exist and was removed.

### How do I migrate "memories" from an existing memory subsystem?

The general shape is one-time, idempotent, and per-record:

```python
for record in existing_memory.iter_all():
    store.put(
        session=session_id_for(record),
        kind=map_to_block_kind(record),
        body=record.text,
        priority=record.importance_score,
    )
```

Concrete starting points by source:

- **LangChain memory:** iterate `chat_history.messages`. `HumanMessage` to `user-message`, `AIMessage` to `assistant-message`, `ToolMessage` to `tool-result` (set `parents=[assistant_id]` so the pager keeps them paired).
- **CrewAI long-term memory:** their backend exposes a search API; iterate or page through it and put each entry with kind `fact`.
- **Raw chat logs (JSONL, transcripts):** parse into role-tagged blocks and put each.
- **Vector DB (Pinecone, Qdrant, pgvector, etc.):** put each document as a `document-chunk` block. If you trust the existing vectors, wire a custom `Embedder` that returns them rather than recomputing. If you want fresh embeddings, just point `OpenAiEmbedder` at the new blocks.

Because the store is content-addressable, re-running an import is safe: identical bytes dedup to the same block id, and re-running with new bytes produces a new block without disturbing the old one.

---

## Sessions and multi-tenancy

### Does it support multiple user sessions?

Yes. Every block belongs to a `SessionId` (a 128-bit value), and every read/write API is session-scoped. Common patterns:

- One session per conversation (chat thread).
- One session per agent instance.
- One session per user.

`list_sessions` enumerates everything in a store. Two sessions in the same store are isolated for retrieval and paging, but share content-hash dedup (identical block bytes are stored once regardless of session).

For stronger isolation between tenants (compliance, key separation, regulatory boundaries), open a separate `LmdbStore` per tenant. Each one is its own LMDB env with its own files.

### A user has many sessions. How does memory span them?

The runtime treats every block as belonging to exactly one `SessionId`, so cross-session memory is something you opt into rather than something that happens by default. Cross-session retrieval is a *retrieval policy*, not a storage capability — blocks remain physically scoped to their original session, and a custom retriever decides whether to surface blocks from other sessions when assembling a working set. Sessions define storage boundaries, not logical memory boundaries.

Three patterns work:

**One session per user, many turns.** Use a single `SessionId` for everything that user does and let the pager surface relevant turns from across the whole history. Simplest model. Works well when "user" and "agent" are the same conceptual thing.

**Many sessions per user, plus a "user-shared" session.** Pick a stable session id derived from the user (`SessionId(user_id_hash)` for example). Write the user's persistent facts to that session. Each conversation gets its own session id. Add a custom retriever that *also* reads from the user-shared session:

```python
class CrossSessionRetriever:
    name = "cross-session"

    def __init__(self, store, shared_session: int, score: float = 0.5):
        self.store = store
        self.shared_session = shared_session
        self.score = score

    def retrieve(self, session, task, limit):
        # Use page() with a giant budget to enumerate the shared
        # session's blocks. (Until a list_blocks method lands.)
        plan = self.store.page(self.shared_session, "gpt-4.1", task)
        return [(bid, self.score) for bid in plan.selected[:limit]]

store.add_python_retriever(CrossSessionRetriever(store, shared_session=0))
```

The pager fans out across all configured retrievers, so this composes with the built-ins.

**Many sessions per user, with the same store.** Simplest variant: every session is isolated by default, and you write the same fact into each session as the agent learns it. No retriever code, but you pay the storage cost (the content-hash dedup keeps it from being awful — same facts share one block id and one body, only the per-session pointer is duplicated).

Pick based on whether facts genuinely are user-scoped (option 1 or 2) or genuinely are conversation-scoped (option 3).

### Can I assert facts that are available to every session?

Yes, with the same patterns. The most common shape is a "global facts" session — pick a sentinel `SessionId` (for example, `SessionId(0)` or a hash of `"global"`) and write facts there:

```python
GLOBAL = 0
store.put(session=GLOBAL, kind="fact", body="The user's name is Mira.")
store.put(session=GLOBAL, kind="fact", body="Always answer in metric units.")
```

Then add a retriever that always reads from that session, regardless of which session the current call is targeting:

```python
class GlobalFactsRetriever:
    name = "global-facts"

    def __init__(self, store, global_session=0):
        self.store = store
        self.global_session = global_session

    def retrieve(self, session, task, limit):
        plan = self.store.page(self.global_session, "gpt-4.1", task)
        return [(bid, 0.7) for bid in plan.selected[:limit]]

store.add_python_retriever(GlobalFactsRetriever(store))
```

The score is a knob: lower it if the global facts should only surface when nothing else is relevant, raise it to bias every prompt toward including them. The pager merges by max score per `BlockId`, so a global fact that's also retrieved by `RecencyRetriever` in the current session keeps the higher of the two scores.

This pattern works at any scope: a global "company-wide instructions" session, a per-user "user profile" session, a per-team "team conventions" session. They're all just specially-named session ids.

### Can multiple agents share the same memory store?

Yes, with two flavors.

**In-process.** Multiple agents in the same Python or Rust process can share a single store cheaply. Open it once, clone the `Arc<LmdbStore>` (or pass the same Python `Store` instance) to every agent. LMDB's MVCC means readers never block each other; writes within one process serialize through an internal mutex. This is the default working assumption.

**Cross-process.** LMDB is designed for multi-process access too. Each process opens its own `LmdbStore` at the same path; committed writes from one are immediately visible to readers in the others. File-level locking serializes writes across processes. The major caveat: this only works on local filesystems with proper mmap semantics. NFS, some networked filesystems, and certain container overlay filesystems don't behave correctly under LMDB's mmap model. POSIX local filesystems (ext4, xfs, APFS) are fine.

**Consistency model.** Reads observe a consistent snapshot taken at transaction start: a long read transaction will not see writes that commit after it began, even from other processes. Writes become visible atomically after commit — there is no partial visibility of a multi-key write. Practically: a `pack` call that opens its read transaction at T sees the store as it was at T, regardless of what concurrent writers are doing.

**Two flavors of "sharing"** are worth distinguishing:

- *Sharing the store, with separate sessions.* Each agent owns one or more `SessionId`s. They never see each other's blocks unless a custom retriever reads across sessions. This is the right pattern for a multi-agent system where each agent has its own memory scope.
- *Sharing the store, with overlapping sessions.* Multiple agents read and write the same `SessionId`. They see every other agent's blocks. This is the right pattern for a "team of agents working on one problem" topology, with the session as the shared workspace.

For strong isolation between tenants (different customers, regulatory boundaries), use a separate store per tenant rather than a separate session. Each store is its own LMDB env, its own files, its own permission boundary.

---

## Retrieval and RAG

### Score normalization

All retriever scores must be normalized to `[0, 1]`. If you mix scoring systems (BM25 raw scores, cosine similarity, recency decay), normalize them inside each retriever before returning candidates. The pager merges by `BlockId` with max-score wins; it assumes comparability and does not fix mismatched scales. A retriever that returns scores in `[0, 100]` will silently drown out one returning `[0, 1]`. If you don't know the right normalization for your retriever, clamp aggressively (`score.clamp(0.0, 1.0)`) and tune the weight upward later.

### How does it work with RAG, and does it store links to blobs/documents?

Both are options.

**Storing the full document.** LMDB handles binary content fine. For small to medium documents (PDFs of a few MB, transcripts), put them as `document-chunk` blocks. For very large documents, chunk them first into ~512-token pieces.

**Storing references.** Put a small block whose body is a URL, a file path, or an external content hash. Write a custom `Packer` that resolves those references at render time. Keeps the store small at the cost of resolve-time latency.

**A vector-RAG flow inside LLM386.** Ingest documents as `document-chunk` blocks. Compute embeddings via `OpenAiEmbedder` (or any custom `Embedder` impl). Use `LinearAnnRetriever` or `HnswAnnRetriever` alongside `RecencyRetriever` and `Bm25Retriever`. The pager merges scores by `BlockId` (max wins), and the packer renders the top hits in the Background section.

If you already run a vector database elsewhere (Pinecone, Qdrant, pgvector), the right integration is a custom `Retriever` that queries it. The next question shows the full shape.

### How do I write a custom retriever for Pinecone, and wire it in?

The `Retriever` trait is three methods: `name`, `retrieve`, and the default-impl `embed_batch`. A Pinecone-backed retriever assumes you've already upserted your block embeddings to a Pinecone index using each block's `BlockId` (32-char hex) as the Pinecone vector id. Then:

```rust
use std::sync::Arc;

use llm386_core::{
    BlockId, Embedder, RetrievalCandidate, RetrievalError, Retriever, SessionId,
};
use serde::{Deserialize, Serialize};

pub struct PineconeRetriever {
    api_key: String,
    index_host: String,        // e.g. "my-index-abc123.svc.us-east-1-aws.pinecone.io"
    embedder: Arc<dyn Embedder>,
    client: reqwest::blocking::Client,
}

impl PineconeRetriever {
    pub fn new(
        api_key: impl Into<String>,
        index_host: impl Into<String>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            index_host: index_host.into(),
            embedder,
            client: reqwest::blocking::Client::new(),
        }
    }
}

impl Retriever for PineconeRetriever {
    fn name(&self) -> &'static str {
        "pinecone"
    }

    fn retrieve(
        &self,
        _session: SessionId,
        task: &str,
        limit: usize,
    ) -> Result<Vec<RetrievalCandidate>, RetrievalError> {
        let vector = self
            .embedder
            .embed(task)
            .map_err(|e| RetrievalError::Failed(format!("embed: {e}")))?;

        #[derive(Serialize)]
        struct Query<'a> {
            vector: &'a [f32],
            #[serde(rename = "topK")]
            top_k: usize,
            #[serde(rename = "includeMetadata")]
            include_metadata: bool,
        }
        #[derive(Deserialize)]
        struct Resp {
            matches: Vec<Match>,
        }
        #[derive(Deserialize)]
        struct Match {
            id: String,
            score: f32,
        }

        let resp: Resp = self
            .client
            .post(format!("https://{}/query", self.index_host))
            .header("Api-Key", &self.api_key)
            .json(&Query { vector: &vector, top_k: limit, include_metadata: false })
            .send()
            .map_err(|e| RetrievalError::Failed(format!("request: {e}")))?
            .error_for_status()
            .map_err(|e| RetrievalError::Failed(format!("status: {e}")))?
            .json()
            .map_err(|e| RetrievalError::Failed(format!("parse: {e}")))?;

        Ok(resp
            .matches
            .into_iter()
            .filter_map(|m| {
                let id = u128::from_str_radix(&m.id, 16).ok()?;
                Some(RetrievalCandidate {
                    block_id: BlockId(id),
                    score: m.score.clamp(0.0, 1.0),
                    source: "pinecone".into(),
                })
            })
            .collect())
    }
}
```

Wire it into the pager alongside the built-ins:

```rust
use llm386_pager::{GreedyPager, RecencyRetriever};
use llm386_retrieve_ann::OpenAiEmbedder;

let embedder = Arc::new(OpenAiEmbedder::from_env()?);
let pinecone = Arc::new(PineconeRetriever::new(
    std::env::var("PINECONE_API_KEY")?,
    "my-index-abc123.svc.us-east-1-aws.pinecone.io",
    embedder,
));

let pager = GreedyPager::new(store, tokenizer)
    .add_retriever(pinecone);
```

The pager fans out across every retriever and merges by `BlockId` (max score wins), so Pinecone sits next to `RecencyRetriever`, `Bm25Retriever`, etc. without further wiring.

Two practical notes:

- **Upsert is your problem, not the runtime's.** Pinecone needs the vectors before `query` returns anything useful. Run a one-time job that walks `list_session(...)`, embeds each block, and upserts to Pinecone with the hex `BlockId` as the vector id. Re-run on new ingest.
- **Score scale.** Pinecone returns cosine similarity in `[-1, 1]` for cosine indexes and dot product for others. Clamp to `[0, 1]` (as above) so the merged scores compose with retrievers that already return that range.

If you'd rather write the retriever in Python, the PyO3 bindings support that too. Define a class with a `name` attribute and a `retrieve(session, task, limit)` method that returns a list of `(block_id_hex, score)` tuples, then register it on the Store:

```python
class PineconeRetriever:
    name = "pinecone"

    def __init__(self, client, index, embedder):
        self.client = client
        self.index = index
        self.embedder = embedder

    def retrieve(self, session, task, limit):
        vec = self.embedder.embed(task)
        matches = self.index.query(vector=vec, top_k=limit, include_metadata=False)
        return [(m["id"], m["score"]) for m in matches["matches"]]

store.add_python_retriever(PineconeRetriever(client, index, embedder))
```

The Rust pager calls back into your Python code on every `page()` / `pack()`. Same composition rules as the Rust retrievers above — the pager merges by `BlockId` (max score wins). Use whichever side fits your stack better; for production-scale latency the Rust path will always win, but the Python path is fine for correctness and easy iteration.

### How do MCP servers and tools work with this? Do tool schemas get committed to memory like other facts?

LLM386 doesn't speak MCP itself, and it doesn't dial out to tools. It is the memory layer your agent reads and writes around the model call. Your agent loop still owns tool dispatch, MCP client connections, and schema discovery. LLM386 just stores the byproducts of those interactions as blocks so they're available on the next turn.

The question of whether tool schemas should live in memory has two answers depending on which schemas you mean:

**Schemas you pass to the model on every call.** OpenAI/Anthropic/etc. take a `tools` array as a separate API parameter, not as prompt text. Don't put those in LLM386. Build the tool list at request time from your MCP client (`list_tools()` per connected server) and pass it to the model alongside whatever LLM386 packed. The model's tool-calling layer is not part of the prompt and not part of the working set.

**Schemas as facts the model needs to reason about.** Sometimes the agent needs to know that a tool exists in order to plan ("if the user asks about Jira, the `create_issue` tool is available"). For that, store a small `Fact` block per tool — name, one-line description, key arguments. Tag it with a label like `tool:jira:create_issue`, set a low priority so it doesn't crowd everything else, and let the lexical/BM25 retriever surface it when the task mentions the relevant domain. Refresh these blocks whenever the MCP server's tool list changes (re-`put` is idempotent thanks to content-hash dedup).

Tool *results* are different and absolutely belong in memory. Store each one as a `ToolResult` block with `parents=[assistant_message_id]` so the pager keeps the call/result paired. Edge-aware paging (`include_parents`) will pull the assistant message back in when only the result was retrieved, and the chat packer renders results with the `tool` role so the model sees them in the right slot. This is the path that most MCP-driven agents care about: the model called `read_file`, the result came back as 4KB of code, you commit it once, and every subsequent turn that retrieves it benefits from the dedup, summarization, and budget-aware packing.

Practical pattern for an MCP-shaped loop:

```python
# 1. Build tool list at request time from your MCP client.
tools = mcp_client.list_tools()  # not stored in LLM386

# 2. Pack memory + send.
result = store.pack(session=sid, model="gpt-4o",
                    task=user_input, chat=True)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": m.role, "content": m.content} for m in result.messages],
    tools=tools,
)

# 3. Persist the assistant turn.
asst_id = store.put(sid, kind="assistant-message",
                    body=response.choices[0].message.content or "")

# 4. Persist each tool result with the assistant turn as parent.
for call in response.choices[0].message.tool_calls or []:
    tool_output = mcp_client.invoke(call)
    store.put(sid, kind="tool-result",
              body=tool_output, parents=[asst_id],
              labels=[f"tool:{call.function.name}"])
```

Tool schemas come from the MCP server. Tool *evidence* — what the tool returned, what the assistant did with it — comes from LLM386. Keep that boundary and the working-set math stays sane.

**Sizing tool outputs.** Tool results may exceed context budgets. A `read_file` against a 200 KB log, a `grep` returning thousands of matches, or a `db.query` returning a wide result set will blow past most section budgets and dominate retrieval on subsequent turns (one giant `ToolResult` block crowds out everything else). Mitigations, in order of preference:

- **Summarize** the result before storing — keep the raw output in a side store and put a short structured summary as the `ToolResult` block.
- **Chunk** the result into multiple `ToolResult` blocks (one per record, page, or logical group) so the pager can pick the relevant subset.
- **Reduce priority** on the raw block (`priority` ≤ 0.1) and rely on retrievers to surface it only when actually relevant; pair with a Summary block for the typical case.

Doing none of the above is fine for small tool outputs (a single API call returning a few hundred tokens) but degrades quickly as outputs grow.

---

## Failure modes

The runtime makes context assembly inspectable; it doesn't prevent you from feeding it nonsense. Common failure modes when running LLM386 in production:

- **Context flooding.** Too many large blocks survive into the working set, the model gets a low-signal prompt, answers degrade. Usually a symptom of unbounded ingest (chat logs, raw tool outputs) and no compensating section budget tightening.
- **Retriever dominance.** One retriever (often a brand-new ANN retriever or a buggy custom one) returns inflated scores and crowds out everything else. Symptoms: every prompt looks similar; the recency retriever stops mattering; `trace diff` shows the same blocks selected turn after turn.
- **Stale facts.** A `Fact` block that was true a month ago keeps getting retrieved by lexical matches and the model parrots it as current. The runtime has no notion of fact expiry — that's an application policy.
- **Over-summarization.** The COLD-tier summary substitution kicks in, an important detail in the original block is missing from the summary, and the model now has *less* useful information than if the original had been omitted entirely.
- **Token fragmentation.** Many small low-value blocks (one fact per sentence, one log line per block) clog the section budgets even when individually each looks cheap.

Mitigations:

- **Normalize and weight retrievers.** Enforce `[0, 1]` scores per retriever (see the [score normalization](#score-normalization) note). Tune retriever weights against a held-out set of representative tasks.
- **Purge or downgrade stale blocks.** Either delete via `purge` or push priority toward `0.0` so they only surface when nothing else is relevant. Application-side TTL policies are easy to add.
- **Summarize cold data.** Run `llm386 summarize --store-summary` periodically. Pair with the pager's `summary_fallback` policy so summaries are substituted only when the original blocks don't fit.
- **Enforce section budgets.** A tight `Recent` budget bounds chat-history bloat; a tight `Tools` budget bounds tool-result bloat. Default budgets are starting points, not invariants — override per workload.

`llm386 trace diff` between a healthy turn and a degraded turn is the fastest way to localize which of these is biting you.

---

## Comparison and adoption

### Why should I use this over other approaches?

Common alternatives and where LLM386 sits relative to them:

- **"Just stuff messages into the prompt".** Fine until you hit a context window. LLM386 starts paying off the moment you have to drop or summarize anything.
- **LangChain or LlamaIndex memory.** Great if you are already in that ecosystem. Both tend to mix flow control, tools, and memory in one stack. LLM386 is just memory and assembly: it sits underneath your existing framework rather than replacing it.
- **Vector DB only.** A vector DB does retrieval. It does not budget tokens, render section-aware prompts, deduplicate, or trace what got included. It is a great `Retriever` backend; it is not a complete answer.
- **Roll your own.** Most teams end up with something LLM386-shaped after a few iterations. Starting from this saves the time you would otherwise spend independently rediscovering "we need section budgets", "we need to dedup on content hash", and "we need to know what was in last week's prompt".

What you give up by adopting it: another dependency, another binary on your servers, the time to learn the model. What you gain: a deterministic, inspectable, replaceable-piece-by-piece foundation that does not tie you to one model family or one agent framework.

If "I need a quick chatbot demo" is your goal, you don't need this. If "I have an agent that works in development but the prompts are now a mess and I can't reason about what the model is seeing" is your goal, this is built for exactly that.
