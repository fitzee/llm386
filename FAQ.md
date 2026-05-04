# Frequently asked questions

- [Performance and sizing](#performance-and-sizing)
  - [How much latency does this add to my agent?](#how-much-latency-does-this-add-to-my-agent)
  - [How big can the memory store get?](#how-big-can-the-memory-store-get)
- [Data lifecycle](#data-lifecycle)
  - [If it gets corrupted, can I rebuild it somehow?](#if-it-gets-corrupted-can-i-rebuild-it-somehow)
  - [Is it ever a good idea to purge memory?](#is-it-ever-a-good-idea-to-purge-memory)
  - [Legal/security asked me to remove customer data. How do I find and remove it?](#legalsecurity-asked-me-to-remove-customer-data-how-do-i-find-and-remove-it)
  - [How do I migrate "memories" from an existing memory subsystem?](#how-do-i-migrate-memories-from-an-existing-memory-subsystem)
- [Sessions and multi-tenancy](#sessions-and-multi-tenancy)
  - [Does it support multiple user sessions?](#does-it-support-multiple-user-sessions)
- [Retrieval and RAG](#retrieval-and-rag)
  - [How does it work with RAG, and does it store links to blobs/documents?](#how-does-it-work-with-rag-and-does-it-store-links-to-blobsdocuments)
  - [How do I write a custom retriever for Pinecone, and wire it in?](#how-do-i-write-a-custom-retriever-for-pinecone-and-wire-it-in)
- [Comparison and adoption](#comparison-and-adoption)
  - [Why should I use this over other approaches?](#why-should-i-use-this-over-other-approaches)

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

---

## Data lifecycle

### If it gets corrupted, can I rebuild it somehow?

LMDB itself is crash-safe. It uses a B+ tree with copy-on-write writes, so a pulled power cord or a `kill -9` during a transaction leaves the store readable; the in-flight transaction is just rolled back.

Application-level corruption (your own code or a future schema migration writing the wrong bytes) is partially recoverable:

- The schema version stamped in the `meta` table prevents older code from opening a newer-format store.
- Each block carries its content hash, so corrupt block bodies are detectable.
- `blocks_by_hash` and `blocks_by_session` are indexes derived from `blocks_by_id`, so they can in principle be rebuilt from the primary table.

There is no `llm386 fsck` or auto-repair subcommand today. The honest answer: keep backups. Copying the store directory while no writer is active is a valid backup. An explicit `repair` and `verify` pair is on the roadmap.

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

**Removing the blocks.** Two paths today, both manual:

- **Whole-session purge** (simplest, when you can sacrifice the session): open a fresh store, copy across every session you want to keep, swap the directories. Atomic at the directory level.
- **Targeted block purge** (more work): same approach, but inside each surviving session, copy block-by-block excluding the offending ids. Preserve `Provenance.parents` references where possible; orphaned references are tolerated by the pager but cosmetic.

A native `llm386 purge --session ... --block ...` subcommand and a matching Python `Store.delete(block_id)` are on the roadmap. Until then, the targeted-purge script is the right answer for compliance work, and you can run it safely while no other writer is active.

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

Cross-session retrieval (for example, "find similar facts across all my agents") isn't built in, but it's a small custom `Retriever` away.

---

## Retrieval and RAG

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

A Python-side trait implementation (writing a `Retriever` in Python that gets called from Rust) is on the roadmap for the v0.3 PyO3 bindings. Until then, custom retrievers live in Rust and the Python SDK exposes the Rust ones.

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
