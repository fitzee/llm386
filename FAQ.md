# Frequently asked questions

## How much latency does this add to my agent?

Specific to your hardware and session size, but here is a baseline from the bundled benchmarks on a 2024 Apple Silicon laptop:

- Pager: 141 µs for 100 blocks, 1.4 ms for 1000 blocks (linear in N).
- Tokenizer (cl100k_base): 56 µs for 2.7 KB, 1.2 ms for 45 KB.
- LMDB put: low single-digit ms.
- LMDB get: sub-millisecond.

For a typical chat-style turn (50 to 100 blocks selected, a few KB of rendered prompt) end-to-end `pack` from the Rust library lands in the 5 to 10 ms range. The model API call itself dominates by orders of magnitude.

The current Python SDK shells to the CLI binary on every call, which adds 30 to 50 ms of process startup per call. The planned PyO3 bindings will close that gap to roughly the Rust numbers above.

If you enable a network-backed summarizer (Anthropic) or embedder (OpenAI), those add their own latency on top of the runtime.

## How big can the memory store get?

The default LMDB `map_size` is 64 GiB. That is a virtual reservation, not an allocation, so the on-disk footprint only grows as you write. Concrete capacity depends on your average block size:

- Chat-style blocks (~200 bytes each): hundreds of millions of blocks.
- Document chunks (~2 KB each): tens of millions.
- Embeddings (1536-dim float32, ~6 KB each): roughly 10 million.

If you need more, pass a larger `map_size` to `StoreConfig`. LMDB's hard ceiling is your platform's address space (effectively unbounded on 64-bit hosts).

There is no built-in size readout yet. `du -sh ./store` is the easy answer.

## If it gets corrupted, can I rebuild it somehow?

LMDB itself is crash-safe. It uses a B+ tree with copy-on-write writes, so a pulled power cord or a `kill -9` during a transaction leaves the store readable; the in-flight transaction is just rolled back.

Application-level corruption (your own code or a future schema migration writing the wrong bytes) is partially recoverable:

- The schema version stamped in the `meta` table prevents older code from opening a newer-format store.
- Each block carries its content hash, so corrupt block bodies are detectable.
- `blocks_by_hash` and `blocks_by_session` are indexes derived from `blocks_by_id`, so they can in principle be rebuilt from the primary table.

There is no `llm386 fsck` or auto-repair subcommand today. The honest answer: keep backups. Copying the store directory while no writer is active is a valid backup. An explicit `repair` and `verify` pair is on the roadmap.

## Is it ever a good idea to purge memory?

Yes, in three cases:

1. **Compliance.** GDPR right-to-be-forgotten and similar regulations require deletion. The pager respects what is in the store, so removing blocks is the right answer.
2. **Hygiene.** Outdated facts (an old address, a stale API endpoint) keep getting retrieved if they stay in the store. Deleting them is more reliable than adding a "this is wrong, ignore it" note that the model will sometimes ignore back.
3. **Privacy boundaries.** In multi-tenant systems where one user's data should never reach another user's context.

Don't purge to "save tokens" or "fit the context window". The pager and section budgets already drop what doesn't fit; that's a runtime concern, not a storage concern. For long-running sessions, summarize old turns instead (the COLD-tier behavior in the pager is built for exactly this).

Blocks are immutable by design and the runtime doesn't expose a `delete` API today. Workaround: stop writing, open the store, copy the blocks you want to keep into a fresh store. A scoped `purge` subcommand is on the roadmap.

## How does it work with RAG, and does it store links to blobs/documents?

Both are options.

**Storing the full document.** LMDB handles binary content fine. For small to medium documents (PDFs of a few MB, transcripts), put them as `document-chunk` blocks. For very large documents, chunk them first into ~512-token pieces.

**Storing references.** Put a small block whose body is a URL, a file path, or an external content hash. Write a custom `Packer` that resolves those references at render time. Keeps the store small at the cost of resolve-time latency.

**A vector-RAG flow inside LLM386.** Ingest documents as `document-chunk` blocks. Compute embeddings via `OpenAiEmbedder` (or any custom `Embedder` impl). Use `LinearAnnRetriever` or `HnswAnnRetriever` alongside `RecencyRetriever` and `Bm25Retriever`. The pager merges scores by `BlockId` (max wins), and the packer renders the top hits in the Background section.

If you already run a vector database elsewhere (Pinecone, Qdrant, pgvector), the right integration is a custom `Retriever` that queries it. The `Retriever` trait has three methods.

## Does it support multiple user sessions?

Yes. Every block belongs to a `SessionId` (a 128-bit value), and every read/write API is session-scoped. Common patterns:

- One session per conversation (chat thread).
- One session per agent instance.
- One session per user.

`list_sessions` enumerates everything in a store. Two sessions in the same store are isolated for retrieval and paging, but share content-hash dedup (identical block bytes are stored once regardless of session).

For stronger isolation between tenants (compliance, key separation, regulatory boundaries), open a separate `LmdbStore` per tenant. Each one is its own LMDB env with its own files.

Cross-session retrieval (for example, "find similar facts across all my agents") isn't built in, but it's a small custom `Retriever` away.

## How do I migrate "memories" from an existing memory subsystem?

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

## Why should I use this over other approaches?

Common alternatives and where LLM386 sits relative to them:

- **"Just stuff messages into the prompt".** Fine until you hit a context window. LLM386 starts paying off the moment you have to drop or summarize anything.
- **LangChain or LlamaIndex memory.** Great if you are already in that ecosystem. Both tend to mix flow control, tools, and memory in one stack. LLM386 is just memory and assembly: it sits underneath your existing framework rather than replacing it.
- **Vector DB only.** A vector DB does retrieval. It does not budget tokens, render section-aware prompts, deduplicate, or trace what got included. It is a great `Retriever` backend; it is not a complete answer.
- **Roll your own.** Most teams end up with something LLM386-shaped after a few iterations. Starting from this saves the time you would otherwise spend independently rediscovering "we need section budgets", "we need to dedup on content hash", and "we need to know what was in last week's prompt".

What you give up by adopting it: another dependency, another binary on your servers, the time to learn the model. What you gain: a deterministic, inspectable, replaceable-piece-by-piece foundation that does not tie you to one model family or one agent framework.

If "I need a quick chatbot demo" is your goal, you don't need this. If "I have an agent that works in development but the prompts are now a mess and I can't reason about what the model is seeing" is your goal, this is built for exactly that.
