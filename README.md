# LLM386

A Rust runtime that manages the external state needed to feed an LLM. It treats the model as a stateless inference function and handles the rest: persistent block storage, retrieval, paging into a model-specific token budget, and deterministic prompt assembly.

The name is a nod to EMM386, the DOS-era memory manager that paged a larger external memory space into a smaller active working set. Same idea, applied to LLM context windows.

## Problem

An LLM call has three properties that make it hard to use directly from application code:

1. The model itself holds no state. Every call re-derives everything from the prompt.
2. The prompt has a hard token budget that varies by model.
3. The data you want to feed in (conversation history, retrieved documents, tool results) almost always exceeds that budget.

Most projects end up reinventing the same pieces per agent or chatbot:

- A storage layer for messages and documents.
- Some way to retrieve relevant content (often just "last N messages").
- Token counting per model family.
- A function that stitches everything into a prompt under the budget.
- Some way to inspect what got included and why when things go wrong.

That code tends to be ad-hoc, model-specific, and hard to test.

## What

LLM386 is the runtime under that surface. The pieces:

- A persistent block store (LMDB-backed) that holds every input the model has seen or might see, keyed and deduplicated by content hash.
- A model registry that knows context windows, output reservations, tokenizers, and capability flags per model.
- A pager that picks which blocks fit the current call, applying per-section budgets and pluggable retrievers (recency, lexical, BM25, embedding ANN, pinned ids).
- A packer that turns the pager's plan into a deterministic prompt string or a list of role-tagged chat messages.
- A tracer that records every page+pack call so you can replay or audit it later.
- A summarizer trait with a pure truncating implementation, plus an Anthropic-backed implementation in a separate crate for LLM-driven summaries.
- A CLI that exposes the whole pipeline.

It is a library first. The CLI is a thin shell over the library.

## Why

A few design choices are worth calling out:

**The model never owns durable state.** Every byte the model sees comes from a block in the store, with provenance attached. Output gets parsed, validated, and committed back as new blocks. The system stays inspectable and recoverable.

**Prompt assembly is deterministic.** Same inputs produce a byte-identical rendered prompt. The trace layer relies on this for replay.

**Budgeting is model-aware, not hand-rolled.** A `ModelProfile` carries `max_context_tokens`, `reserved_output_tokens`, `safety_margin_tokens`, and a `tokenizer` id. The pager and packer respect that contract regardless of which model you swap in.

**Sections, not just blocks.** The pager allocates the input budget across canonical sections (System, Task, State, Plan, Retrieved, Tools, Recent, Background) with a tunable `SectionBudgetTable`. This matches how a chat-style prompt is actually structured.

**Pluggable retrieval.** The default `RecencyRetriever` is fine for chat-style use. Add `LexicalRetriever` or `Bm25Retriever` for keyword search. Add `LinearAnnRetriever` or `HnswAnnRetriever` (with the bundled OpenAI embedder, or your own `Embedder` impl) when you need semantic recall. They compose: the pager fans out across all configured retrievers and merges results by max score per block.

**Storage and serialization are explicit.** LMDB for persistence, postcard for block bodies, hand-rolled big-endian keys. No JSON in the hot path. Content-hash dedup means identical bytes get stored once even across sessions.

**Observability is built in.** Each `pack` call can record a `TraceRecord` (CallId, session, model, plan, prompt hash, duration). Inspect later with `llm386 trace show`.

## How

### Install

Requires Rust 1.95 or newer.

```
cargo build --release
```

The CLI binary is `target/release/llm386`.

### Quick start

```
llm386 init ./store

echo "You are a concise assistant." | llm386 put --store ./store --session 1 --kind system -
echo "What is the capital of Australia?" | llm386 put --store ./store --session 1 --kind user-message -
echo "Canberra." | llm386 put --store ./store --session 1 --kind assistant-message -
echo "It became the capital in 1908." | llm386 put --store ./store --session 1 --kind fact -

llm386 list-models
llm386 page --store ./store --session 1 --model gpt-4o --task "explain Australia's history"
llm386 pack --store ./store --session 1 --model gpt-4o --task "explain Australia's history"
```

`pack` prints the rendered prompt on stdout with a manifest header on stderr. Redirect with `> prompt.txt` to capture just the prompt.

### Variants

Render as JSON for a chat API:

```
llm386 pack --store ./store --session 1 --model gpt-4o --task "..." --chat
```

Record a trace and inspect it later:

```
llm386 pack --store ./store --session 1 --model gpt-4o --task "..." --trace ./traces
llm386 trace show --store ./traces <call-id>
```

Inspect a single block:

```
llm386 show --store ./store <block-id>
llm386 show --store ./store <block-id> --json
```

List sessions in a store:

```
llm386 list-sessions --store ./store
```

Summarize a session:

```
llm386 summarize --store ./store --session 1 --summarizer truncating --max-chars 80
ANTHROPIC_API_KEY=... llm386 summarize --store ./store --session 1 --summarizer anthropic --store-summary
```

### Custom config

A TOML file (passed via `--profiles <path>` or the `LLM386_PROFILES` environment variable) carries three optional sections:

```toml
[[profile]]
name = "my-tiny"
max_context_tokens = 4096
reserved_output_tokens = 1024
tokenizer = "cl100k_base"

[[hf_tokenizer]]
name = "llama-3"
path = "/path/to/llama-3-tokenizer.json"

[[retriever]]
kind = "bm25"
k1 = 1.5

[[retriever]]
kind = "recency"
half_life_secs = 60.0
```

`[[profile]]` adds model profiles on top of the built-ins. `[[hf_tokenizer]]` registers a HuggingFace tokenizer.json (used by Llama, Qwen, Mistral, and similar). `[[retriever]]` replaces the default retriever stack.

### Library

```rust
use std::sync::Arc;
use llm386_core::{PageRequest, SessionId, default_registry};
use llm386_pager::GreedyPager;
use llm386_packer::SimplePacker;
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::cl100k_base;

let store = Arc::new(LmdbStore::open("./store", StoreConfig::default())?);
let tokenizer = Arc::new(cl100k_base()?);
let model = default_registry().get("gpt-4o").unwrap().clone();

let pager = GreedyPager::new(store.clone(), tokenizer.clone());
let packer = SimplePacker::new(store, tokenizer);

let request = PageRequest {
    session_id: SessionId(1),
    task: "answer the user".into(),
    model,
    required_blocks: vec![],
};
let plan = pager.page(request.clone())?;
let prompt = packer.pack(&request, &plan)?;
println!("{}", prompt.rendered);
```

Every component is replaceable: `Pager`, `Packer`, `Retriever`, `Tokenizer`, `Embedder`, `Summarizer`, `BlockStore`, and `TraceSink` are all traits in `llm386-core`.

### Using as a memory layer

LLM386 is the memory and context-assembly layer behind an agent. Your agent loop owns flow control, tool execution, and the LLM call itself. LLM386 owns "what does the model see this turn?" and "what got produced?".

A single agent turn looks like this:

1. `put` the user input as a `UserMessage` block.
2. `pack` the session for the target model and task. You get back a rendered prompt (or, with `--chat`, a list of role-tagged chat messages ready to send to a chat-completion API).
3. Send that to the model.
4. `put` the response as an `AssistantMessage` block. If the model called a tool, `put` each tool result as a `ToolResult` block with `provenance.parents = [assistant_block_id]` so the pager keeps them paired.
5. Repeat.

A Python sketch using the [`llm386` SDK](./python/) (in `python/`):

```python
from llm386 import Store
from openai import OpenAI

store = Store("./store")
client = OpenAI()

def turn(session_id: int, user_input: str) -> str:
    store.put(session_id, kind="user-message", body=user_input)

    result = store.pack(session=session_id, model="gpt-4o",
                         task="answer the user", chat=True)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": m.role, "content": m.content} for m in result.messages],
    )
    reply = response.choices[0].message.content

    asst_id = store.put(session_id, kind="assistant-message", body=reply)
    # for tool_result in response.tool_results:
    #     store.put(session_id, kind="tool-result", body=tool_result,
    #               parents=[asst_id])
    return reply
```

The current SDK shells to the CLI binary on every call; a PyO3-based version with the same surface is planned. See [`python/README.md`](./python/README.md) for the full Python API.

### Framework hooks

Most Python agent frameworks expose a place to plug in custom memory. The pattern is the same in each case: the framework owns flow control and tool execution, LLM386 owns what the model sees.

**LangGraph:** in each node that calls the model, fetch context via `pack` and write the output back via `put`. Use the LangGraph thread id as the LLM386 session id so checkpoints and stored blocks line up.

**CrewAI:** subclass the framework's memory base class and route `save` to `put` and `search` to a `page` call. A `Bm25Retriever` plus `LinearAnnRetriever` (or `HnswAnnRetriever` for larger sessions) is a reasonable default retriever stack for this use.

**AutoGen:** wrap the agent's `generate_reply` so it draws context from `pack` instead of from the agent's local message list. The agent still emits its own messages; you just intercept ingestion and assembly.

`pack --trace ./traces` records each turn so you can later audit exactly what the model saw and why.

## Architecture

```
crates/
  llm386-core                 types and trait seams
  llm386-store-lmdb           LMDB BlockStore impl
  llm386-tokenizer            tiktoken + HuggingFace tokenizer adapters, registry, LRU cache
  llm386-pager                GreedyPager, SectionBudgetTable, retrievers
  llm386-packer               SimplePacker (string and chat-message rendering)
  llm386-trace                LMDB-backed TraceSink
  llm386-compress             pure summarizers (Noop, Truncating)
  llm386-compress-anthropic   Anthropic-backed Summarizer
  llm386-retrieve-ann         LinearAnnRetriever, HnswAnnRetriever, OpenAiEmbedder, EmbeddingCache
  llm386-cli                  the `llm386` binary
```

The dependency direction is one-way: every impl crate depends on `llm386-core` for traits and types, never on a sibling.

## Status

Early. The single-node embedded library and CLI work end to end against real LMDB and real tokenizers. Interfaces are stable enough for downstream consumers to build on, but expect breaking changes as new retrievers, summarizers, and storage backends land.

## Non-goals

- Hosting a chat UI.
- Hiding state inside prompts.
- Treating the model as the source of truth.
- Distributed storage in the initial version.

## License

MIT or Apache-2.0, at your option.
