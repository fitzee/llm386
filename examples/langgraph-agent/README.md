# LangGraph + LLM386 demo

A 5-minute showcase of LLM386 wired into a LangGraph agent. The agent has two stub tools (a calculator and a fake user-profile lookup) and uses LLM386 as its memory layer — every turn, the model's working set is recomputed from the persistent block store rather than carried in process memory.

## What this shows

- **LLM386 owns context.** Each turn the agent calls `store.page()` and `store.pack(chat=True)` to assemble the model's input. No in-memory chat history: the conversation lives in the LMDB store.
- **LangGraph owns the agent loop.** Flow control, tool dispatch, and the LLM call sit on the LangGraph side. The two layers compose without either one having to know about the other's internals.
- **Persistent across restarts.** The store is a Docker volume. Stop the container, restart it next week, and the agent remembers the conversation.
- **Recall via BM25 + recency.** The bundled `llm386.toml` configures a BM25 retriever alongside the default recency one, so the model can recall things mentioned many turns ago, not just the last few exchanges.
- **Typed edges for tool calls.** When the agent calls a tool, the demo persists the result and ties it to the assistant message via `add_edge(..., kind="tool-invocation")`. Edge-aware paging keeps the pair together on subsequent turns.

## Requirements

- Docker (with Compose v2 — the `docker compose` subcommand).
- An Anthropic API key in `ANTHROPIC_API_KEY`.

## Run it

From the project root:

```
export ANTHROPIC_API_KEY=sk-ant-...
docker compose -f examples/langgraph-agent/docker-compose.yml run --rm agent
```

The first build takes a few minutes — Docker compiles the LLM386 Python extension (PyO3 + maturin) and the `llm386` CLI from source. Subsequent runs reuse the cached image and start in seconds.

You'll get a REPL:

```
LLM386 + LangGraph demo. Type 'exit' to quit.
store=/data/store session=1 model=claude-haiku-4-5

you> what's 17 * 23?
[llm386] selected 1 blocks (54 est. tokens, 2 chat messages packed)
bot> 391.

you> look up user u-002
[llm386] selected 3 blocks (98 est. tokens, 4 chat messages packed)
bot> Diego — free tier, America/Bogota.

you> what was my arithmetic question's answer?
[llm386] selected 5 blocks (156 est. tokens, 6 chat messages packed)
bot> 391.
```

The third turn is the interesting one: the model recalls "391" from the conversation history that LLM386 surfaced via the recency retriever, even though no LangGraph state was persisted between turns.

## What happens on each turn

1. Your input is stored as a `user-message` block.
2. `store.page(...)` computes the working set; the demo prints how many blocks were selected and their estimated token cost. `store.pack(...)` renders that working set into role-tagged chat messages.
3. LangGraph runs: the LLM call, then tool dispatch if the model picked a tool, then a follow-up LLM call.
4. Outputs are committed back into LLM386:
   - Text replies → `assistant-message` blocks.
   - Tool invocations → an `assistant-message` marker block (`[calling tools: foo]`).
   - Tool results → `tool-result` blocks linked to the calling assistant block via a `tool-invocation` edge.

## Inspecting the store

Use the `cli` service in the compose file:

```
docker compose -f examples/langgraph-agent/docker-compose.yml run --rm cli \
    list-sessions --store /data/store

docker compose -f examples/langgraph-agent/docker-compose.yml run --rm cli \
    page --store /data/store --session 1 --model claude-haiku-4-5 \
    --task "anything"

docker compose -f examples/langgraph-agent/docker-compose.yml run --rm cli \
    show --store /data/store <block-id>
```

The CLI is the same `llm386` binary documented in [`docs/CLI.md`](../../docs/CLI.md). Anything that works against a host-side store works against the container's mounted volume.

## Resetting

```
docker compose -f examples/langgraph-agent/docker-compose.yml down -v
```

`-v` removes the named volume, wiping the store. Without it the volume survives a `down`.

## Limitations

This is intentionally small. Out of scope:

- **Real RAG ingest.** No document loading, no embeddings, no chunking. See the [Retrieval and RAG FAQ](../../FAQ.md#retrieval-and-rag).
- **MCP tool servers.** The two tools here are LangChain `@tool`-decorated Python functions. For real MCP integration see the [MCP/tools FAQ](../../FAQ.md#how-do-mcp-servers-and-tools-work-with-this-do-tool-schemas-get-committed-to-memory-like-other-facts).
- **Multi-agent / cross-session memory.** Single agent, single session. Multi-agent patterns are in the [Sessions FAQ](../../FAQ.md#sessions-and-multi-tenancy).
- **Production-grade error handling, retries, rate limiting.** None of these are wired in.
- **Trace recording with `update_output`.** Skipped to keep the demo readable. The pattern is in the [Python README](../../python/README.md#trace--replay).

The demo is meant to be runnable in 5 minutes, not to be a starter template. For a production agent you'd want to subclass / replace pieces of `agent.py` rather than fork it.

## Files

| File | Purpose |
| --- | --- |
| `agent.py` | The whole demo: tools, LangGraph wiring, LLM386 bridge, REPL. |
| `requirements.txt` | LangGraph + langchain-anthropic + typing-extensions. The `llm386` wheel comes from the Dockerfile. |
| `llm386.toml` | Retriever stack (BM25 + recency) loaded by `Store(..., profiles="...")`. |
| `Dockerfile` | Multi-stage: Rust builder → Python runtime. Builds the wheel and the CLI from source. |
| `Dockerfile.dockerignore` | Keeps `target/` and friends out of the build context. |
| `docker-compose.yml` | The `agent` service plus a `cli` convenience service. Named volume holds the store. |
