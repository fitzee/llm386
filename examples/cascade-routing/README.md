# Cascade routing demo

A small chatbot that calls **Claude Haiku 4.5** first and escalates to
**Claude Opus 4.7** only when Haiku self-flags low confidence. Same
LLM386 store, same retrieval, same `PagePlan` — `pack_with_plan()`
re-renders the selection for each model's tokenizer and budget without
re-running retrieval.

The demo runs entirely in Docker so you don't need a Rust toolchain
or local Python setup.

## Run it

```
export ANTHROPIC_API_KEY=sk-ant-...
docker compose -f examples/cascade-routing/docker-compose.yml run --rm agent
```

First build takes a few minutes (Rust + maturin compile from source).
Subsequent runs start in seconds.

## What you'll see

```
cascade-routing demo  session=1
  cheap     = claude-haiku-4-5
  expensive = claude-opus-4-7

you> what is LLM386?
  [Haiku] msgs= 4 in= 2107 out=  62 cache_read=    0 cache_write= 1843 boundary=1
bot> LLM386 is a Rust runtime that manages the external state needed to
feed an LLM — persistent block storage, retrieval, paging into a
model-specific token budget, and deterministic prompt assembly. ...

you> what's the EMM386 reference?
  [Haiku] msgs= 5 in=  186 out=  48 cache_read= 1843 cache_write=    0 boundary=1
bot> EMM386 was a DOS-era memory manager that paged a larger external
memory space into a smaller window inside DOS's 640KB conventional
memory region. LLM386 does the same trick for LLM context windows.

you> compare LLM386's pager to a LangGraph checkpointer in two paragraphs.
  [Haiku] msgs= 6 in=  214 out=  10 cache_read= 1843 cache_write=    0 boundary=1
  [Opus ] msgs= 6 in=  191 out= 287 cache_read=    0 cache_write= 1862 boundary=1  ← escalated
bot> [longer, comparative answer from Opus]
```

A few things to read off the trace lines:

- **`cache_read` jumps from 0 → 1843 on turn 2.** Anthropic cached the
  System + Background prefix from turn 1, and turn 2 hit it. From
  here on, every Haiku call pays for cached prefix at 10% of the
  input price plus the small turn-specific tail at full price.
- **`cache_write = 0` on turn 2.** No new prefix was created; the
  existing cache entry was reused.
- **`boundary=1` consistently.** Two messages are in the stable
  prefix: one for `## System`, one for `## Background`. The
  user-message + task messages come after.
- **The Opus call on turn 3 has `cache_read=0`.** Anthropic caches
  per-model. Haiku's cached prefix isn't visible to Opus. Opus
  pays full price the first time it sees the prefix; subsequent
  Opus calls (if any) would hit Opus's own cache.
- **The escalated turn pays for both models' input tokens.** That's
  the cascade tax. Worthwhile only when the cheap model handles
  most turns alone.

## What this demonstrates concretely

1. **`pack_with_plan` reuses one retrieval.** `store.page()` runs
   once per turn. The same `PagePlan` is re-rendered for the cheap
   model and (if escalated) for the expensive model. Different
   tokenizer, different budget, different `cache_boundary` — same
   selected blocks.
2. **`cache_boundary` flows naturally to Anthropic.** `agent.py`
   reads `packed.cache_boundary` and sets
   `cache_control: { type: "ephemeral" }` on that message. The
   stable prefix is cached across turns.
3. **Routing is application code.** LLM386 doesn't pick the model.
   The cheap/expensive choice and the escalation criterion live in
   the agent, not in the runtime. The runtime's job is to render
   *whatever* model you picked correctly.
4. **Persistence across restarts.** The store is a Docker volume.
   Stop and restart the container; the agent picks up where it
   left off, and the cached prefix re-warms on the next turn.

## Inspect the store

The same image carries the `llm386` CLI, useful for poking at what
got stored after a session:

```
docker compose -f examples/cascade-routing/docker-compose.yml run --rm cli \
    list-sessions --store /data/store

docker compose -f examples/cascade-routing/docker-compose.yml run --rm cli \
    pack --store /data/store --session 1 --model claude-haiku-4-5 \
    --task "what is LLM386?" --chat
```

The `--chat` output prints `cache_boundary: N` on stderr — same
boundary the agent uses.

## Tune the cascade

| change                    | knob                                           |
|---------------------------|------------------------------------------------|
| different model tiers     | `LLM386_CHEAP_MODEL` / `LLM386_EXPENSIVE_MODEL`|
| different escalation rule | edit `needs_escalation()` in `agent.py`        |
| more / fewer cached sections | `[cache] stable_sections` in `llm386.toml`  |
| different retrieval mix   | `[[retriever]]` blocks in `llm386.toml`        |
| trace each turn           | pass `trace=...` to `pack_with_plan`           |

## What this deliberately doesn't cover

- **Real confidence scoring.** The escalation trigger is a sentinel
  string the cheap model emits when it can't answer. Production
  cascades use a verifier model, self-consistency over multiple
  cheap calls, or structured-output validation against a schema.
- **Cost accounting.** The `cache_read` / `cache_write` numbers are
  printed but not converted to dollars. Real routing systems log
  per-call cost and alert when `cache_read` stays at 0 for many
  turns (signal that the "stable" prefix isn't actually stable).
- **Three-tier or unified cascades.** The pattern generalizes:
  call cheapest → escalate to mid → escalate to top, or even
  speculative cascades (Google Research's pattern). The core
  primitive — `pack_with_plan` for any model in the registry —
  is the same.
- **Routing across providers.** Crossing OpenAI ↔ Anthropic ↔ Gemini
  also crosses cache pools (each provider caches independently).
  The agent here calls Anthropic-only to keep the demo focused.
- **Tool calls.** The langgraph-agent example covers tool integration;
  the cascade pattern composes with it but isn't shown here.
