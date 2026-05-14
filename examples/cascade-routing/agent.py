"""Cascade routing demo: Haiku-first, escalate to Opus on low confidence.

Pattern: page() once for the cheap model; pack_with_plan() and call cheap;
if cheap returns the "need more capability" sentinel, re-pack the *same plan*
for the expensive model and call it. The same plan is reused — no second
retrieval pass — but the prompt is re-rendered for the new model's
tokenizer, budget, and cache_boundary.

Each call sets `cache_control: ephemeral` on the message at cache_boundary
so Anthropic caches the stable prefix (system + background blocks) across
turns and across the cheap/expensive models independently.
"""

from __future__ import annotations

import os
import sys

from anthropic import Anthropic
from llm386 import Store

STORE_PATH = os.environ.get("LLM386_STORE", "/data/store")
PROFILES = os.environ.get("LLM386_PROFILES", "/app/llm386.toml")
SESSION = int(os.environ.get("LLM386_SESSION", "1"))

CHEAP = os.environ.get("LLM386_CHEAP_MODEL", "claude-haiku-4-5")
EXPENSIVE = os.environ.get("LLM386_EXPENSIVE_MODEL", "claude-opus-4-7")
ESCALATION_SENTINEL = "NEED_MORE_CAPABILITY"

# A long-ish, stable system prompt so the cached prefix comfortably
# clears Anthropic's ~1024-token cache minimum on its own.
SYSTEM_PROMPT = f"""You are a careful research assistant.

You answer the user's questions using the conversation history and
any background documents you have been given. You do not invent
facts. You do not speculate beyond what the context supports.

If — and only if — you cannot give a confident answer from the
context, respond with exactly:

    {ESCALATION_SENTINEL}

and nothing else. Do not apologize, do not explain, do not hedge.
Just emit that sentinel verbatim. The system will then route the
question to a more capable model.

When you DO answer, be concise. Two or three sentences for a
factual question; a short paragraph for a "compare these two
things" question; bullet points for a list.

You will see the conversation history and any retrieved background
under section headers (`## System`, `## Background`, `## Recent`).
The headers are framing for you; do not include them in your reply."""

# A chunk of "background knowledge" the model can answer from.
# Picks two real-but-niche topics so we have something for the
# cheap model to handle (factual lookup) and something for the
# expensive model to handle (synthesis / comparison).
BACKGROUND_DOC = """LLM386 is a Rust runtime that manages context
state for LLM agents. It treats the model as a stateless inference
function and owns the surrounding pieces: a persistent block store
(LMDB-backed), retrieval, paging into a model-specific token
budget, and deterministic prompt assembly.

Key components:

- ContextBlock — the atomic unit of memory. Has a kind (System,
  UserMessage, AssistantMessage, Fact, Summary, ToolResult, Plan,
  State, DocumentChunk, Trace), a body, a priority, and provenance.
- ModelProfile — model-specific context window, output reservation,
  safety margin, tokenizer id, and capability flags.
- Pager — selects which blocks fit the current call. Multiple
  retrievers (recency, lexical, BM25, embedding ANN) feed into a
  greedy selection that respects per-section budgets.
- Packer — renders the selected blocks into a deterministic prompt
  string or chat-message list, in canonical section order.
- Trace — records every page+pack call so it can be replayed,
  audited, or diffed against another call.

The name LLM386 is a homage to EMM386, the DOS-era memory manager
that paged a larger external memory space into a smaller working
set inside the 640KB DOS conventional memory region. LLM386 does
the same trick for LLM context windows: it pages a much larger
persistent memory through the small fixed window the model can
actually see at one time.

Determinism is a load-bearing property. Same store state plus same
PageRequest plus same packer config produces a byte-identical
rendered prompt. The trace layer relies on this for replay; the
prompt-cache hit rate also rides on it (stable prefix → cache hit).

Cascade routing fits naturally on top of LLM386. The expensive
operation in a cascade is context assembly (retrieval, scoring,
edge walk), not the model call itself. PagePlan is computed once;
pack_with_plan re-renders the same selection for any model in the
registry. A cheap-first cascade pays the retrieval cost once and
the cheap model's input tokens 100% of the time, plus the
expensive model's input tokens only on the fraction of turns that
escalate. With a 70% cheap-handle rate and Anthropic Haiku at
roughly one-tenth of Opus's input price, the steady-state cost is
about 0.7*0.1 + 0.3*1.0 = 0.37 of an Opus-only baseline."""

store = Store(STORE_PATH, profiles=PROFILES)
client = Anthropic()

# Idempotent — content-hash dedup means re-running this script
# doesn't add duplicate System or DocumentChunk blocks to the store.
store.put(session=SESSION, kind="system", body=SYSTEM_PROMPT)
store.put(session=SESSION, kind="document-chunk", body=BACKGROUND_DOC)


def to_anthropic_request(packed) -> tuple[list[dict] | None, list[dict]]:
    """Map an LLM386 ChatPrompt to the Anthropic messages.create shape.

    Anthropic puts system messages in a top-level `system` field, not
    in the messages array. cache_control: ephemeral lands on the
    content block at packed.cache_boundary so Anthropic caches the
    prefix `messages[0..=cache_boundary]` (across the system field
    and the messages array uniformly — Anthropic concatenates them
    internally when forming the cache key).
    """
    system_blocks: list[dict] = []
    user_blocks: list[dict] = []
    for i, m in enumerate(packed.messages):
        block: dict = {"type": "text", "text": m.content}
        if packed.cache_boundary is not None and i == packed.cache_boundary:
            block["cache_control"] = {"type": "ephemeral"}
        if m.role == "system":
            system_blocks.append(block)
        else:
            # Anthropic API only knows user/assistant in messages[].
            role = "assistant" if m.role == "assistant" else "user"
            user_blocks.append({"role": role, "content": [block]})
    return (system_blocks or None), user_blocks


def call(model: str, packed) -> tuple[str, dict]:
    system, messages = to_anthropic_request(packed)
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    usage = {
        "input": resp.usage.input_tokens,
        "output": resp.usage.output_tokens,
        "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
    }
    return text, usage


def needs_escalation(text: str) -> bool:
    """Cheap model self-flags via the sentinel defined in the system prompt.

    Real production code would also detect malformed structured output,
    timeouts, or use a verifier model. The sentinel is the simplest
    thing that exposes the cascade plumbing.
    """
    return ESCALATION_SENTINEL in text.strip()


def turn(user_input: str) -> str:
    store.put(session=SESSION, kind="user-message", body=user_input)

    # Page ONCE for the cheap model. The same plan is reused below
    # if we escalate — pack_with_plan skips the retrieval pass.
    plan = store.page(session=SESSION, model=CHEAP, task=user_input)

    cheap_packed = store.pack_with_plan(
        plan, session=SESSION, model=CHEAP, task=user_input, chat=True,
    )
    answer, usage = call(CHEAP, cheap_packed)
    print(
        f"  [Haiku] msgs={len(cheap_packed.messages):>2} "
        f"in={usage['input']:>5} out={usage['output']:>4} "
        f"cache_read={usage['cache_read']:>5} cache_write={usage['cache_write']:>5} "
        f"boundary={cheap_packed.cache_boundary}",
        file=sys.stderr,
    )

    if needs_escalation(answer):
        # SAME plan, new model. No second retrieval pass.
        exp_packed = store.pack_with_plan(
            plan, session=SESSION, model=EXPENSIVE, task=user_input, chat=True,
        )
        answer, usage = call(EXPENSIVE, exp_packed)
        print(
            f"  [Opus ] msgs={len(exp_packed.messages):>2} "
            f"in={usage['input']:>5} out={usage['output']:>4} "
            f"cache_read={usage['cache_read']:>5} cache_write={usage['cache_write']:>5} "
            f"boundary={exp_packed.cache_boundary}  ← escalated",
            file=sys.stderr,
        )

    store.put(session=SESSION, kind="assistant-message", body=answer)
    return answer


def main() -> None:
    print(
        f"cascade-routing demo  session={SESSION}\n"
        f"  cheap     = {CHEAP}\n"
        f"  expensive = {EXPENSIVE}\n"
        "Ask a question. Blank line to quit. The cheap model handles\n"
        "factual lookups; harder questions self-escalate to Opus.\n"
        "Try: 'what is LLM386?' (cheap) then\n"
        "     'compare LLM386 to a LangGraph checkpointer in two paragraphs.' (likely escalates)"
    )
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        try:
            print(f"bot> {turn(q)}")
        except Exception as e:  # noqa: BLE001 - demo error surface
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
