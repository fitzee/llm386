# Example `llm386.toml` profiles

Three TOML configs that tune section budgets and retrievers for different agent shapes. Pick the one closest to your workload, copy it into your project, adjust.

| File | Use case | What stands out |
| --- | --- | --- |
| [`llm386-focused-qa.toml`](./llm386-focused-qa.toml) | Single-fact Q&A, structured extraction, classification | 50% Retrieved, 30% Slack (hard headroom cap), 0% Plan, 0% Background |
| [`llm386-chat-loop.toml`](./llm386-chat-loop.toml) | Long-running chat agent | Balanced split with 30% Recent, 30% Retrieved, 15% Tools |
| [`llm386-rag-heavy.toml`](./llm386-rag-heavy.toml) | RAG over a document corpus | 60% Retrieved, 15% Background, 10% Recent |

## Using one

CLI:

```
llm386 --profiles examples/configs/llm386-focused-qa.toml \
    pack --store ./store --session 1 --model gpt-4o --task "..."
```

Python:

```python
from llm386 import Store
store = Store("./store", profiles="examples/configs/llm386-focused-qa.toml")
```

Or set the env var once for an entire shell session:

```
export LLM386_PROFILES=$PWD/examples/configs/llm386-chat-loop.toml
llm386 pack --store ./store --session 1 --model gpt-4o --task "..."
```

## What `[section_budgets]` does

Every key in `[section_budgets]` is a fraction of the *variable budget* — what's left of the model's input window after `System` (fixed) and `Task` (fixed) and any required pinned blocks have taken their share. Fractions sum to ≤ 1.0; if they sum higher they get normalized down at allocation time, so the per-section budgets never exceed the variable budget.

```toml
[section_budgets]
state      = 0.10   # active agent state (committed by the reducer)
plan       = 0.05   # current plan blocks
recent     = 0.20   # recent user/assistant/tool messages
retrieved  = 0.40   # blocks surfaced by retrievers (facts, document chunks)
tools      = 0.15   # tool result blocks
background = 0.05   # low-priority context
slack      = 0.05   # reserved headroom — never filled
```

`Slack` is special: anything routed to it gets recorded as omitted, never rendered. Setting Slack to 0.30 hard-caps the rendered prompt at 70% of the variable budget no matter how much relevant content exists.

`System` and `Task` are fixed and don't appear in this table — they always pack against the global budget directly.

## Combined with retrievers

Section budgets define *how much* of each section's content gets in. Retrievers define *what* content is even a candidate. The two compose:

- A tight `recent` budget plus a recency retriever with a long half-life ⟹ a small number of older-but-relevant turns.
- A wide `retrieved` budget plus a BM25 retriever ⟹ many keyword-matched facts.
- A tight `retrieved` budget plus an embedding ANN retriever ⟹ a few semantically-strongest matches.

The bundled configs pair each section table with a sensible retriever stack, but they're independent knobs — mix and match.

## Verifying the effect

The fastest way to check whether your tuning did what you expected is to compare two trace records:

```
# Run with one config:
llm386 --profiles examples/configs/llm386-chat-loop.toml \
    pack --store ./store --session 1 --model gpt-4o --task "..." \
    --trace ./traces

# Run with a tighter config:
llm386 --profiles examples/configs/llm386-focused-qa.toml \
    pack --store ./store --session 1 --model gpt-4o --task "..." \
    --trace ./traces

# Diff:
llm386 trace diff --store ./traces <prev-call-id> <next-call-id>
```

The diff shows which blocks were added, which were dropped, and the input-token delta — actionable signal for tightening or relaxing.
