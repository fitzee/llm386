# llm386 (Python)

Python bindings for the [LLM386](../README.md) context virtualization runtime, built with PyO3 + maturin. The whole runtime ships as a native extension; no separate binary or daemon is needed.

## Install

```
pip install llm386
```

## Build from source

```
pip install maturin
cd python
maturin develop
```

## Status

PyO3 bindings (v0.2). The previous v0.1 was a CLI-shelling pure-Python wrapper; the public API is the same so code from v0.1 keeps working.

Custom retrievers, embedders, and summarizers written in Python (subclassing the trait) are on the roadmap for v0.3. For now, only the Rust-side implementations are exposed.

## Quick start

```python
from llm386 import Store, list_models

# Open or initialize an LMDB store at ./store. Idempotent.
store = Store("./store")

block_id = store.put(session=1, kind="user-message", body="What is the capital of Australia?")
store.put(session=1, kind="assistant-message", body="Canberra.")

plan = store.page(session=1, model="gpt-4o", task="explain Australia's history")
print(plan.selected, plan.estimated_tokens)

result = store.pack(session=1, model="gpt-4o", task="explain Australia's history", chat=True)
for msg in result.messages:
    print(f"[{msg.role}] {msg.content}")
```

## Using as a memory layer in an agent loop

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

    # If the model called tools, store each result with the assistant
    # message as a parent so the pager keeps them paired.
    # for tool_result in tool_results:
    #     store.put(session_id, kind="tool-result", body=tool_result,
    #               parents=[asst_id])

    return reply
```

## Trace + replay

```python
result = store.pack(session=1, model="gpt-4o", task="...",
                    chat=True, trace="./traces")

if result.trace_id:
    print(f"recorded trace {result.trace_id}")
```

Inspecting a recorded trace from Python is on the roadmap. For now, use the CLI: `llm386 trace show --store ./traces <call-id>`.

## Custom profiles, tokenizers, retrievers

The PyO3 bindings ship the built-in registries today. Custom profile / tokenizer / retriever loading from a TOML config file is on the roadmap; in the interim, configure them via the CLI and have your Python code use the matching model name.

## Summarization

```python
# Pure (no API call):
print(store.summarize(session=1, summarizer="truncating", max_chars=80))

# Via Anthropic Claude (set ANTHROPIC_API_KEY):
print(store.summarize(session=1, summarizer="anthropic", store_summary=True))
```

## API surface

```python
from llm386 import (
    Store,           # main entry point
    Trace,           # trace store reader
    list_models,     # discover available model profiles

    # Result types
    ChatMessage, ContextBlock, ModelProfile,
    OmittedBlock, PackResult, PagePlan, Provenance,

    LLM386Error,     # raised when the CLI invocation fails
)
```

## License

Apache-2.0.
