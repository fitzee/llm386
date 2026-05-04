# llm386 (Python)

Python wrapper for the [LLM386](../README.md) context virtualization runtime.

## Install

```
pip install llm386
```

You also need the `llm386` binary on your `PATH`. From a checkout of this repo:

```
cargo build --release -p llm386-cli
export PATH="$PWD/target/release:$PATH"
```

## Status

This is the v0 SDK. It shells out to the `llm386` binary for every operation: correct, simple, slow (one process per call). A v1 PyO3-based SDK with the same public surface will replace this once it lands.

Write code against this version and it should keep working when the native bindings ship.

## Quick start

```python
from llm386 import Store, list_models

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
    from llm386 import Trace
    print(Trace("./traces").show(result.trace_id))
```

## Custom profiles, tokenizers, retrievers

Pass a TOML config path via `profiles=`. Same schema as the CLI:

```python
store = Store("./store", profiles="./llm386.toml")
```

```toml
# llm386.toml

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
```

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

MIT or Apache-2.0, at your option.
