"""End-to-end smoke tests. Exercise every public method against
the real `llm386` binary built from the workspace."""

from __future__ import annotations

import json

import pytest

from llm386 import LLM386Error, Store, list_models


def test_put_returns_hex_id(store):
    block_id = store.put(session=1, kind="user-message", body="hello world")
    assert isinstance(block_id, str)
    assert len(block_id) == 32
    int(block_id, 16)  # must parse as hex


def test_show_roundtrips_block(store):
    block_id = store.put(session=1, kind="fact", body="Canberra is the capital.")
    block = store.show(block_id)
    assert block.id == block_id
    assert block.kind == "Fact"
    assert block.body == b"Canberra is the capital."
    assert block.priority == 0.0
    assert isinstance(block.created_at, int)
    assert len(block.hash) == 64  # 32 bytes as hex


def test_put_dedup_returns_same_id(store):
    a = store.put(session=1, kind="fact", body="dup")
    b = store.put(session=1, kind="fact", body="dup")
    assert a == b


def test_list_sessions_enumerates_distinct_sessions(store):
    store.put(session=1, kind="fact", body="x")
    store.put(session=2, kind="fact", body="y")
    store.put(session=2, kind="fact", body="z")
    sessions = store.list_sessions()
    assert "00000000000000000000000000000001" in sessions
    assert "00000000000000000000000000000002" in sessions


def test_page_returns_plan_with_selected_blocks(store):
    store.put(session=1, kind="user-message", body="hi")
    store.put(session=1, kind="fact", body="paris is the capital of france")
    plan = store.page(session=1, model="gpt-4o", task="answer")
    assert len(plan.selected) >= 1
    assert plan.estimated_tokens > 0
    for sid in plan.selected:
        assert len(sid) == 32


def test_pack_prompt_only_returns_rendered_string(store):
    store.put(session=1, kind="user-message", body="say hi")
    result = store.pack(session=1, model="gpt-4o", task="reply briefly")
    assert result.rendered is not None
    assert result.messages is None
    assert "say hi" in result.rendered


def test_pack_chat_returns_message_list(store):
    store.put(session=1, kind="system", body="be concise")
    store.put(session=1, kind="user-message", body="2+2?")
    result = store.pack(session=1, model="gpt-4o", task="answer", chat=True)
    assert result.messages is not None
    assert result.rendered is None
    roles = [m.role for m in result.messages]
    # System message + user task at minimum.
    assert "system" in roles
    assert "user" in roles


def test_pack_with_trace_records_id(store, tmp_path):
    store.put(session=1, kind="user-message", body="x")
    trace_dir = tmp_path / "traces"
    result = store.pack(
        session=1, model="gpt-4o", task="reply", chat=True, trace=trace_dir
    )
    assert result.trace_id is not None
    assert len(result.trace_id) == 32

    from llm386 import Trace

    trace = Trace(trace_dir, binary=store.binary)
    text = trace.show(result.trace_id)
    assert result.trace_id in text


def test_summarize_truncating_returns_text(store):
    for i in range(3):
        store.put(session=1, kind="fact", body=f"fact number {i}")
    out = store.summarize(session=1, summarizer="truncating", max_chars=50)
    assert "fact" in out


def test_list_models_includes_built_ins(llm386_binary):
    models = list_models(binary=llm386_binary)
    names = {m.name for m in models}
    # A few built-ins we know ship.
    assert "gpt-4o" in names
    assert "claude-opus-4-7" in names
    for m in models:
        assert m.max_context_tokens > 0


def test_failed_invocation_raises_llm386_error(store):
    with pytest.raises(LLM386Error):
        store.page(session=1, model="bogus-model-name", task="x")
