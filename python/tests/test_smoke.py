"""End-to-end smoke tests for the PyO3-backed `llm386` package."""

from __future__ import annotations

import pytest

from llm386 import LLM386Error, Store, list_models


def test_put_returns_hex_id(store):
    block_id = store.put(session=1, kind="user-message", body="hello world")
    assert isinstance(block_id, str)
    assert len(block_id) == 32
    int(block_id, 16)  # must parse as hex


def test_put_accepts_str_or_bytes(store):
    a = store.put(session=1, kind="fact", body="hello")
    b = store.put(session=1, kind="fact", body=b"hello")
    # Same content → same id (content-hash dedup).
    assert a == b


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
    assert "system" in roles
    assert "user" in roles


def test_pack_with_trace_records_id(store, tmp_path):
    store.put(session=1, kind="user-message", body="x")
    trace_dir = str(tmp_path / "traces")
    result = store.pack(
        session=1, model="gpt-4o", task="reply", chat=True, trace=trace_dir
    )
    assert result.trace_id is not None
    assert len(result.trace_id) == 32


def test_trace_show_roundtrips_record(store, tmp_path):
    from llm386 import Trace

    store.put(session=42, kind="user-message", body="x")
    trace_dir = str(tmp_path / "traces")
    pack_result = store.pack(
        session=42, model="gpt-4o", task="reply", chat=True, trace=trace_dir
    )
    trace = Trace(trace_dir)
    record = trace.show(pack_result.trace_id)
    assert record.call_id == pack_result.trace_id
    assert record.session.endswith("2a")  # 42 in hex
    assert record.model == "gpt-4o"
    assert record.prompt_tokens > 0
    assert len(record.prompt_hash) == 64
    assert isinstance(record.started_at, int)


def test_trace_show_unknown_call_raises(tmp_path):
    from llm386 import LLM386Error, Trace

    trace = Trace(str(tmp_path / "empty-traces"))
    with pytest.raises(LLM386Error):
        trace.show("0" * 32)


def test_summarize_truncating_returns_text(store):
    for i in range(3):
        store.put(session=1, kind="fact", body=f"fact number {i}")
    out = store.summarize(session=1, summarizer="truncating", max_chars=50)
    assert "fact" in out


def test_summarize_store_summary_persists_block(store):
    for i in range(3):
        store.put(session=1, kind="fact", body=f"fact number {i}")
    before = len(store.list_sessions())
    store.summarize(session=1, summarizer="truncating", store_summary=True)
    # The summary block lands in the same session.
    assert len(store.list_sessions()) == before


def test_list_models_includes_built_ins():
    models = list_models()
    names = {m.name for m in models}
    assert "gpt-4o" in names
    assert "claude-opus-4-7" in names
    for m in models:
        assert m.max_context_tokens > 0


def test_unknown_model_raises_llm386_error(store):
    with pytest.raises(LLM386Error):
        store.page(session=1, model="bogus-model-name", task="x")


def test_show_unknown_block_raises_llm386_error(store):
    with pytest.raises(LLM386Error):
        store.show("0" * 32)


def test_delete_removes_block(store):
    block_id = store.put(session=1, kind="fact", body="to be deleted")
    assert store.delete(block_id) is True
    with pytest.raises(LLM386Error):
        store.show(block_id)


def test_delete_returns_false_for_unknown(store):
    assert store.delete("0" * 32) is False


def test_purge_session_removes_session_blocks(store):
    for i in range(4):
        store.put(session=1, kind="fact", body=f"fact {i}")
    sessions_before = store.list_sessions()
    assert "00000000000000000000000000000001" in sessions_before
    purged = store.purge_session(1)
    assert purged == 4
    sessions_after = store.list_sessions()
    assert "00000000000000000000000000000001" not in sessions_after


def test_purge_session_keeps_blocks_shared_with_other_sessions(store):
    a = store.put(session=1, kind="fact", body="shared")
    b = store.put(session=2, kind="fact", body="shared")
    assert a == b
    store.purge_session(1)
    # The block survives in session 2.
    block = store.show(b)
    assert block.body == b"shared"
