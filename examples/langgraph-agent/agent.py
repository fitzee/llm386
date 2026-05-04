"""LangGraph + LLM386 demo agent.

A small chatbot that uses LangGraph for the agent loop (LLM call + tool
dispatch) and LLM386 as the memory layer (persistent block store, paging,
deterministic prompt assembly).

Run via the bundled Dockerfile:
    docker compose -f examples/langgraph-agent/docker-compose.yml run --rm agent

See README.md in this directory for details.
"""

from __future__ import annotations

import ast
import operator
import os
import sys
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from llm386 import Store


MODEL_NAME = "claude-haiku-4-5"


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2 + 3 * (4 - 1)'.

    Supports +, -, *, /, %, **, parentheses, and decimal numbers. Use this
    when the user asks any arithmetic question — even simple ones — so the
    answer is exact rather than guessed.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:  # noqa: BLE001
        return f"calculator error: {exc}"


_PROFILES: dict[str, dict[str, str]] = {
    "u-001": {"name": "Mira", "tier": "enterprise", "tz": "Europe/Berlin"},
    "u-002": {"name": "Diego", "tier": "free", "tz": "America/Bogota"},
    "u-003": {"name": "Aiko", "tier": "pro", "tz": "Asia/Tokyo"},
}


@tool
def user_profile(user_id: str) -> str:
    """Look up a user's profile by id (e.g. 'u-001').

    Returns name, subscription tier, and timezone. Use whenever the user
    asks about a specific user_id.
    """
    profile = _PROFILES.get(user_id)
    if profile is None:
        return f"no profile for {user_id}"
    return (
        f"name={profile['name']}, "
        f"tier={profile['tier']}, "
        f"tz={profile['tz']}"
    )


TOOLS = [calculator, user_profile]


# --------------------------------------------------------------------------
# LangGraph agent
# --------------------------------------------------------------------------


def build_agent() -> Any:
    llm = ChatAnthropic(model=MODEL_NAME, max_tokens=1024).bind_tools(TOOLS)

    # Use LangGraph's prebuilt `MessagesState`, whose `messages` field
    # carries the `add_messages` reducer. Without that reducer, nodes
    # that return `{"messages": [...]}` *overwrite* the list — and the
    # `ToolNode` returns just the new ToolMessage(s), so the next agent
    # call would see the ToolMessage without its preceding AIMessage,
    # which Anthropic rejects with "tool_result without matching
    # tool_use".
    def call_model(state: MessagesState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    g: StateGraph = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode(TOOLS))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


# --------------------------------------------------------------------------
# LLM386 ↔ LangChain bridge
# --------------------------------------------------------------------------


def to_langchain_messages(packed_messages) -> list[BaseMessage]:
    """Convert LLM386 ChatMessage objects to LangChain message types.

    LLM386 emits role-tagged messages (system / user / assistant / tool).
    LangChain's ToolMessage requires a `tool_call_id` linking back to the
    specific tool call in a prior AIMessage; that linkage isn't preserved
    through `pack()`. So replayed tool results from prior turns are
    surfaced as labeled HumanMessages — the model still sees them, just
    not in the canonical tool-call/result protocol.
    """
    out: list[BaseMessage] = []
    for m in packed_messages:
        if m.role == "system":
            out.append(SystemMessage(content=m.content))
        elif m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
        elif m.role == "tool":
            out.append(HumanMessage(content=f"[prior tool result] {m.content}"))
        else:
            out.append(HumanMessage(content=m.content))
    return out


# --------------------------------------------------------------------------
# Turn
# --------------------------------------------------------------------------


def turn(store: Store, session_id: int, user_input: str, agent: Any) -> str:
    # 1. Persist the user's message.
    store.put(session_id, kind="user-message", body=user_input)

    # 2. Compute the working set (page) and render it (pack).
    #    page() runs first so we can show what LLM386 selected; pack() is
    #    what actually gets sent to the model.
    plan = store.page(session=session_id, model=MODEL_NAME, task=user_input)
    packed = store.pack(
        session=session_id, model=MODEL_NAME, task=user_input, chat=True
    )
    initial = to_langchain_messages(packed.messages)

    print(
        f"[llm386] selected {len(plan.selected)} blocks "
        f"({plan.estimated_tokens} est. tokens, "
        f"{len(initial)} chat messages packed)",
        file=sys.stderr,
    )

    # 3. Run the LangGraph loop.
    result = agent.invoke({"messages": initial})
    new_messages = result["messages"][len(initial):]

    # 4. Persist what the agent produced. Tool results get linked to the
    #    assistant message that called them via a typed edge so the pager
    #    keeps them paired on subsequent turns.
    last_assistant_id: str | None = None
    final_reply: str | None = None

    for m in new_messages:
        if isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            tool_calls = getattr(m, "tool_calls", []) or []
            if content:
                last_assistant_id = store.put(
                    session_id, kind="assistant-message", body=content
                )
                final_reply = content
            elif tool_calls:
                names = ", ".join(c["name"] for c in tool_calls)
                marker = f"[calling tools: {names}]"
                last_assistant_id = store.put(
                    session_id, kind="assistant-message", body=marker
                )
        elif isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            tool_id = store.put(session_id, kind="tool-result", body=content)
            if last_assistant_id is not None:
                store.add_edge(last_assistant_id, tool_id, "tool-invocation")

    return final_reply or "(no response)"


# --------------------------------------------------------------------------
# REPL
# --------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a concise assistant for a demo of LLM386. "
    "You have two tools: a `calculator` for arithmetic and a `user_profile` "
    "lookup. Known profile ids include u-001, u-002, u-003. "
    "When the user asks about anything you've discussed in earlier turns, "
    "rely on the recalled context rather than asking them to repeat themselves."
)


def main() -> None:
    store_path = os.environ.get("LLM386_STORE", "/data/store")
    profiles_path = os.environ.get("LLM386_PROFILES", "/app/llm386.toml")
    session_id = int(os.environ.get("LLM386_SESSION", "1"))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required")

    store = Store(
        store_path,
        profiles=profiles_path if os.path.exists(profiles_path) else None,
    )

    # Persist (or dedup against) the system prompt for this session.
    # Re-running the demo is a no-op for the system block thanks to
    # content-hash dedup.
    store.put(session_id, kind="system", body=SYSTEM_PROMPT, priority=0.9)

    agent = build_agent()

    print("LLM386 + LangGraph demo. Type 'exit' to quit.")
    print(f"store={store_path} session={session_id} model={MODEL_NAME}\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            reply = turn(store, session_id, user_input, agent)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}\n", file=sys.stderr)
            continue
        print(f"bot> {reply}\n")


if __name__ == "__main__":
    main()
