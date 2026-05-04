"""Plain-data types returned by the LLM386 wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelProfile:
    """A model's context constraints."""

    name: str
    max_context_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    tokenizer: str
    supports_system_role: bool = True
    supports_tools: bool = True


@dataclass(frozen=True)
class Provenance:
    source: Optional[str] = None
    parents: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextBlock:
    """A stored block as returned by `Store.show`."""

    id: str
    kind: str
    body: bytes
    priority: float
    created_at: int
    updated_at: int
    hash: str
    provenance: Provenance


@dataclass(frozen=True)
class OmittedBlock:
    block_id: str
    reason: str
    score: float


@dataclass(frozen=True)
class PagePlan:
    """Result of `Store.page`."""

    selected: list[str]
    omitted: list[OmittedBlock]
    estimated_tokens: int


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass(frozen=True)
class PackResult:
    """Result of `Store.pack`. Either `rendered` or `messages` is set
    depending on the `chat` flag, never both."""

    rendered: Optional[str] = None
    messages: Optional[list[ChatMessage]] = None
    trace_id: Optional[str] = None
