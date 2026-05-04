"""Python wrapper for the LLM386 context virtualization runtime.

The v0 implementation shells out to the `llm386` binary on every
call. A v1 PyO3-based implementation with the same public surface
is on the roadmap. Code written against this version should keep
working when the bindings land.
"""

from .api import LLM386Error, Store, Trace, list_models
from .types import (
    ChatMessage,
    ContextBlock,
    ModelProfile,
    OmittedBlock,
    PackResult,
    PagePlan,
    Provenance,
)

__all__ = [
    "ChatMessage",
    "ContextBlock",
    "LLM386Error",
    "ModelProfile",
    "OmittedBlock",
    "PackResult",
    "PagePlan",
    "Provenance",
    "Store",
    "Trace",
    "list_models",
]

__version__ = "0.1.0"
