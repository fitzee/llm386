"""Shared pytest fixtures.

The tests need the `llm386` binary. We build it once per test
session and point the wrapper at the resulting executable via
the `binary` constructor kwarg.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RELEASE_BINARY = WORKSPACE_ROOT / "target" / "release" / "llm386"


@pytest.fixture(scope="session")
def llm386_binary() -> str:
    """Path to the `llm386` binary. Honors `LLM386_BIN` if set;
    otherwise builds (or reuses) the release binary in the
    workspace `target/`."""
    override = os.environ.get("LLM386_BIN")
    if override:
        return override

    if not RELEASE_BINARY.exists():
        subprocess.run(
            ["cargo", "build", "--release", "-p", "llm386-cli"],
            cwd=WORKSPACE_ROOT,
            check=True,
        )
    return str(RELEASE_BINARY)


@pytest.fixture
def store(llm386_binary, tmp_path):
    """A fresh Store rooted in a tempdir, using the test binary."""
    from llm386 import Store

    return Store(tmp_path / "store", binary=llm386_binary)
