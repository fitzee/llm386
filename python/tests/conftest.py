"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def store(tmp_path):
    """A fresh Store rooted in a tempdir."""
    from llm386 import Store

    return Store(str(tmp_path / "store"))
