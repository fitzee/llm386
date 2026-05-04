"""High-level wrapper around the `llm386` CLI binary.

This is the v0 SDK: it shells out to the `llm386` binary for every
operation. It is correct, simple, and slow (one process per call).
A v1 PyO3-based SDK with the same surface is on the roadmap.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

from .types import (
    ChatMessage,
    ContextBlock,
    ModelProfile,
    OmittedBlock,
    PackResult,
    PagePlan,
    Provenance,
)

DEFAULT_BINARY = "llm386"

PathLike = Union[str, os.PathLike[str]]
IdLike = Union[str, int]


class LLM386Error(RuntimeError):
    """Raised when the underlying `llm386` invocation fails."""


def _id_to_hex(n: IdLike) -> str:
    """Normalize an id (int or hex string) to the canonical 32-char
    lowercase hex form the CLI emits."""
    if isinstance(n, int):
        return f"{n:032x}"
    return str(n)


def _bytes_from_ints(arr: list[int]) -> bytes:
    return bytes(arr)


def _hash_to_hex(arr: list[int]) -> str:
    return bytes(arr).hex()


class Store:
    """Wrapper around an LMDB store at `path`. Idempotently
    initialized on construction."""

    def __init__(
        self,
        path: PathLike,
        *,
        binary: str = DEFAULT_BINARY,
        profiles: Optional[PathLike] = None,
    ) -> None:
        self.path = str(Path(path))
        self.binary = binary
        self.profiles = str(profiles) if profiles is not None else None
        # `init` is idempotent; safe to call every construction.
        self._cli("init", self.path)

    # ----------------------- writes -----------------------

    def put(
        self,
        session: int,
        kind: str,
        body: Union[str, bytes],
        *,
        priority: float = 0.0,
    ) -> str:
        """Store `body` as a block. Returns the assigned BlockId
        (32-char lowercase hex)."""
        if isinstance(body, str):
            body = body.encode("utf-8")
        out = self._cli(
            "put",
            "--store",
            self.path,
            "--session",
            str(session),
            "--kind",
            kind,
            "--priority",
            str(priority),
            "-",
            stdin=body,
        )
        return out.strip()

    # ----------------------- reads ------------------------

    def list_sessions(self) -> list[str]:
        """All session ids that have at least one block."""
        out = self._cli("list-sessions", "--store", self.path)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def show(self, block_id: IdLike) -> ContextBlock:
        """Fetch a single block by id."""
        out = self._cli(
            "show", "--store", self.path, _id_to_hex(block_id), "--json"
        )
        data = json.loads(out)
        prov = data.get("provenance") or {}
        return ContextBlock(
            id=_id_to_hex(data["id"]),
            kind=data["kind"],
            body=_bytes_from_ints(data["bytes"]),
            priority=float(data.get("priority", 0.0)),
            created_at=int(data.get("created_at", 0)),
            updated_at=int(data.get("updated_at", 0)),
            hash=_hash_to_hex(data["hash"]),
            provenance=Provenance(
                source=prov.get("source"),
                parents=[_id_to_hex(p) for p in prov.get("parents", [])],
                labels=list(prov.get("labels", [])),
            ),
        )

    # ----------------------- pager ------------------------

    def page(self, session: int, model: str, task: str) -> PagePlan:
        """Run the pager and return the resulting plan."""
        out = self._cli(
            "page",
            "--store",
            self.path,
            "--session",
            str(session),
            "--model",
            model,
            "--task",
            task,
            "--json",
        )
        data = json.loads(out)
        return PagePlan(
            selected=[_id_to_hex(s) for s in data.get("selected", [])],
            omitted=[
                OmittedBlock(
                    block_id=_id_to_hex(o["block_id"]),
                    reason=o["reason"],
                    score=float(o["score"]),
                )
                for o in data.get("omitted", [])
            ],
            estimated_tokens=int(data.get("estimated_tokens", 0)),
        )

    # ----------------------- packer -----------------------

    def pack(
        self,
        session: int,
        model: str,
        task: str,
        *,
        chat: bool = False,
        trace: Optional[PathLike] = None,
    ) -> PackResult:
        """Run page+pack and return the rendered prompt or chat
        messages. With `trace=<path>`, also records a trace and
        returns its id."""
        args = [
            "pack",
            "--store",
            self.path,
            "--session",
            str(session),
            "--model",
            model,
            "--task",
            task,
        ]
        if chat:
            args.append("--chat")
        else:
            args.append("--prompt-only")
        if trace is not None:
            args.extend(["--trace", str(trace)])

        stdout, stderr = self._cli_split(*args)
        trace_id = _extract_trace_id(stderr) if trace is not None else None

        if chat:
            messages_raw = json.loads(stdout)
            messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages_raw]
            return PackResult(rendered=None, messages=messages, trace_id=trace_id)
        return PackResult(rendered=stdout, messages=None, trace_id=trace_id)

    # ----------------------- summarize --------------------

    def summarize(
        self,
        session: int,
        *,
        summarizer: str = "truncating",
        max_chars: int = 80,
        last: Optional[int] = None,
        store_summary: bool = False,
        anthropic_model: Optional[str] = None,
        anthropic_max_tokens: Optional[int] = None,
    ) -> str:
        """Summarize a session and return the summary text."""
        args = [
            "summarize",
            "--store",
            self.path,
            "--session",
            str(session),
            "--summarizer",
            summarizer,
            "--max-chars",
            str(max_chars),
        ]
        if last is not None:
            args.extend(["--last", str(last)])
        if store_summary:
            args.append("--store-summary")
        if anthropic_model is not None:
            args.extend(["--anthropic-model", anthropic_model])
        if anthropic_max_tokens is not None:
            args.extend(["--anthropic-max-tokens", str(anthropic_max_tokens)])
        return self._cli(*args)

    # ----------------------- internals --------------------

    def _cli(self, *args: str, stdin: Optional[bytes] = None) -> str:
        cmd = self._cmd(*args)
        try:
            result = subprocess.run(cmd, input=stdin, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise LLM386Error(_format_cli_error(cmd, e)) from e
        return result.stdout.decode("utf-8")

    def _cli_split(
        self, *args: str, stdin: Optional[bytes] = None
    ) -> tuple[str, str]:
        cmd = self._cmd(*args)
        try:
            result = subprocess.run(cmd, input=stdin, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise LLM386Error(_format_cli_error(cmd, e)) from e
        return result.stdout.decode("utf-8"), result.stderr.decode("utf-8")

    def _cmd(self, *args: str) -> list[str]:
        cmd = [self.binary]
        if self.profiles is not None:
            cmd.extend(["--profiles", self.profiles])
        cmd.extend(args)
        return cmd


class Trace:
    """Read-only wrapper around a trace store created by
    `Store.pack(trace=...)`."""

    def __init__(self, path: PathLike, *, binary: str = DEFAULT_BINARY) -> None:
        self.path = str(Path(path))
        self.binary = binary

    def show(self, call_id: IdLike) -> str:
        """Fetch the human-readable trace record. Returns the raw
        text the CLI emits; structured parsing lands when the CLI
        grows a `--json` flag for `trace show`."""
        try:
            result = subprocess.run(
                [self.binary, "trace", "show", "--store", self.path, _id_to_hex(call_id)],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise LLM386Error(_format_cli_error(["trace show"], e)) from e
        return result.stdout.decode("utf-8")


def list_models(
    *, binary: str = DEFAULT_BINARY, profiles: Optional[PathLike] = None
) -> list[ModelProfile]:
    """List available model profiles."""
    cmd = [binary]
    if profiles is not None:
        cmd.extend(["--profiles", str(profiles)])
    cmd.append("list-models")
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise LLM386Error(_format_cli_error(cmd, e)) from e
    return _parse_list_models_table(result.stdout.decode("utf-8"))


# ----------------------- helpers --------------------------


_TRACE_ID_RE = re.compile(r"#\s*trace_id:\s*([0-9a-fA-F]+)")


def _extract_trace_id(stderr: str) -> Optional[str]:
    m = _TRACE_ID_RE.search(stderr)
    return m.group(1).lower() if m else None


def _parse_list_models_table(output: str) -> list[ModelProfile]:
    """Parse the human-readable `list-models` table.

    The CLI doesn't have a `--json` flag for this subcommand yet, so
    we parse the columnar output. Tolerant of extra whitespace.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return []
    # First line is the header; skip it.
    profiles: list[ModelProfile] = []
    for line in lines[1:]:
        cols = line.split()
        if len(cols) < 5:
            continue
        try:
            profiles.append(
                ModelProfile(
                    name=cols[0],
                    max_context_tokens=int(cols[1]),
                    reserved_output_tokens=int(cols[2]),
                    safety_margin_tokens=int(cols[3]),
                    tokenizer=cols[4],
                )
            )
        except ValueError:
            continue
    return profiles


def _format_cli_error(cmd: list[str], e: subprocess.CalledProcessError) -> str:
    out = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
    err = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
    return (
        f"llm386 invocation failed: {' '.join(map(str, cmd))}\n"
        f"exit code: {e.returncode}\n"
        f"stdout: {out.strip()}\n"
        f"stderr: {err.strip()}"
    )
