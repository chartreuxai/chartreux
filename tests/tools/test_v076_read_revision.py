"""Revision-aware read_file: changed detection, genuine unchanged confirmation,
revert safety, and the conditional-read response contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.core.tools.base import InvokeContext, ToolError
from chartreux.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileConfig,
    ReadFileState,
)
from chartreux.core.tools.io_port import ShellCommandRequest, ShellCommandResult
from chartreux.utils.io import BoundedReadResult, ReadSafeResult, normalize_newlines
from tests.mock.utils import collect_result

CONTENT = "alpha\nbeta\ngamma\n"
CHANGED = "alpha\nBETA\ngamma\n"
ABSENT_REVISION = "blake2b-128:" + "0" * 32


def _tool(tmp_path: Path, config: ReadFileConfig | None = None) -> ReadFile:
    resolved = config or ReadFileConfig()
    return ReadFile(lambda: resolved, ReadFileState(), cwd=tmp_path)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


async def _initial_revision(tool: ReadFile, path: Path) -> str:
    result = await collect_result(tool.run(ReadFileArgs(file_path=str(path))))
    assert result.revision is not None
    return result.revision


class _FakeToolIO:
    """Client-mediated read port: bounded windows, no revision channel."""

    supports_read = True
    supports_write = False
    supports_terminal = False

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files

    async def read_lines(
        self, path: Path, *, start_line: int, limit: int, max_bytes: int
    ) -> BoundedReadResult:
        del max_bytes
        lines = self.files[str(path)].splitlines()
        selected = lines[start_line - 1 : start_line - 1 + limit]
        was_truncated = start_line - 1 + len(selected) < len(lines)
        return BoundedReadResult(selected, len(lines), was_truncated)

    async def read_text(self, path: Path) -> ReadSafeResult:
        text, newline = normalize_newlines(self.files[str(path)])
        return ReadSafeResult(text, "utf-8", newline)

    async def write_text(
        self,
        path: Path,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        del encoding
        self.files[str(path)] = content.replace("\n", newline) if newline else content

    async def run_shell(self, request: ShellCommandRequest) -> ShellCommandResult:
        raise NotImplementedError(request)


@pytest.mark.asyncio
async def test_changed_content_detected(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    target = _write(tmp_path, "notes.txt", CONTENT)
    before = await _initial_revision(tool, target)
    target.write_text(CHANGED, encoding="utf-8")

    result = await collect_result(
        tool.run(ReadFileArgs(file_path=str(target), expected_revision=before))
    )

    assert result.unchanged is False
    assert result.expected_revision == before
    assert result.revision is not None
    assert result.revision != before
    # The change is stated with both revisions, and the current content is
    # served through the normal read path.
    assert "The file changed since revision" in result.content
    assert before in result.content
    assert result.revision in result.content
    assert "BETA" in result.content
    assert result.num_lines == 3


@pytest.mark.asyncio
async def test_unchanged_confirmed_after_checking_current_content(
    tmp_path: Path,
) -> None:
    tool = _tool(tmp_path)
    target = _write(tmp_path, "notes.txt", CONTENT)
    revision = await _initial_revision(tool, target)

    result = await collect_result(
        tool.run(ReadFileArgs(file_path=str(target), expected_revision=revision))
    )

    assert result.unchanged is True
    assert result.revision == revision
    assert result.expected_revision == revision
    assert result.content.startswith("Unchanged: ")
    assert str(target) in result.content
    assert revision in result.content
    # No file body: that omission is the whole point of a conditional read.
    assert "alpha" not in result.content
    # The declaration names both checks it performed.
    assert "read authority" in result.content
    assert "current content through the tool read path" in result.content
    assert "blake2b-128 over" in result.content


@pytest.mark.asyncio
async def test_unchanged_response_contract(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    target = _write(tmp_path, "notes.txt", CONTENT)
    revision = await _initial_revision(tool, target)

    result = await collect_result(
        tool.run(ReadFileArgs(file_path=str(target), expected_revision=revision))
    )

    assert result.checked == [
        "read authority for this path under tool policy",
        "current content through the tool read path (blake2b-128 over "
        "17 bytes, 3 lines)",
    ]
    assert result.num_lines == 0
    assert result.start_line == 1
    assert result.total_lines == 3
    assert result.was_truncated is False
    assert result.end_line is None
    assert result.next_offset is None
    assert result.traversal_id is None
    assert result.requested_offset is None
    assert result.requested_limit is None
    assert result.revision is not None
    assert result.revision.startswith("blake2b-128:")
    assert len(result.revision.split(":", 1)[1]) == 32


@pytest.mark.asyncio
async def test_revert_is_content_derived_not_mtime(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    target = _write(tmp_path, "notes.txt", CONTENT)
    revision = await _initial_revision(tool, target)
    original_mtime = target.stat().st_mtime_ns

    target.write_text(CHANGED, encoding="utf-8")
    target.write_text(CONTENT, encoding="utf-8")
    # Force a metadata change so an mtime-based check would report "changed".
    os.utime(target, ns=(original_mtime + 10**9, original_mtime + 10**9))
    assert target.stat().st_mtime_ns != original_mtime

    result = await collect_result(
        tool.run(ReadFileArgs(file_path=str(target), expected_revision=revision))
    )

    # Identical bytes are genuinely unchanged regardless of the timestamp.
    assert result.unchanged is True
    assert result.revision == revision
    assert "alpha" not in result.content


@pytest.mark.asyncio
async def test_unchanged_never_declared_without_read_authority(tmp_path: Path) -> None:
    tool = _tool(tmp_path, ReadFileConfig(sensitive_patterns=["*.secret"]))
    target = _write(tmp_path, "keys.secret", CONTENT)
    revision = await _initial_revision(tool, target)

    with pytest.raises(ToolError, match="Sensitive file access denied"):
        await collect_result(
            tool.run(ReadFileArgs(file_path=str(target), expected_revision=revision))
        )


@pytest.mark.parametrize(
    "extra",
    [{"offset": 2}, {"limit": 5}, {"traversal_id": "tr-example"}, {"outline": True}],
)
def test_expected_revision_is_a_whole_file_check(
    tmp_path: Path, extra: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="expected_revision"):
        ReadFileArgs(
            file_path=str(tmp_path / "notes.txt"),
            expected_revision=ABSENT_REVISION,
            **extra,
        )


@pytest.mark.asyncio
async def test_client_mediated_read_degrades_to_normal_read(tmp_path: Path) -> None:
    target = _write(tmp_path, "notes.txt", CONTENT)
    tool = _tool(tmp_path)
    ctx = InvokeContext(
        tool_call_id="call", tool_io=_FakeToolIO({str(target): CONTENT})
    )

    result = await collect_result(
        tool.run(
            ReadFileArgs(file_path=str(target), expected_revision=ABSENT_REVISION), ctx
        )
    )

    # A read path with no full-state revision never declares unchanged.
    assert result.unchanged is False
    assert result.expected_revision == ABSENT_REVISION
    assert "Conditional read is unavailable" in result.content
    assert "alpha" in result.content


@pytest.mark.asyncio
async def test_normal_read_is_unchanged(tmp_path: Path) -> None:
    tool = _tool(tmp_path)
    target = _write(tmp_path, "notes.txt", CONTENT)

    result = await collect_result(tool.run(ReadFileArgs(file_path=str(target))))

    assert result.unchanged is False
    assert result.expected_revision is None
    assert result.checked == []
    assert result.revision is not None
    assert result.num_lines == 3
    assert "alpha" in result.content
