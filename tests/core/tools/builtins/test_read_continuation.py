from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins.read_file import (
    CHUNK_TARGET_BYTES,
    MAX_BYTES,
    ReadFile,
    ReadFileArgs,
    ReadFileConfig,
    ReadFileResult,
    ReadFileState,
)
from chartreux.core.tools.io_port import ShellCommandRequest, ShellCommandResult
from chartreux.utils import VIBE_WARNING_TAG
from chartreux.utils.io import BoundedReadResult, ReadSafeResult, normalize_newlines
from tests.mock.utils import collect_result


def _make_read() -> ReadFile:
    return ReadFile(config_getter=lambda: ReadFileConfig(), state=ReadFileState())


def _big_file(tmp_path: Path, line_count: int = 300) -> tuple[Path, list[str]]:
    """A file whose rendered output needs several traversal chunks."""
    lines = [f"line {i:04d} " + "y" * 180 for i in range(1, line_count + 1)]
    target = tmp_path / "big.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target, lines


def _numbered_texts(content: str) -> list[str]:
    """Strip the 10-char line-number prefix (9 digits + arrow) from content."""
    return [rendered[10:] for rendered in content.split("\n")]


def _chunk_texts(result: ReadFileResult) -> list[str]:
    """Content texts of a traversal chunk (notice dropped, markers stripped)."""
    lines = result.content.split("\n")
    assert lines[0].startswith(f"<{VIBE_WARNING_TAG}>")
    texts: list[str] = []
    for rendered in lines[1:]:
        text = rendered[10:]
        if " …[+" in text:
            text = text.split(" …[+", 1)[0]
        texts.append(text)
    return texts


def _continue(tool: ReadFile, target: Path, result: ReadFileResult) -> ReadFileArgs:
    assert result.next_offset is not None
    assert result.traversal_id is not None
    return ReadFileArgs(
        file_path=str(target),
        offset=result.next_offset,
        traversal_id=result.traversal_id,
    )


class _FakeToolIO:
    """Minimal client-mediated read port: bounded windows, no revision channel."""

    supports_read = True
    supports_write = False
    supports_terminal = False

    def __init__(self, files: dict[Path, str]) -> None:
        self.files = files

    async def read_lines(
        self, path: Path, *, start_line: int, limit: int, max_bytes: int
    ) -> BoundedReadResult:
        del max_bytes
        lines = self.files[path].splitlines()
        selected = lines[start_line - 1 : start_line - 1 + limit]
        was_truncated = start_line - 1 + len(selected) < len(lines)
        return BoundedReadResult(selected, len(lines), was_truncated)

    async def read_text(self, path: Path) -> ReadSafeResult:
        text, newline = normalize_newlines(self.files[path])
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
        self.files[path] = content.replace("\n", newline) if newline else content

    async def run_shell(self, request: ShellCommandRequest) -> ShellCommandResult:
        raise NotImplementedError(request)


@pytest.mark.asyncio
async def test_traversal_covers_whole_file_without_gaps_or_dupes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target, lines = _big_file(tmp_path)
    tool = _make_read()

    chunks: list[ReadFileResult] = []
    args = ReadFileArgs(file_path=str(target))
    while True:
        result = await collect_result(tool.run(args))
        assert len(result.content.encode("utf-8")) <= MAX_BYTES
        chunks.append(result)
        if result.next_offset is None:
            break
        args = _continue(tool, target, result)

    # One traversal id, one revision, contiguous gapless duplicate-free chunks.
    assert {chunk.traversal_id for chunk in chunks} == {chunks[0].traversal_id}
    assert {chunk.revision for chunk in chunks} == {chunks[0].revision}
    assert chunks[0].revision is not None
    served: list[str] = []
    expected_start = 1
    for chunk in chunks:
        assert chunk.start_line == expected_start
        assert chunk.end_line is not None
        assert chunk.end_line == chunk.start_line + chunk.num_lines - 1
        assert chunk.total_lines == len(lines)
        served.extend(_chunk_texts(chunk))
        expected_start = chunk.end_line + 1
    assert served == lines


@pytest.mark.asyncio
async def test_traversal_reports_completion_at_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target, lines = _big_file(tmp_path)
    tool = _make_read()

    result = await collect_result(tool.run(ReadFileArgs(file_path=str(target))))
    while result.next_offset is not None:
        result = await collect_result(tool.run(_continue(tool, target, result)))

    assert result.next_offset is None
    assert result.was_truncated is False
    assert "the traversal is complete" in result.content
    assert result.end_line == len(lines)
    # The completed traversal's record is dropped from the registry.
    assert tool.state.traversals.get(result.traversal_id or "") is None


@pytest.mark.asyncio
async def test_traversal_chunk_stays_under_the_output_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target, _lines = _big_file(tmp_path)
    tool = _make_read()

    result = await collect_result(tool.run(ReadFileArgs(file_path=str(target))))

    # The chunk body targets 32 KiB of rendered output; the notice rides along
    # inside the 50 KiB hard cap.
    assert len(result.content.encode("utf-8")) <= CHUNK_TARGET_BYTES + 1024
    assert len(result.content.encode("utf-8")) <= MAX_BYTES
    assert result.num_lines > 1


@pytest.mark.asyncio
async def test_explicit_offset_and_limit_are_honored_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target, lines = _big_file(tmp_path)
    tool = _make_read()

    result = await collect_result(
        tool.run(ReadFileArgs(file_path=str(target), offset=5, limit=3))
    )

    # A bounded request is not an oversize event: exactly the requested
    # window, no traversal fields, no conversion.
    assert _numbered_texts(result.content) == lines[4:7]
    assert result.num_lines == 3
    assert result.start_line == 5
    assert result.end_line == 7
    assert result.next_offset is None
    assert result.traversal_id is None
    assert result.was_truncated is True
    # A truncated window is not a file-state revision.
    assert result.revision is None

    # An explicit limit alone is bounded too.
    limited = await collect_result(
        tool.run(ReadFileArgs(file_path=str(target), limit=5))
    )
    assert limited.num_lines == 5
    assert limited.start_line == 1
    assert limited.next_offset is None
    assert limited.traversal_id is None

    # A bounded read that covers the whole file does carry a revision.
    small = tmp_path / "small.txt"
    small.write_text("a\nb\n", encoding="utf-8")
    whole = await collect_result(
        tool.run(ReadFileArgs(file_path=str(small), offset=1, limit=10))
    )
    assert whole.revision is not None
    assert whole.next_offset is None


@pytest.mark.asyncio
async def test_revision_change_mid_traversal_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target, lines = _big_file(tmp_path)
    tool = _make_read()

    first = await collect_result(tool.run(ReadFileArgs(file_path=str(target))))
    opening_revision = first.revision
    assert opening_revision is not None

    # The file changes under the traversal (a line is inserted at the top).
    target.write_text("inserted\n" + "\n".join(lines) + "\n", encoding="utf-8")

    second = await collect_result(tool.run(_continue(tool, target, first)))

    assert second.restarted is True
    assert "changed" in second.content
    assert "beginning again from line 1" in second.content
    assert second.start_line == 1
    assert second.revision != opening_revision
    # The traversal keeps its identity so the agent continues seamlessly.
    assert second.traversal_id == first.traversal_id
    assert second.next_offset is not None

    third = await collect_result(tool.run(_continue(tool, target, second)))
    assert third.restarted is False
    assert second.end_line is not None
    assert third.start_line == second.end_line + 1


@pytest.mark.asyncio
async def test_unknown_traversal_id_starts_a_fresh_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target, _lines = _big_file(tmp_path)
    tool = _make_read()

    result = await collect_result(
        tool.run(
            ReadFileArgs(
                file_path=str(target), offset=10, traversal_id="tr-does-not-exist"
            )
        )
    )

    assert result.traversal_id is not None
    assert result.traversal_id != "tr-does-not-exist"
    assert "Unknown or expired traversal_id" in result.content
    assert result.start_line == 10
    assert result.next_offset is not None


@pytest.mark.asyncio
async def test_long_line_is_sliced_and_reassembled_without_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    long_line = "z" * (3 * CHUNK_TARGET_BYTES)
    target = tmp_path / "wide.txt"
    target.write_text(f"first\n{long_line}\nlast\n", encoding="utf-8")
    tool = _make_read()

    served: list[str] = []
    args = ReadFileArgs(file_path=str(target))
    calls = 0
    while True:
        result = await collect_result(tool.run(args))
        assert len(result.content.encode("utf-8")) <= MAX_BYTES
        calls += 1
        assert calls < 100  # never loops forever
        texts = _chunk_texts(result)
        assert len(texts) == 1
        served.append(texts[0])
        if result.next_offset is None:
            break
        if result.line_continues:
            assert "bytes in this line" in result.content
            assert result.next_offset == result.start_line
        args = _continue(tool, target, result)

    assert served[0] == "first"
    assert served[-1] == "last"
    # Every slice of the long line reassembles into the original line.
    assert "".join(served[1:-1]) == long_line
    assert len(served) >= 4  # the line really spanned multiple chunks


@pytest.mark.asyncio
async def test_empty_file_starts_no_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")
    tool = _make_read()

    result = await collect_result(tool.run(ReadFileArgs(file_path=str(target))))

    assert result.num_lines == 0
    assert result.total_lines == 0
    assert result.next_offset is None
    assert result.traversal_id is None
    assert result.revision is not None
    assert "empty" in result.content


@pytest.mark.asyncio
async def test_port_path_traversal_degrades_without_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    target, lines = _big_file(tmp_path)
    tool = _make_read()
    tool_io = _FakeToolIO({target: "\n".join(lines) + "\n"})
    ctx = InvokeContext(tool_call_id="call", tool_io=tool_io)

    first = await collect_result(tool.run(ReadFileArgs(file_path=str(target)), ctx))

    # Client-mediated reads cannot provide a content revision: the field is
    # absent (degraded) rather than invented, and the traversal still works.
    assert first.revision is None
    assert first.next_offset is not None
    assert first.traversal_id is not None
    assert first.total_lines == len(lines)

    second = await collect_result(tool.run(_continue(tool, target, first), ctx))
    assert second.revision is None
    assert second.traversal_id == first.traversal_id
    assert first.end_line is not None
    assert second.start_line == first.end_line + 1

    # The traversal completes through the port as well.
    result = second
    while result.next_offset is not None:
        result = await collect_result(tool.run(_continue(tool, target, result), ctx))
    assert result.end_line == len(lines)
    assert "the traversal is complete" in result.content
