from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.file_outline import MAX_OUTLINE_SOURCE_BYTES
from chartreux.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileConfig,
    ReadFileState,
)
from chartreux.utils.io import WindowReadResult, content_revision
from tests.mock.utils import collect_result

PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "chartreux"
    / "core"
    / "tools"
    / "builtins"
    / "prompts"
    / "read_file.md"
)


def _tool() -> ReadFile:
    return ReadFile(lambda: ReadFileConfig(), ReadFileState())


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


@pytest.mark.asyncio
async def test_outline_mode_returns_symbols(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "sample.py",
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "\n"
        "class Beta:\n"
        "    def gamma(self) -> None:\n"
        "        pass\n",
    )

    result = await collect_result(
        _tool().run(ReadFileArgs(file_path=str(path), outline=True))
    )

    assert result.revision is not None
    assert result.next_offset is None
    assert result.traversal_id is None
    assert result.was_truncated is False
    assert "function alpha (lines 1-2)" in result.content
    assert "class Beta (lines 5-7)" in result.content
    assert "function Beta.gamma (lines 6-7)" in result.content
    # Outline-mode honesty: the line count is the rendered outline's, not a
    # symbol count, and pagination is not reported as requested.
    assert result.num_lines == len(result.content.splitlines())
    assert result.requested_limit is None


@pytest.mark.asyncio
async def test_outline_revision_pairing_survives_concurrent_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(tmp_path, "sample.py", "def alpha():\n    return 1\n")
    original_revision = content_revision(path.read_bytes())

    async def mutating_read_window(
        self: ReadFile, file_path: Path, **kwargs: object
    ) -> WindowReadResult:
        # A concurrent edit lands between the outline read and the revision
        # read of the old two-read scheme.
        path.write_text("def beta():\n    return 2\n")
        return WindowReadResult(
            lines=["def beta():"],
            total_lines=2,
            was_truncated=False,
            revision=content_revision(path.read_bytes()),
            raw_bytes=None,
            first_line_truncated=False,
            last_line_may_be_partial=False,
        )

    monkeypatch.setattr(ReadFile, "_read_window", mutating_read_window)
    result = await collect_result(
        _tool().run(ReadFileArgs(file_path=str(path), outline=True))
    )

    assert "function alpha (lines 1-2)" in result.content
    assert result.revision == original_revision


@pytest.mark.asyncio
async def test_outline_mode_non_python_falls_back(tmp_path: Path) -> None:
    path = _write(tmp_path, "notes.txt", "hello\nworld\n")

    result = await collect_result(
        _tool().run(ReadFileArgs(file_path=str(path), outline=True))
    )

    assert "vibe_warning" in result.content
    assert "does not outline" in result.content
    assert "1\u2192hello" in result.content
    assert "2\u2192world" in result.content


@pytest.mark.asyncio
async def test_outline_mode_over_cap_file_falls_back_without_full_read(
    tmp_path: Path,
) -> None:
    # A source past the outline cap must fall back at the guard, never
    # loading its full content into memory first.

    path = tmp_path / "huge.py"
    path.write_bytes(b"x = 1\n" * ((MAX_OUTLINE_SOURCE_BYTES // 6) + 2))

    result = await collect_result(
        _tool().run(ReadFileArgs(file_path=str(path), outline=True))
    )

    assert "vibe_warning" in result.content
    assert "does not outline" in result.content


@pytest.mark.asyncio
async def test_outline_mode_unparseable_falls_back(tmp_path: Path) -> None:
    path = _write(tmp_path, "broken.py", "def broken(:\n")

    result = await collect_result(
        _tool().run(ReadFileArgs(file_path=str(path), outline=True))
    )

    assert "vibe_warning" in result.content
    assert "does not outline" in result.content
    assert "1\u2192def broken(:" in result.content


@pytest.mark.asyncio
async def test_outline_mode_sensitive_path_falls_back(tmp_path: Path) -> None:
    path = _write(tmp_path, ".env.py", "def secret():\n    return 1\n")

    result = await collect_result(
        _tool().run(ReadFileArgs(file_path=str(path), outline=True))
    )

    assert "vibe_warning" in result.content
    assert "skipped for sensitive paths" in result.content
    assert "1\u2192def secret():" in result.content
    assert "function secret" not in result.content


@pytest.mark.asyncio
async def test_outline_mode_default_off_matches_plain_read(tmp_path: Path) -> None:
    path = _write(tmp_path, "plain.py", "x = 1\ny = 2\n")

    plain = await collect_result(_tool().run(ReadFileArgs(file_path=str(path))))
    explicit_off = await collect_result(
        _tool().run(ReadFileArgs(file_path=str(path), outline=False))
    )

    assert plain == explicit_off


def test_outline_mode_rejects_pagination_args() -> None:
    with pytest.raises(ValidationError):
        ReadFileArgs(file_path="/tmp/x.py", outline=True, offset=2)
    with pytest.raises(ValidationError):
        ReadFileArgs(file_path="/tmp/x.py", outline=True, limit=10)
    with pytest.raises(ValidationError):
        ReadFileArgs(file_path="/tmp/x.py", outline=True, traversal_id="abc")


def test_outline_prompt_anchor() -> None:
    text = PROMPT_PATH.read_text()

    assert "`outline: true`" in text
    assert "off by default" in text
