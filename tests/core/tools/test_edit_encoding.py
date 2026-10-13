from __future__ import annotations

import os
from pathlib import Path
import re
import stat

import pytest

from chartreux.core.tools.base import BaseToolState, ToolError
from chartreux.core.tools.builtins.edit import Edit, EditArgs, EditConfig
from chartreux.utils.io import content_revision
from tests.mock.utils import collect_result


@pytest.mark.asyncio
async def test_edit_rewrites_with_detected_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "utf16.txt"
    original = "line one café\nline two été\n"
    path.write_bytes(original.encode("utf-16"))

    tool = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())
    await collect_result(
        tool.run(
            EditArgs(
                file_path=str(path),
                old_string="line one café",
                new_string="LINE ONE CAFÉ",
            )
        )
    )

    assert path.read_bytes().startswith(b"\xff\xfe")
    assert path.read_text(encoding="utf-16") == "LINE ONE CAFÉ\nline two été\n"


@pytest.mark.asyncio
async def test_edit_preserves_file_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "script.sh"
    path.write_text("#!/bin/sh\necho old\n")
    os.chmod(path, 0o755)

    tool = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())
    await collect_result(
        tool.run(EditArgs(file_path=str(path), old_string="old", new_string="new"))
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o755


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\r\n", "\r", "\n"])
async def test_edit_preserves_line_endings(
    newline: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "f.txt"
    original = newline.join(["alpha", "beta", "gamma"])
    path.write_bytes(original.encode("utf-8"))

    tool = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())
    await collect_result(
        tool.run(EditArgs(file_path=str(path), old_string="beta", new_string="BETA"))
    )

    assert path.read_bytes() == newline.join(["alpha", "BETA", "gamma"]).encode("utf-8")


@pytest.mark.asyncio
async def test_edit_binary_file_raises_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "blob.bin"
    # PNG header + some non-text payload.
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4)

    tool = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())
    with pytest.raises(ToolError, match="not valid text"):
        await collect_result(
            tool.run(EditArgs(file_path=str(path), old_string="PNG", new_string="JPG"))
        )


def _parse_window(context: str | None) -> dict[int, str]:
    """Parse the ``N→text`` entries of a rendered context window."""
    assert context is not None
    entries: dict[int, str] = {}
    for line in context.splitlines():
        match = re.match(r"^\s*(\d+)→(.*)$", line)
        if match:
            entries[int(match.group(1))] = match.group(2)
    return entries


@pytest.mark.asyncio
async def test_edit_evidence_revision_hashes_raw_bytes_for_utf16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "utf16.txt"
    original = "line one café\nline two été\n"
    path.write_bytes(original.encode("utf-16"))

    tool = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())
    result = await collect_result(
        tool.run(
            EditArgs(
                file_path=str(path),
                old_string="line one café",
                new_string="LINE ONE CAFÉ",
            )
        )
    )

    # The revision identifies the raw post-edit bytes (BOM included), not the
    # decoded text and not host-side metadata.
    raw = path.read_bytes()
    assert result.revision is not None
    assert result.revision.revision == content_revision(raw)
    assert result.revision.size_bytes == len(raw)
    current = path.read_text(encoding="utf-16").splitlines()
    window = _parse_window(result.context)
    assert window
    for number, text in window.items():
        assert current[number - 1] == text
    assert window[1] == "LINE ONE CAFÉ"


@pytest.mark.asyncio
async def test_edit_evidence_revision_hashes_crlf_raw_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "f.txt"
    path.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

    tool = Edit(config_getter=lambda: EditConfig(), state=BaseToolState())
    result = await collect_result(
        tool.run(EditArgs(file_path=str(path), old_string="beta", new_string="BETA"))
    )

    raw = path.read_bytes()
    assert raw == b"alpha\r\nBETA\r\ngamma\r\n"
    assert result.revision is not None
    assert result.revision.revision == content_revision(raw)
    assert result.revision.size_bytes == len(raw)
    # The window shows the normalized current text with current line numbers.
    current = path.read_text().splitlines()
    assert current == ["alpha", "BETA", "gamma"]
    window = _parse_window(result.context)
    assert window
    for number, text in window.items():
        assert current[number - 1] == text
    assert window[2] == "BETA"
