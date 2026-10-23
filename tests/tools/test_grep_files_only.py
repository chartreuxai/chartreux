from __future__ import annotations

import pytest

from chartreux.core.tools.base import BaseToolState
from chartreux.core.tools.builtins.grep import Grep, GrepArgs, GrepToolConfig
from tests.mock.utils import collect_result


@pytest.fixture
def grep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig()
    return Grep(config_getter=lambda: config, state=BaseToolState())


@pytest.mark.asyncio
async def test_files_only_lists_paths(grep, tmp_path):
    (tmp_path / "beta.txt").write_text("other\nneedle again\n")
    (tmp_path / "alpha.txt").write_text("needle here\n")
    (tmp_path / "gamma.txt").write_text("nothing to see\n")

    result = await collect_result(grep.run(GrepArgs(pattern="needle", files_only=True)))

    assert result.matches.splitlines() == [
        str(tmp_path / "alpha.txt"),
        str(tmp_path / "beta.txt"),
    ]
    assert result.match_count == 2
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_files_only_dedupes(grep, tmp_path):
    (tmp_path / "single.txt").write_text("needle\nmore needle\nlast needle\n")
    (tmp_path / "empty.txt").write_text("nothing\n")

    result = await collect_result(grep.run(GrepArgs(pattern="needle", files_only=True)))

    assert result.matches.splitlines() == [str(tmp_path / "single.txt")]
    assert result.match_count == 3
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_files_only_caps_distinct_files(grep, tmp_path):
    for index in range(6):
        (tmp_path / f"{index:02d}-file.txt").write_text("needle\n")

    result = await collect_result(
        grep.run(GrepArgs(pattern="needle", files_only=True, max_matches=3))
    )

    # Which files are kept depends on the backend's traversal order; only
    # the count, distinctness, and truncation flag are deterministic.
    lines = result.matches.splitlines()
    assert len(lines) == 3
    assert set(lines) <= {str(tmp_path / f"{index:02d}-file.txt") for index in range(6)}
    assert result.match_count == 4
    assert result.was_truncated


@pytest.mark.asyncio
async def test_files_only_caps_output_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = GrepToolConfig(max_output_bytes=20)
    grep = Grep(config_getter=lambda: config, state=BaseToolState())
    names = [f"{index:02d}-with-a-long-name.txt" for index in range(5)]
    for name in names:
        (tmp_path / name).write_text("needle\n")
    full_listing = "\n".join(str(tmp_path / name) for name in names)
    assert len(full_listing) > 20

    result = await collect_result(grep.run(GrepArgs(pattern="needle", files_only=True)))

    assert result.matches == full_listing[:20]
    assert result.was_truncated


@pytest.mark.asyncio
async def test_files_only_byte_cap_multibyte(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = ["日本語-001.txt", "日本語-002.txt", "日本語-003.txt"]
    for name in names:
        (tmp_path / name).write_text("needle\n", encoding="utf-8")
    prefix = f"{tmp_path}/"
    # The old character-slice rule promises bytes but spends characters: two
    # characters past the ASCII prefix already exceed the cap in bytes.
    cap = len(prefix) + 2
    assert len((prefix + names[0][:2]).encode("utf-8")) > cap
    config = GrepToolConfig(max_output_bytes=cap)
    grep = Grep(config_getter=lambda: config, state=BaseToolState())

    result = await collect_result(grep.run(GrepArgs(pattern="needle", files_only=True)))

    assert len(result.matches.encode("utf-8")) <= cap
    assert result.was_truncated
    # A whole character that no longer fits is dropped, never split.
    assert result.matches == prefix


@pytest.mark.asyncio
async def test_plain_path_byte_cap_multibyte(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "日本語.txt").write_text("needle\n", encoding="utf-8")
    path_line = f"{tmp_path / '日本語.txt'}:1:needle"
    cut = path_line.index("日")
    assert path_line[:cut].isascii()
    # The cap reaches just past the first CJK character: its 3 UTF-8 bytes
    # exceed the character count, so the old character-slice rule (which
    # would keep prefix + "日本" here) violates the byte cap.
    cap = len(path_line[: cut + 1].encode("utf-8")) - 1
    config = GrepToolConfig(max_output_bytes=cap)
    grep = Grep(config_getter=lambda: config, state=BaseToolState())

    result = await collect_result(grep.run(GrepArgs(pattern="needle")))

    assert len(result.matches.encode("utf-8")) <= cap
    assert result.was_truncated
    assert result.match_count == 1
    # The whole character that no longer fits is dropped, never split.
    assert result.matches == path_line[:cut]


@pytest.mark.asyncio
async def test_files_only_default_off_matches_plain(grep, tmp_path):
    (tmp_path / "code.py").write_text("alpha needle\nmiddle\nbeta needle\n")

    default_result = await collect_result(grep.run(GrepArgs(pattern="needle")))
    explicit_result = await collect_result(
        grep.run(GrepArgs(pattern="needle", files_only=False))
    )

    assert default_result == explicit_result
    # The default output keeps the plain matched-line shape: path:line:content
    # with context lines, never a bare file list.
    assert default_result.matches.splitlines() == [
        f"{tmp_path / 'code.py'}:1:alpha needle",
        f"{tmp_path / 'code.py'}-2-middle",
        f"{tmp_path / 'code.py'}:3:beta needle",
    ]
    assert default_result.match_count == 2
    assert not default_result.was_truncated
