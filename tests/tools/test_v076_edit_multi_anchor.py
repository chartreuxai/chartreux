"""P6c gates: expected_site_count and multi_anchor edit modes.

The edit tool gained two parameters (design v11): ``expected_site_count``,
which composes only with ``replace_all=true`` and turns a count mismatch into a
no-op with a bounded, line-keyed error, and ``multi_anchor``, an ordered list of
single-occurrence replacements that are all validated before one atomic write.
The combination matrix is mutually exclusive by construction: multi_anchor with
replace_all or expected_site_count is a parameter error, and the default mode is
unchanged.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pytest

from chartreux.core.tools.base import BaseToolState, ToolError
from chartreux.core.tools.builtins._edit_result_context import (
    EDIT_DIFF_MAX_SITES,
    EDIT_DIFF_NOTE_MAX_BYTES,
)
from chartreux.core.tools.builtins.edit import (
    AnchorEdit,
    Edit,
    EditArgs,
    EditConfig,
    EditResult,
)
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.utils import ToolPermission
from tests.mock.utils import collect_result


def _make_edit() -> Edit:
    return Edit(config_getter=lambda: EditConfig(), state=BaseToolState())


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def _hidden_note(diff: str) -> str:
    """Return the trailing hidden-site note, including its leading ellipsis."""
    marker = "\n… "
    return diff[diff.index(marker) + 1 :]


def _single(*anchors: tuple[str, str]) -> list[AnchorEdit]:
    return [AnchorEdit(old_string=old, new_string=new) for old, new in anchors]


@pytest.mark.asyncio
async def test_default_mode_still_replaces_only_the_first_occurrence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "a b c\n")

    result = await collect_result(
        _make_edit().run(EditArgs(file_path="f.txt", old_string="a", new_string="A"))
    )

    assert path.read_text() == "A b c\n"
    assert result.replaced_count == 1
    assert result.message == "The file has been updated successfully."


@pytest.mark.asyncio
async def test_default_mode_multiple_matches_is_still_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "a a\n")

    with pytest.raises(ToolError, match="Found 2 matches"):
        await collect_result(
            _make_edit().run(
                EditArgs(file_path="f.txt", old_string="a", new_string="A")
            )
        )

    assert path.read_text() == "a a\n"


@pytest.mark.asyncio
async def test_expected_site_count_match_replaces_every_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "tgt\ntgt\ntgt\ntgt\n")

    result = await collect_result(
        _make_edit().run(
            EditArgs(
                file_path="f.txt",
                old_string="tgt",
                new_string="TGT",
                replace_all=True,
                expected_site_count=4,
            )
        )
    )

    assert path.read_text() == "TGT\nTGT\nTGT\nTGT\n"
    assert result.replaced_count == 4
    assert result.message == (
        "The file has been updated. All occurrences were successfully replaced"
    )


@pytest.mark.asyncio
async def test_expected_site_count_mismatch_is_a_bounded_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "".join(f"tgt {index}\n" for index in range(12))
    path = _write(tmp_path, "f.txt", content)

    with pytest.raises(ToolError) as excinfo:
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    old_string="tgt",
                    new_string="TGT",
                    replace_all=True,
                    expected_site_count=5,
                )
            )
        )

    message = str(excinfo.value)
    # Nothing was written: a count mismatch is a no-op, not a partial edit.
    assert path.read_text() == content
    assert "expected_site_count is 5" in message
    assert "matched 12 site(s)" in message
    assert "no changes were made" in message
    # The error names the first K sites by line number and stays bounded.
    assert f"First {EDIT_DIFF_MAX_SITES} site(s) at lines " in message
    assert "1, 2, 3, 4, 5, 6, 7, 8" in message
    assert "(+4 more)" in message
    assert len(message.encode("utf-8")) <= EDIT_DIFF_NOTE_MAX_BYTES


@pytest.mark.asyncio
async def test_expected_site_count_error_bound_holds_for_many_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "".join(f"tgt {index}\n" for index in range(500))
    path = _write(tmp_path, "f.txt", content)

    with pytest.raises(ToolError) as excinfo:
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    old_string="tgt",
                    new_string="TGT",
                    replace_all=True,
                    expected_site_count=1,
                )
            )
        )

    message = str(excinfo.value)
    assert path.read_text() == content
    assert "matched 500 site(s)" in message
    # The echo cannot grow with the number of sites: K named lines + a remainder.
    assert f"First {EDIT_DIFF_MAX_SITES} site(s) at lines " in message
    assert "(+492 more)" in message
    assert len(message.encode("utf-8")) <= EDIT_DIFF_NOTE_MAX_BYTES


@pytest.mark.asyncio
async def test_multi_anchor_applies_every_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "alpha\nbeta\ngamma\n")

    result = await collect_result(
        _make_edit().run(
            EditArgs(
                file_path="f.txt",
                multi_anchor=_single(("alpha", "ALPHA"), ("gamma", "GAMMA")),
            )
        )
    )

    assert path.read_text() == "ALPHA\nbeta\nGAMMA\n"
    assert result.replaced_count == 2
    assert result.message == (
        "The file has been updated. Every anchor was replaced exactly once"
    )
    assert result.ui_start_lines == [1, 3]


@pytest.mark.asyncio
async def test_multi_anchor_lands_in_one_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "one\ntwo\nthree\n")
    writes: list[str] = []
    original_write = Edit._write_file

    async def recording_write(
        self: Edit,
        file_path: Path,
        content: str,
        encoding: str,
        newline: str,
        tool_io: ToolIOPort | None = None,
    ) -> None:
        writes.append(content)
        await original_write(self, file_path, content, encoding, newline, tool_io)

    monkeypatch.setattr(Edit, "_write_file", recording_write)

    await collect_result(
        _make_edit().run(
            EditArgs(
                file_path="f.txt", multi_anchor=_single(("one", "1"), ("three", "3"))
            )
        )
    )

    # The whole edit is assembled first and written exactly once.
    assert writes == ["1\ntwo\n3\n"]


@pytest.mark.asyncio
async def test_multi_anchor_any_failure_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "alpha\nbeta\nbeta\ngamma\n"
    path = _write(tmp_path, "f.txt", content)
    writes: list[str] = []

    async def recording_write(
        self: Edit,
        file_path: Path,
        written: str,
        encoding: str,
        newline: str,
        tool_io: ToolIOPort | None = None,
    ) -> None:
        writes.append(written)

    monkeypatch.setattr(Edit, "_write_file", recording_write)

    with pytest.raises(ToolError) as excinfo:
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    multi_anchor=_single(("alpha", "ALPHA"), ("beta", "BETA")),
                )
            )
        )

    message = str(excinfo.value)
    assert "multi_anchor[1] must match exactly once but matched 2 site(s)" in message
    assert "no changes were made" in message
    assert "First 2 site(s) at lines 2, 3." in message
    # The first anchor was valid, yet nothing was written at all.
    assert writes == []
    assert path.read_text() == content


@pytest.mark.asyncio
async def test_multi_anchor_zero_match_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "alpha\n"
    path = _write(tmp_path, "f.txt", content)

    with pytest.raises(
        ToolError, match=r"multi_anchor\[1\] must match exactly once but matched 0"
    ):
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    multi_anchor=_single(("alpha", "A"), ("missing", "M")),
                )
            )
        )

    assert path.read_text() == content


@pytest.mark.asyncio
async def test_multi_anchor_requires_file_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "first\nsecond\n"
    path = _write(tmp_path, "f.txt", content)

    with pytest.raises(ToolError, match="not in file order"):
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    multi_anchor=_single(("second", "S"), ("first", "F")),
                )
            )
        )

    assert path.read_text() == content


@pytest.mark.asyncio
async def test_multi_anchor_rejects_overlapping_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "abcdef\n"
    path = _write(tmp_path, "f.txt", content)

    with pytest.raises(ToolError, match="overlaps multi_anchor\\[0\\]"):
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    multi_anchor=_single(("abcd", "X"), ("cdef", "Y")),
                )
            )
        )

    assert path.read_text() == content


@pytest.mark.asyncio
async def test_multi_anchor_write_failure_leaves_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "alpha\nbeta\n"
    path = _write(tmp_path, "f.txt", content)

    def failing_replace(
        file_path: Path, new_content: str, *, encoding: str, newline: str
    ) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        "chartreux.core.tools.builtins.edit.atomic_replace", failing_replace
    )

    with pytest.raises(ToolError, match="OS error accessing"):
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    multi_anchor=_single(("alpha", "A"), ("beta", "B")),
                )
            )
        )

    assert path.read_text() == content


@pytest.mark.asyncio
async def test_multi_anchor_crash_after_replace_leaves_whole_new_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "alpha\nbeta\n")
    module = "chartreux.core.tools.builtins.edit"
    real_replace = __import__(module, fromlist=["atomic_replace"]).atomic_replace

    async def crashing_replace(
        file_path: Path, new_content: str, *, encoding: str, newline: str
    ) -> None:
        await real_replace(file_path, new_content, encoding=encoding, newline=newline)
        raise OSError("crash after the atomic rename")

    monkeypatch.setattr(f"{module}.atomic_replace", crashing_replace)

    with pytest.raises(ToolError, match="OS error accessing"):
        await collect_result(
            _make_edit().run(
                EditArgs(
                    file_path="f.txt",
                    multi_anchor=_single(("alpha", "A"), ("beta", "B")),
                )
            )
        )

    # The atomic write landed completely: never a partially spliced file.
    assert path.read_text() == "A\nB\n"


@pytest.mark.asyncio
async def test_multi_anchor_evidence_uses_current_line_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "alpha\nbeta\ngamma\ndelta\n")

    result = await collect_result(
        _make_edit().run(
            EditArgs(
                file_path="f.txt",
                multi_anchor=_single(
                    ("alpha", "A"), ("gamma", "G1\nG2"), ("delta", "D")
                ),
            )
        )
    )

    assert path.read_text() == "A\nbeta\nG1\nG2\nD\n"
    assert result.replaced_count == 3
    assert result.applied_diff is not None
    # Sites are keyed to current (post-edit) line numbers: the second anchor
    # inserts a line, so the third anchor has moved down by one.
    assert "@@ line 1 (current) @@" in result.applied_diff
    assert "@@ line 3 (current) @@" in result.applied_diff
    assert "@@ line 5 (current) @@" in result.applied_diff
    assert result.context is not None
    assert "1→A" in result.context


@pytest.mark.asyncio
async def test_multi_anchor_hidden_sites_are_named_by_current_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    lines = "".join(f"m{index:02d}\n" for index in range(12))
    path = _write(tmp_path, "f.txt", lines)
    anchors = tuple((f"m{index:02d}", f"M{index:02d}\nextra") for index in range(12))

    result = await collect_result(
        _make_edit().run(EditArgs(file_path="f.txt", multi_anchor=_single(*anchors)))
    )

    assert result.replaced_count == 12
    assert result.applied_diff is not None
    assert result.applied_diff.count("@@ line") == EDIT_DIFF_MAX_SITES
    note = _hidden_note(result.applied_diff)
    assert note.startswith("… 4 more replaced site(s) not shown (total 12).")
    # Every earlier anchor inserted a line, so site i now sits at 2*i+1.
    assert "Hidden lines: 17, 19, 21, 23." in note
    current = path.read_text().splitlines()
    for line in (17, 19, 21, 23):
        assert current[line - 1].startswith("M")


@pytest.mark.asyncio
async def test_expected_count_mode_publishes_hidden_line_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "".join(f"tgt {index}\n" for index in range(12)))

    result = await collect_result(
        _make_edit().run(
            EditArgs(
                file_path="f.txt",
                old_string="tgt",
                new_string="TGT",
                replace_all=True,
                expected_site_count=12,
            )
        )
    )

    assert result.replaced_count == 12
    assert result.applied_diff is not None
    note = _hidden_note(result.applied_diff)
    assert note.startswith("… 4 more replaced site(s) not shown (total 12).")
    assert "Hidden lines: 9, 10, 11, 12." in note


_MULTI = [
    {"old_string": "x1", "new_string": "A"},
    {"old_string": "x2", "new_string": "B"},
]


@pytest.mark.parametrize(
    ("args_kwargs", "expected_error"),
    [
        ({"old_string": "x1", "new_string": "X1"}, None),
        ({"old_string": "x", "new_string": "X", "replace_all": True}, None),
        (
            {"old_string": "x", "new_string": "X", "expected_site_count": 3},
            "expected_site_count requires replace_all=true",
        ),
        (
            {
                "old_string": "x",
                "new_string": "X",
                "replace_all": True,
                "expected_site_count": 3,
            },
            None,
        ),
        (
            {
                "old_string": "x",
                "new_string": "X",
                "replace_all": True,
                "expected_site_count": 2,
            },
            "expected_site_count is 2",
        ),
        (
            {
                "old_string": "x",
                "new_string": "X",
                "replace_all": True,
                "expected_site_count": 0,
            },
            "expected_site_count must be at least 1",
        ),
        ({"multi_anchor": _MULTI}, None),
        (
            {"multi_anchor": _MULTI, "replace_all": True},
            "multi_anchor cannot be combined with replace_all=true",
        ),
        (
            {"multi_anchor": _MULTI, "expected_site_count": 3},
            "multi_anchor cannot be combined with expected_site_count",
        ),
        (
            {"multi_anchor": _MULTI, "old_string": "x1", "new_string": "A"},
            "multi_anchor replaces old_string/new_string",
        ),
        (
            {"multi_anchor": _MULTI, "new_string": "A"},
            "multi_anchor replaces old_string/new_string",
        ),
        ({"multi_anchor": []}, "multi_anchor must contain at least one anchor"),
        (
            {"multi_anchor": [{"old_string": "x1", "new_string": "x1"}]},
            "makes no change",
        ),
        (
            {"multi_anchor": [{"old_string": "", "new_string": "A"}]},
            "old_string cannot be empty",
        ),
        ({}, "old_string and new_string are required unless multi_anchor is used"),
        (
            {"old_string": "x1"},
            "old_string and new_string are required unless multi_anchor is used",
        ),
    ],
)
@pytest.mark.asyncio
async def test_mode_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args_kwargs: dict[str, Any],
    expected_error: str | None,
) -> None:
    """Every parameter combination resolves to its one defined semantics."""
    monkeypatch.chdir(tmp_path)
    content = "x1\nx2\nx3\n"
    path = _write(tmp_path, "f.txt", content)
    args = EditArgs(file_path="f.txt", **args_kwargs)

    if expected_error is None:
        result = await collect_result(_make_edit().run(args))
        assert result.replaced_count >= 1
        assert path.read_text() != content
        return

    with pytest.raises(ToolError, match=re.escape(expected_error)):
        await collect_result(_make_edit().run(args))
    assert path.read_text() == content


def test_authority_checks_are_unchanged_for_the_new_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    inside = _write(tmp_path, "f.txt", "x\n")
    edit = _make_edit()
    new_mode_args = [
        {"multi_anchor": _MULTI},
        {
            "old_string": "x",
            "new_string": "X",
            "replace_all": True,
            "expected_site_count": 3,
        },
    ]

    for kwargs in new_mode_args:
        denied = edit.resolve_permission(EditArgs(file_path="/tmp/file.py", **kwargs))
        assert isinstance(denied, PermissionContext)
        assert denied.permission is ToolPermission.NEVER
        assert denied.reason

        allowed = edit.resolve_permission(EditArgs(file_path=str(inside), **kwargs))
        assert isinstance(allowed, PermissionContext)
        assert allowed.permission is ToolPermission.ALWAYS


def test_multi_anchor_call_display_lists_anchors() -> None:
    args = EditArgs(file_path="/abs/foo.py", multi_anchor=_single(("a", "b")))

    display = Edit.format_call_display(args)

    assert display.summary == "Editing /abs/foo.py"
    assert display.content is not None
    assert "multi_anchor[0]: 'a' -> 'b'" in display.content
    assert "old_string: None" not in display.content


def test_multi_anchor_result_projection_uses_anchor_occurrences() -> None:
    result = EditResult(file="/x", message="m", old_string="", new_string="")
    result._ui_occurrences = [(1, "a", "A"), (4, "b", "B")]

    assert Edit.project_result(result) == {
        "file": "/x",
        "old_string": "",
        "new_string": "",
        "occurrences": [
            {"start_line": 1, "old_text": "a", "new_text": "A"},
            {"start_line": 4, "old_text": "b", "new_text": "B"},
        ],
    }


def test_description_teaches_exactly_the_shipped_parameters() -> None:
    description = Edit.get_full_description()

    assert description
    for field in EditArgs.model_fields:
        assert f"`{field}`" in description
    assert "exactly once" in description
    assert "file order" in description
    assert "atomic write" in description
    assert "cannot be combined" in description

    # Every snake_case identifier the description names is a shipped schema
    # field (or the read_file tool), so the text cannot drift from the schema.
    shipped = set(EditArgs.model_fields) | set(EditResult.model_fields)
    shipped.add("read_file")
    for identifier in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", description):
        assert identifier in shipped, identifier


def test_schema_exposes_the_shipped_parameters() -> None:
    parameters = Edit.get_parameters()

    assert set(parameters["properties"]) == set(EditArgs.model_fields)
    assert "expected_site_count" in parameters["properties"]
    assert "multi_anchor" in parameters["properties"]
    assert "AnchorEdit" in parameters["$defs"]
