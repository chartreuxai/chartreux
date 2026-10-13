from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
import re

import pytest

from chartreux.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from chartreux.core.events import ToolResultEvent
from chartreux.core.file_revision import full_revision
from chartreux.core.tools.base import BaseToolState, ToolError
from chartreux.core.tools.builtins._edit_result_context import (
    EDIT_DIFF_MAX_SITES,
    EDIT_WINDOW_MAX_LINES,
)
from chartreux.core.tools.builtins.edit import Edit, EditArgs, EditConfig, EditResult
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay
from chartreux.core.trusted_folders import trusted_folders_manager
from chartreux.utils.io import content_revision
from tests.mock.utils import collect_result


def _make_edit() -> Edit:
    return Edit(config_getter=lambda: EditConfig(), state=BaseToolState())


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


@pytest.mark.asyncio
async def test_exact_match_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "hello world\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="hello", new_string="goodbye"))
    )

    assert (tmp_path / "f.txt").read_text() == "goodbye world\n"
    assert result.file == str(tmp_path / "f.txt")
    assert result.message == "The file has been updated successfully."


@pytest.mark.asyncio
async def test_replace_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "aaa bbb aaa\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(
                file_path="f.txt", old_string="aaa", new_string="ccc", replace_all=True
            )
        )
    )

    assert (tmp_path / "f.txt").read_text() == "ccc bbb ccc\n"
    assert result.message == (
        "The file has been updated. All occurrences were successfully replaced"
    )


@pytest.mark.asyncio
async def test_not_found_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "hello world\n")
    edit = _make_edit()

    with pytest.raises(ToolError, match="String to replace not found in file"):
        await collect_result(
            edit.run(EditArgs(file_path="f.txt", old_string="missing", new_string="x"))
        )


@pytest.mark.asyncio
async def test_multiple_matches_without_replace_all_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "aaa bbb aaa\n")
    edit = _make_edit()

    with pytest.raises(ToolError, match="Found 2 matches"):
        await collect_result(
            edit.run(EditArgs(file_path="f.txt", old_string="aaa", new_string="ccc"))
        )


@pytest.mark.asyncio
async def test_old_equals_new_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "hello\n")
    edit = _make_edit()

    with pytest.raises(ToolError, match="No changes to make"):
        await collect_result(
            edit.run(
                EditArgs(file_path="f.txt", old_string="hello", new_string="hello")
            )
        )


@pytest.mark.asyncio
async def test_empty_old_string_rejected_for_nonempty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "hello\n")
    edit = _make_edit()

    with pytest.raises(ToolError, match="only when populating an empty file"):
        await collect_result(
            edit.run(EditArgs(file_path="f.txt", old_string="", new_string="x"))
        )


@pytest.mark.asyncio
async def test_empty_old_string_populates_existing_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "")

    result = await collect_result(
        _make_edit().run(
            EditArgs(file_path="f.txt", old_string="", new_string="content")
        )
    )

    assert (tmp_path / "f.txt").read_text() == "content"
    assert result.old_string == ""


@pytest.mark.asyncio
async def test_file_not_found_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    edit = _make_edit()

    with pytest.raises(ToolError, match="File does not exist"):
        await collect_result(
            edit.run(
                EditArgs(
                    file_path="/nonexistent/file.py", old_string="x", new_string="y"
                )
            )
        )


@pytest.mark.asyncio
async def test_empty_file_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    edit = _make_edit()

    with pytest.raises(ToolError, match="File path cannot be empty"):
        await collect_result(
            edit.run(EditArgs(file_path="", old_string="x", new_string="y"))
        )


@pytest.mark.asyncio
async def test_deletion_removes_exact_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "line1\nline2\nline3\n")
    edit = _make_edit()

    await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="line2\n", new_string=""))
    )

    assert (tmp_path / "f.txt").read_text() == "line1\nline3\n"


@pytest.mark.asyncio
async def test_parallel_edits_same_file_all_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "A\nB\nC\nD\n")
    edit = _make_edit()

    await asyncio.gather(
        collect_result(
            edit.run(EditArgs(file_path="f.txt", old_string="A", new_string="A1"))
        ),
        collect_result(
            edit.run(EditArgs(file_path="f.txt", old_string="B", new_string="B1"))
        ),
        collect_result(
            edit.run(EditArgs(file_path="f.txt", old_string="C", new_string="C1"))
        ),
        collect_result(
            edit.run(EditArgs(file_path="f.txt", old_string="D", new_string="D1"))
        ),
    )

    assert (tmp_path / "f.txt").read_text() == "A1\nB1\nC1\nD1\n"


@pytest.mark.asyncio
async def test_relative_path_resolved_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir(exist_ok=True)
    _write(tmp_path / "sub", "f.txt", "old")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="sub/f.txt", old_string="old", new_string="new"))
    )

    assert (tmp_path / "sub" / "f.txt").read_text() == "new"
    assert result.file == str(tmp_path / "sub" / "f.txt")


def test_format_call_display() -> None:
    args = EditArgs(file_path="/abs/foo.py", old_string="old", new_string="new")
    display = Edit.format_call_display(args)

    assert isinstance(display, ToolCallDisplay)
    assert display.summary == "Editing /abs/foo.py"
    assert display.verb == "Editing"
    assert display.message == "/abs/foo.py"
    assert display.settled_verb == "Edited"
    assert display.settled_message == "/abs/foo.py"


def test_get_result_display() -> None:
    result = EditResult(
        file="/path/to/foo.py",
        message="The file has been updated successfully.",
        old_string="old",
        new_string="new",
    )
    event = ToolResultEvent(
        tool_call_id="test", tool_name="edit", tool_class=None, result=result
    )
    display = Edit.get_result_display(event)

    assert isinstance(display, ToolResultDisplay)
    assert display.success is True
    assert "foo.py" in display.message


def test_ui_hints_not_part_of_model_contract() -> None:
    result = EditResult(file="/x", message="m", old_string="a", new_string="b")
    result._ui_occurrences = [(42, "a", "b")]

    assert result.ui_start_lines == [42]
    assert result.ui_occurrences == [(42, "a", "b")]
    for key in ("ui_start_lines", "ui_occurrences", "_ui_occurrences"):
        assert key not in result.model_dump()
        assert key not in result.model_dump_json()
        assert key not in dict(result)
    assert "ui_occurrences" not in EditResult.model_fields
    assert "ui_occurrences" not in EditResult.model_json_schema().get("properties", {})


@pytest.mark.asyncio
async def test_ui_start_lines_computed_at_edit_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "alpha\nbeta\ngamma\ndelta\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="gamma", new_string="GAMMA"))
    )

    assert result.ui_start_lines == [3]


@pytest.mark.asyncio
async def test_ui_start_lines_set_for_pure_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "keep1\nkeep2\nremove\nkeep3\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="remove\n", new_string=""))
    )

    assert result.ui_start_lines == [3]


@pytest.mark.asyncio
async def test_ui_start_lines_lists_all_occurrences_for_replace_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "x\ntgt\ny\ntgt\nz\ntgt\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(
                file_path="f.txt", old_string="tgt", new_string="TGT", replace_all=True
            )
        )
    )

    assert result.ui_start_lines == [2, 4, 6]


@pytest.mark.asyncio
async def test_ui_start_lines_single_entry_without_replace_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "a\nuniq\nb\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="uniq", new_string="UNIQ"))
    )

    assert result.ui_start_lines == [2]


def test_ui_occurrences_fall_back_to_snippet_when_unset() -> None:
    result = EditResult(file="/x", message="m", old_string="a", new_string="b")

    assert result.ui_occurrences == [(None, "a", "b")]


@pytest.mark.asyncio
async def test_ui_occurrences_expand_mid_line_snippet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "foo = bar + baz\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="bar", new_string="qux"))
    )

    assert result.ui_occurrences == [(1, "foo = bar + baz", "foo = qux + baz")]


@pytest.mark.asyncio
async def test_ui_occurrences_for_pure_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "keep1\nkeep2\nremove\nkeep3\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="remove\n", new_string=""))
    )

    assert result.ui_occurrences == [(3, "remove\n", "")]


@pytest.mark.asyncio
async def test_ui_occurrences_use_per_occurrence_lines_for_replace_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "x = bar + 1\ny = bar - 2\nz = bar\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(
                file_path="f.txt", old_string="bar", new_string="qux", replace_all=True
            )
        )
    )

    assert result.ui_occurrences == [
        (1, "x = bar + 1", "x = qux + 1"),
        (2, "y = bar - 2", "y = qux - 2"),
        (3, "z = bar", "z = qux"),
    ]


@pytest.fixture()
def _setup_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(trusted_folders_manager, "is_trusted", lambda _: True)
    monkeypatch.setattr(
        trusted_folders_manager, "find_trust_root", lambda _: tmp_path.resolve()
    )
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    yield
    reset_harness_files_manager()


@pytest.mark.usefixtures("_setup_manager")
def test_format_call_display_relative_to_cwd(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "config.py"
    target.write_text("x", encoding="utf-8")

    args = EditArgs(file_path=str(target), old_string="x", new_string="y")
    display = Edit.format_call_display(args)

    assert display.summary == "Editing pkg/config.py"
    assert display.message == "pkg/config.py"
    assert display.settled_message == "pkg/config.py"


@pytest.mark.usefixtures("_setup_manager")
def test_get_result_display_relative_to_cwd(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "config.py"
    target.write_text("x", encoding="utf-8")

    result = EditResult(
        file=str(target),
        message="The file has been updated successfully.",
        old_string="x",
        new_string="y",
    )
    event = ToolResultEvent(
        tool_call_id="test", tool_name="edit", tool_class=None, result=result
    )
    display = Edit.get_result_display(event)

    assert display.success is True
    assert display.message == "pkg/config.py"
    assert display.message != "config.py"


def _parse_window(context: str | None) -> dict[int, str]:
    """Parse the ``N→text`` entries of a rendered context window."""
    assert context is not None
    entries: dict[int, str] = {}
    for line in context.splitlines():
        match = re.match(r"^\s*(\d+)→(.*)$", line)
        if match:
            entries[int(match.group(1))] = match.group(2)
    return entries


def _assert_window_matches_current_file(
    context: str | None, current_lines: list[str]
) -> dict[int, str]:
    """Every numbered window line points at the current text on disk."""
    window = _parse_window(context)
    assert window, "context window rendered no numbered lines"
    for number, text in window.items():
        assert current_lines[number - 1] == text
    return window


@pytest.mark.asyncio
async def test_evidence_window_line_numbers_after_newline_insertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "alpha\nbeta\ngamma\ndelta\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(file_path="f.txt", old_string="beta", new_string="BETA1\nBETA2")
        )
    )

    current = path.read_text().splitlines()
    assert current == ["alpha", "BETA1", "BETA2", "gamma", "delta"]
    window = _assert_window_matches_current_file(result.context, current)
    assert window[2] == "BETA1"
    assert window[3] == "BETA2"
    assert window[4] == "gamma"


@pytest.mark.asyncio
async def test_evidence_window_line_numbers_after_newline_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "a\nb\nc\nd\ne\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="b\nc\n", new_string=""))
    )

    current = path.read_text().splitlines()
    assert current == ["a", "d", "e"]
    assert result.applied_diff == "@@ line 2 (current) @@\n- b\n- c"
    window = _assert_window_matches_current_file(result.context, current)
    # "d" moved up from line 4 to line 2; the window numbers it correctly.
    assert window[2] == "d"
    assert window[3] == "e"


@pytest.mark.asyncio
async def test_evidence_diff_describes_actual_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "foo = bar + baz\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="bar", new_string="qux"))
    )

    assert result.applied_diff == (
        "@@ line 1 (current) @@\n- foo = bar + baz\n+ foo = qux + baz"
    )
    assert result.replaced_count == 1


@pytest.mark.asyncio
async def test_evidence_replace_all_names_every_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "x = bar + 1\ny = bar - 2\nz = bar\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(
                file_path="f.txt", old_string="bar", new_string="qux", replace_all=True
            )
        )
    )

    assert result.replaced_count == 3
    assert result.applied_diff == (
        "@@ line 1 (current) @@\n"
        "- x = bar + 1\n"
        "+ x = qux + 1\n"
        "@@ line 2 (current) @@\n"
        "- y = bar - 2\n"
        "+ y = qux - 2\n"
        "@@ line 3 (current) @@\n"
        "- z = bar\n"
        "+ z = qux"
    )
    current = path.read_text().splitlines()
    _assert_window_matches_current_file(result.context, current)


@pytest.mark.asyncio
async def test_evidence_replace_all_sites_shift_after_newline_insertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "a\ntgt\nb\ntgt\nc\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(
                file_path="f.txt",
                old_string="tgt",
                new_string="t1\nt2",
                replace_all=True,
            )
        )
    )

    assert result.replaced_count == 2
    current = path.read_text().splitlines()
    assert current == ["a", "t1", "t2", "b", "t1", "t2", "c"]
    # The second site moved down one line because of the first replacement.
    assert result.applied_diff == (
        "@@ line 2 (current) @@\n"
        "- tgt\n"
        "+ t1\n"
        "+ t2\n"
        "@@ line 5 (current) @@\n"
        "- tgt\n"
        "+ t1\n"
        "+ t2"
    )
    window = _assert_window_matches_current_file(result.context, current)
    assert window[2] == "t1"


@pytest.mark.asyncio
async def test_evidence_revision_matches_independent_post_edit_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "hello world\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="hello", new_string="goodbye"))
    )

    raw = path.read_bytes()
    assert result.revision is not None
    assert result.revision.revision == content_revision(raw)
    assert result.revision == full_revision(raw, line_count=1)
    assert result.revision.scope == "full"
    assert result.revision.size_bytes == len(raw)


@pytest.mark.asyncio
async def test_evidence_revision_changes_after_second_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "one\ntwo\n")
    edit = _make_edit()

    first = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="one", new_string="ONE"))
    )
    second = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="two", new_string="TWO"))
    )

    assert first.revision is not None
    assert second.revision is not None
    assert first.revision.revision != second.revision.revision
    assert second.revision.revision == content_revision(path.read_bytes())


@pytest.mark.asyncio
async def test_failed_edit_publishes_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "hello world\n")
    edit = _make_edit()

    yielded: list[object] = []
    with pytest.raises(ToolError, match="String to replace not found in file"):
        async for item in edit.run(
            EditArgs(file_path="f.txt", old_string="missing", new_string="x")
        ):
            yielded.append(item)

    assert yielded == []
    assert path.read_text() == "hello world\n"


@pytest.mark.asyncio
async def test_ambiguous_edit_publishes_no_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "aaa bbb aaa\n")
    edit = _make_edit()

    yielded: list[object] = []
    with pytest.raises(ToolError, match="Found 2 matches"):
        async for item in edit.run(
            EditArgs(file_path="f.txt", old_string="aaa", new_string="ccc")
        ):
            yielded.append(item)

    assert yielded == []
    assert path.read_text() == "aaa bbb aaa\n"


@pytest.mark.asyncio
async def test_evidence_serialized_through_loop_public_field_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The agent loop serializes tool results with
    # result_model.model_dump(mode="json") and renders "k: v" lines
    # (chartreux/core/agent_loop/_loop.py); the evidence must ride along.
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "alpha\nbeta\ngamma\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="beta", new_string="BETA"))
    )

    result_dict = result.model_dump(mode="json")
    text = "\n".join(f"{k}: {v}" for k, v in result_dict.items())
    assert {"applied_diff", "context", "revision", "replaced_count"} <= (
        result_dict.keys()
    )
    assert "_ui_occurrences" not in result_dict
    assert "ui_occurrences" not in result_dict
    assert result_dict["applied_diff"] == result.applied_diff
    assert result_dict["context"] == result.context
    assert result_dict["replaced_count"] == 1
    assert result_dict["revision"]["revision"] == content_revision(path.read_bytes())
    assert result_dict["revision"]["scope"] == "full"
    assert "@@ line 2 (current) @@" in text
    assert "2→BETA" in text


@pytest.mark.asyncio
async def test_evidence_for_empty_file_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "")

    result = await collect_result(
        _make_edit().run(
            EditArgs(file_path="f.txt", old_string="", new_string="content")
        )
    )

    assert result.replaced_count == 1
    assert result.applied_diff == "@@ line 1 (current) @@\n+ content"
    assert _parse_window(result.context) == {1: "content"}
    assert result.revision is not None
    assert result.revision.revision == content_revision(path.read_bytes())


@pytest.mark.asyncio
async def test_evidence_diff_caps_rendered_sites_but_counts_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "f.txt", "".join(f"tgt {i}\n" for i in range(10)))
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(
                file_path="f.txt", old_string="tgt", new_string="TGT", replace_all=True
            )
        )
    )

    assert result.replaced_count == 10
    assert result.applied_diff is not None
    assert result.applied_diff.count("@@ line") == EDIT_DIFF_MAX_SITES
    assert "… 2 more replaced site(s) not shown (total 10)." in result.applied_diff


@pytest.mark.asyncio
async def test_evidence_window_caps_lines_for_wide_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path, "f.txt", "head\nmid\ntail\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(
            EditArgs(
                file_path="f.txt",
                old_string="mid",
                new_string="\n".join(f"m{i}" for i in range(30)),
            )
        )
    )

    current = path.read_text().splitlines()
    window = _assert_window_matches_current_file(result.context, current)
    assert len(window) <= EDIT_WINDOW_MAX_LINES
    assert window[2] == "m0"
    assert result.applied_diff is not None
    assert "… [+18 line(s)]" in result.applied_diff


@pytest.mark.asyncio
async def test_evidence_window_elides_oversized_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    long_line = "x" * 2000
    path = _write(tmp_path, "f.txt", f"before\n{long_line}\nafter\n")
    edit = _make_edit()

    result = await collect_result(
        edit.run(EditArgs(file_path="f.txt", old_string="before", new_string="BEFORE"))
    )

    current = path.read_text().splitlines()
    assert current[1] == long_line
    window = _parse_window(result.context)
    assert window[1] == "BEFORE"
    # The oversized neighbour is elided to a byte-safe prefix with a marker,
    # never cut mid-line and never silently truncated.
    assert window[2].startswith("x" * 512)
    assert window[2].endswith("…[+1488 bytes]")
    assert window[3] == "after"
