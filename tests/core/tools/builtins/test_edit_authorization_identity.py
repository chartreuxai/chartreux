from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins import edit as edit_module
from chartreux.core.tools.builtins.edit import Edit, EditArgs, EditConfig
from chartreux.core.tools.permissions import PermissionContext
from tests.mock.utils import collect_result


def _make_edit(
    cwd: Path, *, denylist: list[str] | None = None, scratchpad_dir: Path | None = None
) -> Edit:
    config = EditConfig(
        permission=ToolPermission.ALWAYS, denylist=denylist or [], sensitive_patterns=[]
    )
    return Edit(
        config_getter=lambda: config,
        state=BaseToolState(),
        cwd=cwd,
        scratchpad_dir=scratchpad_dir,
    )


@pytest.mark.parametrize("padding", [" ", "\t", " \t"])
def test_padded_path_cannot_bypass_synthetic_denylist(
    tmp_path: Path, padding: str
) -> None:
    target = tmp_path / "blocked.txt"
    target.write_text("before", encoding="utf-8")
    edit = _make_edit(tmp_path, denylist=[str(target)])

    args = EditArgs(
        file_path=f"{padding}{target}{padding}", old_string="before", new_string="after"
    )
    permission = edit.resolve_permission(args)

    assert args.file_path == str(target)
    assert isinstance(permission, PermissionContext)
    assert permission.permission is ToolPermission.NEVER
    assert target.read_text(encoding="utf-8") == "before"


@pytest.mark.asyncio
async def test_permission_snapshot_and_execution_share_normalized_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    edit = _make_edit(tmp_path)
    seen: dict[str, str] = {}

    def permission_spy(path: str, **_: object) -> PermissionContext:
        seen["permission"] = path
        return PermissionContext(permission=ToolPermission.ALWAYS)

    def snapshot_spy(path: str):
        seen["snapshot"] = path
        return None

    original_resolve = edit_module.resolve_tool_path

    def execution_spy(path: str, cwd: Path) -> Path:
        seen["execution"] = path
        return original_resolve(path, cwd)

    monkeypatch.setattr(edit_module, "resolve_file_tool_permission", permission_spy)
    monkeypatch.setattr(edit, "get_file_snapshot_for_path", snapshot_spy)
    monkeypatch.setattr(edit_module, "resolve_tool_path", execution_spy)

    args = EditArgs(
        file_path=f" \t{target}\t ", old_string="before", new_string="after"
    )
    edit.resolve_permission(args)
    edit.get_file_snapshot(args)
    await collect_result(edit.run(args))

    assert seen == {
        "permission": str(target),
        "snapshot": str(target),
        "execution": str(target),
    }


@pytest.mark.asyncio
async def test_denied_padded_invocation_takes_no_snapshot_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "blocked.txt"
    target.write_text("before", encoding="utf-8")
    edit = _make_edit(tmp_path, denylist=[str(target)])
    snapshot_called = False
    write_called = False

    def snapshot_spy(_: EditArgs):
        nonlocal snapshot_called
        snapshot_called = True
        return None

    async def write_spy(*_: object, **__: object) -> None:
        nonlocal write_called
        write_called = True

    monkeypatch.setattr(edit, "get_file_snapshot", snapshot_spy)
    monkeypatch.setattr(edit, "_write_file", write_spy)

    args = EditArgs(file_path=f"{target} \t", old_string="before", new_string="after")
    permission = edit.resolve_permission(args)
    assert permission is not None
    if permission.permission is not ToolPermission.NEVER:
        edit.get_file_snapshot(args)
        await collect_result(edit.run(args))

    assert permission.permission is ToolPermission.NEVER
    assert snapshot_called is False
    assert write_called is False
    assert target.read_text(encoding="utf-8") == "before"


@pytest.mark.asyncio
async def test_allowed_padded_path_edits_target_and_preserves_diff(
    tmp_path: Path,
) -> None:
    target = tmp_path / "allowed.txt"
    target.write_text("prefix before suffix\n", encoding="utf-8")
    edit = _make_edit(tmp_path)

    args = EditArgs(
        file_path=f"\t {target} \t", old_string="before", new_string="after"
    )
    result = await collect_result(edit.run(args))

    assert target.read_text(encoding="utf-8") == "prefix after suffix\n"
    assert result.file == str(target)
    assert result.ui_occurrences == [(1, "prefix before suffix", "prefix after suffix")]


@pytest.mark.parametrize("file_path", [" ", "\t", " \t "])
def test_whitespace_only_path_fails_argument_validation(file_path: str) -> None:
    with pytest.raises(ValidationError, match="File path cannot be empty"):
        EditArgs(file_path=file_path, old_string="before", new_string="after")


def test_padded_scratchpad_path_retains_permission(tmp_path: Path) -> None:
    scratchpad = (tmp_path / "scratchpad").resolve()
    scratchpad.mkdir()
    target = scratchpad / "notes.txt"
    target.write_text("before", encoding="utf-8")
    edit = _make_edit(tmp_path, scratchpad_dir=scratchpad)

    permission = edit.resolve_permission(
        EditArgs(file_path=f" \t{target}\t ", old_string="before", new_string="after")
    )

    assert permission is not None
    assert permission.permission is ToolPermission.ALWAYS
