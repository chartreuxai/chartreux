"""Scoped file adapter contract; no script, extension, or filesystem-race sandbox."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.base import BaseTool, ToolPermission
from chartreux.core.tools.builtins.edit import Edit
from chartreux.core.tools.builtins.grep import Grep
from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileArgs
from chartreux.core.tools.builtins.write_file import WriteFile
from chartreux.core.workspace import Workspace
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.parametrize("tool_class", [ReadFile, WriteFile, Edit, Grep])
@pytest.mark.parametrize("configured", [ToolPermission.ASK, ToolPermission.ALWAYS])
@pytest.mark.parametrize(
    ("target", "denied", "sensitive", "expected"),
    [
        ("project/plain.txt", False, False, ToolPermission.ALWAYS),
        ("scratch/plain.txt", False, False, ToolPermission.ALWAYS),
        ("outside/plain.txt", False, False, ToolPermission.NEVER),
        ("project/../outside/plain.txt", False, False, ToolPermission.NEVER),
        ("sibling-scratch/plain.txt", False, False, ToolPermission.NEVER),
        ("project/plain.txt", True, False, ToolPermission.NEVER),
        ("scratch/plain.txt", True, False, ToolPermission.NEVER),
        ("project/plain.txt", False, True, ToolPermission.NEVER),
        ("scratch/plain.txt", False, True, ToolPermission.NEVER),
    ],
)
def test_file_guards_precede_every_automatic_allowance(
    tmp_path: Path,
    tool_class: type[BaseTool],
    configured: ToolPermission,
    target: str,
    denied: bool,
    sensitive: bool,
    expected: ToolPermission,
) -> None:
    project = tmp_path / "project"
    scratch = tmp_path / "scratch"
    project.mkdir()
    scratch.mkdir()
    config = tool_class._get_tool_config_class()(
        permission=configured,
        allowlist=["*"],
        denylist=["*/plain.txt"] if denied else [],
        sensitive_patterns=["**/plain.txt"] if sensitive else [],
    )
    tool = tool_class.from_config(lambda: config, cwd=project, scratchpad_dir=scratch)
    path = str(tmp_path / target)
    raw = (
        {"pattern": "hello", "path": path}
        if tool_class is Grep
        else {
            "file_path": path,
            "content": "hello",
            "old_string": "a",
            "new_string": "b",
        }
    )
    decision = tool.resolve_permission(tool.validate_arguments(raw))
    assert decision is not None
    assert decision.permission == expected
    if expected == ToolPermission.NEVER:
        assert decision.reason


def test_discovery_roots_are_not_tool_authority(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    harness = HarnessFilesManager(sources=(), cwd=project).for_session(
        project, workspace_roots=[tmp_path]
    )
    tool = ReadFile.from_config(
        ReadFile._get_tool_config_class(), cwd=project, harness_files=harness
    )
    assert tool.workspace.authorized_roots == (project.resolve(),)


def test_symlink_escape_is_denied_with_an_allowlist(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "link").symlink_to(outside, target_is_directory=True)
    config = ReadFile._get_tool_config_class()(allowlist=["*"])
    tool = ReadFile.from_config(lambda: config, cwd=project)
    decision = tool.resolve_permission(
        ReadFileArgs(file_path=str(project / "link" / "secret.txt"))
    )
    assert decision is not None and decision.permission == ToolPermission.NEVER


def test_retargeted_authorized_root_does_not_redefine_authority(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    workspace = Workspace.for_session(root)
    moved = tmp_path / "moved"
    root.rename(moved)
    root.symlink_to(tmp_path, target_is_directory=True)
    assert not workspace.allows(root / "secret.txt")


@pytest.mark.asyncio
async def test_denied_read_then_useful_write_has_zero_approval_callbacks(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    destination = project / "result.txt"
    calls = [
        ToolCall(
            id="blocked",
            index=0,
            function=FunctionCall(
                name="read_file",
                arguments=json.dumps({"file_path": str(tmp_path / "outside.txt")}),
            ),
        ),
        ToolCall(
            id="continued",
            index=0,
            function=FunctionCall(
                name="write_file",
                arguments=json.dumps({
                    "file_path": str(destination),
                    "content": "continued",
                }),
            ),
        ),
    ]
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[calls[0]])],
        [mock_llm_chunk(tool_calls=[calls[1]])],
        [mock_llm_chunk(content="Done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read_file", "write_file"]),
        backend=backend,
        cwd=project,
    )
    results: list[ToolResultEvent] = []
    async for event in loop.act("Do useful work within the project"):
        assert "approval" not in type(event).__name__.lower()
        if isinstance(event, ToolResultEvent):
            results.append(event)
    assert len(results) == 2
    assert results[0].error or results[0].skip_reason
    assert results[1].error is None and results[1].result is not None
    assert destination.read_text() == "continued"
