"""Shared parametrized suite for the agent-loop grant-lifecycle twins (W6b, stage 15a).

The twin files (``test_dynamic_root_grants.py`` on the session-root-grant side
vs ``test_dangerous_command_lifecycle.py`` on the dangerous-command side)
copy-adapted the same mid-turn grant lifecycle twice: a grantable denial opens
a user question through the request broker, the answer installs runtime
authority (a session grant; the dangerous side additionally has once permits),
the original call is fully re-resolved, and only then does it resume. Refusals
are remembered; "Always for this project" reports the persistence port's
outcome separately from the live grant; noninteractive sessions and child
sessions deny without prompting. This harness runs each migrated concern once
per feature. :class:`GrantLifecycleFeature` carries the feature's tools,
declined-feedback text, config knob, prompt options, scenario builders, store
assertion helpers, and persistence-port adapters, so every case asserts the
UNION of both twins' assertions. Feature-specific concerns (the sensitive-path
short-circuit, root proposal targets, the broad-root warning, the cached
workspace check, waiter cancellation, the cancelled-save propagation, allow-once
permit semantics, refusal-memory identity rules, scope binding, the
recheck/admission gates, once-permit cleanup, session reload/resume/clear
lifetimes, the parent ceiling, prompt failure modes, the consequential/mixed
noninteractive texts) stay in the feature-specific modules. The suite is
collected through ``test_grant_lifecycle_harness.py``, which re-exports these
cases under stable node IDs; per-pair difference audits and survival mappings
live in the stage-15 report.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.config._dangerous_command_persistence import SavedGrantsRead
from chartreux.core.config._root_persistence import SavedRootsRead
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_commands import RepositoryScope
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.core.workspace import Workspace
from chartreux.questions import UserAnswer, UserQuestionRequest, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

VISION_MODEL = ModelConfig(
    name="vision", provider="mistral", alias="vision", supports_images=True
)

PUSH = "git.push"
PROMPT_TEXT = "Exercise grant lifecycle"

ROOTS_PLAIN_DENIAL = (
    "File read is outside authorized workspace and session scratch roots "
    "and is not an instruction file injected into this agent's context. "
    "Other paths require an explicit user scope change."
)

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVQI12NoAAAAggCB3UNq9"
    "AAAAABJRU5ErkJggg=="
)


def _config() -> ChartreuxConfigSchema:
    return build_test_vibe_config(active_model="vision", models=[VISION_MODEL])


def _call(name: str, args: dict[str, Any], call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name=name, arguments=json.dumps(args)),
    )


async def _drive(
    agent: AgentLoop, answer: str = "Deny", *, cancelled: bool = False
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one turn, answering every grant prompt with *answer*."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act(PROMPT_TEXT):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                result = (
                    UserQuestionResult(answers=[], cancelled=True)
                    if cancelled
                    else UserQuestionResult(
                        answers=[UserAnswer(question="q", answer=answer)]
                    )
                )
                agent.resolve_user_input_request(event.request_id, result)
            events.append(event)
    return events, requests


def _results(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _tool_messages(agent: AgentLoop) -> list[str]:
    return [
        message.content or ""
        for message in agent.messages
        if message.role == Role.tool and message.content
    ]


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def scope_of(path: Path) -> RepositoryScope:
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


def _authorities(agent: AgentLoop) -> list[tuple[frozenset[str], RepositoryScope]]:
    return [
        (authority.operations, authority.scope)
        for authority in agent._dangerous_command_grants.dangerous_command_authorities()
    ]


@dataclass(frozen=True)
class _GrantScenario:
    """One granted call's world: the loop's cwd, the granted target, its args."""

    project: Path
    outside: Path | None
    target: Path | None
    args: dict[str, Any]


@dataclass(frozen=True)
class _ConcurrentScenario:
    """Two concurrent grant-needing calls issued in one assistant turn."""

    project: Path
    outside: Path | None
    calls: tuple[tuple[str, dict[str, Any]], ...]


def _prepare_tool(tool: str, outside: Path) -> tuple[Path, dict[str, str]]:
    """Create the out-of-root target and tool arguments for one file tool."""
    if tool == "read_file":
        target = outside / "note.txt"
        target.write_text("granted content")
        return target, {"file_path": str(target)}
    if tool == "write_file":
        target = outside / "created.txt"
        return target, {"file_path": str(target), "content": "granted write"}
    if tool == "edit":
        target = outside / "edit.txt"
        target.write_text("old content")
        return target, {
            "file_path": str(target),
            "old_string": "old content",
            "new_string": "new content",
        }
    if tool == "grep":
        target = outside / "hits.txt"
        target.write_text("granted needle\n")
        return target, {"pattern": "needle", "path": str(target)}
    if tool == "read_image":
        target = outside / "pixel.png"
        target.write_bytes(PNG_BYTES)
        return target, {"file_path": str(target)}
    raise AssertionError(f"unknown tool {tool}")


def _roots_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    return project


def _roots_scenario(
    tmp_path: Path, tool: str, command: str = "git push"
) -> _GrantScenario:
    project = _roots_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target, args = _prepare_tool(tool, outside)
    return _GrantScenario(project=project, outside=outside, target=target, args=args)


def _dangerous_scenario(
    tmp_path: Path, tool: str, command: str = "git push"
) -> _GrantScenario:
    project = make_repo(tmp_path / "project")
    return _GrantScenario(
        project=project, outside=None, target=None, args={"command": command}
    )


def _roots_granted(agent: AgentLoop, scenario: _GrantScenario) -> None:
    assert scenario.outside is not None
    assert agent._session_root_grants.roots == {scenario.outside.resolve()}
    assert agent._session_root_grants.denied_roots == set()
    assert agent._session_root_grants.pending == {}


def _roots_refused(agent: AgentLoop, scenario: _GrantScenario) -> None:
    assert scenario.outside is not None
    assert agent._session_root_grants.roots == set()
    assert agent._session_root_grants.denied_roots == {scenario.outside.resolve()}


def _roots_no_authority(agent: AgentLoop) -> None:
    assert agent._session_root_grants.roots == set()


def _roots_executed(result: ToolResultEvent) -> None:
    assert not result.skipped
    assert result.result is not None


def _roots_grant_side_effects(scenario: _GrantScenario, tool: str) -> None:
    if tool == "write_file":
        assert scenario.target is not None
        assert scenario.target.read_text() == "granted write"
    elif tool == "edit":
        assert scenario.target is not None
        assert scenario.target.read_text() == "new content"


def _roots_question_content(question: Any, scenario: _GrantScenario) -> None:
    assert scenario.outside is not None
    assert str(scenario.outside.resolve()) in question.question
    assert "reading and writing" in question.question
    assert "shell" in question.question
    assert "home directory" not in question.question


def _roots_saved_outcome_fragments(scenario: _GrantScenario) -> list[str]:
    assert scenario.outside is not None
    return [
        "applies to this session",
        "saved in your user config",
        str(scenario.project.resolve()),
        str(scenario.outside.resolve()),
    ]


def _roots_grant_effective(agent: AgentLoop, scenario: _GrantScenario) -> None:
    tool = agent.tool_manager.get("read_file")
    args = tool.validate_arguments(scenario.args)
    decision = tool.resolve_permission(args)
    assert decision is not None and decision.permission == ToolPermission.ALWAYS
    assert scenario.target is not None
    assert agent.tool_manager.workspace.allows(scenario.target)


def _roots_child_workspace(project: Path, scenario: _GrantScenario) -> Workspace:
    assert scenario.outside is not None
    return Workspace.for_session(project, authorized_roots=[scenario.outside])


def _roots_child_bounded(reason: str) -> None:
    assert "outside authorized workspace" in reason


def _roots_plain_denial(reason: str) -> None:
    assert reason == ROOTS_PLAIN_DENIAL


def _dangerous_granted(agent: AgentLoop, scenario: _GrantScenario) -> None:
    assert _authorities(agent) == [(frozenset({PUSH}), scope_of(scenario.project))]


def _dangerous_refused(agent: AgentLoop, scenario: _GrantScenario) -> None:
    assert _authorities(agent) == []


def _dangerous_no_authority(agent: AgentLoop) -> None:
    assert _authorities(agent) == []


def _dangerous_executed(result: ToolResultEvent) -> None:
    assert not result.skipped


def _dangerous_grant_side_effects(scenario: _GrantScenario, tool: str) -> None:
    return None


def _dangerous_question_content(question: Any, scenario: _GrantScenario) -> None:
    assert scenario.args["command"] in question.question
    assert str(scenario.project.resolve()) in question.question
    assert str(scope_of(scenario.project).work_tree) in question.question
    assert "permission authority" in question.question
    assert "standing intent to publish" in question.question


def _dangerous_saved_outcome_fragments(scenario: _GrantScenario) -> list[str]:
    return [
        "applies to this session",
        "saved in your user config",
        str(scenario.project.resolve()),
    ]


def _dangerous_grant_effective(agent: AgentLoop, scenario: _GrantScenario) -> None:
    tool = agent.tool_manager.get("bash")
    args = tool.validate_arguments({"command": "git push"})
    decision = tool.resolve_permission(args)
    assert decision is not None and decision.permission == ToolPermission.ALWAYS


def _dangerous_child_workspace(project: Path, scenario: _GrantScenario) -> Workspace:
    return Workspace.for_session(project)


def _dangerous_child_bounded(reason: str) -> None:
    assert "requires approval for a dangerous git operation" in reason


def _dangerous_plain_denial(reason: str) -> None:
    assert "requires approval for a dangerous git operation" in reason
    assert PUSH in reason


class _StubSavePort:
    """Runtime port stub: applies grants locally and records save attempts."""

    def __init__(self, loop: AgentLoop, result: ConfigSaveResult) -> None:
        self._loop = loop
        self._result = result
        self.saves: list[tuple[str, Path, str]] = []

    async def grant_root(self, session_id: str, root: Path) -> None:
        assert session_id == self._loop.session_id
        self._loop.apply_root_grant(root)

    async def save_root(
        self, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        assert session_id == self._loop.session_id
        self.saves.append((session_id, root, expected_revision))
        return self._result


class _StubGrantPort:
    """Runtime persistence port stub: records save attempts, returns a result."""

    def __init__(self, result: ConfigSaveResult) -> None:
        self._result = result
        self.saves: list[tuple[str, Path, str, str]] = []

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        self.saves.append((session_id, project, operation_id, expected_revision))
        return self._result


class _StubRootGrantPort:
    """Runtime marker only; a no-capability session never calls it."""

    async def grant_root(self, session_id: str, root: Path) -> None:
        raise AssertionError("A session without user input must not grant roots")

    async def save_root(
        self, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        raise AssertionError("A session without user input must not save roots")


class _NeverCalledGrantPort:
    """Runtime marker only; a session that never approves never saves."""

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        raise AssertionError("A session without an approval must not save grants")


def _roots_save_port(loop: AgentLoop, result: ConfigSaveResult) -> Any:
    return _StubSavePort(loop, result)


def _dangerous_save_port(loop: AgentLoop, result: ConfigSaveResult) -> Any:
    return _StubGrantPort(result)


def _roots_never_called_port(loop: AgentLoop) -> Any:
    return _StubRootGrantPort()


def _dangerous_never_called_port(loop: AgentLoop) -> Any:
    return _NeverCalledGrantPort()


def _roots_bind_port(agent: AgentLoop, port: Any) -> None:
    agent.bind_root_grant_port(port)


def _dangerous_bind_port(agent: AgentLoop, port: Any) -> None:
    agent.bind_dangerous_command_grant_port(port)


def _roots_expected_save(
    agent: AgentLoop, scenario: _GrantScenario, revision: str
) -> tuple[Any, ...]:
    assert scenario.outside is not None
    return (agent.session_id, scenario.outside.resolve(), revision)


def _dangerous_expected_save(
    agent: AgentLoop, scenario: _GrantScenario, revision: str
) -> tuple[Any, ...]:
    return (agent.session_id, scenario.project.resolve(), PUSH, revision)


def _roots_patch_saved_surface(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Point the loop's saved-roots read surface at a controlled revision."""
    read = AsyncMock(return_value=SavedRootsRead(roots=(), user_revision="user-rev-1"))
    monkeypatch.setattr("chartreux.core.agent_loop._loop.read_saved_roots", read)
    return read


def _dangerous_patch_saved_surface(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Point the loop's saved-grants read surface at a controlled revision."""
    read = AsyncMock(
        return_value=SavedGrantsRead(operations=(), user_revision="user-rev-1")
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_dangerous_command_grants", read
    )
    return read


def _roots_patch_unavailable_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_roots",
        AsyncMock(return_value=SavedRootsRead(unavailable="no_user_source")),
    )


def _dangerous_patch_unavailable_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_dangerous_command_grants",
        AsyncMock(return_value=SavedGrantsRead(unavailable="no_user_source")),
    )


def _roots_concurrent_scenario(tmp_path: Path, tool: str) -> _ConcurrentScenario:
    project = _roots_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    first = outside / "first.txt"
    first.write_text("first granted content")
    second = outside / "second.txt"
    second.write_text("second granted content")
    return _ConcurrentScenario(
        project=project,
        outside=outside,
        calls=(
            ("read_file", {"file_path": str(first)}),
            ("read_file", {"file_path": str(second)}),
        ),
    )


def _dangerous_concurrent_scenario(tmp_path: Path, tool: str) -> _ConcurrentScenario:
    project = make_repo(tmp_path / "project")
    args = {"command": "git push"}
    return _ConcurrentScenario(
        project=project, outside=None, calls=((tool, args), (tool, args))
    )


def _roots_concurrent_granted(agent: AgentLoop, scenario: _ConcurrentScenario) -> None:
    assert scenario.outside is not None
    assert agent._session_root_grants.roots == {scenario.outside.resolve()}
    assert agent._session_root_grants.pending == {}


def _dangerous_concurrent_granted(
    agent: AgentLoop, scenario: _ConcurrentScenario
) -> None:
    assert _authorities(agent) == []


def _roots_revision_scenarios(
    tmp_path: Path, tool: str
) -> tuple[_GrantScenario, _GrantScenario]:
    project = _roots_project(tmp_path)
    first_root = tmp_path / "first-outside"
    first_root.mkdir()
    second_root = tmp_path / "second-outside"
    second_root.mkdir()
    first = first_root / "note.txt"
    first.write_text("first content")
    second = second_root / "note.txt"
    second.write_text("second content")
    return (
        _GrantScenario(
            project=project,
            outside=first_root,
            target=first,
            args={"file_path": str(first)},
        ),
        _GrantScenario(
            project=project,
            outside=second_root,
            target=second,
            args={"file_path": str(second)},
        ),
    )


def _dangerous_revision_scenarios(
    tmp_path: Path, tool: str
) -> tuple[_GrantScenario, _GrantScenario]:
    make_repo(tmp_path / "a")
    make_repo(tmp_path / "b")
    return (
        _GrantScenario(
            project=tmp_path,
            outside=None,
            target=None,
            args={"command": "git -C a push; true"},
        ),
        _GrantScenario(
            project=tmp_path,
            outside=None,
            target=None,
            args={"command": "git -C b push; true"},
        ),
    )


def _roots_revision_state(
    agent: AgentLoop,
    port: Any,
    scenarios: tuple[_GrantScenario, _GrantScenario],
    tmp_path: Path,
) -> None:
    first, second = scenarios
    assert first.outside is not None and second.outside is not None
    assert agent._session_root_grants.roots == {
        first.outside.resolve(),
        second.outside.resolve(),
    }


def _dangerous_revision_state(
    agent: AgentLoop,
    port: Any,
    scenarios: tuple[_GrantScenario, _GrantScenario],
    tmp_path: Path,
) -> None:
    assert {save[1] for save in port.saves} == {tmp_path.resolve()}


@dataclass(frozen=True)
class GrantLifecycleFeature:
    """One grant-lifecycle feature as the shared suite parameterizes it.

    ``make_scenario`` builds the feature's granted call (the roots side a file
    tool on an out-of-root target; the dangerous side a shell tool running a
    git command in a repository). The ``assert_*`` helpers state the feature's
    share of each migrated concern's union of assertions; the port adapters
    wire the feature's persistence port and saved-surface read.
    """

    name: str
    tools: tuple[str, ...]
    declined_feedback: str
    config_knob: str
    prompt_options: tuple[str, ...]
    coalesces_concurrent_prompts: bool
    concurrent_answer: str
    cannot_save_fragment: str
    make_scenario: Callable[[Path, str, str], _GrantScenario]
    assert_granted: Callable[[AgentLoop, _GrantScenario], None]
    assert_refused: Callable[[AgentLoop, _GrantScenario], None]
    assert_no_authority: Callable[[AgentLoop], None]
    assert_executed: Callable[[ToolResultEvent], None]
    assert_grant_side_effects: Callable[[_GrantScenario, str], None]
    assert_question_content: Callable[[Any, _GrantScenario], None]
    saved_outcome_fragments: Callable[[_GrantScenario], list[str]]
    assert_grant_effective: Callable[[AgentLoop, _GrantScenario], None]
    child_workspace: Callable[[Path, _GrantScenario], Workspace]
    assert_child_bounded: Callable[[str], None]
    assert_plain_denial: Callable[[str], None]
    make_save_port: Callable[[AgentLoop, ConfigSaveResult], Any]
    make_never_called_port: Callable[[AgentLoop], Any]
    bind_port: Callable[[AgentLoop, Any], None]
    expected_save: Callable[[AgentLoop, _GrantScenario, str], tuple[Any, ...]]
    patch_saved_surface: Callable[[pytest.MonkeyPatch], AsyncMock]
    patch_unavailable_surface: Callable[[pytest.MonkeyPatch], None]
    make_concurrent_scenario: Callable[[Path, str], _ConcurrentScenario]
    assert_concurrent_granted: Callable[[AgentLoop, _ConcurrentScenario], None]
    make_revision_scenarios: Callable[
        [Path, str], tuple[_GrantScenario, _GrantScenario]
    ]
    assert_revision_state: Callable[
        [AgentLoop, Any, tuple[_GrantScenario, _GrantScenario], Path], None
    ]


ROOTS = GrantLifecycleFeature(
    name="roots",
    tools=("read_file", "write_file", "edit", "grep", "read_image"),
    declined_feedback="user declined; do not retry this path",
    config_knob="authorized_roots_by_project",
    prompt_options=("Allow this session", "Always for this project", "Deny"),
    coalesces_concurrent_prompts=True,
    concurrent_answer="Allow this session",
    cannot_save_fragment="cannot save roots",
    make_scenario=_roots_scenario,
    assert_granted=_roots_granted,
    assert_refused=_roots_refused,
    assert_no_authority=_roots_no_authority,
    assert_executed=_roots_executed,
    assert_grant_side_effects=_roots_grant_side_effects,
    assert_question_content=_roots_question_content,
    saved_outcome_fragments=_roots_saved_outcome_fragments,
    assert_grant_effective=_roots_grant_effective,
    child_workspace=_roots_child_workspace,
    assert_child_bounded=_roots_child_bounded,
    assert_plain_denial=_roots_plain_denial,
    make_save_port=_roots_save_port,
    make_never_called_port=_roots_never_called_port,
    bind_port=_roots_bind_port,
    expected_save=_roots_expected_save,
    patch_saved_surface=_roots_patch_saved_surface,
    patch_unavailable_surface=_roots_patch_unavailable_surface,
    make_concurrent_scenario=_roots_concurrent_scenario,
    assert_concurrent_granted=_roots_concurrent_granted,
    make_revision_scenarios=_roots_revision_scenarios,
    assert_revision_state=_roots_revision_state,
)

DANGEROUS = GrantLifecycleFeature(
    name="dangerous",
    tools=("bash", "bash_start"),
    declined_feedback="user declined; do not retry this command",
    config_knob="dangerous_command_grants_by_project",
    prompt_options=(
        "Deny",
        "Allow once",
        "Allow this session",
        "Always for this project",
    ),
    coalesces_concurrent_prompts=False,
    concurrent_answer="Allow once",
    cannot_save_fragment="cannot save dangerous-command grants",
    make_scenario=_dangerous_scenario,
    assert_granted=_dangerous_granted,
    assert_refused=_dangerous_refused,
    assert_no_authority=_dangerous_no_authority,
    assert_executed=_dangerous_executed,
    assert_grant_side_effects=_dangerous_grant_side_effects,
    assert_question_content=_dangerous_question_content,
    saved_outcome_fragments=_dangerous_saved_outcome_fragments,
    assert_grant_effective=_dangerous_grant_effective,
    child_workspace=_dangerous_child_workspace,
    assert_child_bounded=_dangerous_child_bounded,
    assert_plain_denial=_dangerous_plain_denial,
    make_save_port=_dangerous_save_port,
    make_never_called_port=_dangerous_never_called_port,
    bind_port=_dangerous_bind_port,
    expected_save=_dangerous_expected_save,
    patch_saved_surface=_dangerous_patch_saved_surface,
    patch_unavailable_surface=_dangerous_patch_unavailable_surface,
    make_concurrent_scenario=_dangerous_concurrent_scenario,
    assert_concurrent_granted=_dangerous_concurrent_granted,
    make_revision_scenarios=_dangerous_revision_scenarios,
    assert_revision_state=_dangerous_revision_state,
)

# The active migration frontier: the cases below are parametrized over these
# features. 15a-ii migrates the dynamic-root-grants side (FEATURES = (ROOTS,));
# 15a-iii extends it with the lifecycle side (FEATURES = (ROOTS, DANGEROUS)),
# which appends the dangerous combos without renaming the roots ones.
FEATURES: tuple[GrantLifecycleFeature, ...] = (ROOTS, DANGEROUS)

# The smoke case proves the machinery for both features from the start (the
# 12c-i SMOKE_ADAPTERS pattern: the smoke frontier is independent of the cases'
# migration frontier).
SMOKE_FEATURES: tuple[GrantLifecycleFeature, ...] = (ROOTS, DANGEROUS)
SMOKE_FEATURE_IDS = [feature.name for feature in SMOKE_FEATURES]


@pytest.mark.asyncio
@pytest.mark.parametrize("feature", SMOKE_FEATURES, ids=SMOKE_FEATURE_IDS)
async def test_harness_smoke_grant_lifecycle_round_trip(
    feature: GrantLifecycleFeature, tmp_path: Path
) -> None:
    """Harness smoke: the feature machinery drives a real accepted grant."""
    tool = feature.tools[0]
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "smoke")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        feature.assert_executed(results[0])
        feature.assert_granted(agent, scenario)
    finally:
        await agent.aclose()


# ---------------------------------------------------------------------------
# Migrated twin pairs (added by 15a-ii from test_dynamic_root_grants.py and
# extended by 15a-iii with the test_dangerous_command_lifecycle.py side; each
# case asserts the union of both twins' assertions — see the stage-15 report
# for the per-pair audits and the deleted -> harness survival mappings).
# ---------------------------------------------------------------------------

_ALL_ROOTS_TOOLS = ("read_file", "write_file", "edit", "grep", "read_image")
_BOTH_SHELL_TOOLS = ("bash", "bash_start")
_READ_FILE_ONLY = ("read_file",)
_BASH_ONLY = ("bash",)


def _combos(
    roots_tools: tuple[str, ...], dangerous_tools: tuple[str, ...]
) -> list[tuple[GrantLifecycleFeature, str]]:
    """The active (feature, tool) combos: one per migrated twin item."""
    combos: list[tuple[GrantLifecycleFeature, str]] = []
    for feature in FEATURES:
        tools = roots_tools if feature is ROOTS else dangerous_tools
        combos.extend((feature, tool) for tool in tools)
    return combos


def _combo_ids(
    roots_tools: tuple[str, ...], dangerous_tools: tuple[str, ...]
) -> list[str]:
    return [
        f"{feature.name}-{tool}"
        for feature, tool in _combos(roots_tools, dangerous_tools)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_ALL_ROOTS_TOOLS, _BOTH_SHELL_TOOLS),
    ids=_combo_ids(_ALL_ROOTS_TOOLS, _BOTH_SHELL_TOOLS),
)
async def test_allow_session_grant_resumes_the_tool_call(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """Allow-session: one prompt, the call resumes, the grant covers a repeat."""
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "grant-me")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "grant-again")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        feature.assert_executed(results[0])
        feature.assert_granted(agent, scenario)
        feature.assert_grant_side_effects(scenario, tool)
        # The identical second call resumes without prompting.
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert not results[0].skipped
        feature.assert_granted(agent, scenario)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_ALL_ROOTS_TOOLS, _BOTH_SHELL_TOOLS),
    ids=_combo_ids(_ALL_ROOTS_TOOLS, _BOTH_SHELL_TOOLS),
)
async def test_deny_skips_the_tool_call_and_remembers_the_refusal(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """Deny: the call is skipped with the declined feedback, the refusal is
    remembered, and the identical retry never re-prompts.
    """
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    before = None
    if scenario.target is not None:
        before = scenario.target.read_bytes() if scenario.target.exists() else None
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "deny-me")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "retry")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == feature.declined_feedback
        assert results[0].result is None
        feature.assert_refused(agent, scenario)
        if scenario.target is not None:
            if before is None:
                assert not scenario.target.exists()
            else:
                assert scenario.target.read_bytes() == before
        # The decided refusal answers the identical retry: no re-prompt,
        # even when the retry would otherwise be grantable.
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == feature.declined_feedback
        feature.assert_refused(agent, scenario)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BOTH_SHELL_TOOLS),
    ids=_combo_ids(_READ_FILE_ONLY, _BOTH_SHELL_TOOLS),
)
async def test_cancelled_prompt_denies_and_records_refusal(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """A cancelled prompt denies with the declined feedback and records the
    refusal, so the identical call never re-prompts.
    """
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "esc-1")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "esc-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, cancelled=True)
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == feature.declined_feedback
        feature.assert_refused(agent, scenario)
        # Escape counts as a decision: the refusal is remembered.
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        feature.assert_refused(agent, scenario)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_prompt_shape_states_access_and_choices(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """The grant question states the access and offers the feature's choices."""
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "shape")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        question = request.questions[0]
        assert len(question.header) <= 20
        assert question.hide_other is True
        assert question.multi_select is False
        assert [option.label for option in question.options] == list(
            feature.prompt_options
        )
        assert request.footer_note
        feature.assert_question_content(question, scenario)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_always_applies_the_grant_and_reports_the_saved_outcome(
    feature: GrantLifecycleFeature,
    tool: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Always-for-this-project: the live grant applies and the save outcome is
    reported with the saved result.
    """
    scenario = feature.make_scenario(tmp_path, tool, "git push; true")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "always")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    port = feature.make_save_port(
        agent, ConfigSaveResult("user", "saved", "unchanged", "rev-2")
    )
    feature.bind_port(agent, port)
    read = feature.patch_saved_surface(monkeypatch)
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert not results[0].skipped
        # The session grant applies exactly like Allow-session does.
        feature.assert_granted(agent, scenario)
        # The user revision is fetched from the read surface at decision time.
        assert read.call_count == 1
        assert read.call_args.kwargs["project"] == scenario.project.resolve()
        assert port.saves == [feature.expected_save(agent, scenario, "user-rev-1")]
        fragments = feature.saved_outcome_fragments(scenario)
        assert any(
            all(fragment in text for fragment in fragments)
            for text in _tool_messages(agent)
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_always_reports_a_conflict_without_retrying(
    feature: GrantLifecycleFeature,
    tool: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conflicting save is reported session-only: no retry, no re-prompt."""
    scenario = feature.make_scenario(tmp_path, tool, "git push; true")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "conflict")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    port = feature.make_save_port(
        agent, ConfigSaveResult("user", "not_saved", "unchanged", error="conflict")
    )
    feature.bind_port(agent, port)
    feature.patch_saved_surface(monkeypatch)
    try:
        events, requests = await _drive(agent, "Always for this project")
        # No automatic retry and no automatic re-prompt.
        assert len(requests) == 1
        assert len(port.saves) == 1
        assert not _results(events)[0].skipped
        # The session grant is still applied; the save is reported session-only.
        feature.assert_granted(agent, scenario)
        assert any(
            "applies to this session only" in text and "retry" in text
            for text in _tool_messages(agent)
        )
    finally:
        await agent.aclose()


_SAVE_FAILURE_CASES: list[tuple[ConfigSaveResult, str]] = [
    (
        ConfigSaveResult("user", "not_saved", "unchanged", error="write"),
        "saving it to your user config failed",
    ),
    (
        ConfigSaveResult("user", "durability_uncertain", "unchanged", "rev"),
        "could not be confirmed",
    ),
    (
        ConfigSaveResult("user", "saved", "unchanged", "rev", "cancelled"),
        "was cancelled",
    ),
]
_SAVE_FAILURE_IDS = ["write-failure", "durability-uncertain", "cancelled-save"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected"), _SAVE_FAILURE_CASES, ids=_SAVE_FAILURE_IDS
)
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_always_reports_save_failures_honestly(
    feature: GrantLifecycleFeature,
    tool: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: ConfigSaveResult,
    expected: str,
) -> None:
    """A failed save is reported honestly; the session grant still applies."""
    scenario = feature.make_scenario(tmp_path, tool, "git push; true")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "failure")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    port = feature.make_save_port(agent, result)
    feature.bind_port(agent, port)
    feature.patch_saved_surface(monkeypatch)
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        # The session-only authorization still applies.
        feature.assert_granted(agent, scenario)
        assert any(
            expected in text and "applies to this session" in text
            for text in _tool_messages(agent)
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_always_without_a_save_transport_reports_it(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """Without a save transport the always outcome reports it cannot save."""
    scenario = feature.make_scenario(tmp_path, tool, "git push; true")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "bare")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        feature.assert_granted(agent, scenario)
        assert any(
            feature.cannot_save_fragment in text for text in _tool_messages(agent)
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_always_fetches_the_user_revision_at_decision_time(
    feature: GrantLifecycleFeature,
    tool: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user revision is read per decision and never cached across turns."""
    first, second = feature.make_revision_scenarios(tmp_path, tool)
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, first.args, "one")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, second.args, "two")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=first.project, backend=backend, user_input_capability=True
    )
    port = feature.make_save_port(
        agent, ConfigSaveResult("user", "saved", "unchanged", "rev")
    )
    feature.bind_port(agent, port)
    read = feature.patch_saved_surface(monkeypatch)
    try:
        await _drive(agent, "Always for this project")
        await _drive(agent, "Always for this project")
        # The revision is read per decision and never cached across turns.
        assert read.call_count == 2
        assert [save[-1] for save in port.saves] == ["user-rev-1", "user-rev-1"]
        feature.assert_revision_state(agent, port, (first, second), tmp_path)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_always_without_a_readable_saved_surface_skips_the_save(
    feature: GrantLifecycleFeature,
    tool: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable saved surface skips the save and says so."""
    scenario = feature.make_scenario(tmp_path, tool, "git push; true")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "unreadable")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    port = feature.make_save_port(
        agent, ConfigSaveResult("user", "saved", "unchanged", "rev")
    )
    feature.bind_port(agent, port)
    feature.patch_unavailable_surface(monkeypatch)
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        feature.assert_granted(agent, scenario)
        assert port.saves == []
        assert any("could not be read" in text for text in _tool_messages(agent))
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_session_grant_survives_compaction(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """A session grant survives compaction and stays effective."""
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "grant")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(content="<summary>summary</summary>")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        feature.assert_granted(agent, scenario)
        summary = await agent.compact()
        assert summary == "summary"
        feature.assert_granted(agent, scenario)
        feature.assert_grant_effective(agent, scenario)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BOTH_SHELL_TOOLS),
    ids=_combo_ids(_READ_FILE_ONLY, _BOTH_SHELL_TOOLS),
)
async def test_noninteractive_session_denies_with_remediation(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """A noninteractive session denies promptly with the config remediation."""
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "no-cap")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=False,
    )
    try:
        feature.bind_port(agent, feature.make_never_called_port(agent))
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert feature.config_knob in results[0].skip_reason
        assert "config.toml" in results[0].skip_reason
        feature.assert_no_authority(agent)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_bare_loop_without_a_port_keeps_the_plain_denial(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """A bare loop without a runtime port keeps the plain denial text."""
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "bare")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=False,
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        feature.assert_plain_denial(results[0].skip_reason)
        feature.assert_no_authority(agent)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BASH_ONLY),
    ids=_combo_ids(_READ_FILE_ONLY, _BASH_ONLY),
)
async def test_child_session_is_bounded_and_never_prompts(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """A child session never prompts on its own; it is bounded and denied."""
    scenario = feature.make_scenario(tmp_path, tool, "git push")
    parent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=FakeBackend(),
        user_input_capability=True,
    )
    child = AgentLoop(
        config_orchestrator=parent.config_orchestrator.copy(),
        cwd=scenario.project,
        backend=FakeBackend([
            [mock_llm_chunk(tool_calls=[_call(tool, scenario.args, "child-call")])],
            [mock_llm_chunk(content="done")],
        ]),
        is_subagent=True,
        inherited_workspace=feature.child_workspace(scenario.project, scenario),
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(child, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        feature.assert_child_bounded(results[0].skip_reason)
        feature.assert_no_authority(child)
    finally:
        await child.aclose()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("feature", "tool"),
    _combos(_READ_FILE_ONLY, _BOTH_SHELL_TOOLS),
    ids=_combo_ids(_READ_FILE_ONLY, _BOTH_SHELL_TOOLS),
)
async def test_concurrent_calls_coalesce_or_need_separate_approvals(
    feature: GrantLifecycleFeature, tool: str, tmp_path: Path
) -> None:
    """Two concurrent grant-needing calls: root prompts coalesce into one
    question and one grant; dangerous once approvals never coalesce.
    """
    scenario = feature.make_concurrent_scenario(tmp_path, tool)
    calls = [
        _call(name, args, f"concurrent-{index}", index)
        for index, (name, args) in enumerate(scenario.calls)
    ]
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=calls)],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(),
        cwd=scenario.project,
        backend=backend,
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(agent, feature.concurrent_answer)
        results = _results(events)
        assert len(results) == 2
        if feature.coalesces_concurrent_prompts:
            # Root prompts coalesce: one question, one grant covers both calls.
            assert len(requests) == 1
        else:
            # Dangerous once approvals never coalesce: one question per call.
            assert len(requests) == 2
            assert {request.request_id for request in requests} == {
                requests[0].request_id,
                requests[1].request_id,
            }
        for result in results:
            feature.assert_executed(result)
        feature.assert_concurrent_granted(agent, scenario)
    finally:
        await agent.aclose()
