"""Dangerous-command lifecycle — grants concern (stage 20c split target).

Grant lifecycle semantics: what an approval authorizes, scope binding and
isolation, grant integrity, the recheck and admission gates, and once-permit
cleanup. Moved here from ``test_dangerous_command_lifecycle.py`` in the
20c-ii-1 concern commit: allow-once, scope binding and isolation, forged
answers, duplicate callbacks, consequential and multi-repo grants, the
recheck and admission gates, once-permit cleanup, and their single-concern
helpers (``_patch_saved_grants``, ``_script_recheck``, ``_spy_invalidations``,
``_revoke_live_grants``, ``_drive_through_window``, the grant-port stubs, the
blocking snapshot, and the recording tool IO). The module re-registers the
marks the population had in the original — ``pytestmark =
pytest.mark.asyncio``; the original file defines no module-local fixtures,
so there is no fixture registration to carry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config._dangerous_command_persistence import SavedGrantsRead
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.events import BaseEvent, UserInputRequestEvent
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.io_port import ShellCommandResult
from chartreux.core.tools.permissions import PermissionContext
from chartreux.questions import UserAnswer, UserQuestionRequest, UserQuestionResult
from tests.agent_loop._lifecycle_test_support import (
    DECLINED_FEEDBACK,
    PROMPT_TEXT,
    _authorities,
    _bash_args,
    _call,
    _config,
    _drive,
    _results,
    make_repo,
    scope_of,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


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


class _WorldMutatingGrantPort:
    """Stub port that mutates the world inside the save await (TOCTOU probe)."""

    def __init__(self, agent: AgentLoop, mutate: Callable[[AgentLoop], None]) -> None:
        self._agent = agent
        self._mutate = mutate
        self.saves: list[tuple[str, Path, str, str]] = []

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        self.saves.append((session_id, project, operation_id, expected_revision))
        self._mutate(self._agent)
        return ConfigSaveResult("user", "saved", "unchanged", "rev-2")


class _BlockingGrantPort:
    """Stub port whose save await blocks until released (cancellation probe)."""

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        self._entered = entered
        self._release = release
        self.saves: list[tuple[str, Path, str, str]] = []

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        self.saves.append((session_id, project, operation_id, expected_revision))
        self._entered.set()
        await self._release.wait()
        return ConfigSaveResult("user", "saved", "unchanged", "rev-2")


def _patch_saved_grants(
    monkeypatch: pytest.MonkeyPatch, revision: str = "user-rev-1"
) -> AsyncMock:
    """Point the loop's saved-grants read surface at a controlled revision."""
    read = AsyncMock(
        return_value=SavedGrantsRead(operations=(), user_revision=revision)
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_dangerous_command_grants", read
    )
    return read


def _script_recheck(
    monkeypatch: pytest.MonkeyPatch, tool: Any, recheck: Callable[[Any], Any]
) -> list[int]:
    """Script only the post-approval recheck of a tool's resolution.

    The first resolution — the original denial that opens the approval
    prompt — stays real; every later resolution runs *recheck*, so the
    approval flow's full recheck is controlled without touching the denial
    that started it. Returns the resolution call counter.
    """
    real = tool.resolve_permission
    calls = [0]

    def resolve(args: Any) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            return real(args)
        return recheck(args)

    monkeypatch.setattr(tool, "resolve_permission", resolve)
    return calls


def _spy_invalidations(monkeypatch: pytest.MonkeyPatch, agent: AgentLoop) -> list[int]:
    """Count dangerous-command authority invalidations on the tool manager."""
    manager = agent.tool_manager
    real = manager._invalidate_dangerous_command_authority
    calls = [0]

    def invalidate() -> None:
        calls[0] += 1
        real()

    monkeypatch.setattr(manager, "_invalidate_dangerous_command_authority", invalidate)
    return calls


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_allow_once_resumes_exactly_one_original_call(
    tmp_path: Path, tool: str
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    command = f"git push; touch {marker.name}"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "once-1")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "once-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow once")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert not results[0].skipped
        if tool == "bash":
            assert marker.exists()
        # The once permit authorized exactly this invocation: no authority
        # remains, so the identical call prompts again instead of resuming.
        assert _authorities(agent) == []
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == DECLINED_FEEDBACK
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_session_grant_is_bound_to_the_equivalent_scope(tmp_path: Path) -> None:
    repo_a = make_repo(tmp_path / "a")
    make_repo(tmp_path / "b")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git -C a push"), "scope-1")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git -C b push"), "scope-2")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=tmp_path, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(repo_a))]
        # A sibling repository is a different scope: it still prompts.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(repo_a))]
    finally:
        await agent.aclose()


async def test_sibling_worktree_scope_is_isolated(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )
    worktree = project / "wt"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-q", str(worktree), "-b", "wt"],
        check=True,
    )
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git -C wt push"), "wt-1")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "proj-1")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(worktree))]
        # The main worktree is a different repository identity: it prompts.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
    finally:
        await agent.aclose()


async def test_forged_consequential_always_is_rejected(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git stash drop"), "forged")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        # The consequential class never offers the persistent choice.
        assert "Always for this project" not in {
            option.label for option in request.questions[0].options
        }
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == DECLINED_FEEDBACK
        # The forged answer installed no authority at all.
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_duplicate_callback_completion_cannot_execute_another_call(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "dup")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow once")
        assert len(requests) == 1
        assert len(_results(events)) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == []
        # A duplicate completion of the settled callback is a no-op: it cannot
        # install authority or execute another call.
        agent.resolve_user_input_request(
            requests[0].request_id,
            UserQuestionResult(
                answers=[UserAnswer(question="q", answer="Allow this session")]
            ),
        )
        assert _authorities(agent) == []
        assert len(_results(events)) == 1
    finally:
        await agent.aclose()


async def test_consequential_session_grant_does_not_authorize_publication(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git push --force"), "cons-1")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "cons-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [
            (frozenset({"git.push.force"}), scope_of(project))
        ]
        # Ordinary publication was never approved: it prompts again.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == [
            (frozenset({"git.push.force"}), scope_of(project))
        ]
    finally:
        await agent.aclose()


async def test_multi_repo_approval_covers_all_operations_and_scopes(
    tmp_path: Path,
) -> None:
    repo_a = make_repo(tmp_path / "a")
    repo_b = make_repo(tmp_path / "b")
    command = "git -C a push && git -C b checkout"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args(command), "multi")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=tmp_path, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        question = request.questions[0]
        # The proposal is complete (every operation), the scope representative.
        assert "git.push" in question.question
        assert "git.checkout" in question.question
        assert not _results(events)[0].skipped
        # The approval covers every operation at every scope: the recheck
        # passes only when both repositories carry their operation.
        assert set(_authorities(agent)) == {
            (frozenset({"git.push"}), scope_of(repo_a)),
            (frozenset({"git.checkout"}), scope_of(repo_b)),
        }
    finally:
        await agent.aclose()


class TestRecheckFailure:
    """A world change between the denial and the approval executes nothing."""

    async def test_new_denylist_entry_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call(
                            "bash",
                            _bash_args(f"git push; touch {marker.name}"),
                            "recheck",
                        )
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )

        async def add_denylist(event: UserInputRequestEvent) -> None:
            agent.config.tools.setdefault("bash", {})["denylist"] = ["git"]

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=add_denylist)
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "denylist" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_new_disablement_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def disable(event: UserInputRequestEvent) -> None:
            agent.config.tools.setdefault("bash", {})["permission"] = "never"

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=disable)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_new_sensitivity_rule_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def sensitize(event: UserInputRequestEvent) -> None:
            agent.config.tools.setdefault("bash", {})["sensitive_patterns"] = ["git"]

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=sensitize)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert "sensitive" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_new_path_restriction_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        sub = project / "sub"
        sub.mkdir()
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git -C sub push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def relocate(event: UserInputRequestEvent) -> None:
            shutil.rmtree(sub)
            sub.symlink_to("/etc")

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=relocate)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_changed_repo_identity_fails_the_recheck(
        self, tmp_path: Path
    ) -> None:
        project = make_repo(tmp_path / "project")
        subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "init",
            ],
            check=True,
        )
        worktree = project / "wt"
        subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "worktree",
                "add",
                "-q",
                str(worktree),
                "-b",
                "wt",
            ],
            check=True,
        )
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git -C wt push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def dissolve(event: UserInputRequestEvent) -> None:
            # The approved worktree identity no longer resolves to itself.
            (worktree / ".git").unlink()

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=dissolve)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_retired_manager_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def retire(event: UserInputRequestEvent) -> None:
            agent.tool_manager._authority_retired = True

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=retire)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert "retired" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()


class TestPersistAwaitAdmissionGate:
    """The persist await is a world-change window; the recheck after it gates.

    For "Always for this project" the approval flow persists BETWEEN its
    recheck and execution, so the world can change while the save runs. The
    full resolution runs again after the await — the final admission gate,
    exactly as the roots flow rechecks last — and a mid-await change skips
    the call instead of executing it.
    """

    async def test_denylist_added_inside_the_save_await_skips_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args(command), "toctou")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        def deny_git(loop: AgentLoop) -> None:
            loop.config.tools.setdefault("bash", {})["denylist"] = ["git"]

        port = _WorldMutatingGrantPort(agent, deny_git)
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            # git became denylisted during the save await: the post-await
            # recheck refuses the call and nothing executes.
            assert results[0].skipped
            assert "denylist" in (results[0].skip_reason or "")
            assert not marker.exists()
            # The approval still installed the live session grant and the
            # save still ran; only this invocation is refused.
            assert len(port.saves) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
        finally:
            await agent.aclose()

    async def test_retired_manager_inside_the_save_await_skips_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args(command), "retire")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        def retire(loop: AgentLoop) -> None:
            loop.tool_manager._authority_retired = True

        port = _WorldMutatingGrantPort(agent, retire)
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            # A mid-await manager retirement is a clean SKIP, not an
            # invoke-time error.
            assert results[0].skipped
            assert "retired" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert len(port.saves) == 1
        finally:
            await agent.aclose()


class _BlockingSnapshot:
    """Deterministically block the pre-invoke snapshot await, thread-side.

    The loop awaits ``get_file_snapshot`` through ``asyncio.to_thread``
    before invoking the tool, so blocking here holds the call inside the
    world-change window between the admission pre-check and execution.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, args: Any) -> Any:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10)
        return None


class _RecordingToolIO:
    """Client-terminal delegation port that records every shell request."""

    supports_terminal = True

    def __init__(self) -> None:
        self.shell_requests: list[Any] = []

    async def run_shell(self, request: Any) -> Any:
        self.shell_requests.append(request)
        return ShellCommandResult(stdout="host output", stderr="", returncode=0)


def _revoke_live_grants(agent: AgentLoop) -> None:
    """Revoke every live dangerous-command grant, as the endpoint does."""
    store = agent._dangerous_command_grants
    requirements = {
        (operation_id, authority.scope)
        for authority in store.dangerous_command_authorities()
        for operation_id in authority.operations
    }
    store.revoke_session(requirements)
    agent.tool_manager._invalidate_dangerous_command_authority()


async def _drive_through_window(
    agent: AgentLoop,
    answer: str | Callable[[UserInputRequestEvent], str],
    *,
    block: _BlockingSnapshot,
    mutate: Callable[[AgentLoop], None],
    prompt_text: str = PROMPT_TEXT,
    tool_io: Any = None,
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one turn; ``mutate`` runs while the snapshot await is blocked."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []

    async def consume() -> None:
        async for event in agent.act(prompt_text, tool_io=tool_io):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                label = answer(event) if callable(answer) else answer
                agent.resolve_user_input_request(
                    event.request_id,
                    UserQuestionResult(
                        answers=[UserAnswer(question="q", answer=label)]
                    ),
                )
            events.append(event)

    task = asyncio.create_task(consume())
    try:
        async with asyncio.timeout(10):
            while not block.entered.is_set():
                await asyncio.sleep(0.005)
            mutate(agent)
            block.release.set()
        await asyncio.wait_for(task, 10)
    finally:
        block.release.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    return events, requests


class TestExecutionAdmissionGate:
    """The snapshot await is a revocation window; the handoff gate blocks.

    The approval flow's final recheck returns EXECUTE, but the loop then
    awaits the tool's file snapshot before invoking it. Authority revoked
    while that await is blocked must be enforced at the execution handoff —
    after the snapshot, immediately before invoke — for every grant type,
    with the denial accounted as a rejected call and never re-prompted.
    """

    @pytest.mark.parametrize("tool", ["bash", "bash_start"])
    async def test_session_grant_revoked_in_the_window_blocks_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call(tool, _bash_args(command), "gate-sess")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get(tool), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent, "Allow this session", block=block, mutate=_revoke_live_grants
            )
            # The revoked grant is not re-prompted: the gate fails closed.
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            # A POLICY-class denial: the blocked requirement stated plainly,
            # with no alternative offered.
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            assert "Instead," not in (results[0].skip_reason or "")
            assert block.calls == 1
            assert not marker.exists()
            assert _authorities(agent) == []
            # The denial is accounted as a rejected call, not an execution.
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
            assert agent.stats.tool_calls_failed == 0
        finally:
            await agent.aclose()

    async def test_project_grant_revoked_in_the_window_blocks_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-proj")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(ConfigSaveResult("user", "saved", "unchanged", "rev-2"))
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent,
                "Always for this project",
                block=block,
                mutate=_revoke_live_grants,
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            assert not marker.exists()
            # The save still ran; only this invocation is refused.
            assert len(port.saves) == 1
            assert _authorities(agent) == []
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
        finally:
            await agent.aclose()

    async def test_previously_granted_call_revoked_in_the_window_blocks_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        first = project / "first.txt"
        second = project / "second.txt"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call(
                            "bash",
                            _bash_args(f"git push; touch {first.name}"),
                            "gate-prev-1",
                        )
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call(
                            "bash",
                            _bash_args(f"git push; touch {second.name}"),
                            "gate-prev-2",
                        )
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        try:
            # Turn 1: the session grant is approved and executes.
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert first.exists()
            # Turn 2: the operation is already granted, so no prompt arrives;
            # the revocation inside the snapshot window still blocks it.
            monkeypatch.setattr(
                agent.tool_manager.get("bash"), "get_file_snapshot", block
            )
            events, requests = await _drive_through_window(
                agent, "Deny", block=block, mutate=_revoke_live_grants
            )
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            assert not second.exists()
            assert first.exists()
            assert _authorities(agent) == []
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_succeeded == 1
        finally:
            await agent.aclose()

    @pytest.mark.parametrize(
        "answer", ["Allow once", "Allow this session", "Always for this project"]
    )
    async def test_a_still_valid_grant_executes_through_the_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-ok")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(ConfigSaveResult("user", "saved", "unchanged", "rev-2"))
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent, answer, block=block, mutate=lambda loop: None
            )
            # One approval, no re-prompt at the gate.
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            assert marker.exists()
            assert block.calls == 1
            assert agent.stats.tool_calls_agreed == 1
            assert agent.stats.tool_calls_rejected == 0
            assert agent.stats.tool_calls_succeeded == 1
            if answer == "Allow once":
                # The gate consumed the once authority for exactly this
                # invocation: nothing is reusable.
                assert _authorities(agent) == []
                assert agent._dangerous_command_grants._once == []
            else:
                # A session/project grant is not consumed by the gate.
                assert _authorities(agent) == [
                    (frozenset({"git.push"}), scope_of(project))
                ]
        finally:
            await agent.aclose()

    async def test_concurrent_identical_call_is_not_authorized_in_the_once_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call("bash", _bash_args(command), "once-a", 0),
                        _call("bash", _bash_args(command), "once-b", 1),
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        answers = iter(["Allow once", "Deny"])
        try:
            events, requests = await _drive_through_window(
                agent,
                lambda event: next(answers),
                block=block,
                mutate=lambda loop: None,
            )
            # Two prompts: the concurrent identical call did not inherit the
            # first call's once authority while its snapshot await was blocked.
            assert len(requests) == 2
            results = _results(events)
            assert len(results) == 2
            executed = [result for result in results if not result.skipped]
            denied = [result for result in results if result.skipped]
            assert len(executed) == 1
            assert len(denied) == 1
            assert denied[0].skip_reason == DECLINED_FEEDBACK
            assert marker.exists()
            # Only the authorized invocation reached the snapshot; the gate
            # consumed the once authority for exactly that invocation.
            assert block.calls == 1
            assert _authorities(agent) == []
            assert agent._dangerous_command_grants._once == []
            assert agent.stats.tool_calls_agreed == 1
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_succeeded == 1
        finally:
            await agent.aclose()

    async def test_client_delegation_gets_the_same_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-io")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        tool_io = _RecordingToolIO()
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent,
                "Allow this session",
                block=block,
                mutate=_revoke_live_grants,
                tool_io=tool_io,
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            # The delegated execution never ran.
            assert tool_io.shell_requests == []
            assert not marker.exists()
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
        finally:
            await agent.aclose()

    async def test_client_delegation_executes_when_the_grant_is_still_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        command = "git push"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-io-ok")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        tool_io = _RecordingToolIO()
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent,
                "Allow this session",
                block=block,
                mutate=lambda loop: None,
                tool_io=tool_io,
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            # The delegated execution ran exactly once through the gate.
            assert len(tool_io.shell_requests) == 1
            assert agent.stats.tool_calls_agreed == 1
            assert agent.stats.tool_calls_rejected == 0
            assert agent.stats.tool_calls_succeeded == 1
        finally:
            await agent.aclose()

    @pytest.mark.parametrize("answer", ["Allow once", "Allow this session"])
    @pytest.mark.parametrize(
        "kind, fragment",
        [
            ("disable", "disabled by policy"),
            ("denylist", "denylist"),
            ("retire", "retired"),
            ("sensitize", "sensitive"),
        ],
    )
    async def test_policy_revocation_in_the_window_blocks_execution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        answer: str,
        kind: str,
        fragment: str,
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-pol")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)

        def mutate(loop: AgentLoop) -> None:
            if kind == "disable":
                loop.config.tools.setdefault("bash", {})["permission"] = "never"
            elif kind == "denylist":
                loop.config.tools.setdefault("bash", {})["denylist"] = ["git"]
            elif kind == "retire":
                loop.tool_manager._authority_retired = True
            else:
                loop.config.tools.setdefault("bash", {})["sensitive_patterns"] = ["git"]

        try:
            events, requests = await _drive_through_window(
                agent, answer, block=block, mutate=mutate
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert fragment in (results[0].skip_reason or "")
            assert not marker.exists()
            assert block.calls == 1
            # The denial is accounted, and no once authority survived the
            # gate whether the revalidation passed or failed.
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
            assert agent.stats.tool_calls_failed == 0
            assert agent._dangerous_command_grants._once == []
        finally:
            await agent.aclose()

    async def test_ungranted_call_is_denied_at_the_gate_when_policy_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call("bash", _bash_args(f"touch {marker.name}"), "gate-plain")
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)

        def mutate(loop: AgentLoop) -> None:
            loop.config.tools.setdefault("bash", {})["denylist"] = ["touch"]

        try:
            events, requests = await _drive_through_window(
                agent, "Deny", block=block, mutate=mutate
            )
            # A routine call never prompts; the gate still enforces the
            # world change that happened inside the snapshot window.
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "denylist" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
        finally:
            await agent.aclose()


class TestOncePermitCleanup:
    """An installed once permit never outlives the resume, whatever exits it.

    The approval flow installs a once permit, fully re-resolves the original
    call, and only a resumed call consumes the permit. A resolver that
    raises, a NEVER recheck, or a cancellation must all leave the store
    without the permit: a leaked permit would authorize any later matching
    call without a new grant.
    """

    async def _bound_tool_and_args(
        self, agent: AgentLoop, command: str
    ) -> tuple[Any, Any]:
        tool = agent.tool_manager.get("bash")
        args = tool.validate_arguments({"command": command})
        return tool, args

    async def _answer_prompt(
        self, agent: AgentLoop, queue: asyncio.Queue, label: str
    ) -> None:
        event = await asyncio.wait_for(queue.get(), 5)
        assert isinstance(event, UserInputRequestEvent)
        agent.resolve_user_input_request(
            event.request_id,
            UserQuestionResult(answers=[UserAnswer(question="q", answer=label)]),
        )

    async def test_resolver_exception_after_install_once_destroys_the_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recheck that raises must not leak the installed once permit."""
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(
                agent, f"git push; touch {marker.name}"
            )
            generation_before = agent._dangerous_command_grants.authority_generation
            counts_at_raise: list[int] = []

            def explode(_args: Any) -> Any:
                counts_at_raise.append(invalidations[0])
                raise RuntimeError("resolver exploded during the recheck")

            _script_recheck(monkeypatch, tool, explode)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "explode")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            with pytest.raises(RuntimeError, match="resolver exploded"):
                await task
            # The permit installed for the approval did not survive the
            # failed recheck: the store holds no once permit and no authority.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            # Install bumped the generation once and the cleanup destroy
            # bumped it again; a leaked permit would leave it one bump short.
            assert (
                agent._dangerous_command_grants.authority_generation
                == generation_before + 2
            )
            # The cached authority the permit fed was invalidated after the
            # failure, not only when the permit was installed.
            assert counts_at_raise == [1]
            assert invalidations[0] == 2
            assert not marker.exists()
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_never_verdict_at_the_recheck_destroys_the_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recheck that denies executes nothing and leaves no permit."""
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(
                agent, f"git push; touch {marker.name}"
            )
            denial = PermissionContext(
                permission=ToolPermission.NEVER, reason="recheck denied"
            )
            _script_recheck(monkeypatch, tool, lambda _args: denial)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "recheck-never")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            decision = await asyncio.wait_for(task, 5)
            assert decision.verdict.name == "SKIP"
            assert decision.feedback == "recheck denied"
            # Nothing executed and no once authority survived the denial.
            assert not marker.exists()
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            assert invalidations[0] == 2
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_cancellation_at_the_recheck_destroys_the_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CancelledError is not an Exception: the finally still cleans up.

        The recheck itself is synchronous, so a task cancellation is
        delivered at an await; raising CancelledError at the scripted
        recheck reproduces exactly that delivery while the permit is live.
        """
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(agent, "git push")
            counts_at_cancel: list[int] = []

            def cancel(_args: Any) -> Any:
                counts_at_cancel.append(invalidations[0])
                raise asyncio.CancelledError()

            _script_recheck(monkeypatch, tool, cancel)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "recheck-cancel")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            with pytest.raises(asyncio.CancelledError):
                await task
            # The permit installed for the approval did not survive the
            # cancelled recheck: no once permit, no authority, no execution.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            assert counts_at_cancel == [1]
            assert invalidations[0] == 2
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_cancellation_during_the_persist_await_leaves_no_once_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The allow_project path installs no once permit, so cancelling the
        save await leaks no permit; the intentional session grant remains.
        """
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        save_entered = asyncio.Event()
        release = asyncio.Event()
        port = _BlockingGrantPort(save_entered, release)
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        try:
            tool, args = await self._bound_tool_and_args(
                agent, f"git push; touch {marker.name}"
            )
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "persist-cancel")
            )
            await self._answer_prompt(agent, queue, "Always for this project")
            await asyncio.wait_for(save_entered.wait(), 5)
            # The approval installed a session grant, never a once permit:
            # there is no permit to leak while the save await is suspended.
            assert agent._dangerous_command_grants._once == []
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Cleanup is total: still no once permit, and the session grant
            # is intentional and persistent — it is not rolled back.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            assert not marker.exists()
        finally:
            release.set()
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_matching_call_after_a_failed_recheck_needs_a_new_grant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression pin: a permit leaked by a failed recheck would
        authorize a later identical call without a new grant; it must not.
        """
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        try:
            tool, args = await self._bound_tool_and_args(agent, "git push")

            def explode(_args: Any) -> Any:
                raise RuntimeError("resolver exploded during the recheck")

            _script_recheck(monkeypatch, tool, explode)
            task = asyncio.create_task(agent._should_execute_tool(tool, args, "leak-1"))
            await self._answer_prompt(agent, queue, "Allow once")
            with pytest.raises(RuntimeError, match="resolver exploded"):
                await task
            assert agent._dangerous_command_grants._once == []
            # The leaked-permit scenario is over: the real resolver is back.
            monkeypatch.delattr(tool, "resolve_permission")
            # The identical call is denied again: with no residual authority
            # the resolver proposes a fresh approval instead of resuming, so
            # a prompt arrives (a leaked permit would skip it and execute).
            task = asyncio.create_task(agent._should_execute_tool(tool, args, "leak-2"))
            await self._answer_prompt(agent, queue, "Deny")
            decision = await asyncio.wait_for(task, 5)
            assert decision.verdict.name == "SKIP"
            assert decision.feedback == DECLINED_FEEDBACK
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_successful_allow_once_consumes_the_permit_before_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Normal path unchanged: the recheck passes, the permit is consumed
        before the call resumes, and exactly one execution is authorized.
        """
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(agent, "git push")
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "allow-once-ok")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            decision = await asyncio.wait_for(task, 5)
            assert decision.verdict.name == "EXECUTE"
            # The permit authorized exactly this invocation and is gone: no
            # authority remains for a later matching call.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            # Install and consume each invalidated once; the cleanup finally
            # added no extra invalidation on the success path.
            assert invalidations[0] == 2
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()
